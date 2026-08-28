//===- CapstoneS12MovcLdcHazard.cpp - Avoid the S-12 movc/ldc register match -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
//
// WORKAROUND for a SILICON defect (S-12) on the Capstone CVA6 FPGA. This is not a
// codegen fix: the code it rewrites is correct. It exists because the hardware
// mishandles one specific register relationship.
//
// STATUS: DEFAULT OFF. The pass is validated as SAFE (Capstone lit 58/58, byte-identical
// output when disabled against a pre-pass compiler, 126 renames in the SQLite amalgamation)
// but its EFFICACY IS NOT ESTABLISHED, and the mechanism below has been RETRACTED.
//
// THE RETRACTION, recorded here so nobody re-derives it: the claim was that a register match
// between a `movc rD, zero` and a later `ldc rD` is required. It is not. The three-byte patch
// that motivated this pass CURED while still containing that pairing -- `movc a4, zero` at
// 0x1047f0 writes a4, and a4 is undisturbed into `ldc a4` at 0x104810. The patch in fact
// severs TWO relations at once (the movc's adjacency to the ldc, and the source register of
// the adjacent stc) and cannot attribute between them. The supporting statistics were also
// wrong: Fisher exact is ~1e-3, not 1e-5, and no baseline draw existed in the slot
// configuration actually used.
//
// What survives is narrower: a `stc` of the same null capability, to the same address, at the
// same position and distance, sourced from a DIFFERENT register, did not wedge in 4/4 valid
// draws -- so the store's presence alone is not sufficient. This pass implements exactly that
// transform, which is why it is kept; it is NOT known to remove the fault.
//
// ENABLE IT ONLY with `-capstone-s12-movc-ldc-workaround=true`, and only after a slot-1 board
// arm shows the compiler-generated build returning against a same-configuration baseline.
//
// THE ORIGINAL RATIONALE, RETRACTED, follows. A `movc rD, zero`
// whose destination rD is subsequently the destination of an `ldc rD` feeding a
// capability consumer can leave the consumer reading movc's `{cursor 0, NOT_CAP}`
// instead of the loaded capability. The consumer then raises UNEXPECTED_OPERAND
// (mcause 25) with tval = 0 -- which is exactly movc's own value, and is what makes
// this diagnosable at all.
//
// THE EVIDENCE. Patching the faulting SQLite domain by THREE BYTES -- rewriting
// `movc a4, zero` and its dependent `stc a4, 0(a5)` to use a6, leaving the
// `ldc a4` untouched -- cured the fault 0 wedges in 4 draws, against a baseline of
// 14 confirmed + 1 unconfirmed wedge in 16 draws (~0.94/draw), p ~ 1.3e-5. Every
// address, the instruction count, the store-to-reload distance and the stored value
// were held identical, so the register match is the operative variable. The
// register's IDENTITY is irrelevant: the fault also occurs on a5.
//
// WHY RENAMING AND NOT PADDING. A 4-byte `nop` inserted at the same point, giving a
// byte-identical symbol table, does NOT cure it (wedged 4/4). A `fence rw,rw` does
// cure, but a fence is expensive and its effect could not be separated from timing.
// Renaming is the minimal change that removes the measured condition.
//
// SCOPE AND SAFETY. The transform is a pure register rename of a MOVC's destination
// and its uses up to the redefining LDC, using a register proven free over that
// range. It changes no addresses, no instruction count and no program semantics. If
// no free register exists, or rD is redefined or live-out in the window, the
// candidate is left alone -- the pass never inserts, deletes or reorders anything.
//
// This is a WORKAROUND. The RTL structure that mishandles the relationship has not
// been identified, so this must not be described as a fix, and it should be removed
// when the hardware is corrected. See
// capstone/tests/fpga-repros/S12-wherecode-notcap-operand-vs-memory/.
//
//===----------------------------------------------------------------------===//

#include "Capstone.h"
#include "CapstoneInstrInfo.h"
#include "CapstoneSubtarget.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "capstone-s12-movc-ldc"
#define CAPSTONE_S12_NAME "Capstone S-12 movc/ldc register-match workaround"

STATISTIC(NumRenamed, "Number of MOVC destinations renamed to avoid S-12");
STATISTIC(NumNoFreeReg, "S-12 candidates left alone: no free register");

static cl::opt<bool> EnableS12Workaround(
    "capstone-s12-movc-ldc-workaround", cl::Hidden, cl::init(false),
    cl::desc("Rename movc/ldc destination collisions associated with the S-12 "
             "silicon defect. DEFAULT OFF: the pass is validated as SAFE and inert "
             "when disabled, but its EFFICACY is not established -- see the header."));

// How far ahead of the MOVC to look for a redefining LDC. The measured window is 2
// instructions; this is deliberately wider so scheduling changes cannot hide the
// pattern, and narrow enough that renaming stays cheap.
static cl::opt<unsigned> S12Window(
    "capstone-s12-window", cl::Hidden, cl::init(16),
    cl::desc("Instructions to scan ahead of a movc for a redefining ldc"));

namespace {
class CapstoneS12MovcLdcHazard : public MachineFunctionPass {
public:
  static char ID;
  CapstoneS12MovcLdcHazard() : MachineFunctionPass(ID) {}
  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
  StringRef getPassName() const override { return CAPSTONE_S12_NAME; }

private:
  bool runOnBlock(MachineBasicBlock &MBB, const TargetRegisterInfo &TRI);
};
} // end anonymous namespace

char CapstoneS12MovcLdcHazard::ID = 0;
INITIALIZE_PASS(CapstoneS12MovcLdcHazard, DEBUG_TYPE, CAPSTONE_S12_NAME, false,
                false)

// Is this `movc rD, zero`? Only the zero form is implicated: the measured fault
// delivers cursor 0 with the tag clear, which is what MOVC from X0 produces.
static bool isMovcZero(const MachineInstr &MI) {
  return MI.getOpcode() == Capstone::MOVC && MI.getNumOperands() >= 2 &&
         MI.getOperand(1).isReg() && MI.getOperand(1).getReg() == Capstone::X0;
}

bool CapstoneS12MovcLdcHazard::runOnBlock(MachineBasicBlock &MBB,
                                          const TargetRegisterInfo &TRI) {
  bool Changed = false;

  for (auto MovcIt = MBB.begin(); MovcIt != MBB.end(); ++MovcIt) {
    if (!isMovcZero(*MovcIt) || !MovcIt->getOperand(0).isReg())
      continue;
    Register RD = MovcIt->getOperand(0).getReg();
    if (!RD.isPhysical() || RD == Capstone::X0)
      continue;

    // Find a redefining LDC within the window. Bail on anything that redefines RD
    // first, or on a terminator/call -- the rename must stay inside one straight
    // stretch where liveness is simple to reason about.
    MachineBasicBlock::iterator LdcIt = MBB.end();
    SmallVector<MachineInstr *, 8> Users;
    unsigned Steps = 0;
    for (auto It = std::next(MovcIt); It != MBB.end() && Steps < S12Window;
         ++It, ++Steps) {
      if (It->isCall() || It->isTerminator() || It->isInlineAsm())
        break;
      if (It->getOpcode() == Capstone::LDC && It->getNumOperands() >= 1 &&
          It->getOperand(0).isReg() && It->getOperand(0).getReg() == RD) {
        LdcIt = It;
        break;
      }
      if (It->modifiesRegister(RD, &TRI))
        break; // redefined by something else -- not our pattern
      if (It->readsRegister(RD, &TRI))
        Users.push_back(&*It);
    }
    if (LdcIt == MBB.end())
      continue;

    // Pick a replacement that is free across [movc, ldc]. LivePhysRegs stepped
    // backwards from the LDC gives exactly the set live at the point the rename
    // must survive to.
    LivePhysRegs LPR(TRI);
    LPR.addLiveOuts(MBB);
    for (auto It = MBB.rbegin(); It != MBB.rend(); ++It) {
      if (&*It == &*LdcIt)
        break;
      LPR.stepBackward(*It);
    }

    Register NewReg = Capstone::NoRegister;
    for (MCPhysReg Cand : {Capstone::X16, Capstone::X17, Capstone::X28,
                           Capstone::X29, Capstone::X30, Capstone::X31}) {
      if (Cand == RD || !LPR.available(MBB.getParent()->getRegInfo(), Cand))
        continue;
      // The candidate must also be untouched between the movc and the ldc.
      bool Clash = false;
      for (auto It = MovcIt; It != LdcIt; ++It)
        if (It->readsRegister(Cand, &TRI) || It->modifiesRegister(Cand, &TRI)) {
          Clash = true;
          break;
        }
      if (LdcIt->readsRegister(Cand, &TRI))
        Clash = true;
      if (!Clash) {
        NewReg = Cand;
        break;
      }
    }
    if (NewReg == Capstone::NoRegister) {
      ++NumNoFreeReg;
      continue;
    }

    MovcIt->getOperand(0).setReg(NewReg);
    for (MachineInstr *U : Users)
      for (MachineOperand &MO : U->operands())
        if (MO.isReg() && MO.getReg() == RD && !MO.isDef())
          MO.setReg(NewReg);

    ++NumRenamed;
    Changed = true;
  }
  return Changed;
}

bool CapstoneS12MovcLdcHazard::runOnMachineFunction(MachineFunction &MF) {
  // NOTE: deliberately NOT gated on skipFunction(). At -O0 clang marks every function
  // `optnone`, which makes skipFunction() true -- and the SQLite domain that exhibits
  // S-12 is built at -O0, so honouring it disabled the workaround exactly where the
  // fault lives. This is a correctness workaround for a silicon defect, not an
  // optimisation, so it must run at every optimisation level. Caught by the positive
  // control: the first version compiled, ran, and transformed nothing.
  if (!EnableS12Workaround)
    return false;
  const TargetRegisterInfo &TRI = *MF.getSubtarget().getRegisterInfo();
  bool Changed = false;
  for (MachineBasicBlock &MBB : MF)
    Changed |= runOnBlock(MBB, TRI);
  return Changed;
}

FunctionPass *llvm::createCapstoneS12MovcLdcHazardPass() {
  return new CapstoneS12MovcLdcHazard();
}
