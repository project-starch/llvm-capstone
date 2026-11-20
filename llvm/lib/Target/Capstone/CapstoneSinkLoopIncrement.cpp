//===-- CapstoneSinkLoopIncrement.cpp - Sink a loop increment below its base's last use -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A byte-copy loop leaves instruction selection as
//
//   %old = PHI %init, %bb.pre, %new, %bb.loop
//   %new = CIncOffsetImm %old, 1
//   SB %v, %old, 0
//   ... exit test on %new ...
//
// %old is still read after %new is made, so the two cannot share a register, and
// the register allocator closes every iteration with a copy (`movc a0, a4`). Plain
// RISC-V clang emits the same shape (`addi a4, a0, 1; sb a3, 0(a0); mv a0, a4`); it
// is generic LLVM, not Capstone's. Measured against GCC-built riscv-base images, such
// loop-carried copies ran +0.65 % of perlbench test's instructions and +0.60 % of
// xz test's (history 20-11-2026_18-39-20_c4-move-census-per-benchmark.md and the C4
// loop-copy census).
//
// The pass moves the increment below the last use of %old in its block, when no
// instruction in between reads %new. %old then dies where %new is born, the
// coalescer joins them with the PHI, and the copy goes. The instructions are the
// same, only their order changes, and no access is ever made through %new, so the
// cursor of %new on the last iteration (possibly at or past the object's end) is
// never dereferenced, exactly as before.
//
// Traps. CIncOffset and CIncOffsetImm raise UNEXPECTED_CAP_TYPE (26) on a sealed or
// uninitialised %old (CapstoneInstrInfo.cpp, trapsOnUntaggedOperand). Moving one
// below other instructions can change which instruction traps first, and so the
// trap's cause and PC. The pass therefore sinks a trapping increment only when the
// first instruction after it that is not pure (cannot trap, no side effect) is a
// LOAD whose base is %old itself:
//  - if %old is sealed or uninitialised, that load raises the same cause, 26: the
//    RTL rejects every type but linear and non-linear (load_store_unit.sv, the
//    cause-26 branch) and QEMU every type but linear, non-linear and a
//    synchronous sealed-return (op_helper.c, type_ok), and CIncOffset traps on
//    uninitialised and sealed alone. Only the trap's PC moves, to the load;
//  - if the load completes, %old was not uninitialised or sealed, so the
//    increment would not have trapped either, and nothing after the load can
//    observe the order.
// A STORE does not qualify: QEMU accepts a store through an uninitialised
// capability, so a sunk increment would let the store write and then trap. The
// RTL rejects that store with 26, so the two models differ there; until they
// agree, stores keep the order (-capstone-sink-loop-increment-past-stores lifts
// this, for measurement only). An ADDI cannot trap and is sunk below anything
// that does not read its result.
//
// Off by default (-capstone-enable-sink-loop-increment).
//
//===----------------------------------------------------------------------===//

#include "Capstone.h"
#include "CapstoneInstrInfo.h"
#include "CapstoneSubtarget.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "capstone-sink-loop-increment"
#define CAPSTONE_SINK_LOOP_INCREMENT_NAME "Capstone sink loop increment"

STATISTIC(NumSunk, "Loop increments sunk below their base's last use");
STATISTIC(NumKeptTrapOrder, "Trapping increments kept in place to preserve trap order");

// Measurement only: lets a store through the old base settle the trap order too
// (see the file comment for why that is not safe on QEMU's profile).
static cl::opt<bool> PastStores(
    "capstone-sink-loop-increment-past-stores", cl::Hidden, cl::init(false),
    cl::desc("Also sink a trapping loop increment below a store through its "
             "base (measurement only)"));

namespace {

class CapstoneSinkLoopIncrement : public MachineFunctionPass {
public:
  static char ID;
  CapstoneSinkLoopIncrement() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<MachineLoopInfoWrapperPass>();
    AU.addPreserved<MachineLoopInfoWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override {
    return CAPSTONE_SINK_LOOP_INCREMENT_NAME;
  }

private:
  const CapstoneInstrInfo *TII = nullptr;
  MachineRegisterInfo *MRI = nullptr;

  bool isPure(const MachineInstr &MI) const;
  bool sinkOne(MachineInstr &PHI, MachineLoop &L);
};

} // end anonymous namespace

char CapstoneSinkLoopIncrement::ID = 0;

INITIALIZE_PASS_BEGIN(CapstoneSinkLoopIncrement, DEBUG_TYPE,
                      CAPSTONE_SINK_LOOP_INCREMENT_NAME, false, false)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_END(CapstoneSinkLoopIncrement, DEBUG_TYPE,
                    CAPSTONE_SINK_LOOP_INCREMENT_NAME, false, false)

FunctionPass *llvm::createCapstoneSinkLoopIncrementPass() {
  return new CapstoneSinkLoopIncrement();
}

// Cannot trap and has no effect another instruction could observe.
bool CapstoneSinkLoopIncrement::isPure(const MachineInstr &MI) const {
  return !MI.mayLoadOrStore() && !MI.isCall() && !MI.isTerminator() &&
         !MI.hasUnmodeledSideEffects() && !MI.isPHI() &&
         !TII->canTrap(MI) && !MI.hasOrderedMemoryRef();
}

static bool isIncrement(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case Capstone::CIncOffsetImm:
  case Capstone::CIncOffset:
  case Capstone::ADDI:
    return true;
  default:
    return false;
  }
}

// A load (or, with PastStores, a store) addressed as `base, imm` with base Reg
// (operand 1, as every scalar and capability load and store of this target is
// laid out).
static bool isAccessThrough(const MachineInstr &MI, Register Reg) {
  if (MI.isCall() || MI.hasUnmodeledSideEffects() || MI.getNumOperands() < 3 ||
      !MI.getOperand(1).isReg() || MI.getOperand(1).getReg() != Reg ||
      !MI.getOperand(2).isImm())
    return false;
  if (MI.mayLoad() && !MI.mayStore())
    return true;
  return PastStores && MI.mayStore() && !MI.mayLoad();
}

bool CapstoneSinkLoopIncrement::sinkOne(MachineInstr &PHI, MachineLoop &L) {
  Register Old = PHI.getOperand(0).getReg();
  // The incoming value from inside the loop.
  Register New;
  for (unsigned I = 1, E = PHI.getNumOperands(); I + 1 < E; I += 2) {
    MachineBasicBlock *Pred = PHI.getOperand(I + 1).getMBB();
    if (L.contains(Pred)) {
      if (New.isValid())
        return false; // more than one latch value
      New = PHI.getOperand(I).getReg();
    }
  }
  if (!New.isValid() || !New.isVirtual() || !MRI->hasOneDef(New))
    return false;
  MachineInstr *Inc = MRI->getVRegDef(New);
  if (!Inc || !isIncrement(*Inc) || !Inc->getOperand(1).isReg() ||
      Inc->getOperand(1).getReg() != Old)
    return false;
  MachineBasicBlock *MBB = Inc->getParent();
  if (!L.contains(MBB))
    return false;

  // Every use of Old other than Inc must sit in Inc's block; the last one after
  // Inc is where Inc goes.
  MachineInstr *Last = nullptr;
  for (MachineInstr &U : MRI->use_nodbg_instructions(Old)) {
    if (&U == Inc)
      continue;
    if (U.getParent() != MBB)
      return false;
  }
  for (auto It = std::next(Inc->getIterator()); It != MBB->end(); ++It) {
    if (It->isDebugInstr())
      continue;
    if (It->isTerminator())
      break;
    for (const MachineOperand &MO : It->uses())
      if (MO.isReg() && MO.getReg() == Old)
        Last = &*It;
  }
  if (!Last)
    return false;

  // Nothing between Inc and Last may read New, and any other operand of Inc
  // (CIncOffset's offset register) is already defined above Inc.
  bool Trapping = TII->canTrap(*Inc);
  bool OrderSettled = !Trapping;
  for (auto It = std::next(Inc->getIterator());; ++It) {
    MachineInstr &MI = *It;
    if (!MI.isDebugInstr()) {
      for (const MachineOperand &MO : MI.uses())
        if (MO.isReg() && MO.getReg() == New)
          return false;
      if (!OrderSettled) {
        if (isAccessThrough(MI, Old))
          OrderSettled = true;
        else if (!isPure(MI)) {
          ++NumKeptTrapOrder;
          return false;
        }
      }
    }
    if (&MI == Last)
      break;
  }
  if (!OrderSettled) {
    ++NumKeptTrapOrder;
    return false;
  }

  MBB->splice(std::next(Last->getIterator()), MBB, Inc->getIterator());
  // Old no longer lives past Inc; kill flags on its earlier uses are now wrong
  // only in the direction of being conservative, but clear them all to be exact.
  MRI->clearKillFlags(Old);
  ++NumSunk;
  return true;
}

bool CapstoneSinkLoopIncrement::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;
  TII = MF.getSubtarget<CapstoneSubtarget>().getInstrInfo();
  MRI = &MF.getRegInfo();
  if (!MRI->isSSA())
    return false;
  MachineLoopInfo &MLI = getAnalysis<MachineLoopInfoWrapperPass>().getLI();

  bool Changed = false;
  SmallVector<MachineLoop *, 16> Work(MLI.begin(), MLI.end());
  while (!Work.empty()) {
    MachineLoop *L = Work.pop_back_val();
    Work.append(L->begin(), L->end());
    // Collected first: phis() ends at the first non-PHI, which can be the very
    // increment sinkOne moves.
    SmallVector<MachineInstr *, 8> PHIs;
    for (MachineInstr &PHI : L->getHeader()->phis())
      PHIs.push_back(&PHI);
    for (MachineInstr *PHI : PHIs)
      Changed |= sinkOne(*PHI, *L);
  }
  return Changed;
}
