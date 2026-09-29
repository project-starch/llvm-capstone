//===-- CapstoneSetAddressExpand.cpp - __intcap address replacement --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Expands PseudoSetAddr, the selection of llvm.capstone.cap.set.address, into a
// dispatch on the operand's type. __intcap arithmetic produces a value whose
// address is new and whose authority is the operand's, but the operand may hold
// a plain integer: `scc` alone would trap on it on the RTL (and on QEMU), and
// `cincoffset` likewise. So:
//
//   head:    t = lcc cap, 1           ; selector 1 is total: untagged -> 7
//            d = addi t, -1           ; NONLIN -> 0
//            bne d, x0, bridge
//   nonlin:  r1 = scc cap, addr       ; keep the authority, move the cursor
//            j done
//   bridge:  r2 = PseudoBRIDGE_CAP addr ; an untagged value holding the address
//   done:    r = phi [r1, nonlin], [r2, bridge]
//
// Anything that is not NONLIN takes the bridge: an integer (the common case, a
// Datum that holds a number), and also a sealed or linear capability. CHERI
// likewise clears the tag when the address of a sealed capability changes; a
// linear one never belongs in an __intcap under the Tier 4.1 contract.
//
// It runs before register allocation, after the SSA-level MachineLICM and
// MachineCSE, which therefore saw one side-effect-free pseudo. The scc it
// emits is in trapsOnUntaggedOperand, so the post-RA passes that move code
// (C-66) keep it behind its branch. It runs at every optimization level.
//
//===----------------------------------------------------------------------===//

#include "Capstone.h"
#include "CapstoneInstrInfo.h"
#include "CapstoneSubtarget.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

using namespace llvm;

#define DEBUG_TYPE "capstone-set-address-expand"
#define PASS_NAME "Capstone __intcap address replacement expansion"

STATISTIC(NumExpanded, "PseudoSetAddr expanded into a type dispatch");

// LCC selector 1 answers the type minus one; NONLIN is type 2.
static constexpr int64_t LccFieldType = 1;
static constexpr int64_t NonLinTypeValue = 1;

namespace {
class CapstoneSetAddressExpand : public MachineFunctionPass {
public:
  static char ID;
  CapstoneSetAddressExpand() : MachineFunctionPass(ID) {}
  StringRef getPassName() const override { return PASS_NAME; }
  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  void expand(MachineInstr &MI);
};
} // end anonymous namespace

char CapstoneSetAddressExpand::ID = 0;
INITIALIZE_PASS(CapstoneSetAddressExpand, DEBUG_TYPE, PASS_NAME, false, false)

FunctionPass *llvm::createCapstoneSetAddressExpandPass() {
  return new CapstoneSetAddressExpand();
}

void CapstoneSetAddressExpand::expand(MachineInstr &MI) {
  MachineBasicBlock *HeadMBB = MI.getParent();
  MachineFunction &MF = *HeadMBB->getParent();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const CapstoneInstrInfo &TII =
      *MF.getSubtarget<CapstoneSubtarget>().getInstrInfo();
  const DebugLoc &DL = MI.getDebugLoc();

  Register Dst = MI.getOperand(0).getReg();
  Register Cap = MI.getOperand(1).getReg();
  Register Addr = MI.getOperand(2).getReg();
  assert(Dst.isVirtual() && Cap.isVirtual() && Addr.isVirtual() &&
         "PseudoSetAddr is expanded before register allocation");

  // head | nonlin | bridge | done, with the rest of the block in done.
  const BasicBlock *BB = HeadMBB->getBasicBlock();
  MachineBasicBlock *NonLinMBB = MF.CreateMachineBasicBlock(BB);
  MachineBasicBlock *BridgeMBB = MF.CreateMachineBasicBlock(BB);
  MachineBasicBlock *DoneMBB = MF.CreateMachineBasicBlock(BB);
  MachineFunction::iterator It = std::next(HeadMBB->getIterator());
  MF.insert(It, NonLinMBB);
  MF.insert(It, BridgeMBB);
  MF.insert(It, DoneMBB);

  DoneMBB->splice(DoneMBB->begin(), HeadMBB, std::next(MI.getIterator()),
                  HeadMBB->end());
  DoneMBB->transferSuccessorsAndUpdatePHIs(HeadMBB);
  HeadMBB->addSuccessor(NonLinMBB);
  HeadMBB->addSuccessor(BridgeMBB);
  NonLinMBB->addSuccessor(DoneMBB);
  BridgeMBB->addSuccessor(DoneMBB);

  // The operands are now read in two blocks; neither read may claim a kill.
  MRI.clearKillFlags(Cap);
  MRI.clearKillFlags(Addr);

  const TargetRegisterClass *IntRC = &Capstone::GPRRegClass;
  const TargetRegisterClass *CapRC = MRI.getRegClass(Dst);
  Register TypeReg = MRI.createVirtualRegister(IntRC);
  Register DiffReg = MRI.createVirtualRegister(IntRC);
  BuildMI(*HeadMBB, MI, DL, TII.get(Capstone::LCC), TypeReg)
      .addReg(Cap)
      .addImm(LccFieldType);
  BuildMI(*HeadMBB, MI, DL, TII.get(Capstone::ADDI), DiffReg)
      .addReg(TypeReg)
      .addImm(-NonLinTypeValue);
  BuildMI(*HeadMBB, MI, DL, TII.get(Capstone::BNE))
      .addReg(DiffReg)
      .addReg(Capstone::X0)
      .addMBB(BridgeMBB);

  Register NonLinReg = MRI.createVirtualRegister(CapRC);
  BuildMI(NonLinMBB, DL, TII.get(Capstone::SCC), NonLinReg)
      .addReg(Cap)
      .addReg(Addr);
  BuildMI(NonLinMBB, DL, TII.get(Capstone::PseudoBR)).addMBB(DoneMBB);

  Register BridgeReg = MRI.createVirtualRegister(CapRC);
  BuildMI(BridgeMBB, DL, TII.get(Capstone::PseudoBRIDGE_CAP), BridgeReg)
      .addReg(Addr);

  BuildMI(*DoneMBB, DoneMBB->begin(), DL, TII.get(TargetOpcode::PHI), Dst)
      .addReg(NonLinReg)
      .addMBB(NonLinMBB)
      .addReg(BridgeReg)
      .addMBB(BridgeMBB);

  MI.eraseFromParent();
  ++NumExpanded;
}

bool CapstoneSetAddressExpand::runOnMachineFunction(MachineFunction &MF) {
  SmallVector<MachineInstr *, 8> Worklist;
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : MBB)
      if (MI.getOpcode() == Capstone::PseudoSetAddr)
        Worklist.push_back(&MI);
  for (MachineInstr *MI : Worklist)
    expand(*MI);
  return !Worklist.empty();
}
