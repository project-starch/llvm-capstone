//===-- CapstoneLateBranchOpt.cpp - Late Stage Branch Optimization -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file provides Capstone specific target optimizations, currently it's
/// limited to convert conditional branches into unconditional branches when
/// the condition can be statically evaluated.
///
//===----------------------------------------------------------------------===//

#include "CapstoneInstrInfo.h"
#include "CapstoneSubtarget.h"

using namespace llvm;

#define Capstone_LATE_BRANCH_OPT_NAME "Capstone Late Branch Optimisation Pass"

namespace {

struct CapstoneLateBranchOpt : public MachineFunctionPass {
  static char ID;

  CapstoneLateBranchOpt() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return Capstone_LATE_BRANCH_OPT_NAME; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &Fn) override;

private:
  bool runOnBasicBlock(MachineBasicBlock &MBB) const;

  const CapstoneInstrInfo *RII = nullptr;
};
} // namespace

char CapstoneLateBranchOpt::ID = 0;
INITIALIZE_PASS(CapstoneLateBranchOpt, "capstone-late-branch-opt",
                Capstone_LATE_BRANCH_OPT_NAME, false, false)

bool CapstoneLateBranchOpt::runOnBasicBlock(MachineBasicBlock &MBB) const {
  MachineBasicBlock *TBB, *FBB;
  SmallVector<MachineOperand, 4> Cond;
  if (RII->analyzeBranch(MBB, TBB, FBB, Cond, /*AllowModify=*/false))
    return false;

  if (!TBB || Cond.size() != 3)
    return false;

  CapstoneCC::CondCode CC = CapstoneInstrInfo::getCondFromBranchOpc(Cond[0].getImm());
  assert(CC != CapstoneCC::COND_INVALID);

  MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();

  // Try and convert a conditional branch that can be evaluated statically
  // into an unconditional branch.
  int64_t C0, C1;
  if (!CapstoneInstrInfo::isFromLoadImm(MRI, Cond[1], C0) ||
      !CapstoneInstrInfo::isFromLoadImm(MRI, Cond[2], C1))
    return false;

  MachineBasicBlock *Folded =
      CapstoneInstrInfo::evaluateCondBranch(CC, C0, C1) ? TBB : FBB;

  // At this point, its legal to optimize.
  RII->removeBranch(MBB);

  // Only need to insert a branch if we're not falling through.
  if (Folded) {
    DebugLoc DL = MBB.findBranchDebugLoc();
    RII->insertBranch(MBB, Folded, nullptr, {}, DL);
  }

  // Update the successors. Remove them all and add back the correct one.
  while (!MBB.succ_empty())
    MBB.removeSuccessor(MBB.succ_end() - 1);

  // If it's a fallthrough, we need to figure out where MBB is going.
  if (!Folded) {
    MachineFunction::iterator Fallthrough = ++MBB.getIterator();
    if (Fallthrough != MBB.getParent()->end())
      MBB.addSuccessor(&*Fallthrough);
  } else
    MBB.addSuccessor(Folded);

  return true;
}

bool CapstoneLateBranchOpt::runOnMachineFunction(MachineFunction &Fn) {
  if (skipFunction(Fn.getFunction()))
    return false;

  auto &ST = Fn.getSubtarget<CapstoneSubtarget>();
  RII = ST.getInstrInfo();

  bool Changed = false;
  for (MachineBasicBlock &MBB : Fn)
    Changed |= runOnBasicBlock(MBB);
  return Changed;
}

FunctionPass *llvm::createCapstoneLateBranchOptPass() {
  return new CapstoneLateBranchOpt();
}
