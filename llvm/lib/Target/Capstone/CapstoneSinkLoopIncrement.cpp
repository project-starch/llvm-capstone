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
// below an access through %old would turn trap-before-read into read-then-trap on
// silicon: the RTL checks a scalar access's base type only in M-mode with
// capability mode on (load_store_unit.sv, the cap_violation_detection gate), so in a
// domain the load or store through a sealed or uninitialised base goes ahead
// (ISSUES Q-36, R-30, R-31). A trapping increment is therefore moved only when it
// cannot trap: when %old is PROVEN a linear or non-linear capability on every path,
// i.e. every value that reaches the loop PHI is derived in this function, through
// copies, offset increments, narrowings (SHRINK, SHRINKTO) and PHIs alone, from a stack slot or a global's base
// (PseudoCapGlobalBase), never through a load or an argument. An increment keeps
// the type, so such a base stays usable and the increment cannot raise 26. ADDI
// cannot trap at all. Every other trapping increment keeps its place.
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
STATISTIC(NumSunkProven, "Trapping increments sunk because their base is proven usable");
STATISTIC(NumKeptTrapOrder, "Trapping increments kept in place: base not proven usable");

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

  bool provenUsable(Register Reg, SmallPtrSetImpl<MachineInstr *> &Seen) const;
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

// Reg holds a linear or non-linear capability on every path: it is a stack slot's
// address or a global's base, or reaches one through copies, offset increments,
// narrowings and PHIs. A cycle through a PHI adds no new source, so a PHI already being visited
// counts as proven.
bool CapstoneSinkLoopIncrement::provenUsable(
    Register Reg, SmallPtrSetImpl<MachineInstr *> &Seen) const {
  if (!Reg.isVirtual())
    return false;
  MachineInstr *Def = MRI->getVRegDef(Reg);
  if (!Def)
    return false;
  switch (Def->getOpcode()) {
  case Capstone::PseudoCapGlobalBase:
    return true;
  case Capstone::CIncOffsetImm:
  case Capstone::CIncOffset:
  case Capstone::SHRINK:
  case Capstone::SHRINKTO:
    // An offset change or a narrowing keeps the type of a usable capability.
    if (Def->getOperand(1).isFI())
      return true;
    return Def->getOperand(1).isReg() &&
           provenUsable(Def->getOperand(1).getReg(), Seen);
  case TargetOpcode::COPY:
    return Def->getOperand(1).isReg() &&
           provenUsable(Def->getOperand(1).getReg(), Seen);
  case TargetOpcode::PHI:
    if (!Seen.insert(Def).second)
      return true;
    for (unsigned I = 1, E = Def->getNumOperands(); I < E; I += 2)
      if (!Def->getOperand(I).isReg() ||
          !provenUsable(Def->getOperand(I).getReg(), Seen))
        return false;
    return true;
  default:
    return false;
  }
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

  // A trapping increment moves only when its base is proven usable, so it
  // cannot trap (see the file comment).
  bool Trapping = TII->canTrap(*Inc);
  if (Trapping) {
    SmallPtrSet<MachineInstr *, 8> Seen;
    if (!provenUsable(Old, Seen)) {
      ++NumKeptTrapOrder;
      return false;
    }
  }
  // Nothing between Inc and Last may read New; any other operand of Inc
  // (CIncOffset's offset register) is already defined above Inc.
  for (auto It = std::next(Inc->getIterator());; ++It) {
    MachineInstr &MI = *It;
    if (!MI.isDebugInstr())
      for (const MachineOperand &MO : MI.uses())
        if (MO.isReg() && MO.getReg() == New)
          return false;
    if (&MI == Last)
      break;
  }

  MBB->splice(std::next(Last->getIterator()), MBB, Inc->getIterator());
  // Old no longer lives past Inc; kill flags on its earlier uses are now wrong
  // only in the direction of being conservative, but clear them all to be exact.
  MRI->clearKillFlags(Old);
  ++NumSunk;
  if (Trapping)
    ++NumSunkProven;
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
