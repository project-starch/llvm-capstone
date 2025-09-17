//===-- CapstoneMoveMerger.cpp - Capstone move merge pass ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a pass that performs move related peephole optimizations
// as Zcmp has specified. This pass should be run after register allocation.
//
// This pass also supports Xqccmp, which has identical instructions.
//
//===----------------------------------------------------------------------===//

#include "CapstoneInstrInfo.h"
#include "CapstoneSubtarget.h"

using namespace llvm;

#define Capstone_MOVE_MERGE_NAME "Capstone Zcmp move merging pass"

namespace {
struct CapstoneMoveMerge : public MachineFunctionPass {
  static char ID;

  CapstoneMoveMerge() : MachineFunctionPass(ID) {}

  const CapstoneSubtarget *ST;
  const CapstoneInstrInfo *TII;
  const TargetRegisterInfo *TRI;

  // Track which register units have been modified and used.
  LiveRegUnits ModifiedRegUnits, UsedRegUnits;

  bool isCandidateToMergeMVA01S(const DestSourcePair &RegPair);
  bool isCandidateToMergeMVSA01(const DestSourcePair &RegPair);
  // Merge the two instructions indicated into a single pair instruction.
  MachineBasicBlock::iterator
  mergePairedInsns(MachineBasicBlock::iterator I,
                   MachineBasicBlock::iterator Paired, bool MoveFromSToA);

  // Look for C.MV instruction that can be combined with
  // the given instruction into CM.MVA01S or CM.MVSA01. Return the matching
  // instruction if one exists.
  MachineBasicBlock::iterator
  findMatchingInst(MachineBasicBlock::iterator &MBBI, bool MoveFromSToA,
                   const DestSourcePair &RegPair);
  bool mergeMoveSARegPair(MachineBasicBlock &MBB);
  bool runOnMachineFunction(MachineFunction &Fn) override;

  StringRef getPassName() const override { return Capstone_MOVE_MERGE_NAME; }
};

char CapstoneMoveMerge::ID = 0;

} // end of anonymous namespace

INITIALIZE_PASS(CapstoneMoveMerge, "capstone-move-merge", Capstone_MOVE_MERGE_NAME,
                false, false)

static unsigned getMoveFromSToAOpcode(const CapstoneSubtarget &ST) {
  if (ST.hasStdExtZcmp())
    return Capstone::CM_MVA01S;

  if (ST.hasVendorXqccmp())
    return Capstone::QC_CM_MVA01S;

  llvm_unreachable("Unhandled subtarget with paired A to S move.");
}

static unsigned getMoveFromAToSOpcode(const CapstoneSubtarget &ST) {
  if (ST.hasStdExtZcmp())
    return Capstone::CM_MVSA01;

  if (ST.hasVendorXqccmp())
    return Capstone::QC_CM_MVSA01;

  llvm_unreachable("Unhandled subtarget with paired S to A move");
}

// Check if registers meet CM.MVA01S constraints.
bool CapstoneMoveMerge::isCandidateToMergeMVA01S(const DestSourcePair &RegPair) {
  Register Destination = RegPair.Destination->getReg();
  Register Source = RegPair.Source->getReg();
  // If destination is not a0 or a1.
  if ((Destination == Capstone::X10 || Destination == Capstone::X11) &&
      Capstone::SR07RegClass.contains(Source))
    return true;
  return false;
}

// Check if registers meet CM.MVSA01 constraints.
bool CapstoneMoveMerge::isCandidateToMergeMVSA01(const DestSourcePair &RegPair) {
  Register Destination = RegPair.Destination->getReg();
  Register Source = RegPair.Source->getReg();
  // If Source is s0 - s7.
  if ((Source == Capstone::X10 || Source == Capstone::X11) &&
      Capstone::SR07RegClass.contains(Destination))
    return true;
  return false;
}

MachineBasicBlock::iterator
CapstoneMoveMerge::mergePairedInsns(MachineBasicBlock::iterator I,
                                 MachineBasicBlock::iterator Paired,
                                 bool MoveFromSToA) {
  const MachineOperand *Sreg1, *Sreg2;
  MachineBasicBlock::iterator E = I->getParent()->end();
  MachineBasicBlock::iterator NextI = next_nodbg(I, E);
  DestSourcePair FirstPair = TII->isCopyInstrImpl(*I).value();
  DestSourcePair PairedRegs = TII->isCopyInstrImpl(*Paired).value();

  if (NextI == Paired)
    NextI = next_nodbg(NextI, E);
  DebugLoc DL = I->getDebugLoc();

  // Make a copy so we can update the kill flag in the MoveFromSToA case. The
  // copied operand needs to be scoped outside the if since we make a pointer
  // to it.
  MachineOperand PairedSource = *PairedRegs.Source;

  // The order of S-reg depends on which instruction holds A0, instead of
  // the order of register pair.
  // e,g.
  //   mv a1, s1
  //   mv a0, s2    =>  cm.mva01s s2,s1
  //
  //   mv a0, s2
  //   mv a1, s1    =>  cm.mva01s s2,s1
  unsigned Opcode;
  if (MoveFromSToA) {
    // We are moving one of the copies earlier so its kill flag may become
    // invalid. Clear the copied kill flag if there are any reads of the
    // register between the new location and the old location.
    for (auto It = std::next(I); It != Paired && PairedSource.isKill(); ++It)
      if (It->readsRegister(PairedSource.getReg(), TRI))
        PairedSource.setIsKill(false);

    Opcode = getMoveFromSToAOpcode(*ST);
    Sreg1 = FirstPair.Source;
    Sreg2 = &PairedSource;
    if (FirstPair.Destination->getReg() != Capstone::X10)
      std::swap(Sreg1, Sreg2);
  } else {
    Opcode = getMoveFromAToSOpcode(*ST);
    Sreg1 = FirstPair.Destination;
    Sreg2 = PairedRegs.Destination;
    if (FirstPair.Source->getReg() != Capstone::X10)
      std::swap(Sreg1, Sreg2);
  }

  BuildMI(*I->getParent(), I, DL, TII->get(Opcode)).add(*Sreg1).add(*Sreg2);

  I->eraseFromParent();
  Paired->eraseFromParent();
  return NextI;
}

MachineBasicBlock::iterator
CapstoneMoveMerge::findMatchingInst(MachineBasicBlock::iterator &MBBI,
                                 bool MoveFromSToA,
                                 const DestSourcePair &RegPair) {
  MachineBasicBlock::iterator E = MBBI->getParent()->end();

  // Track which register units have been modified and used between the first
  // insn and the second insn.
  ModifiedRegUnits.clear();
  UsedRegUnits.clear();

  for (MachineBasicBlock::iterator I = next_nodbg(MBBI, E); I != E;
       I = next_nodbg(I, E)) {

    MachineInstr &MI = *I;

    if (auto SecondPair = TII->isCopyInstrImpl(MI)) {
      Register SourceReg = SecondPair->Source->getReg();
      Register DestReg = SecondPair->Destination->getReg();

      bool IsCandidate = MoveFromSToA ? isCandidateToMergeMVA01S(*SecondPair)
                                      : isCandidateToMergeMVSA01(*SecondPair);
      if (IsCandidate) {
        // Second destination must be different.
        if (RegPair.Destination->getReg() == DestReg)
          return E;

        // For AtoS the source must also be different.
        if (!MoveFromSToA && RegPair.Source->getReg() == SourceReg)
          return E;

        // If paired destination register was modified or used, the source reg
        // was modified, there is no possibility of finding matching
        // instruction so exit early.
        if (!ModifiedRegUnits.available(DestReg) ||
            !UsedRegUnits.available(DestReg) ||
            !ModifiedRegUnits.available(SourceReg))
          return E;

        return I;
      }
    }
    // Update modified / used register units.
    LiveRegUnits::accumulateUsedDefed(MI, ModifiedRegUnits, UsedRegUnits, TRI);
  }
  return E;
}

// Finds instructions, which could be represented as C.MV instructions and
// merged into CM.MVA01S or CM.MVSA01.
bool CapstoneMoveMerge::mergeMoveSARegPair(MachineBasicBlock &MBB) {
  bool Modified = false;

  for (MachineBasicBlock::iterator MBBI = MBB.begin(), E = MBB.end();
       MBBI != E;) {
    // Check if the instruction can be compressed to C.MV instruction. If it
    // can, return Dest/Src register pair.
    auto RegPair = TII->isCopyInstrImpl(*MBBI);
    if (RegPair.has_value()) {
      bool MoveFromSToA = isCandidateToMergeMVA01S(*RegPair);
      if (!MoveFromSToA && !isCandidateToMergeMVSA01(*RegPair)) {
        ++MBBI;
        continue;
      }

      MachineBasicBlock::iterator Paired =
          findMatchingInst(MBBI, MoveFromSToA, RegPair.value());
      // If matching instruction can be found merge them.
      if (Paired != E) {
        MBBI = mergePairedInsns(MBBI, Paired, MoveFromSToA);
        Modified = true;
        continue;
      }
    }
    ++MBBI;
  }
  return Modified;
}

bool CapstoneMoveMerge::runOnMachineFunction(MachineFunction &Fn) {
  if (skipFunction(Fn.getFunction()))
    return false;

  ST = &Fn.getSubtarget<CapstoneSubtarget>();
  if (!ST->hasStdExtZcmp() && !ST->hasVendorXqccmp())
    return false;

  TII = ST->getInstrInfo();
  TRI = ST->getRegisterInfo();
  // Resize the modified and used register unit trackers.  We do this once
  // per function and then clear the register units each time we optimize a
  // move.
  ModifiedRegUnits.init(*TRI);
  UsedRegUnits.init(*TRI);
  bool Modified = false;
  for (auto &MBB : Fn)
    Modified |= mergeMoveSARegPair(MBB);
  return Modified;
}

/// createCapstoneMoveMergePass - returns an instance of the
/// move merge pass.
FunctionPass *llvm::createCapstoneMoveMergePass() { return new CapstoneMoveMerge(); }
