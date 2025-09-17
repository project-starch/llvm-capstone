//===------- CapstonePushPopOptimizer.cpp - Capstone Push/Pop opt. pass --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a pass that replaces Zcmp POP instructions with
// POPRET[Z] where possible.
//
//===----------------------------------------------------------------------===//

#include "CapstoneInstrInfo.h"
#include "CapstoneMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineInstr.h"

using namespace llvm;

#define Capstone_PUSH_POP_OPT_NAME "Capstone Zcmp Push/Pop optimization pass"

namespace {
struct CapstonePushPopOpt : public MachineFunctionPass {
  static char ID;

  CapstonePushPopOpt() : MachineFunctionPass(ID) {}

  const CapstoneInstrInfo *TII;
  const TargetRegisterInfo *TRI;

  // Track which register units have been modified and used.
  LiveRegUnits ModifiedRegUnits, UsedRegUnits;

  bool usePopRet(MachineBasicBlock::iterator &MBBI,
                 MachineBasicBlock::iterator &NextI, bool IsReturnZero);
  bool adjustRetVal(MachineBasicBlock::iterator &MBBI);
  bool runOnMachineFunction(MachineFunction &Fn) override;

  StringRef getPassName() const override { return Capstone_PUSH_POP_OPT_NAME; }
};

char CapstonePushPopOpt::ID = 0;

} // end of anonymous namespace

INITIALIZE_PASS(CapstonePushPopOpt, "capstone-push-pop-opt", Capstone_PUSH_POP_OPT_NAME,
                false, false)

static bool isPop(unsigned Opcode) {
  switch (Opcode) {
  case Capstone::CM_POP:
  case Capstone::QC_CM_POP:
    return true;
  default:
    return false;
  }
}

static unsigned getPopRetOpcode(unsigned PopOpcode, bool IsReturnZero) {
  assert(isPop(PopOpcode) && "Unexpected Pop Opcode");

  switch (PopOpcode) {
  case Capstone::CM_POP:
    return IsReturnZero ? Capstone::CM_POPRETZ : Capstone::CM_POPRET;
  case Capstone::QC_CM_POP:
    return IsReturnZero ? Capstone::QC_CM_POPRETZ : Capstone::QC_CM_POPRET;
  default:
    llvm_unreachable("Unhandled Pop Opcode");
  }
}

bool CapstonePushPopOpt::usePopRet(MachineBasicBlock::iterator &MBBI,
                                MachineBasicBlock::iterator &NextI,
                                bool IsReturnZero) {
  // Since Pseudo instruction lowering happen later in the pipeline,
  // this will detect all ret instruction.
  DebugLoc DL = NextI->getDebugLoc();
  unsigned Opc = getPopRetOpcode(MBBI->getOpcode(), IsReturnZero);
  MachineInstrBuilder PopRetBuilder =
      BuildMI(*NextI->getParent(), NextI, DL, TII->get(Opc))
          .add(MBBI->getOperand(0))
          .add(MBBI->getOperand(1))
          .setMIFlag(MachineInstr::FrameDestroy);

  // Copy over the variable implicit uses and defs from the CM_POP. They depend
  // on what register list has been picked during frame lowering.
  const MCInstrDesc &PopDesc = MBBI->getDesc();
  unsigned FirstNonDeclaredOp = PopDesc.getNumOperands() +
                                PopDesc.NumImplicitUses +
                                PopDesc.NumImplicitDefs;
  for (unsigned i = FirstNonDeclaredOp; i < MBBI->getNumOperands(); ++i)
    PopRetBuilder.add(MBBI->getOperand(i));

  MBBI->eraseFromParent();
  NextI->eraseFromParent();
  return true;
}

// Search for last assignment to a0 and if possible use ret_val slot of POP to
// store return value.
bool CapstonePushPopOpt::adjustRetVal(MachineBasicBlock::iterator &MBBI) {
  MachineBasicBlock::reverse_iterator RE = MBBI->getParent()->rend();
  // Track which register units have been modified and used between the POP
  // insn and the last assignment to register a0.
  ModifiedRegUnits.clear();
  UsedRegUnits.clear();
  // Since POP instruction is in Epilogue no normal instructions will follow
  // after it. Therefore search only previous ones to find the return value.
  for (MachineBasicBlock::reverse_iterator I =
           next_nodbg(MBBI.getReverse(), RE);
       I != RE; I = next_nodbg(I, RE)) {
    MachineInstr &MI = *I;
    if (auto OperandPair = TII->isCopyInstrImpl(MI)) {
      Register DestReg = OperandPair->Destination->getReg();
      Register Source = OperandPair->Source->getReg();
      if (DestReg == Capstone::X10 && Source == Capstone::X0) {
        MI.removeFromParent();
        return true;
      }
    }
    // Update modified / used register units.
    LiveRegUnits::accumulateUsedDefed(MI, ModifiedRegUnits, UsedRegUnits, TRI);
    // If a0 was modified or used, there is no possibility
    // of using ret_val slot of POP instruction.
    if (!ModifiedRegUnits.available(Capstone::X10) ||
        !UsedRegUnits.available(Capstone::X10))
      return false;
  }
  return false;
}

bool CapstonePushPopOpt::runOnMachineFunction(MachineFunction &Fn) {
  if (skipFunction(Fn.getFunction()))
    return false;

  // If Zcmp extension is not supported, abort.
  const CapstoneSubtarget *Subtarget = &Fn.getSubtarget<CapstoneSubtarget>();
  if (!Subtarget->hasStdExtZcmp() && !Subtarget->hasVendorXqccmp())
    return false;

  TII = Subtarget->getInstrInfo();
  TRI = Subtarget->getRegisterInfo();

  // Resize the modified and used register unit trackers.  We do this once
  // per function and then clear the register units each time we determine
  // correct return value for the POP.
  ModifiedRegUnits.init(*TRI);
  UsedRegUnits.init(*TRI);

  bool Modified = false;
  for (auto &MBB : Fn) {
    // RET should be the only terminator.
    auto RetMBBI = MBB.getFirstTerminator();
    if (RetMBBI == MBB.end() || RetMBBI->getOpcode() != Capstone::PseudoRET ||
        RetMBBI == MBB.begin())
      continue;

    // The previous instruction should be a POP.
    auto PopMBBI = prev_nodbg(RetMBBI, MBB.begin());
    if (isPop(PopMBBI->getOpcode()) &&
        PopMBBI->getFlag(MachineInstr::FrameDestroy))
      Modified |= usePopRet(PopMBBI, RetMBBI, adjustRetVal(PopMBBI));
  }

  return Modified;
}

/// createCapstonePushPopOptimizationPass - returns an instance of the
/// Push/Pop optimization pass.
FunctionPass *llvm::createCapstonePushPopOptimizationPass() {
  return new CapstonePushPopOpt();
}
