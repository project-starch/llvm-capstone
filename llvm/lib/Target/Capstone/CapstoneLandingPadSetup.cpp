//===------------ CapstoneLandingPadSetup.cpp ---------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a Capstone pass to setup landing pad labels for indirect jumps.
// Currently this pass only supports fixed labels.
//
//===----------------------------------------------------------------------===//

#include "Capstone.h"
#include "CapstoneInstrInfo.h"
#include "CapstoneSubtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

using namespace llvm;

#define DEBUG_TYPE "capstone-lpad-setup"
#define PASS_NAME "Capstone Landing Pad Setup"

extern cl::opt<uint32_t> PreferredLandingPadLabel;

namespace {

class CapstoneLandingPadSetup : public MachineFunctionPass {
public:
  static char ID;

  CapstoneLandingPadSetup() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &F) override;

  StringRef getPassName() const override { return PASS_NAME; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

bool CapstoneLandingPadSetup::runOnMachineFunction(MachineFunction &MF) {
  const auto &STI = MF.getSubtarget<CapstoneSubtarget>();
  const CapstoneInstrInfo &TII = *STI.getInstrInfo();

  if (!STI.hasStdExtZicfilp())
    return false;

  uint32_t Label = 0;
  if (PreferredLandingPadLabel.getNumOccurrences() > 0) {
    if (!isUInt<20>(PreferredLandingPadLabel))
      report_fatal_error("capstone-landing-pad-label=<val>, <val> needs to fit in "
                         "unsigned 20-bits");
    Label = PreferredLandingPadLabel;
  }

  // Zicfilp does not check X7 if landing pad label is zero.
  if (Label == 0)
    return false;

  bool Changed = false;
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : llvm::make_early_inc_range(MBB)) {
      if (MI.getOpcode() != Capstone::PseudoBRINDNonX7 &&
          MI.getOpcode() != Capstone::PseudoCALLIndirectNonX7 &&
          MI.getOpcode() != Capstone::PseudoTAILIndirectNonX7)
        continue;
      BuildMI(MBB, MI, MI.getDebugLoc(), TII.get(Capstone::LUI), Capstone::X7)
          .addImm(Label);
      MachineInstrBuilder(MF, &MI).addUse(Capstone::X7, RegState::ImplicitKill);
      Changed = true;
    }

  return Changed;
}

INITIALIZE_PASS(CapstoneLandingPadSetup, DEBUG_TYPE, PASS_NAME, false, false)

char CapstoneLandingPadSetup::ID = 0;

FunctionPass *llvm::createCapstoneLandingPadSetupPass() {
  return new CapstoneLandingPadSetup();
}
