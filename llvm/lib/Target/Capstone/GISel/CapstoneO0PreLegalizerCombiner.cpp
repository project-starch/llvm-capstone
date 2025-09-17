//=== CapstoneO0PreLegalizerCombiner.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass does combining of machine instructions at the generic MI level,
// before the legalizer.
//
//===----------------------------------------------------------------------===//

#include "CapstoneSubtarget.h"
#include "llvm/CodeGen/GlobalISel/Combiner.h"
#include "llvm/CodeGen/GlobalISel/CombinerHelper.h"
#include "llvm/CodeGen/GlobalISel/CombinerInfo.h"
#include "llvm/CodeGen/GlobalISel/GIMatchTableExecutorImpl.h"
#include "llvm/CodeGen/GlobalISel/GISelValueTracking.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/TargetPassConfig.h"

#define GET_GICOMBINER_DEPS
#include "CapstoneGenO0PreLegalizeGICombiner.inc"
#undef GET_GICOMBINER_DEPS

#define DEBUG_TYPE "capstone-O0-prelegalizer-combiner"

using namespace llvm;

namespace {
#define GET_GICOMBINER_TYPES
#include "CapstoneGenO0PreLegalizeGICombiner.inc"
#undef GET_GICOMBINER_TYPES

class CapstoneO0PreLegalizerCombinerImpl : public Combiner {
protected:
  const CombinerHelper Helper;
  const CapstoneO0PreLegalizerCombinerImplRuleConfig &RuleConfig;
  const CapstoneSubtarget &STI;

public:
  CapstoneO0PreLegalizerCombinerImpl(
      MachineFunction &MF, CombinerInfo &CInfo, const TargetPassConfig *TPC,
      GISelValueTracking &VT, GISelCSEInfo *CSEInfo,
      const CapstoneO0PreLegalizerCombinerImplRuleConfig &RuleConfig,
      const CapstoneSubtarget &STI);

  static const char *getName() { return "CapstoneO0PreLegalizerCombiner"; }

  bool tryCombineAll(MachineInstr &I) const override;

private:
#define GET_GICOMBINER_CLASS_MEMBERS
#include "CapstoneGenO0PreLegalizeGICombiner.inc"
#undef GET_GICOMBINER_CLASS_MEMBERS
};

#define GET_GICOMBINER_IMPL
#include "CapstoneGenO0PreLegalizeGICombiner.inc"
#undef GET_GICOMBINER_IMPL

CapstoneO0PreLegalizerCombinerImpl::CapstoneO0PreLegalizerCombinerImpl(
    MachineFunction &MF, CombinerInfo &CInfo, const TargetPassConfig *TPC,
    GISelValueTracking &VT, GISelCSEInfo *CSEInfo,
    const CapstoneO0PreLegalizerCombinerImplRuleConfig &RuleConfig,
    const CapstoneSubtarget &STI)
    : Combiner(MF, CInfo, TPC, &VT, CSEInfo),
      Helper(Observer, B, /*IsPreLegalize*/ true, &VT), RuleConfig(RuleConfig),
      STI(STI),
#define GET_GICOMBINER_CONSTRUCTOR_INITS
#include "CapstoneGenO0PreLegalizeGICombiner.inc"
#undef GET_GICOMBINER_CONSTRUCTOR_INITS
{
}

// Pass boilerplate
// ================

class CapstoneO0PreLegalizerCombiner : public MachineFunctionPass {
public:
  static char ID;

  CapstoneO0PreLegalizerCombiner();

  StringRef getPassName() const override {
    return "CapstoneO0PreLegalizerCombiner";
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;

private:
  CapstoneO0PreLegalizerCombinerImplRuleConfig RuleConfig;
};
} // end anonymous namespace

void CapstoneO0PreLegalizerCombiner::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<TargetPassConfig>();
  AU.setPreservesCFG();
  getSelectionDAGFallbackAnalysisUsage(AU);
  AU.addRequired<GISelValueTrackingAnalysisLegacy>();
  AU.addPreserved<GISelValueTrackingAnalysisLegacy>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

CapstoneO0PreLegalizerCombiner::CapstoneO0PreLegalizerCombiner()
    : MachineFunctionPass(ID) {
  if (!RuleConfig.parseCommandLineOption())
    report_fatal_error("Invalid rule identifier");
}

bool CapstoneO0PreLegalizerCombiner::runOnMachineFunction(MachineFunction &MF) {
  if (MF.getProperties().hasFailedISel())
    return false;
  auto &TPC = getAnalysis<TargetPassConfig>();

  const Function &F = MF.getFunction();
  GISelValueTracking *VT =
      &getAnalysis<GISelValueTrackingAnalysisLegacy>().get(MF);

  const CapstoneSubtarget &ST = MF.getSubtarget<CapstoneSubtarget>();

  CombinerInfo CInfo(/*AllowIllegalOps*/ true, /*ShouldLegalizeIllegal*/ false,
                     /*LegalizerInfo*/ nullptr, /*EnableOpt*/ false,
                     F.hasOptSize(), F.hasMinSize());
  // Disable fixed-point iteration in the Combiner. This improves compile-time
  // at the cost of possibly missing optimizations. See PR#94291 for details.
  CInfo.MaxIterations = 1;

  CapstoneO0PreLegalizerCombinerImpl Impl(MF, CInfo, &TPC, *VT,
                                       /*CSEInfo*/ nullptr, RuleConfig, ST);
  return Impl.combineMachineInstrs();
}

char CapstoneO0PreLegalizerCombiner::ID = 0;
INITIALIZE_PASS_BEGIN(CapstoneO0PreLegalizerCombiner, DEBUG_TYPE,
                      "Combine Capstone machine instrs before legalization",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_DEPENDENCY(GISelValueTrackingAnalysisLegacy)
INITIALIZE_PASS_DEPENDENCY(GISelCSEAnalysisWrapperPass)
INITIALIZE_PASS_END(CapstoneO0PreLegalizerCombiner, DEBUG_TYPE,
                    "Combine Capstone machine instrs before legalization", false,
                    false)

FunctionPass *llvm::createCapstoneO0PreLegalizerCombiner() {
  return new CapstoneO0PreLegalizerCombiner();
}
