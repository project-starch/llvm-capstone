//===----- CapstoneZacasABIFix.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass implements a fence insertion for an atomic cmpxchg in a case that
// isn't easy to do with the current AtomicExpandPass hooks API.
//
//===----------------------------------------------------------------------===//

#include "Capstone.h"
#include "CapstoneTargetMachine.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"

using namespace llvm;

#define DEBUG_TYPE "capstone-zacas-abi-fix"
#define PASS_NAME "Capstone Zacas ABI fix"

namespace {

class CapstoneZacasABIFix : public FunctionPass,
                         public InstVisitor<CapstoneZacasABIFix, bool> {
  const CapstoneSubtarget *ST;

public:
  static char ID;

  CapstoneZacasABIFix() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override;

  StringRef getPassName() const override { return PASS_NAME; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<TargetPassConfig>();
  }

  bool visitInstruction(Instruction &I) { return false; }
  bool visitAtomicCmpXchgInst(AtomicCmpXchgInst &I);
};

} // end anonymous namespace

// Insert a leading fence (needed for broadest atomics ABI compatibility)
// only if the Zacas extension is enabled and the AtomicCmpXchgInst has a
// SequentiallyConsistent failure ordering.
bool CapstoneZacasABIFix::visitAtomicCmpXchgInst(AtomicCmpXchgInst &I) {
  assert(ST->hasStdExtZacas() && "only necessary to run in presence of zacas");
  IRBuilder<> Builder(&I);
  if (I.getFailureOrdering() != AtomicOrdering::SequentiallyConsistent)
    return false;

  Builder.CreateFence(AtomicOrdering::SequentiallyConsistent);
  return true;
}

bool CapstoneZacasABIFix::runOnFunction(Function &F) {
  auto &TPC = getAnalysis<TargetPassConfig>();
  auto &TM = TPC.getTM<CapstoneTargetMachine>();
  ST = &TM.getSubtarget<CapstoneSubtarget>(F);

  if (skipFunction(F) || !ST->hasStdExtZacas())
    return false;

  bool MadeChange = false;
  for (auto &BB : F)
    for (Instruction &I : llvm::make_early_inc_range(BB))
      MadeChange |= visit(I);

  return MadeChange;
}

INITIALIZE_PASS_BEGIN(CapstoneZacasABIFix, DEBUG_TYPE, PASS_NAME, false, false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(CapstoneZacasABIFix, DEBUG_TYPE, PASS_NAME, false, false)

char CapstoneZacasABIFix::ID = 0;

FunctionPass *llvm::createCapstoneZacasABIFixPass() {
  return new CapstoneZacasABIFix();
}
