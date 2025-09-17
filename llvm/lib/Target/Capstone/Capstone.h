//===-- Capstone.h - Top-level interface for Capstone ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in the LLVM
// Capstone back-end.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_Capstone_Capstone_H
#define LLVM_LIB_TARGET_Capstone_Capstone_H

#include "MCTargetDesc/CapstoneBaseInfo.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
class FunctionPass;
class InstructionSelector;
class PassRegistry;
class CapstoneRegisterBankInfo;
class CapstoneSubtarget;
class CapstoneTargetMachine;

FunctionPass *createCapstoneCodeGenPreparePass();
void initializeCapstoneCodeGenPreparePass(PassRegistry &);

FunctionPass *createCapstoneDeadRegisterDefinitionsPass();
void initializeCapstoneDeadRegisterDefinitionsPass(PassRegistry &);

FunctionPass *createCapstoneIndirectBranchTrackingPass();
void initializeCapstoneIndirectBranchTrackingPass(PassRegistry &);

FunctionPass *createCapstoneLandingPadSetupPass();
void initializeCapstoneLandingPadSetupPass(PassRegistry &);

FunctionPass *createCapstoneISelDag(CapstoneTargetMachine &TM,
                                 CodeGenOptLevel OptLevel);

FunctionPass *createCapstoneLateBranchOptPass();
void initializeCapstoneLateBranchOptPass(PassRegistry &);

FunctionPass *createCapstoneMakeCompressibleOptPass();
void initializeCapstoneMakeCompressibleOptPass(PassRegistry &);

FunctionPass *createCapstoneGatherScatterLoweringPass();
void initializeCapstoneGatherScatterLoweringPass(PassRegistry &);

FunctionPass *createCapstoneVectorPeepholePass();
void initializeCapstoneVectorPeepholePass(PassRegistry &);

FunctionPass *createCapstoneOptWInstrsPass();
void initializeCapstoneOptWInstrsPass(PassRegistry &);

FunctionPass *createCapstoneFoldMemOffsetPass();
void initializeCapstoneFoldMemOffsetPass(PassRegistry &);

FunctionPass *createCapstoneMergeBaseOffsetOptPass();
void initializeCapstoneMergeBaseOffsetOptPass(PassRegistry &);

FunctionPass *createCapstoneExpandPseudoPass();
void initializeCapstoneExpandPseudoPass(PassRegistry &);

FunctionPass *createCapstonePreRAExpandPseudoPass();
void initializeCapstonePreRAExpandPseudoPass(PassRegistry &);

FunctionPass *createCapstoneExpandAtomicPseudoPass();
void initializeCapstoneExpandAtomicPseudoPass(PassRegistry &);

FunctionPass *createCapstoneInsertVSETVLIPass();
void initializeCapstoneInsertVSETVLIPass(PassRegistry &);
extern char &CapstoneInsertVSETVLIID;

FunctionPass *createCapstonePostRAExpandPseudoPass();
void initializeCapstonePostRAExpandPseudoPass(PassRegistry &);
FunctionPass *createCapstoneInsertReadWriteCSRPass();
void initializeCapstoneInsertReadWriteCSRPass(PassRegistry &);

FunctionPass *createCapstoneInsertWriteVXRMPass();
void initializeCapstoneInsertWriteVXRMPass(PassRegistry &);

FunctionPass *createCapstoneRedundantCopyEliminationPass();
void initializeCapstoneRedundantCopyEliminationPass(PassRegistry &);

FunctionPass *createCapstoneMoveMergePass();
void initializeCapstoneMoveMergePass(PassRegistry &);

FunctionPass *createCapstonePushPopOptimizationPass();
void initializeCapstonePushPopOptPass(PassRegistry &);
FunctionPass *createCapstoneLoadStoreOptPass();
void initializeCapstoneLoadStoreOptPass(PassRegistry &);

FunctionPass *createCapstoneZacasABIFixPass();
void initializeCapstoneZacasABIFixPass(PassRegistry &);

InstructionSelector *
createCapstoneInstructionSelector(const CapstoneTargetMachine &,
                               const CapstoneSubtarget &,
                               const CapstoneRegisterBankInfo &);
void initializeCapstoneDAGToDAGISelLegacyPass(PassRegistry &);

FunctionPass *createCapstonePostLegalizerCombiner();
void initializeCapstonePostLegalizerCombinerPass(PassRegistry &);

FunctionPass *createCapstoneO0PreLegalizerCombiner();
void initializeCapstoneO0PreLegalizerCombinerPass(PassRegistry &);

FunctionPass *createCapstonePreLegalizerCombiner();
void initializeCapstonePreLegalizerCombinerPass(PassRegistry &);

FunctionPass *createCapstoneVLOptimizerPass();
void initializeCapstoneVLOptimizerPass(PassRegistry &);

FunctionPass *createCapstoneVMV0EliminationPass();
void initializeCapstoneVMV0EliminationPass(PassRegistry &);

void initializeCapstoneAsmPrinterPass(PassRegistry &);
} // namespace llvm

#endif
