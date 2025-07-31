//===- CapstoneTargetTransformInfo.cpp - Capstone specific TTI pass ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
/// \file
/// This file implements a TargetTransformInfo analysis pass specific to the
/// Capstone target machine. It uses the target's detailed information to provide
/// more precise answers to certain TTI queries, while letting the target
/// independent and default TTI implementations handle the rest.
///
//===----------------------------------------------------------------------===//

#include "CapstoneTargetTransformInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/IntrinsicsCapstone.h"

using namespace llvm;

#define DEBUG_TYPE "capstonetti"

unsigned CapstoneTTIImpl::getLoadVectorFactor(unsigned VF, unsigned LoadSize,
                                           unsigned ChainSizeInBytes,
                                           VectorType *VecTy) const {
  // We support <2 x i16> loads.
  unsigned ElemSize = VecTy->getScalarSizeInBits();
  if (ElemSize != 16)
    return 0;

  return std::min(VF, 2u);
}

InstructionCost
CapstoneTTIImpl::getIntrinsicInstrCost(const IntrinsicCostAttributes &ICA,
                                    TTI::TargetCostKind CostKind) const {
  // Extending the input values of a widening multiply is more expensive than a
  // regular instruction.
  // For code size, though, this is the same.
  if (CostKind != TargetTransformInfo::TCK_CodeSize &&
      ICA.getID() == Intrinsic::capstone_widening_smul)
    return TargetTransformInfo::TCC_Expensive;

  return BaseT::getIntrinsicInstrCost(ICA, CostKind);
}
