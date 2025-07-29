//==- CapstoneTargetTransformInfo.cpp - Capstone specific TTI pass -*- C++ -*-==//
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

#ifndef LLVM_LIB_TARGET_Capstone_CapstoneTARGETTRANSFORMINFO_H
#define LLVM_LIB_TARGET_Capstone_CapstoneTARGETTRANSFORMINFO_H

#include "CapstoneSubtarget.h"
#include "CapstoneTargetMachine.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/BasicTTIImpl.h"

namespace llvm {

class CapstoneTTIImpl : public BasicTTIImplBase<CapstoneTTIImpl> {
  using BaseT = BasicTTIImplBase<CapstoneTTIImpl>;
  using TTI = TargetTransformInfo;

  friend BaseT;

  // Supply the minimum required APIs.
  const CapstoneSubtarget &ST;
  const CapstoneTargetLowering &TLI;

  const CapstoneSubtarget *getST() const { return &ST; }
  const CapstoneTargetLowering *getTLI() const { return &TLI; }

public:
  explicit CapstoneTTIImpl(const CapstoneTargetMachine *TM, const Function &F)
      : BaseT(TM, F.getDataLayout()), ST(*TM->getSubtargetImpl(F)),
        TLI(*ST.getTargetLowering()) {}

  /// \name Vector TTI Implementations
  /// @{
  unsigned getLoadVectorFactor(unsigned VF, unsigned LoadSize,
                               unsigned ChainSizeInBytes,
                               VectorType *VecTy) const override;
  /// @}

  InstructionCost getIntrinsicInstrCost(const IntrinsicCostAttributes &ICA,
                                        TTI::TargetCostKind CostKind) const override;
};

} // end namespace llvm
#endif // LLVM_LIB_TARGET_Capstone_CapstoneTARGETTRANSFORMINFO_H
