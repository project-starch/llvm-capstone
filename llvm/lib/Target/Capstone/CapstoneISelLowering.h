//===-- CapstoneISelLowering.h - Capstone DAG Lowering Interface --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that Capstone uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_CAPSTONE_CAPSTONEISELLOWERING_H
#define LLVM_LIB_TARGET_CAPSTONE_CAPSTONEISELLOWERING_H

#include "llvm/CodeGen/TargetLowering.h"

namespace llvm {

class CapstoneSubtarget;
class CapstoneTargetMachine;

class CapstoneTargetLowering : public TargetLowering {
public:
  explicit CapstoneTargetLowering(const TargetMachine &TM);
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_CAPSTONE_CAPSTONEISELLOWERING_H
