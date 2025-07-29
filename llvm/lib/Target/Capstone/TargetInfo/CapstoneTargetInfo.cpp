//===-- CapstoneTargetInfo.cpp - Capstone Target Implementation -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CapstoneTargetInfo.h"
#include "llvm/MC/TargetRegistry.h" // For RegisterTarget.
#include "llvm/Support/Compiler.h"  // For LLVM_EXTERNAL_VISIBILITY.
#include "llvm/TextAPI/Target.h"    // For Target class.

using namespace llvm;

Target &llvm::getTheCapstoneTarget() {
  static Target TheCapstoneTarget;
  return TheCapstoneTarget;
}

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeCapstoneTargetInfo() {
  RegisterTarget<Triple::capstone, /*HasJIT=*/false> X(
      getTheCapstoneTarget(), /*Name=*/"capstone",
      /*Desc=*/"Capstone project compiler",
      /*BackendName=*/"H2BLB");
}
