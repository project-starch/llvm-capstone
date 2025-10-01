//===-- CapstoneTargetInfo.cpp - Capstone Target Implementation ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TargetInfo/CapstoneTargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Compiler.h"
using namespace llvm;

Target &llvm::getTheCapstone32Target() {
  static Target TheCapstone32Target;
  return TheCapstone32Target;
}

Target &llvm::getTheCapstone64Target() {
  static Target TheCapstone64Target;
  return TheCapstone64Target;
}

Target &llvm::getTheCapstone32beTarget() {
  static Target TheCapstone32beTarget;
  return TheCapstone32beTarget;
}

Target &llvm::getTheCapstone64beTarget() {
  static Target TheCapstone64beTarget;
  return TheCapstone64beTarget;
}

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void
LLVMInitializeCapstoneTargetInfo() {
  RegisterTarget<Triple::capstone32, /*HasJIT=*/true> X(
      getTheCapstone32Target(), "capstone32", "32-bit Capstone", "Capstone");
  RegisterTarget<Triple::capstone64, /*HasJIT=*/true> Y(
      getTheCapstone64Target(), "capstone64", "64-bit Capstone", "Capstone");
  // RegisterTarget<Triple::capstone32be> A(getTheCapstone32beTarget(), "capstone32be",
  //                                     "32-bit big endian Capstone", "Capstone");
  // RegisterTarget<Triple::capstone64be> B(getTheCapstone64beTarget(), "capstone64be",
  //                                     "64-bit big endian Capstone", "Capstone");
}
