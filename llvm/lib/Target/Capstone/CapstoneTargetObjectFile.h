//===-- CapstoneTargetObjectFile.h - Capstone Object Info -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_Capstone_CapstoneTARGETOBJECTFILE_H
#define LLVM_LIB_TARGET_Capstone_CapstoneTARGETOBJECTFILE_H

#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"

namespace llvm {

/// This implementation is used for Capstone ELF targets (Linux in particular).
class Capstone_ELFTargetObjectFile : public TargetLoweringObjectFileELF {
public:
  Capstone_ELFTargetObjectFile();
};

/// This TLOF implementation is used for Darwin.
class Capstone_MachoTargetObjectFile : public TargetLoweringObjectFileMachO {
public:
  Capstone_MachoTargetObjectFile();
};

} // end namespace llvm

#endif
