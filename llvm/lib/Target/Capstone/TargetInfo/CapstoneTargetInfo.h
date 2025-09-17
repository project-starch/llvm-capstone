//===-- CapstoneTargetInfo.h - Capstone Target Implementation --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_Capstone_TARGETINFO_CapstoneTARGETINFO_H
#define LLVM_LIB_TARGET_Capstone_TARGETINFO_CapstoneTARGETINFO_H

namespace llvm {

class Target;

Target &getTheCapstone32Target();
Target &getTheCapstone64Target();
Target &getTheCapstone32beTarget();
Target &getTheCapstone64beTarget();

} // namespace llvm

#endif // LLVM_LIB_TARGET_Capstone_TARGETINFO_CapstoneTARGETINFO_H
