//===-- MCTargetDesc/CapstoneMCAsmInfo.h - Capstone MCAsm Interface ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// Declaration of the Capstone MCAsmInfos.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_CAPSTONE_MCTARGETDESC_CAPSTONEMCASMINFO_H
#define LLVM_LIB_TARGET_CAPSTONE_MCTARGETDESC_CAPSTONEMCASMINFO_H

#include "llvm/MC/MCAsmInfoDarwin.h"
#include "llvm/MC/MCAsmInfoELF.h"
namespace llvm {

class Triple;

class CapstoneMCAsmInfoELF : public MCAsmInfoELF {
public:
  explicit CapstoneMCAsmInfoELF(const Triple &TT, const MCTargetOptions &Options);
};

class CapstoneMCAsmInfoDarwin : public MCAsmInfoDarwin {
public:
  explicit CapstoneMCAsmInfoDarwin(const Triple &TT,
                                const MCTargetOptions &Options);
};
} // namespace llvm
#endif
