//===-- CapstoneMCObjectFileInfo.h - Capstone object file Info ------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the CapstoneMCObjectFileInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_Capstone_MCTARGETDESC_CapstoneMCOBJECTFILEINFO_H
#define LLVM_LIB_TARGET_Capstone_MCTARGETDESC_CapstoneMCOBJECTFILEINFO_H

#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"

namespace llvm {

class CapstoneMCObjectFileInfo : public MCObjectFileInfo {
public:
  static unsigned getTextSectionAlignment(const MCSubtargetInfo &STI);
  unsigned getTextSectionAlignment() const override;
};

} // namespace llvm

#endif
