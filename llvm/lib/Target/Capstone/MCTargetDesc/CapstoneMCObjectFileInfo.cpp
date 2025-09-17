//===-- CapstoneMCObjectFileInfo.cpp - Capstone object file properties ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the CapstoneMCObjectFileInfo properties.
//
//===----------------------------------------------------------------------===//

#include "CapstoneMCObjectFileInfo.h"
#include "CapstoneMCTargetDesc.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSubtargetInfo.h"

using namespace llvm;

unsigned
CapstoneMCObjectFileInfo::getTextSectionAlignment(const MCSubtargetInfo &STI) {
  return STI.hasFeature(Capstone::FeatureStdExtZca) ? 2 : 4;
}

unsigned CapstoneMCObjectFileInfo::getTextSectionAlignment() const {
  return getTextSectionAlignment(*getContext().getSubtargetInfo());
}
