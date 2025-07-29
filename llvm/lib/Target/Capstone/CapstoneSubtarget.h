//===- CapstoneSubtarget.h - Define Subtarget for the Capstone ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Capstone specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_CAPSTONE_CAPSTONESUBTARGET_H
#define LLVM_LIB_TARGET_CAPSTONE_CAPSTONESUBTARGET_H

#include "CapstoneISelLowering.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"

namespace llvm {

class TargetMachine;
class Triple;

class CapstoneSubtarget : public TargetSubtargetInfo {
  virtual void anchor();
  CapstoneTargetLowering TLInfo;

public:
  CapstoneSubtarget(const Triple &TT, StringRef CPU, StringRef FS,
                 const TargetMachine &TM);
  const CapstoneTargetLowering *getTargetLowering() const override {
    return &TLInfo;
  }

  /// Return the target's register information.
  const TargetRegisterInfo *getRegisterInfo() const override {
    return nullptr;
  }
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_CAPSTONE_CAPSTONESUBTARGET_H
