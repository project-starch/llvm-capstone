//===- CapstoneSubtarget.cpp - Capstone Subtarget Information ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Capstone specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#include "CapstoneSubtarget.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;
// Pin the vtable to this file.
void CapstoneSubtarget::anchor() {}

CapstoneSubtarget::CapstoneSubtarget(const Triple &TT, StringRef CPU, StringRef FS,
                               const TargetMachine &TM)
    : TargetSubtargetInfo(TT, CPU, /*TuneCPU=*/"", FS, /*PN=*/{}, /*PF=*/{},
                          /*PD=*/{},
                          /*WPR=*/nullptr,
                          /*WL=*/nullptr,
                          /*RA=*/nullptr, /*IS=*/nullptr,
                          /*OC=*/nullptr, /*FP=*/nullptr),
      TLInfo(TM) {}
