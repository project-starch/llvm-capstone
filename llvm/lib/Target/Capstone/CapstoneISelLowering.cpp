//===-- CapstoneISelLowering.cpp - Capstone DAG Lowering Implementation -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the interfaces that Capstone uses to lower LLVM code
// into a selection DAG.
//
//===----------------------------------------------------------------------===//

#include "CapstoneISelLowering.h"
#include "CapstoneSubtarget.h"
#include "CapstoneTargetMachine.h"

using namespace llvm;

#define DEBUG_TYPE "capstone-lowering"

CapstoneTargetLowering::CapstoneTargetLowering(const TargetMachine &TM)
    : TargetLowering(TM) {}
