//===-- Capstone.td - Describe the Capstone Target Machine ------*- tablegen -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level entry point for the Capstone target.
//
//===----------------------------------------------------------------------===//

// Include the generic target definitions.
include "llvm/Target/Target.td"

def : ProcessorModel<"generic", NoSchedModel, /*Features=*/[]>;

def CapstoneInstrInfo : InstrInfo;

def Capstone : Target {
  let InstructionSet = CapstoneInstrInfo;
}
