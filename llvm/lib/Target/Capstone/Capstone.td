//===-- Capstone.td - Describe the Capstone Target Machine ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Target/Target.td"

//===----------------------------------------------------------------------===//
// Capstone subtarget features and instruction predicates.
//===----------------------------------------------------------------------===//

include "CapstoneFeatures.td"

//===----------------------------------------------------------------------===//
// Capstone profiles supported.
//===----------------------------------------------------------------------===//

include "CapstoneProfiles.td"

//===----------------------------------------------------------------------===//
// Named operands for CSR instructions.
//===----------------------------------------------------------------------===//

include "CapstoneSystemOperands.td"

//===----------------------------------------------------------------------===//
// Registers, calling conventions, instruction descriptions.
//===----------------------------------------------------------------------===//

include "CapstoneRegisterInfo.td"
include "CapstoneSchedule.td"
include "CapstoneCallingConv.td"
include "CapstoneInstrInfo.td"
include "GISel/CapstoneRegisterBanks.td"

//===----------------------------------------------------------------------===//
// Instruction predicates
//===----------------------------------------------------------------------===//

include "CapstoneInstrPredicates.td"

//===----------------------------------------------------------------------===//
// Capstone macro fusions.
//===----------------------------------------------------------------------===//

include "CapstoneMacroFusion.td"

//===----------------------------------------------------------------------===//
// Capstone Scheduling Models
//===----------------------------------------------------------------------===//
include "CapstoneSchedAndes45.td"
include "CapstoneSchedGenericOOO.td"
include "CapstoneSchedMIPSP8700.td"
include "CapstoneSchedRocket.td"
include "CapstoneSchedSiFive7.td"
include "CapstoneSchedSiFiveP400.td"
include "CapstoneSchedSiFiveP500.td"
include "CapstoneSchedSiFiveP600.td"
include "CapstoneSchedSiFiveP800.td"
include "CapstoneSchedSpacemitX60.td"
include "CapstoneSchedSyntacoreSCR1.td"
include "CapstoneSchedSyntacoreSCR345.td"
include "CapstoneSchedSyntacoreSCR7.td"
include "CapstoneSchedTTAscalonD8.td"
include "CapstoneSchedXiangShanNanHu.td"

//===----------------------------------------------------------------------===//
// Capstone processors supported.
//===----------------------------------------------------------------------===//

include "CapstoneProcessors.td"

//===----------------------------------------------------------------------===//
// Pfm Counters
//===----------------------------------------------------------------------===//

include "CapstonePfmCounters.td"

//===----------------------------------------------------------------------===//
// Define the Capstone target.
//===----------------------------------------------------------------------===//

def CapstoneInstrInfo : InstrInfo {
  let guessInstructionProperties = 0;
}

def CapstoneAsmParser : AsmParser {
  let ShouldEmitMatchRegisterAltName = 1;
  let AllowDuplicateRegisterNames = 1;
}

def CapstoneAsmWriter : AsmWriter {
  int PassSubtarget = 1;
}

def Capstone : Target {
  let InstructionSet = CapstoneInstrInfo;
  let AssemblyParsers = [CapstoneAsmParser];
  let AssemblyWriters = [CapstoneAsmWriter];
  let AllowRegisterRenaming = 1;
}
