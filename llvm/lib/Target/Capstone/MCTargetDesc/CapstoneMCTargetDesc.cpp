//===-- CapstoneMCTargetDesc.cpp - Capstone Target Descriptions -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides Capstone specific target descriptions.
//
//===----------------------------------------------------------------------===//

#include "CapstoneMCTargetDesc.h"
#include "CapstoneMCAsmInfo.h"
#include "TargetInfo/CapstoneTargetInfo.h" // For getTheCapstoneTarget.
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Compiler.h"  // For LLVM_EXTERNAL_VISIBILITY.
#include "llvm/Support/ErrorHandling.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;

#define GET_SUBTARGETINFO_MC_DESC
#include "CapstoneGenSubtargetInfo.inc"

static MCRegisterInfo *createCapstoneMCRegisterInfo(const Triple &Triple) {
  MCRegisterInfo *X = new MCRegisterInfo();
  // TODO: Fill out the register info.
  return X;
}

static MCInstrInfo *createCapstoneMCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  // TODO: Fill out the instr info.
  return X;
}

static MCSubtargetInfo *
createCapstoneMCSubtargetInfo(const Triple &TT, StringRef CPU, StringRef FS) {
  return createCapstoneMCSubtargetInfoImpl(TT, CPU, /*TuneCPU*/ CPU, FS);
}

static MCAsmInfo *createCapstoneMCAsmInfo(const MCRegisterInfo &MRI,
                                       const Triple &TheTriple,
                                       const MCTargetOptions &Options) {
  MCAsmInfo *MAI;
  if (TheTriple.isOSBinFormatMachO())
    MAI = new CapstoneMCAsmInfoDarwin(TheTriple, Options);
  else if (TheTriple.isOSBinFormatELF())
    MAI = new CapstoneMCAsmInfoELF(TheTriple, Options);
  else
    report_fatal_error("Binary format not supported");

  return MAI;
}

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeCapstoneTargetMC() {
  Target &TheTarget = getTheCapstoneTarget();

  // Register the MC asm info.
  RegisterMCAsmInfoFn X(TheTarget, createCapstoneMCAsmInfo);

  // Register the MC instruction info.
  TargetRegistry::RegisterMCInstrInfo(TheTarget, createCapstoneMCInstrInfo);

  // Register the MC register info.
  TargetRegistry::RegisterMCRegInfo(TheTarget, createCapstoneMCRegisterInfo);

  // Register the MC subtarget info.
  TargetRegistry::RegisterMCSubtargetInfo(TheTarget,
                                          createCapstoneMCSubtargetInfo);
}
