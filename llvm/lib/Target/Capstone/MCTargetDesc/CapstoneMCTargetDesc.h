//===-- CapstoneMCTargetDesc.h - Capstone Target Descriptions --------*- C++ -*-===//
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

#ifndef LLVM_LIB_TARGET_Capstone_MCTARGETDESC_CapstoneMCTARGETDESC_H
#define LLVM_LIB_TARGET_Capstone_MCTARGETDESC_CapstoneMCTARGETDESC_H

#include "llvm/MC/MCTargetOptions.h"
#include "llvm/Support/DataTypes.h"
#include <memory>

namespace llvm {
class MCAsmBackend;
class MCCodeEmitter;
class MCContext;
class MCInstrInfo;
class MCObjectTargetWriter;
class MCRegisterInfo;
class MCSubtargetInfo;
class Target;

MCCodeEmitter *createCapstoneMCCodeEmitter(const MCInstrInfo &MCII,
                                        MCContext &Ctx);

MCAsmBackend *createCapstoneAsmBackend(const Target &T, const MCSubtargetInfo &STI,
                                    const MCRegisterInfo &MRI,
                                    const MCTargetOptions &Options);

std::unique_ptr<MCObjectTargetWriter> createCapstoneELFObjectWriter(uint8_t OSABI,
                                                                 bool Is64Bit);
} // namespace llvm

// Defines symbolic names for Capstone registers.
#define GET_REGINFO_ENUM
#include "CapstoneGenRegisterInfo.inc"

// Defines symbolic names for Capstone instructions.
#define GET_INSTRINFO_ENUM
#define GET_INSTRINFO_MC_HELPER_DECLS
#include "CapstoneGenInstrInfo.inc"

#define GET_SUBTARGETINFO_ENUM
#include "CapstoneGenSubtargetInfo.inc"

#endif
