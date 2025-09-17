//===-- CapstoneTargetObjectFile.h - Capstone Object Info ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_Capstone_CapstoneTARGETOBJECTFILE_H
#define LLVM_LIB_TARGET_Capstone_CapstoneTARGETOBJECTFILE_H

#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"

namespace llvm {

/// This implementation is used for Capstone ELF targets.
class CapstoneELFTargetObjectFile : public TargetLoweringObjectFileELF {
  MCSection *SmallDataSection;
  MCSection *SmallRODataSection;
  MCSection *SmallROData4Section;
  MCSection *SmallROData8Section;
  MCSection *SmallROData16Section;
  MCSection *SmallROData32Section;
  MCSection *SmallBSSSection;
  unsigned SSThreshold = 0;

public:
  unsigned getTextSectionAlignment() const override;

  void Initialize(MCContext &Ctx, const TargetMachine &TM) override;

  /// Return true if this global address should be placed into small data/bss
  /// section.
  bool isGlobalInSmallSection(const GlobalObject *GO,
                              const TargetMachine &TM) const;

  MCSection *SelectSectionForGlobal(const GlobalObject *GO, SectionKind Kind,
                                    const TargetMachine &TM) const override;

  /// Return true if this constant should be placed into small data section.
  bool isConstantInSmallSection(const DataLayout &DL, const Constant *CN) const;

  MCSection *getSectionForConstant(const DataLayout &DL, SectionKind Kind,
                                   const Constant *C,
                                   Align &Alignment) const override;

  void getModuleMetadata(Module &M) override;

  bool isInSmallSection(uint64_t Size) const;

  const MCExpr *getIndirectSymViaGOTPCRel(const GlobalValue *GV,
                                          const MCSymbol *Sym,
                                          const MCValue &MV, int64_t Offset,
                                          MachineModuleInfo *MMI,
                                          MCStreamer &Streamer) const override;
};

} // end namespace llvm

#endif
