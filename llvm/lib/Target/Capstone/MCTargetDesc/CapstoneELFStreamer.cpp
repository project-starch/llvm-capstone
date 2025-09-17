//===-- CapstoneELFStreamer.cpp - Capstone ELF Target Streamer Methods ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides Capstone specific target streamer methods.
//
//===----------------------------------------------------------------------===//

#include "CapstoneELFStreamer.h"
#include "CapstoneAsmBackend.h"
#include "CapstoneBaseInfo.h"
#include "CapstoneMCTargetDesc.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCSubtargetInfo.h"

using namespace llvm;

// This part is for ELF object output.
CapstoneTargetELFStreamer::CapstoneTargetELFStreamer(MCStreamer &S,
                                               const MCSubtargetInfo &STI)
    : CapstoneTargetStreamer(S), CurrentVendor("capstone") {
  MCAssembler &MCA = getStreamer().getAssembler();
  const FeatureBitset &Features = STI.getFeatureBits();
  auto &MAB = static_cast<CapstoneAsmBackend &>(MCA.getBackend());
  setTargetABI(CapstoneABI::computeTargetABI(STI.getTargetTriple(), Features,
                                          MAB.getTargetOptions().getABIName()));
  setFlagsFromFeatures(STI);
}

CapstoneELFStreamer::CapstoneELFStreamer(MCContext &C,
                                   std::unique_ptr<MCAsmBackend> MAB,
                                   std::unique_ptr<MCObjectWriter> MOW,
                                   std::unique_ptr<MCCodeEmitter> MCE)
    : MCELFStreamer(C, std::move(MAB), std::move(MOW), std::move(MCE)) {}

CapstoneELFStreamer &CapstoneTargetELFStreamer::getStreamer() {
  return static_cast<CapstoneELFStreamer &>(Streamer);
}

void CapstoneTargetELFStreamer::emitDirectiveOptionExact() {}
void CapstoneTargetELFStreamer::emitDirectiveOptionNoExact() {}
void CapstoneTargetELFStreamer::emitDirectiveOptionPIC() {}
void CapstoneTargetELFStreamer::emitDirectiveOptionNoPIC() {}
void CapstoneTargetELFStreamer::emitDirectiveOptionPop() {}
void CapstoneTargetELFStreamer::emitDirectiveOptionPush() {}
void CapstoneTargetELFStreamer::emitDirectiveOptionRelax() {}
void CapstoneTargetELFStreamer::emitDirectiveOptionNoRelax() {}
void CapstoneTargetELFStreamer::emitDirectiveOptionRVC() {}
void CapstoneTargetELFStreamer::emitDirectiveOptionNoRVC() {}

void CapstoneTargetELFStreamer::emitAttribute(unsigned Attribute, unsigned Value) {
  getStreamer().setAttributeItem(Attribute, Value, /*OverwriteExisting=*/true);
}

void CapstoneTargetELFStreamer::emitTextAttribute(unsigned Attribute,
                                               StringRef String) {
  getStreamer().setAttributeItem(Attribute, String, /*OverwriteExisting=*/true);
}

void CapstoneTargetELFStreamer::emitIntTextAttribute(unsigned Attribute,
                                                  unsigned IntValue,
                                                  StringRef StringValue) {
  getStreamer().setAttributeItems(Attribute, IntValue, StringValue,
                                  /*OverwriteExisting=*/true);
}

void CapstoneTargetELFStreamer::finishAttributeSection() {
  CapstoneELFStreamer &S = getStreamer();
  if (S.Contents.empty())
    return;

  S.emitAttributesSection(CurrentVendor, ".capstone.attributes",
                          ELF::SHT_Capstone_ATTRIBUTES, AttributeSection);
}

void CapstoneTargetELFStreamer::finish() {
  CapstoneTargetStreamer::finish();
  ELFObjectWriter &W = getStreamer().getWriter();
  CapstoneABI::ABI ABI = getTargetABI();

  unsigned EFlags = W.getELFHeaderEFlags();

  if (hasRVC())
    EFlags |= ELF::EF_Capstone_RVC;
  if (hasTSO())
    EFlags |= ELF::EF_Capstone_TSO;

  switch (ABI) {
  case CapstoneABI::ABI_ILP32:
  case CapstoneABI::ABI_LP64:
    break;
  case CapstoneABI::ABI_ILP32F:
  case CapstoneABI::ABI_LP64F:
    EFlags |= ELF::EF_Capstone_FLOAT_ABI_SINGLE;
    break;
  case CapstoneABI::ABI_ILP32D:
  case CapstoneABI::ABI_LP64D:
    EFlags |= ELF::EF_Capstone_FLOAT_ABI_DOUBLE;
    break;
  case CapstoneABI::ABI_ILP32E:
  case CapstoneABI::ABI_LP64E:
    EFlags |= ELF::EF_Capstone_RVE;
    break;
  case CapstoneABI::ABI_Unknown:
    llvm_unreachable("Improperly initialised target ABI");
  }

  W.setELFHeaderEFlags(EFlags);
}

void CapstoneTargetELFStreamer::reset() {
  AttributeSection = nullptr;
}

void CapstoneTargetELFStreamer::emitDirectiveVariantCC(MCSymbol &Symbol) {
  getStreamer().getAssembler().registerSymbol(Symbol);
  static_cast<MCSymbolELF &>(Symbol).setOther(ELF::STO_Capstone_VARIANT_CC);
}

void CapstoneELFStreamer::reset() {
  static_cast<CapstoneTargetStreamer *>(getTargetStreamer())->reset();
  MCELFStreamer::reset();
  LastMappingSymbols.clear();
  LastEMS = EMS_None;
}

void CapstoneELFStreamer::emitDataMappingSymbol() {
  if (LastEMS == EMS_Data)
    return;
  emitMappingSymbol("$d");
  LastEMS = EMS_Data;
}

void CapstoneELFStreamer::emitInstructionsMappingSymbol() {
  if (LastEMS == EMS_Instructions)
    return;
  emitMappingSymbol("$x");
  LastEMS = EMS_Instructions;
}

void CapstoneELFStreamer::emitMappingSymbol(StringRef Name) {
  auto *Symbol =
      static_cast<MCSymbolELF *>(getContext().createLocalSymbol(Name));
  emitLabel(Symbol);
  Symbol->setType(ELF::STT_NOTYPE);
  Symbol->setBinding(ELF::STB_LOCAL);
}

void CapstoneELFStreamer::changeSection(MCSection *Section, uint32_t Subsection) {
  // We have to keep track of the mapping symbol state of any sections we
  // use. Each one should start off as EMS_None, which is provided as the
  // default constructor by DenseMap::lookup.
  LastMappingSymbols[getPreviousSection().first] = LastEMS;
  LastEMS = LastMappingSymbols.lookup(Section);

  MCELFStreamer::changeSection(Section, Subsection);
}

void CapstoneELFStreamer::emitInstruction(const MCInst &Inst,
                                       const MCSubtargetInfo &STI) {
  emitInstructionsMappingSymbol();
  MCELFStreamer::emitInstruction(Inst, STI);
}

void CapstoneELFStreamer::emitBytes(StringRef Data) {
  emitDataMappingSymbol();
  MCELFStreamer::emitBytes(Data);
}

void CapstoneELFStreamer::emitFill(const MCExpr &NumBytes, uint64_t FillValue,
                                SMLoc Loc) {
  emitDataMappingSymbol();
  MCELFStreamer::emitFill(NumBytes, FillValue, Loc);
}

void CapstoneELFStreamer::emitValueImpl(const MCExpr *Value, unsigned Size,
                                     SMLoc Loc) {
  emitDataMappingSymbol();
  MCELFStreamer::emitValueImpl(Value, Size, Loc);
}

MCStreamer *llvm::createCapstoneELFStreamer(const Triple &, MCContext &C,
                                         std::unique_ptr<MCAsmBackend> &&MAB,
                                         std::unique_ptr<MCObjectWriter> &&MOW,
                                         std::unique_ptr<MCCodeEmitter> &&MCE) {
  return new CapstoneELFStreamer(C, std::move(MAB), std::move(MOW),
                              std::move(MCE));
}
