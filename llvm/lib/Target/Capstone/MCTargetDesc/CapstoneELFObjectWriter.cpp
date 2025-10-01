//===-- CapstoneELFObjectWriter.cpp - Capstone ELF Writer ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/CapstoneFixupKinds.h"
#include "MCTargetDesc/CapstoneMCAsmInfo.h"
#include "MCTargetDesc/CapstoneMCTargetDesc.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace {
class CapstoneELFObjectWriter : public MCELFObjectTargetWriter {
public:
  CapstoneELFObjectWriter(uint8_t OSABI, bool Is64Bit);

  ~CapstoneELFObjectWriter() override;

  // Return true if the given relocation must be with a symbol rather than
  // section plus offset.
  bool needsRelocateWithSymbol(const MCValue &, unsigned Type) const override {
    // TODO: this is very conservative, update once Capstone psABI requirements
    //       are clarified.
    return true;
  }

protected:
  unsigned getRelocType(const MCFixup &, const MCValue &,
                        bool IsPCRel) const override;
};
}

CapstoneELFObjectWriter::CapstoneELFObjectWriter(uint8_t OSABI, bool Is64Bit)
    : MCELFObjectTargetWriter(Is64Bit, OSABI, ELF::EM_CAPSTONE,
                              /*HasRelocationAddend*/ true) {}

CapstoneELFObjectWriter::~CapstoneELFObjectWriter() = default;

unsigned CapstoneELFObjectWriter::getRelocType(const MCFixup &Fixup,
                                            const MCValue &Target,
                                            bool IsPCRel) const {
  auto Kind = Fixup.getKind();
  auto Spec = Target.getSpecifier();
  switch (Spec) {
  case ELF::R_Capstone_TPREL_HI20:
  case ELF::R_Capstone_TLS_GOT_HI20:
  case ELF::R_Capstone_TLS_GD_HI20:
  case ELF::R_Capstone_TLSDESC_HI20:
    if (auto *SA = const_cast<MCSymbol *>(Target.getAddSym()))
      static_cast<MCSymbolELF *>(SA)->setType(ELF::STT_TLS);
    break;
  case ELF::R_Capstone_PLT32:
  case ELF::R_Capstone_GOT32_PCREL:
    if (Kind == FK_Data_4)
      break;
    reportError(Fixup.getLoc(), "%" + Capstone::getSpecifierName(Spec) +
                                    " can only be used in a .word directive");
    return ELF::R_Capstone_NONE;
  default:
    break;
  }

  // Extract the relocation type from the fixup kind, after applying STT_TLS as
  // needed.
  if (mc::isRelocation(Fixup.getKind()))
    return Kind;

  if (IsPCRel) {
    switch (Kind) {
    default:
      reportError(Fixup.getLoc(), "unsupported relocation type");
      return ELF::R_Capstone_NONE;
    case FK_Data_4:
      return ELF::R_Capstone_32_PCREL;
    case Capstone::fixup_capstone_pcrel_hi20:
      return ELF::R_Capstone_PCREL_HI20;
    case Capstone::fixup_capstone_pcrel_lo12_i:
      return ELF::R_Capstone_PCREL_LO12_I;
    case Capstone::fixup_capstone_pcrel_lo12_s:
      return ELF::R_Capstone_PCREL_LO12_S;
    case Capstone::fixup_capstone_jal:
      return ELF::R_Capstone_JAL;
    case Capstone::fixup_capstone_branch:
      return ELF::R_Capstone_BRANCH;
    case Capstone::fixup_capstone_rvc_jump:
      return ELF::R_Capstone_RVC_JUMP;
    case Capstone::fixup_capstone_rvc_branch:
      return ELF::R_Capstone_RVC_BRANCH;
    case Capstone::fixup_capstone_call:
      return ELF::R_Capstone_CALL_PLT;
    case Capstone::fixup_capstone_call_plt:
      return ELF::R_Capstone_CALL_PLT;
    case Capstone::fixup_capstone_qc_e_branch:
      return ELF::R_Capstone_QC_E_BRANCH;
    case Capstone::fixup_capstone_qc_e_call_plt:
      return ELF::R_Capstone_QC_E_CALL_PLT;
    case Capstone::fixup_capstone_nds_branch_10:
      return ELF::R_Capstone_NDS_BRANCH_10;
    }
  }

  switch (Kind) {
  default:
    reportError(Fixup.getLoc(), "unsupported relocation type");
    return ELF::R_Capstone_NONE;

  case FK_Data_1:
    reportError(Fixup.getLoc(), "1-byte data relocations not supported");
    return ELF::R_Capstone_NONE;
  case FK_Data_2:
    reportError(Fixup.getLoc(), "2-byte data relocations not supported");
    return ELF::R_Capstone_NONE;
  case FK_Data_4:
    switch (Spec) {
    case ELF::R_Capstone_32_PCREL:
    case ELF::R_Capstone_GOT32_PCREL:
    case ELF::R_Capstone_PLT32:
      return Spec;
    }
    return ELF::R_Capstone_32;
  case FK_Data_8:
    return ELF::R_Capstone_64;
  case Capstone::fixup_capstone_hi20:
    return ELF::R_Capstone_HI20;
  case Capstone::fixup_capstone_lo12_i:
    return ELF::R_Capstone_LO12_I;
  case Capstone::fixup_capstone_lo12_s:
    return ELF::R_Capstone_LO12_S;
  case Capstone::fixup_capstone_rvc_imm:
    reportError(Fixup.getLoc(), "No relocation for CI-type instructions");
    return ELF::R_Capstone_NONE;
  case Capstone::fixup_capstone_qc_e_32:
    return ELF::R_Capstone_QC_E_32;
  case Capstone::fixup_capstone_qc_abs20_u:
    return ELF::R_Capstone_QC_ABS20_U;
  }
}

std::unique_ptr<MCObjectTargetWriter>
llvm::createCapstoneELFObjectWriter(uint8_t OSABI, bool Is64Bit) {
  return std::make_unique<CapstoneELFObjectWriter>(OSABI, Is64Bit);
}
