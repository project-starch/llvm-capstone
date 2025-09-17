//===-- CapstoneMCAsmInfo.cpp - Capstone Asm properties ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the CapstoneMCAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "CapstoneMCAsmInfo.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/TargetParser/Triple.h"
using namespace llvm;

void CapstoneMCAsmInfo::anchor() {}

CapstoneMCAsmInfo::CapstoneMCAsmInfo(const Triple &TT) {
  IsLittleEndian = TT.isLittleEndian();
  CodePointerSize = CalleeSaveStackSlotSize = TT.isArch64Bit() ? 8 : 4;
  CommentString = "#";
  AlignmentIsInBytes = false;
  SupportsDebugInformation = true;
  ExceptionsType = ExceptionHandling::DwarfCFI;
  UseAtForSpecifier = false;
  Data16bitsDirective = "\t.half\t";
  Data32bitsDirective = "\t.word\t";
}

const MCExpr *CapstoneMCAsmInfo::getExprForFDESymbol(const MCSymbol *Sym,
                                                  unsigned Encoding,
                                                  MCStreamer &Streamer) const {
  if (!(Encoding & dwarf::DW_EH_PE_pcrel))
    return MCAsmInfo::getExprForFDESymbol(Sym, Encoding, Streamer);

  // The default symbol subtraction results in an ADD/SUB relocation pair.
  // Processing this relocation pair is problematic when linker relaxation is
  // enabled, so we follow binutils in using the R_Capstone_32_PCREL relocation
  // for the FDE initial location.
  MCContext &Ctx = Streamer.getContext();
  const MCExpr *ME = MCSymbolRefExpr::create(Sym, Ctx);
  assert(Encoding & dwarf::DW_EH_PE_sdata4 && "Unexpected encoding");
  return MCSpecifierExpr::create(ME, ELF::R_Capstone_32_PCREL, Ctx);
}

void CapstoneMCAsmInfo::printSpecifierExpr(raw_ostream &OS,
                                        const MCSpecifierExpr &Expr) const {
  auto S = Expr.getSpecifier();
  bool HasSpecifier = S != 0 && S != ELF::R_Capstone_CALL_PLT;
  if (HasSpecifier)
    OS << '%' << Capstone::getSpecifierName(S) << '(';
  printExpr(OS, *Expr.getSubExpr());
  if (HasSpecifier)
    OS << ')';
}
