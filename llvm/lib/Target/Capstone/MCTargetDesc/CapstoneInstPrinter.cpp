//===-- CapstoneInstPrinter.cpp - Convert Capstone MCInst to asm syntax --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This class prints an Capstone MCInst to a .s file.
//
//===----------------------------------------------------------------------===//

#include "CapstoneInstPrinter.h"
#include "CapstoneBaseInfo.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
using namespace llvm;

#define DEBUG_TYPE "asm-printer"

// Include the auto-generated portion of the assembly writer.
#define PRINT_ALIAS_INSTR
#include "CapstoneGenAsmWriter.inc"

static cl::opt<bool>
    NoAliases("capstone-no-aliases",
              cl::desc("Disable the emission of assembler pseudo instructions"),
              cl::init(false), cl::Hidden);

static cl::opt<bool> EmitX8AsFP("capstone-emit-x8-as-fp",
                                cl::desc("Emit x8 as fp instead of s0"),
                                cl::init(false), cl::Hidden);

// Print architectural register names rather than the ABI names (such as x2
// instead of sp).
// TODO: Make CapstoneInstPrinter::getRegisterName non-static so that this can a
// member.
static bool ArchRegNames;

// The command-line flags above are used by llvm-mc and llc. They can be used by
// `llvm-objdump`, but we override their values here to handle options passed to
// `llvm-objdump` with `-M` (which matches GNU objdump). There did not seem to
// be an easier way to allow these options in all these tools, without doing it
// this way.
bool CapstoneInstPrinter::applyTargetSpecificCLOption(StringRef Opt) {
  if (Opt == "no-aliases") {
    PrintAliases = false;
    return true;
  }
  if (Opt == "numeric") {
    ArchRegNames = true;
    return true;
  }
  if (Opt == "emit-x8-as-fp") {
    if (!ArchRegNames)
      EmitX8AsFP = true;
    return true;
  }

  return false;
}

void CapstoneInstPrinter::printInst(const MCInst *MI, uint64_t Address,
                                 StringRef Annot, const MCSubtargetInfo &STI,
                                 raw_ostream &O) {
  bool Res = false;
  const MCInst *NewMI = MI;
  MCInst UncompressedMI;
  if (PrintAliases && !NoAliases)
    Res = CapstoneRVC::uncompress(UncompressedMI, *MI, STI);
  if (Res)
    NewMI = &UncompressedMI;
  if (!PrintAliases || NoAliases || !printAliasInstr(NewMI, Address, STI, O))
    printInstruction(NewMI, Address, STI, O);
  printAnnotation(O, Annot);
}

void CapstoneInstPrinter::printRegName(raw_ostream &O, MCRegister Reg) {
  markup(O, Markup::Register) << getRegisterName(Reg);
}

void CapstoneInstPrinter::printOperand(const MCInst *MI, unsigned OpNo,
                                    const MCSubtargetInfo &STI,
                                    raw_ostream &O) {
  const MCOperand &MO = MI->getOperand(OpNo);

  if (MO.isReg()) {
    printRegName(O, MO.getReg());
    return;
  }

  if (MO.isImm()) {
    markup(O, Markup::Immediate) << formatImm(MO.getImm());
    return;
  }

  assert(MO.isExpr() && "Unknown operand kind in printOperand");
  MAI.printExpr(O, *MO.getExpr());
}

void CapstoneInstPrinter::printBranchOperand(const MCInst *MI, uint64_t Address,
                                          unsigned OpNo,
                                          const MCSubtargetInfo &STI,
                                          raw_ostream &O) {
  const MCOperand &MO = MI->getOperand(OpNo);
  if (!MO.isImm())
    return printOperand(MI, OpNo, STI, O);

  if (PrintBranchImmAsAddress) {
    uint64_t Target = Address + MO.getImm();
    if (!STI.hasFeature(Capstone::Feature64Bit))
      Target &= 0xffffffff;
    markup(O, Markup::Target) << formatHex(Target);
  } else {
    markup(O, Markup::Target) << formatImm(MO.getImm());
  }
}

void CapstoneInstPrinter::printCSRSystemRegister(const MCInst *MI, unsigned OpNo,
                                              const MCSubtargetInfo &STI,
                                              raw_ostream &O) {
  unsigned Imm = MI->getOperand(OpNo).getImm();
  auto Range = CapstoneSysReg::lookupSysRegByEncoding(Imm);
  for (auto &Reg : Range) {
    if (Reg.IsAltName || Reg.IsDeprecatedName)
      continue;
    if (Reg.haveRequiredFeatures(STI.getFeatureBits())) {
      markup(O, Markup::Register) << Reg.Name;
      return;
    }
  }
  markup(O, Markup::Register) << formatImm(Imm);
}

void CapstoneInstPrinter::printFenceArg(const MCInst *MI, unsigned OpNo,
                                     const MCSubtargetInfo &STI,
                                     raw_ostream &O) {
  unsigned FenceArg = MI->getOperand(OpNo).getImm();
  assert (((FenceArg >> 4) == 0) && "Invalid immediate in printFenceArg");

  if ((FenceArg & CapstoneFenceField::I) != 0)
    O << 'i';
  if ((FenceArg & CapstoneFenceField::O) != 0)
    O << 'o';
  if ((FenceArg & CapstoneFenceField::R) != 0)
    O << 'r';
  if ((FenceArg & CapstoneFenceField::W) != 0)
    O << 'w';
  if (FenceArg == 0)
    O << "0";
}

void CapstoneInstPrinter::printFRMArg(const MCInst *MI, unsigned OpNo,
                                   const MCSubtargetInfo &STI, raw_ostream &O) {
  auto FRMArg =
      static_cast<CapstoneFPRndMode::RoundingMode>(MI->getOperand(OpNo).getImm());
  if (PrintAliases && !NoAliases && FRMArg == CapstoneFPRndMode::RoundingMode::DYN)
    return;
  O << ", " << CapstoneFPRndMode::roundingModeToString(FRMArg);
}

void CapstoneInstPrinter::printFRMArgLegacy(const MCInst *MI, unsigned OpNo,
                                         const MCSubtargetInfo &STI,
                                         raw_ostream &O) {
  auto FRMArg =
      static_cast<CapstoneFPRndMode::RoundingMode>(MI->getOperand(OpNo).getImm());
  // Never print rounding mode if it's the default 'rne'. This ensures the
  // output can still be parsed by older tools that erroneously failed to
  // accept a rounding mode.
  if (FRMArg == CapstoneFPRndMode::RoundingMode::RNE)
    return;
  O << ", " << CapstoneFPRndMode::roundingModeToString(FRMArg);
}

void CapstoneInstPrinter::printFPImmOperand(const MCInst *MI, unsigned OpNo,
                                         const MCSubtargetInfo &STI,
                                         raw_ostream &O) {
  unsigned Imm = MI->getOperand(OpNo).getImm();
  if (Imm == 1) {
    markup(O, Markup::Immediate) << "min";
  } else if (Imm == 30) {
    markup(O, Markup::Immediate) << "inf";
  } else if (Imm == 31) {
    markup(O, Markup::Immediate) << "nan";
  } else {
    float FPVal = CapstoneLoadFPImm::getFPImm(Imm);
    // If the value is an integer, print a .0 fraction. Otherwise, use %g to
    // which will not print trailing zeros and will use scientific notation
    // if it is shorter than printing as a decimal. The smallest value requires
    // 12 digits of precision including the decimal.
    if (FPVal == (int)(FPVal))
      markup(O, Markup::Immediate) << format("%.1f", FPVal);
    else
      markup(O, Markup::Immediate) << format("%.12g", FPVal);
  }
}

void CapstoneInstPrinter::printZeroOffsetMemOp(const MCInst *MI, unsigned OpNo,
                                            const MCSubtargetInfo &STI,
                                            raw_ostream &O) {
  const MCOperand &MO = MI->getOperand(OpNo);

  assert(MO.isReg() && "printZeroOffsetMemOp can only print register operands");
  O << "(";
  printRegName(O, MO.getReg());
  O << ")";
}

void CapstoneInstPrinter::printVTypeI(const MCInst *MI, unsigned OpNo,
                                   const MCSubtargetInfo &STI, raw_ostream &O) {
  unsigned Imm = MI->getOperand(OpNo).getImm();
  // Print the raw immediate for reserved values: vlmul[2:0]=4, vsew[2:0]=0b1xx,
  // altfmt=1 without zvfbfa extension, or non-zero in bits 9 and above.
  if (CapstoneVType::getVLMUL(Imm) == CapstoneVType::VLMUL::LMUL_RESERVED ||
      CapstoneVType::getSEW(Imm) > 64 ||
      (CapstoneVType::isAltFmt(Imm) &&
       !STI.hasFeature(Capstone::FeatureStdExtZvfbfa)) ||
      (Imm >> 9) != 0) {
    O << formatImm(Imm);
    return;
  }
  // Print the text form.
  CapstoneVType::printVType(Imm, O);
}

void CapstoneInstPrinter::printXSfmmVType(const MCInst *MI, unsigned OpNo,
                                       const MCSubtargetInfo &STI,
                                       raw_ostream &O) {
  unsigned Imm = MI->getOperand(OpNo).getImm();
  assert(CapstoneVType::isValidXSfmmVType(Imm));
  unsigned SEW = CapstoneVType::getSEW(Imm);
  O << "e" << SEW;
  bool AltFmt = CapstoneVType::isAltFmt(Imm);
  if (AltFmt)
    O << "alt";
  unsigned Widen = CapstoneVType::getXSfmmWiden(Imm);
  O << ", w" << Widen;
}

// Print a Zcmp RList. If we are printing architectural register names rather
// than ABI register names, we need to print "{x1, x8-x9, x18-x27}" for all
// registers. Otherwise, we print "{ra, s0-s11}".
void CapstoneInstPrinter::printRegList(const MCInst *MI, unsigned OpNo,
                                    const MCSubtargetInfo &STI, raw_ostream &O) {
  unsigned Imm = MI->getOperand(OpNo).getImm();

  assert(Imm >= CapstoneZC::RLISTENCODE::RA &&
         Imm <= CapstoneZC::RLISTENCODE::RA_S0_S11 && "Invalid Rlist");

  O << "{";
  printRegName(O, Capstone::X1);

  if (Imm >= CapstoneZC::RLISTENCODE::RA_S0) {
    O << ", ";
    printRegName(O, Capstone::X8);
  }

  if (Imm >= CapstoneZC::RLISTENCODE::RA_S0_S1) {
    O << '-';
    if (Imm == CapstoneZC::RLISTENCODE::RA_S0_S1 || ArchRegNames)
      printRegName(O, Capstone::X9);
  }

  if (Imm >= CapstoneZC::RLISTENCODE::RA_S0_S2) {
    if (ArchRegNames)
      O << ", ";
    if (Imm == CapstoneZC::RLISTENCODE::RA_S0_S2 || ArchRegNames)
      printRegName(O, Capstone::X18);
  }

  if (Imm >= CapstoneZC::RLISTENCODE::RA_S0_S3) {
    if (ArchRegNames)
      O << '-';
    unsigned Offset = (Imm - CapstoneZC::RLISTENCODE::RA_S0_S3);
    // Encodings for S3-S9 are contiguous. There is no encoding for S10, so we
    // must skip to S11(X27).
    if (Imm == CapstoneZC::RLISTENCODE::RA_S0_S11)
      ++Offset;
    printRegName(O, Capstone::X19 + Offset);
  }

  O << "}";
}

void CapstoneInstPrinter::printRegReg(const MCInst *MI, unsigned OpNo,
                                   const MCSubtargetInfo &STI, raw_ostream &O) {
  const MCOperand &OffsetMO = MI->getOperand(OpNo + 1);

  assert(OffsetMO.isReg() && "printRegReg can only print register operands");
  printRegName(O, OffsetMO.getReg());

  O << "(";
  const MCOperand &BaseMO = MI->getOperand(OpNo);
  assert(BaseMO.isReg() && "printRegReg can only print register operands");
  printRegName(O, BaseMO.getReg());
  O << ")";
}

void CapstoneInstPrinter::printStackAdj(const MCInst *MI, unsigned OpNo,
                                     const MCSubtargetInfo &STI, raw_ostream &O,
                                     bool Negate) {
  int64_t Imm = MI->getOperand(OpNo).getImm();
  bool IsRV64 = STI.hasFeature(Capstone::Feature64Bit);
  int64_t StackAdj = 0;
  auto RlistVal = MI->getOperand(0).getImm();
  auto Base = CapstoneZC::getStackAdjBase(RlistVal, IsRV64);
  StackAdj = Imm + Base;
  assert((StackAdj >= Base && StackAdj <= Base + 48) &&
         "Incorrect stack adjust");
  if (Negate)
    StackAdj = -StackAdj;

  // RAII guard for ANSI color escape sequences
  WithMarkup ScopedMarkup = markup(O, Markup::Immediate);
  O << StackAdj;
}

void CapstoneInstPrinter::printVMaskReg(const MCInst *MI, unsigned OpNo,
                                     const MCSubtargetInfo &STI,
                                     raw_ostream &O) {
  const MCOperand &MO = MI->getOperand(OpNo);

  assert(MO.isReg() && "printVMaskReg can only print register operands");
  if (MO.getReg() == Capstone::NoRegister)
    return;
  O << ", ";
  printRegName(O, MO.getReg());
  O << ".t";
}

const char *CapstoneInstPrinter::getRegisterName(MCRegister Reg) {
  // When PrintAliases is enabled, and EmitX8AsFP is enabled, x8 will be printed
  // as fp instead of s0. Note that these similar registers are not replaced:
  // - X8_H: used for f16 register in zhinx
  // - X8_W: used for f32 register in zfinx
  // - X8_X9: used for GPR Pair
  if (!ArchRegNames && EmitX8AsFP && Reg == Capstone::X8)
    return "fp";
  return getRegisterName(Reg, ArchRegNames ? Capstone::NoRegAltName
                                           : Capstone::ABIRegAltName);
}
