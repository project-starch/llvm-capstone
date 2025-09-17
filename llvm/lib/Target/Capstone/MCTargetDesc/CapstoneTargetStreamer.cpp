//===-- CapstoneTargetStreamer.cpp - Capstone Target Streamer Methods ----------===//
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

#include "CapstoneTargetStreamer.h"
#include "CapstoneBaseInfo.h"
#include "CapstoneMCTargetDesc.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/CapstoneAttributes.h"
#include "llvm/TargetParser/CapstoneISAInfo.h"

using namespace llvm;

// This option controls whether or not we emit ELF attributes for ABI features,
// like Capstone atomics or X3 usage.
static cl::opt<bool> CapstoneAbiAttr(
    "capstone-abi-attributes",
    cl::desc("Enable emitting Capstone ELF attributes for ABI features"),
    cl::Hidden);

CapstoneTargetStreamer::CapstoneTargetStreamer(MCStreamer &S) : MCTargetStreamer(S) {}

void CapstoneTargetStreamer::finish() { finishAttributeSection(); }
void CapstoneTargetStreamer::reset() {}

void CapstoneTargetStreamer::emitDirectiveOptionArch(
    ArrayRef<CapstoneOptionArchArg> Args) {}
void CapstoneTargetStreamer::emitDirectiveOptionExact() {}
void CapstoneTargetStreamer::emitDirectiveOptionNoExact() {}
void CapstoneTargetStreamer::emitDirectiveOptionPIC() {}
void CapstoneTargetStreamer::emitDirectiveOptionNoPIC() {}
void CapstoneTargetStreamer::emitDirectiveOptionPop() {}
void CapstoneTargetStreamer::emitDirectiveOptionPush() {}
void CapstoneTargetStreamer::emitDirectiveOptionRelax() {}
void CapstoneTargetStreamer::emitDirectiveOptionNoRelax() {}
void CapstoneTargetStreamer::emitDirectiveOptionRVC() {}
void CapstoneTargetStreamer::emitDirectiveOptionNoRVC() {}
void CapstoneTargetStreamer::emitDirectiveVariantCC(MCSymbol &Symbol) {}
void CapstoneTargetStreamer::emitAttribute(unsigned Attribute, unsigned Value) {}
void CapstoneTargetStreamer::finishAttributeSection() {}
void CapstoneTargetStreamer::emitTextAttribute(unsigned Attribute,
                                            StringRef String) {}
void CapstoneTargetStreamer::emitIntTextAttribute(unsigned Attribute,
                                               unsigned IntValue,
                                               StringRef StringValue) {}

void CapstoneTargetStreamer::emitNoteGnuPropertySection(
    const uint32_t Feature1And) {
  MCStreamer &OutStreamer = getStreamer();
  MCContext &Ctx = OutStreamer.getContext();

  const Triple &Triple = Ctx.getTargetTriple();
  Align NoteAlign;
  uint64_t DescSize;
  if (Triple.isArch64Bit()) {
    NoteAlign = Align(8);
    DescSize = 16;
  } else {
    assert(Triple.isArch32Bit());
    NoteAlign = Align(4);
    DescSize = 12;
  }

  assert(Ctx.getObjectFileType() == MCContext::Environment::IsELF);
  MCSection *const NoteSection =
      Ctx.getELFSection(".note.gnu.property", ELF::SHT_NOTE, ELF::SHF_ALLOC);
  OutStreamer.pushSection();
  OutStreamer.switchSection(NoteSection);

  // Emit the note header
  OutStreamer.emitValueToAlignment(NoteAlign);
  OutStreamer.emitIntValue(4, 4);                           // n_namsz
  OutStreamer.emitIntValue(DescSize, 4);                    // n_descsz
  OutStreamer.emitIntValue(ELF::NT_GNU_PROPERTY_TYPE_0, 4); // n_type
  OutStreamer.emitBytes(StringRef("GNU", 4));               // n_name

  // Emit n_desc field

  // Emit the feature_1_and property
  OutStreamer.emitIntValue(ELF::GNU_PROPERTY_Capstone_FEATURE_1_AND, 4); // pr_type
  OutStreamer.emitIntValue(4, 4);              // pr_datasz
  OutStreamer.emitIntValue(Feature1And, 4);    // pr_data
  OutStreamer.emitValueToAlignment(NoteAlign); // pr_padding

  OutStreamer.popSection();
}

void CapstoneTargetStreamer::setTargetABI(CapstoneABI::ABI ABI) {
  assert(ABI != CapstoneABI::ABI_Unknown && "Improperly initialized target ABI");
  TargetABI = ABI;
}

void CapstoneTargetStreamer::setFlagsFromFeatures(const MCSubtargetInfo &STI) {
  HasRVC = STI.hasFeature(Capstone::FeatureStdExtZca);
  HasTSO = STI.hasFeature(Capstone::FeatureStdExtZtso);
}

void CapstoneTargetStreamer::emitTargetAttributes(const MCSubtargetInfo &STI,
                                               bool EmitStackAlign) {
  if (EmitStackAlign) {
    unsigned StackAlign;
    if (TargetABI == CapstoneABI::ABI_ILP32E)
      StackAlign = 4;
    else if (TargetABI == CapstoneABI::ABI_LP64E)
      StackAlign = 8;
    else
      StackAlign = 16;
    emitAttribute(CapstoneAttrs::STACK_ALIGN, StackAlign);
  }

  auto ParseResult = CapstoneFeatures::parseFeatureBits(
      STI.hasFeature(Capstone::Feature64Bit), STI.getFeatureBits());
  if (!ParseResult) {
    report_fatal_error(ParseResult.takeError());
  } else {
    auto &ISAInfo = *ParseResult;
    emitTextAttribute(CapstoneAttrs::ARCH, ISAInfo->toString());
  }

  if (CapstoneAbiAttr && STI.hasFeature(Capstone::FeatureStdExtA)) {
    unsigned AtomicABITag;
    if (STI.hasFeature(Capstone::FeatureStdExtZalasr))
      AtomicABITag = static_cast<unsigned>(CapstoneAttrs::CapstoneAtomicAbiTag::A7);
    else if (STI.hasFeature(Capstone::FeatureNoTrailingSeqCstFence))
      AtomicABITag = static_cast<unsigned>(CapstoneAttrs::CapstoneAtomicAbiTag::A6C);
    else
      AtomicABITag = static_cast<unsigned>(CapstoneAttrs::CapstoneAtomicAbiTag::A6S);
    emitAttribute(CapstoneAttrs::ATOMIC_ABI, AtomicABITag);
  }
}

// This part is for ascii assembly output
CapstoneTargetAsmStreamer::CapstoneTargetAsmStreamer(MCStreamer &S,
                                               formatted_raw_ostream &OS)
    : CapstoneTargetStreamer(S), OS(OS) {}

void CapstoneTargetAsmStreamer::emitDirectiveOptionPush() {
  OS << "\t.option\tpush\n";
}

void CapstoneTargetAsmStreamer::emitDirectiveOptionPop() {
  OS << "\t.option\tpop\n";
}

void CapstoneTargetAsmStreamer::emitDirectiveOptionPIC() {
  OS << "\t.option\tpic\n";
}

void CapstoneTargetAsmStreamer::emitDirectiveOptionNoPIC() {
  OS << "\t.option\tnopic\n";
}

void CapstoneTargetAsmStreamer::emitDirectiveOptionRVC() {
  OS << "\t.option\trvc\n";
}

void CapstoneTargetAsmStreamer::emitDirectiveOptionNoRVC() {
  OS << "\t.option\tnorvc\n";
}

void CapstoneTargetAsmStreamer::emitDirectiveOptionExact() {
  OS << "\t.option\texact\n";
}

void CapstoneTargetAsmStreamer::emitDirectiveOptionNoExact() {
  OS << "\t.option\tnoexact\n";
}

void CapstoneTargetAsmStreamer::emitDirectiveOptionRelax() {
  OS << "\t.option\trelax\n";
}

void CapstoneTargetAsmStreamer::emitDirectiveOptionNoRelax() {
  OS << "\t.option\tnorelax\n";
}

void CapstoneTargetAsmStreamer::emitDirectiveOptionArch(
    ArrayRef<CapstoneOptionArchArg> Args) {
  OS << "\t.option\tarch";
  for (const auto &Arg : Args) {
    OS << ", ";
    switch (Arg.Type) {
    case CapstoneOptionArchArgType::Full:
      break;
    case CapstoneOptionArchArgType::Plus:
      OS << "+";
      break;
    case CapstoneOptionArchArgType::Minus:
      OS << "-";
      break;
    }
    OS << Arg.Value;
  }
  OS << "\n";
}

void CapstoneTargetAsmStreamer::emitDirectiveVariantCC(MCSymbol &Symbol) {
  OS << "\t.variant_cc\t" << Symbol.getName() << "\n";
}

void CapstoneTargetAsmStreamer::emitAttribute(unsigned Attribute, unsigned Value) {
  OS << "\t.attribute\t" << Attribute << ", " << Twine(Value) << "\n";
}

void CapstoneTargetAsmStreamer::emitTextAttribute(unsigned Attribute,
                                               StringRef String) {
  OS << "\t.attribute\t" << Attribute << ", \"" << String << "\"\n";
}

void CapstoneTargetAsmStreamer::emitIntTextAttribute(unsigned Attribute,
                                                  unsigned IntValue,
                                                  StringRef StringValue) {}

void CapstoneTargetAsmStreamer::finishAttributeSection() {}
