//===-- CapstoneTargetObjectFile.cpp - Capstone Object Info --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CapstoneTargetObjectFile.h"
#include "MCTargetDesc/CapstoneMCObjectFileInfo.h"
#include "CapstoneTargetMachine.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

// Small data for VARIABLES on the Capstone target, whose clang emits no
// SmallDataLimit module flag: a defined variable of at most this many bytes goes
// to .sdata/.sbss, where the image link script puts it inside the 4 KiB that
// __global_pointer$ reaches (lld gp relaxation, -capstone-mark-gp-relaxable).
// Constants are not affected: under the image-gp ABI the constant pool stays in
// .rodata (getSectionForConstant). 0 (the default) changes nothing.
static cl::opt<unsigned> CapstoneSmallDataLimit(
    "capstone-small-data-limit", cl::Hidden, cl::init(0),
    cl::desc("Place defined variables of at most this many bytes in "
             ".sdata/.sbss (constants unaffected); default 0"));

bool capstoneImageGpAbiActive(); // CapstoneISelDAGToDAG.cpp

unsigned CapstoneELFTargetObjectFile::getTextSectionAlignment() const {
  return CapstoneMCObjectFileInfo::getTextSectionAlignment(
      *getContext().getSubtargetInfo());
}

void CapstoneELFTargetObjectFile::Initialize(MCContext &Ctx,
                                          const TargetMachine &TM) {
  TargetLoweringObjectFileELF::Initialize(Ctx, TM);

  PLTPCRelativeSpecifier = ELF::R_Capstone_PLT32;
  SupportIndirectSymViaGOTPCRel = true;

  SmallDataSection = getContext().getELFSection(
      ".sdata", ELF::SHT_PROGBITS, ELF::SHF_WRITE | ELF::SHF_ALLOC);
  SmallBSSSection = getContext().getELFSection(".sbss", ELF::SHT_NOBITS,
                                               ELF::SHF_WRITE | ELF::SHF_ALLOC);
  SmallRODataSection =
      getContext().getELFSection(".srodata", ELF::SHT_PROGBITS, ELF::SHF_ALLOC);
  SmallROData4Section = getContext().getELFSection(
      ".srodata.cst4", ELF::SHT_PROGBITS, ELF::SHF_ALLOC | ELF::SHF_MERGE, 4);
  SmallROData8Section = getContext().getELFSection(
      ".srodata.cst8", ELF::SHT_PROGBITS, ELF::SHF_ALLOC | ELF::SHF_MERGE, 8);
  SmallROData16Section = getContext().getELFSection(
      ".srodata.cst16", ELF::SHT_PROGBITS, ELF::SHF_ALLOC | ELF::SHF_MERGE, 16);
  SmallROData32Section = getContext().getELFSection(
      ".srodata.cst32", ELF::SHT_PROGBITS, ELF::SHF_ALLOC | ELF::SHF_MERGE, 32);
}

const MCExpr *CapstoneELFTargetObjectFile::getIndirectSymViaGOTPCRel(
    const GlobalValue *GV, const MCSymbol *Sym, const MCValue &MV,
    int64_t Offset, MachineModuleInfo *MMI, MCStreamer &Streamer) const {
  auto &Ctx = getContext();
  const MCExpr *Res = MCSymbolRefExpr::create(Sym, Ctx);
  Res = MCBinaryExpr::createAdd(
      Res, MCConstantExpr::create(Offset + MV.getConstant(), Ctx), Ctx);
  return MCSpecifierExpr::create(Res, ELF::R_Capstone_GOT32_PCREL, Ctx);
}

// A address must be loaded from a small section if its size is less than the
// small section size threshold. Data in this section could be addressed by
// using gp_rel operator.
bool CapstoneELFTargetObjectFile::isInSmallSection(uint64_t Size) const {
  // gcc has traditionally not treated zero-sized objects as small data, so this
  // is effectively part of the ABI.
  return Size > 0 && Size <= SSThreshold;
}

// Return true if this global address should be placed into small data/bss
// section.
bool CapstoneELFTargetObjectFile::isGlobalInSmallSection(
    const GlobalObject *GO, const TargetMachine &TM) const {
  // Only global variables, not functions.
  const GlobalVariable *GVA = dyn_cast<GlobalVariable>(GO);
  if (!GVA)
    return false;

  // If the variable has an explicit section, it is placed in that section.
  if (GVA->hasSection()) {
    StringRef Section = GVA->getSection();

    // Explicitly placing any variable in the small data section overrides
    // the global -G value.
    if (Section == ".sdata" || Section == ".sbss")
      return true;

    // Otherwise reject putting the variable to small section if it has an
    // explicit section name.
    return false;
  }

  if (((GVA->hasExternalLinkage() && GVA->isDeclaration()) ||
       GVA->hasCommonLinkage()))
    return false;

  Type *Ty = GVA->getValueType();
  // It is possible that the type of the global is unsized, i.e. a declaration
  // of a extern struct. In this case don't presume it is in the small data
  // section. This happens e.g. when building the FreeBSD kernel.
  if (!Ty->isSized())
    return false;

  uint64_t Size = GVA->getDataLayout().getTypeAllocSize(Ty);
  if (CapstoneSmallDataLimit && Size > 0 && Size <= CapstoneSmallDataLimit)
    return true;
  return isInSmallSection(Size);
}

MCSection *CapstoneELFTargetObjectFile::SelectSectionForGlobal(
    const GlobalObject *GO, SectionKind Kind, const TargetMachine &TM) const {
  // Handle Small Section classification here.
  if (isGlobalInSmallSection(GO, TM)) {
    // Emit to an unique sdata/sbss section when -fdata-section is set.
    // However, if a symbol has an explicit sdata/sbss section, place it in that
    // section.
    bool EmitUniquedSection = TM.getDataSections() && !GO->hasSection();

    if (Kind.isBSS()) {
      if (EmitUniquedSection) {
        SmallString<128> Name(".sbss.");
        Name.append(GO->getName());
        return getContext().getELFSection(Name.str(), ELF::SHT_NOBITS,
                                          ELF::SHF_WRITE | ELF::SHF_ALLOC);
      }

      return SmallBSSSection;
    }

    if (Kind.isData()) {
      if (EmitUniquedSection) {
        SmallString<128> Name(".sdata.");
        Name.append(GO->getName());
        return getContext().getELFSection(Name.str(), ELF::SHT_PROGBITS,
                                          ELF::SHF_WRITE | ELF::SHF_ALLOC);
      }

      return SmallDataSection;
    }
  }

  // Otherwise, we work the same as ELF.
  return TargetLoweringObjectFileELF::SelectSectionForGlobal(GO, Kind, TM);
}

void CapstoneELFTargetObjectFile::getModuleMetadata(Module &M) {
  TargetLoweringObjectFileELF::getModuleMetadata(M);
  SmallVector<Module::ModuleFlagEntry, 8> ModuleFlags;
  M.getModuleFlagsMetadata(ModuleFlags);

  for (const auto &MFE : ModuleFlags) {
    StringRef Key = MFE.Key->getString();
    if (Key == "SmallDataLimit") {
      SSThreshold = mdconst::extract<ConstantInt>(MFE.Val)->getZExtValue();
      break;
    }
  }
}

/// Return true if this constant should be placed into small data section.
bool CapstoneELFTargetObjectFile::isConstantInSmallSection(
    const DataLayout &DL, const Constant *CN) const {
  return isInSmallSection(DL.getTypeAllocSize(CN->getType()));
}

MCSection *CapstoneELFTargetObjectFile::getSectionForConstant(
    const DataLayout &DL, SectionKind Kind, const Constant *C,
    Align &Alignment) const {
  if (C && isConstantInSmallSection(DL, C)) {
    if (Kind.isMergeableConst4())
      return SmallROData4Section;
    if (Kind.isMergeableConst8())
      return SmallROData8Section;
    if (Kind.isMergeableConst16())
      return SmallROData16Section;
    if (Kind.isMergeableConst32())
      return SmallROData32Section;
    // LLVM only generate up to .rodata.cst32, and use .rodata section if more
    // than 32 bytes, so just use .srodata here.
    return SmallRODataSection;
  }

  // Under the image-gp ABI, emit mergeable constant-pool entries into .rodata
  // rather than the SHF_MERGE .rodata.cst* sections, so that a function's
  // entries stay contiguous, in constant-pool order, at the byte offsets
  // AsmPrinter::emitConstantPool assigns (alignment padding between entries),
  // which the linker never splits or reorders. CapstoneFoldMemOffset
  // (rebaseConstantPool) relies on that layout to address a function's whole
  // pool through one gp-derived base capability. The price is the merge
  // section's deduplication of identical constants across functions.
  if (capstoneImageGpAbiActive() && Kind.isMergeableConst())
    return ReadOnlySection;

  // Otherwise, we work the same as ELF.
  return TargetLoweringObjectFileELF::getSectionForConstant(DL, Kind, C,
                                                            Alignment);
}

MCSection *CapstoneELFTargetObjectFile::getSectionForConstant(
    const DataLayout &DL, SectionKind Kind, const Constant *C, Align &Alignment,
    StringRef SectionSuffix) const {
  // AsmPrinter::emitConstantPool goes through this form, with the static-data
  // partition's "hot"/"unlikely" suffix when -partition-static-data-sections
  // and a profile are active. The ELF default would then place an entry in
  // .rodata.cst8.hot. while an unprofiled neighbour stays in .rodata, and the
  // pool layout rebaseConstantPool predicted (through the four-argument form,
  // which has no profile) would no longer be the one emitted. Under the
  // image-gp ABI the suffix is therefore ignored for every entry: the layout
  // is the four-argument form's whatever the profile says.
  if (capstoneImageGpAbiActive())
    return getSectionForConstant(DL, Kind, C, Alignment);
  return TargetLoweringObjectFileELF::getSectionForConstant(DL, Kind, C,
                                                            Alignment,
                                                            SectionSuffix);
}
