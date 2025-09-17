//===------------------- CapstoneCustomBehaviour.cpp ---------------*-C++ -* -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file implements methods from the CapstoneCustomBehaviour class.
///
//===----------------------------------------------------------------------===//

#include "CapstoneCustomBehaviour.h"
#include "MCTargetDesc/CapstoneMCTargetDesc.h"
#include "Capstone.h"
#include "TargetInfo/CapstoneTargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "llvm-mca-capstone-custombehaviour"

namespace llvm::Capstone {
struct VXMemOpInfo {
  unsigned Log2IdxEEW : 3;
  unsigned IsOrdered : 1;
  unsigned IsStore : 1;
  unsigned NF : 4;
  unsigned BaseInstr;
};

#define GET_CapstoneBaseVXMemOpTable_IMPL
#include "CapstoneGenSearchableTables.inc"
} // namespace llvm::Capstone

namespace llvm {
namespace mca {

const llvm::StringRef CapstoneLMULInstrument::DESC_NAME = "Capstone-LMUL";

bool CapstoneLMULInstrument::isDataValid(llvm::StringRef Data) {
  // Return true if not one of the valid LMUL strings
  return StringSwitch<bool>(Data)
      .Cases("M1", "M2", "M4", "M8", "MF2", "MF4", "MF8", true)
      .Default(false);
}

uint8_t CapstoneLMULInstrument::getLMUL() const {
  // assertion prevents us from needing llvm_unreachable in the StringSwitch
  // below
  assert(isDataValid(getData()) &&
         "Cannot get LMUL because invalid Data value");
  // These are the LMUL values that are used in Capstone tablegen
  return StringSwitch<uint8_t>(getData())
      .Case("M1", 0b000)
      .Case("M2", 0b001)
      .Case("M4", 0b010)
      .Case("M8", 0b011)
      .Case("MF2", 0b111)
      .Case("MF4", 0b110)
      .Case("MF8", 0b101);
}

const llvm::StringRef CapstoneSEWInstrument::DESC_NAME = "Capstone-SEW";

bool CapstoneSEWInstrument::isDataValid(llvm::StringRef Data) {
  // Return true if not one of the valid SEW strings
  return StringSwitch<bool>(Data)
      .Cases("E8", "E16", "E32", "E64", true)
      .Default(false);
}

uint8_t CapstoneSEWInstrument::getSEW() const {
  // assertion prevents us from needing llvm_unreachable in the StringSwitch
  // below
  assert(isDataValid(getData()) && "Cannot get SEW because invalid Data value");
  // These are the LMUL values that are used in Capstone tablegen
  return StringSwitch<uint8_t>(getData())
      .Case("E8", 8)
      .Case("E16", 16)
      .Case("E32", 32)
      .Case("E64", 64);
}

bool CapstoneInstrumentManager::supportsInstrumentType(
    llvm::StringRef Type) const {
  return Type == CapstoneLMULInstrument::DESC_NAME ||
         Type == CapstoneSEWInstrument::DESC_NAME;
}

UniqueInstrument
CapstoneInstrumentManager::createInstrument(llvm::StringRef Desc,
                                         llvm::StringRef Data) {
  if (Desc == CapstoneLMULInstrument::DESC_NAME) {
    if (!CapstoneLMULInstrument::isDataValid(Data)) {
      LLVM_DEBUG(dbgs() << "RVCB: Bad data for instrument kind " << Desc << ": "
                        << Data << '\n');
      return nullptr;
    }
    return std::make_unique<CapstoneLMULInstrument>(Data);
  }

  if (Desc == CapstoneSEWInstrument::DESC_NAME) {
    if (!CapstoneSEWInstrument::isDataValid(Data)) {
      LLVM_DEBUG(dbgs() << "RVCB: Bad data for instrument kind " << Desc << ": "
                        << Data << '\n');
      return nullptr;
    }
    return std::make_unique<CapstoneSEWInstrument>(Data);
  }

  LLVM_DEBUG(dbgs() << "RVCB: Unknown instrumentation Desc: " << Desc << '\n');
  return nullptr;
}

SmallVector<UniqueInstrument>
CapstoneInstrumentManager::createInstruments(const MCInst &Inst) {
  if (Inst.getOpcode() == Capstone::VSETVLI ||
      Inst.getOpcode() == Capstone::VSETIVLI) {
    LLVM_DEBUG(dbgs() << "RVCB: Found VSETVLI and creating instrument for it: "
                      << Inst << "\n");
    unsigned VTypeI = Inst.getOperand(2).getImm();
    CapstoneVType::VLMUL VLMUL = CapstoneVType::getVLMUL(VTypeI);

    StringRef LMUL;
    switch (VLMUL) {
    case CapstoneVType::LMUL_1:
      LMUL = "M1";
      break;
    case CapstoneVType::LMUL_2:
      LMUL = "M2";
      break;
    case CapstoneVType::LMUL_4:
      LMUL = "M4";
      break;
    case CapstoneVType::LMUL_8:
      LMUL = "M8";
      break;
    case CapstoneVType::LMUL_F2:
      LMUL = "MF2";
      break;
    case CapstoneVType::LMUL_F4:
      LMUL = "MF4";
      break;
    case CapstoneVType::LMUL_F8:
      LMUL = "MF8";
      break;
    case CapstoneVType::LMUL_RESERVED:
      llvm_unreachable("Cannot create instrument for LMUL_RESERVED");
    }
    SmallVector<UniqueInstrument> Instruments;
    Instruments.emplace_back(
        createInstrument(CapstoneLMULInstrument::DESC_NAME, LMUL));

    unsigned SEW = CapstoneVType::getSEW(VTypeI);
    StringRef SEWStr;
    switch (SEW) {
    case 8:
      SEWStr = "E8";
      break;
    case 16:
      SEWStr = "E16";
      break;
    case 32:
      SEWStr = "E32";
      break;
    case 64:
      SEWStr = "E64";
      break;
    default:
      llvm_unreachable("Cannot create instrument for SEW");
    }
    Instruments.emplace_back(
        createInstrument(CapstoneSEWInstrument::DESC_NAME, SEWStr));

    return Instruments;
  }
  return SmallVector<UniqueInstrument>();
}

static std::pair<uint8_t, uint8_t>
getEEWAndEMUL(unsigned Opcode, CapstoneVType::VLMUL LMUL, uint8_t SEW) {
  uint8_t EEW;
  switch (Opcode) {
  case Capstone::VLM_V:
  case Capstone::VSM_V:
  case Capstone::VLE8_V:
  case Capstone::VSE8_V:
  case Capstone::VLSE8_V:
  case Capstone::VSSE8_V:
    EEW = 8;
    break;
  case Capstone::VLE16_V:
  case Capstone::VSE16_V:
  case Capstone::VLSE16_V:
  case Capstone::VSSE16_V:
    EEW = 16;
    break;
  case Capstone::VLE32_V:
  case Capstone::VSE32_V:
  case Capstone::VLSE32_V:
  case Capstone::VSSE32_V:
    EEW = 32;
    break;
  case Capstone::VLE64_V:
  case Capstone::VSE64_V:
  case Capstone::VLSE64_V:
  case Capstone::VSSE64_V:
    EEW = 64;
    break;
  default:
    llvm_unreachable("Could not determine EEW from Opcode");
  }

  auto EMUL = CapstoneVType::getSameRatioLMUL(SEW, LMUL, EEW);
  if (!EEW)
    llvm_unreachable("Invalid SEW or LMUL for new ratio");
  return std::make_pair(EEW, *EMUL);
}

static bool opcodeHasEEWAndEMULInfo(unsigned short Opcode) {
  return Opcode == Capstone::VLM_V || Opcode == Capstone::VSM_V ||
         Opcode == Capstone::VLE8_V || Opcode == Capstone::VSE8_V ||
         Opcode == Capstone::VLE16_V || Opcode == Capstone::VSE16_V ||
         Opcode == Capstone::VLE32_V || Opcode == Capstone::VSE32_V ||
         Opcode == Capstone::VLE64_V || Opcode == Capstone::VSE64_V ||
         Opcode == Capstone::VLSE8_V || Opcode == Capstone::VSSE8_V ||
         Opcode == Capstone::VLSE16_V || Opcode == Capstone::VSSE16_V ||
         Opcode == Capstone::VLSE32_V || Opcode == Capstone::VSSE32_V ||
         Opcode == Capstone::VLSE64_V || Opcode == Capstone::VSSE64_V;
}

unsigned CapstoneInstrumentManager::getSchedClassID(
    const MCInstrInfo &MCII, const MCInst &MCI,
    const llvm::SmallVector<Instrument *> &IVec) const {
  unsigned short Opcode = MCI.getOpcode();
  unsigned SchedClassID = MCII.get(Opcode).getSchedClass();

  // Unpack all possible Capstone instruments from IVec.
  CapstoneLMULInstrument *LI = nullptr;
  CapstoneSEWInstrument *SI = nullptr;
  for (auto &I : IVec) {
    if (I->getDesc() == CapstoneLMULInstrument::DESC_NAME)
      LI = static_cast<CapstoneLMULInstrument *>(I);
    else if (I->getDesc() == CapstoneSEWInstrument::DESC_NAME)
      SI = static_cast<CapstoneSEWInstrument *>(I);
  }

  // Need LMUL or LMUL, SEW in order to override opcode. If no LMUL is provided,
  // then no option to override.
  if (!LI) {
    LLVM_DEBUG(
        dbgs() << "RVCB: Did not use instrumentation to override Opcode.\n");
    return SchedClassID;
  }
  uint8_t LMUL = LI->getLMUL();

  // getBaseInfo works with (Opcode, LMUL, 0) if no SEW instrument,
  // or (Opcode, LMUL, SEW) if SEW instrument is active, and depends on LMUL
  // and SEW, or (Opcode, LMUL, 0) if does not depend on SEW.
  uint8_t SEW = SI ? SI->getSEW() : 0;

  std::optional<unsigned> VPOpcode;
  if (const auto *VXMO = Capstone::getVXMemOpInfo(Opcode)) {
    // Calculate the expected index EMUL. For indexed operations,
    // the DataEEW and DataEMUL are equal to SEW and LMUL, respectively.
    unsigned IndexEMUL = ((1 << VXMO->Log2IdxEEW) * LMUL) / SEW;

    if (!VXMO->NF) {
      // Indexed Load / Store.
      if (VXMO->IsStore) {
        if (const auto *VXP = Capstone::getVSXPseudo(
                /*Masked=*/0, VXMO->IsOrdered, VXMO->Log2IdxEEW, LMUL,
                IndexEMUL))
          VPOpcode = VXP->Pseudo;
      } else {
        if (const auto *VXP = Capstone::getVLXPseudo(
                /*Masked=*/0, VXMO->IsOrdered, VXMO->Log2IdxEEW, LMUL,
                IndexEMUL))
          VPOpcode = VXP->Pseudo;
      }
    } else {
      // Segmented Indexed Load / Store.
      if (VXMO->IsStore) {
        if (const auto *VXP =
                Capstone::getVSXSEGPseudo(VXMO->NF, /*Masked=*/0, VXMO->IsOrdered,
                                       VXMO->Log2IdxEEW, LMUL, IndexEMUL))
          VPOpcode = VXP->Pseudo;
      } else {
        if (const auto *VXP =
                Capstone::getVLXSEGPseudo(VXMO->NF, /*Masked=*/0, VXMO->IsOrdered,
                                       VXMO->Log2IdxEEW, LMUL, IndexEMUL))
          VPOpcode = VXP->Pseudo;
      }
    }
  } else if (opcodeHasEEWAndEMULInfo(Opcode)) {
    CapstoneVType::VLMUL VLMUL = static_cast<CapstoneVType::VLMUL>(LMUL);
    auto [EEW, EMUL] = getEEWAndEMUL(Opcode, VLMUL, SEW);
    if (const auto *RVV =
            CapstoneVInversePseudosTable::getBaseInfo(Opcode, EMUL, EEW))
      VPOpcode = RVV->Pseudo;
  } else {
    // Check if it depends on LMUL and SEW
    const auto *RVV = CapstoneVInversePseudosTable::getBaseInfo(Opcode, LMUL, SEW);
    // Check if it depends only on LMUL
    if (!RVV)
      RVV = CapstoneVInversePseudosTable::getBaseInfo(Opcode, LMUL, 0);

    if (RVV)
      VPOpcode = RVV->Pseudo;
  }

  // Not a RVV instr
  if (!VPOpcode) {
    LLVM_DEBUG(
        dbgs() << "RVCB: Could not find PseudoInstruction for Opcode "
               << MCII.getName(Opcode)
               << ", LMUL=" << (LI ? LI->getData() : "Unspecified")
               << ", SEW=" << (SI ? SI->getData() : "Unspecified")
               << ". Ignoring instrumentation and using original SchedClassID="
               << SchedClassID << '\n');
    return SchedClassID;
  }

  // Override using pseudo
  LLVM_DEBUG(dbgs() << "RVCB: Found Pseudo Instruction for Opcode "
                    << MCII.getName(Opcode) << ", LMUL=" << LI->getData()
                    << ", SEW=" << (SI ? SI->getData() : "Unspecified")
                    << ". Overriding original SchedClassID=" << SchedClassID
                    << " with " << MCII.getName(*VPOpcode) << '\n');
  return MCII.get(*VPOpcode).getSchedClass();
}

} // namespace mca
} // namespace llvm

using namespace llvm;
using namespace mca;

static InstrumentManager *
createCapstoneInstrumentManager(const MCSubtargetInfo &STI,
                             const MCInstrInfo &MCII) {
  return new CapstoneInstrumentManager(STI, MCII);
}

/// Extern function to initialize the targets for the Capstone backend
extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void
LLVMInitializeCapstoneTargetMCA() {
  TargetRegistry::RegisterInstrumentManager(getTheCapstone32Target(),
                                            createCapstoneInstrumentManager);
  TargetRegistry::RegisterInstrumentManager(getTheCapstone64Target(),
                                            createCapstoneInstrumentManager);
}
