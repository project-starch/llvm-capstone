//===-- CapstoneDisassembler.cpp - Disassembler for Capstone -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the CapstoneDisassembler class.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/CapstoneBaseInfo.h"
#include "MCTargetDesc/CapstoneMCTargetDesc.h"
#include "TargetInfo/CapstoneTargetInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDecoder.h"
#include "llvm/MC/MCDecoderOps.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Endian.h"

using namespace llvm;
using namespace llvm::MCD;

#define DEBUG_TYPE "capstone-disassembler"

typedef MCDisassembler::DecodeStatus DecodeStatus;

namespace {
class CapstoneDisassembler : public MCDisassembler {
  std::unique_ptr<MCInstrInfo const> const MCII;

public:
  CapstoneDisassembler(const MCSubtargetInfo &STI, MCContext &Ctx,
                    MCInstrInfo const *MCII)
      : MCDisassembler(STI, Ctx), MCII(MCII) {}

  DecodeStatus getInstruction(MCInst &Instr, uint64_t &Size,
                              ArrayRef<uint8_t> Bytes, uint64_t Address,
                              raw_ostream &CStream) const override;

private:
  DecodeStatus getInstruction48(MCInst &Instr, uint64_t &Size,
                                ArrayRef<uint8_t> Bytes, uint64_t Address,
                                raw_ostream &CStream) const;

  DecodeStatus getInstruction32(MCInst &Instr, uint64_t &Size,
                                ArrayRef<uint8_t> Bytes, uint64_t Address,
                                raw_ostream &CStream) const;
  DecodeStatus getInstruction16(MCInst &Instr, uint64_t &Size,
                                ArrayRef<uint8_t> Bytes, uint64_t Address,
                                raw_ostream &CStream) const;
};
} // end anonymous namespace

static MCDisassembler *createCapstoneDisassembler(const Target &T,
                                               const MCSubtargetInfo &STI,
                                               MCContext &Ctx) {
  return new CapstoneDisassembler(STI, Ctx, T.createMCInstrInfo());
}

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void
LLVMInitializeCapstoneDisassembler() {
  // Register the disassembler for each target.
  TargetRegistry::RegisterMCDisassembler(getTheCapstone32Target(),
                                         createCapstoneDisassembler);
  TargetRegistry::RegisterMCDisassembler(getTheCapstone64Target(),
                                         createCapstoneDisassembler);
  TargetRegistry::RegisterMCDisassembler(getTheCapstone32beTarget(),
                                         createCapstoneDisassembler);
  TargetRegistry::RegisterMCDisassembler(getTheCapstone64beTarget(),
                                         createCapstoneDisassembler);
}

static DecodeStatus DecodeGPRRegisterClass(MCInst &Inst, uint32_t RegNo,
                                           uint64_t Address,
                                           const MCDisassembler *Decoder) {
  bool IsRVE = Decoder->getSubtargetInfo().hasFeature(Capstone::FeatureStdExtE);

  if (RegNo >= 32 || (IsRVE && RegNo >= 16))
    return MCDisassembler::Fail;

  MCRegister Reg = Capstone::X0 + RegNo;
  Inst.addOperand(MCOperand::createReg(Reg));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeGPRF16RegisterClass(MCInst &Inst, uint32_t RegNo,
                                              uint64_t Address,
                                              const MCDisassembler *Decoder) {
  bool IsRVE = Decoder->getSubtargetInfo().hasFeature(Capstone::FeatureStdExtE);

  if (RegNo >= 32 || (IsRVE && RegNo >= 16))
    return MCDisassembler::Fail;

  MCRegister Reg = Capstone::X0_H + RegNo;
  Inst.addOperand(MCOperand::createReg(Reg));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeGPRF32RegisterClass(MCInst &Inst, uint32_t RegNo,
                                              uint64_t Address,
                                              const MCDisassembler *Decoder) {
  bool IsRVE = Decoder->getSubtargetInfo().hasFeature(Capstone::FeatureStdExtE);

  if (RegNo >= 32 || (IsRVE && RegNo >= 16))
    return MCDisassembler::Fail;

  MCRegister Reg = Capstone::X0_W + RegNo;
  Inst.addOperand(MCOperand::createReg(Reg));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeGPRX1X5RegisterClass(MCInst &Inst, uint32_t RegNo,
                                               uint64_t Address,
                                               const MCDisassembler *Decoder) {
  MCRegister Reg = Capstone::X0 + RegNo;
  if (Reg != Capstone::X1 && Reg != Capstone::X5)
    return MCDisassembler::Fail;

  Inst.addOperand(MCOperand::createReg(Reg));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeFPR16RegisterClass(MCInst &Inst, uint32_t RegNo,
                                             uint64_t Address,
                                             const MCDisassembler *Decoder) {
  if (RegNo >= 32)
    return MCDisassembler::Fail;

  MCRegister Reg = Capstone::F0_H + RegNo;
  Inst.addOperand(MCOperand::createReg(Reg));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeFPR32RegisterClass(MCInst &Inst, uint32_t RegNo,
                                             uint64_t Address,
                                             const MCDisassembler *Decoder) {
  if (RegNo >= 32)
    return MCDisassembler::Fail;

  MCRegister Reg = Capstone::F0_F + RegNo;
  Inst.addOperand(MCOperand::createReg(Reg));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeFPR32CRegisterClass(MCInst &Inst, uint32_t RegNo,
                                              uint64_t Address,
                                              const MCDisassembler *Decoder) {
  if (RegNo >= 8) {
    return MCDisassembler::Fail;
  }
  MCRegister Reg = Capstone::F8_F + RegNo;
  Inst.addOperand(MCOperand::createReg(Reg));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeFPR64RegisterClass(MCInst &Inst, uint32_t RegNo,
                                             uint64_t Address,
                                             const MCDisassembler *Decoder) {
  if (RegNo >= 32)
    return MCDisassembler::Fail;

  MCRegister Reg = Capstone::F0_D + RegNo;
  Inst.addOperand(MCOperand::createReg(Reg));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeFPR64CRegisterClass(MCInst &Inst, uint32_t RegNo,
                                              uint64_t Address,
                                              const MCDisassembler *Decoder) {
  if (RegNo >= 8) {
    return MCDisassembler::Fail;
  }
  MCRegister Reg = Capstone::F8_D + RegNo;
  Inst.addOperand(MCOperand::createReg(Reg));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeFPR128RegisterClass(MCInst &Inst, uint32_t RegNo,
                                              uint64_t Address,
                                              const MCDisassembler *Decoder) {
  if (RegNo >= 32)
    return MCDisassembler::Fail;

  MCRegister Reg = Capstone::F0_Q + RegNo;
  Inst.addOperand(MCOperand::createReg(Reg));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeGPRX1RegisterClass(MCInst &Inst,
                                             const MCDisassembler *Decoder) {
  Inst.addOperand(MCOperand::createReg(Capstone::X1));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeSPRegisterClass(MCInst &Inst,
                                          const MCDisassembler *Decoder) {
  Inst.addOperand(MCOperand::createReg(Capstone::X2));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeGPRX5RegisterClass(MCInst &Inst,
                                             const MCDisassembler *Decoder) {
  Inst.addOperand(MCOperand::createReg(Capstone::X5));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeGPRNoX0RegisterClass(MCInst &Inst, uint32_t RegNo,
                                               uint64_t Address,
                                               const MCDisassembler *Decoder) {
  if (RegNo == 0)
    return MCDisassembler::Fail;

  return DecodeGPRRegisterClass(Inst, RegNo, Address, Decoder);
}

static DecodeStatus DecodeGPRNoX2RegisterClass(MCInst &Inst, uint64_t RegNo,
                                               uint32_t Address,
                                               const MCDisassembler *Decoder) {
  if (RegNo == 2)
    return MCDisassembler::Fail;

  return DecodeGPRRegisterClass(Inst, RegNo, Address, Decoder);
}

static DecodeStatus DecodeGPRNoX31RegisterClass(MCInst &Inst, uint32_t RegNo,
                                                uint64_t Address,
                                                const MCDisassembler *Decoder) {
  if (RegNo == 31) {
    return MCDisassembler::Fail;
  }

  return DecodeGPRRegisterClass(Inst, RegNo, Address, Decoder);
}

static DecodeStatus DecodeGPRCRegisterClass(MCInst &Inst, uint32_t RegNo,
                                            uint64_t Address,
                                            const MCDisassembler *Decoder) {
  if (RegNo >= 8)
    return MCDisassembler::Fail;

  MCRegister Reg = Capstone::X8 + RegNo;
  Inst.addOperand(MCOperand::createReg(Reg));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeGPRPairRegisterClass(MCInst &Inst, uint32_t RegNo,
                                               uint64_t Address,
                                               const MCDisassembler *Decoder) {
  if (RegNo >= 32 || RegNo % 2)
    return MCDisassembler::Fail;

  const CapstoneDisassembler *Dis =
      static_cast<const CapstoneDisassembler *>(Decoder);
  const MCRegisterInfo *RI = Dis->getContext().getRegisterInfo();
  MCRegister Reg = RI->getMatchingSuperReg(
      Capstone::X0 + RegNo, Capstone::sub_gpr_even,
      &CapstoneMCRegisterClasses[Capstone::GPRPairRegClassID]);
  Inst.addOperand(MCOperand::createReg(Reg));
  return MCDisassembler::Success;
}

static DecodeStatus
DecodeGPRPairNoX0RegisterClass(MCInst &Inst, uint32_t RegNo, uint64_t Address,
                               const MCDisassembler *Decoder) {
  if (RegNo == 0)
    return MCDisassembler::Fail;

  return DecodeGPRPairRegisterClass(Inst, RegNo, Address, Decoder);
}

static DecodeStatus DecodeGPRPairCRegisterClass(MCInst &Inst, uint32_t RegNo,
                                                uint64_t Address,
                                                const MCDisassembler *Decoder) {
  if (RegNo >= 8 || RegNo % 2)
    return MCDisassembler::Fail;

  const CapstoneDisassembler *Dis =
      static_cast<const CapstoneDisassembler *>(Decoder);
  const MCRegisterInfo *RI = Dis->getContext().getRegisterInfo();
  MCRegister Reg = RI->getMatchingSuperReg(
      Capstone::X8 + RegNo, Capstone::sub_gpr_even,
      &CapstoneMCRegisterClasses[Capstone::GPRPairCRegClassID]);
  Inst.addOperand(MCOperand::createReg(Reg));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeSR07RegisterClass(MCInst &Inst, uint32_t RegNo,
                                            uint64_t Address,
                                            const void *Decoder) {
  if (RegNo >= 8)
    return MCDisassembler::Fail;

  MCRegister Reg = (RegNo < 2) ? (RegNo + Capstone::X8) : (RegNo - 2 + Capstone::X18);
  Inst.addOperand(MCOperand::createReg(Reg));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeVRRegisterClass(MCInst &Inst, uint32_t RegNo,
                                          uint64_t Address,
                                          const MCDisassembler *Decoder) {
  if (RegNo >= 32)
    return MCDisassembler::Fail;

  MCRegister Reg = Capstone::V0 + RegNo;
  Inst.addOperand(MCOperand::createReg(Reg));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeVRM2RegisterClass(MCInst &Inst, uint32_t RegNo,
                                            uint64_t Address,
                                            const MCDisassembler *Decoder) {
  if (RegNo >= 32 || RegNo % 2)
    return MCDisassembler::Fail;

  const CapstoneDisassembler *Dis =
      static_cast<const CapstoneDisassembler *>(Decoder);
  const MCRegisterInfo *RI = Dis->getContext().getRegisterInfo();
  MCRegister Reg =
      RI->getMatchingSuperReg(Capstone::V0 + RegNo, Capstone::sub_vrm1_0,
                              &CapstoneMCRegisterClasses[Capstone::VRM2RegClassID]);

  Inst.addOperand(MCOperand::createReg(Reg));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeVRM4RegisterClass(MCInst &Inst, uint32_t RegNo,
                                            uint64_t Address,
                                            const MCDisassembler *Decoder) {
  if (RegNo >= 32 || RegNo % 4)
    return MCDisassembler::Fail;

  const CapstoneDisassembler *Dis =
      static_cast<const CapstoneDisassembler *>(Decoder);
  const MCRegisterInfo *RI = Dis->getContext().getRegisterInfo();
  MCRegister Reg =
      RI->getMatchingSuperReg(Capstone::V0 + RegNo, Capstone::sub_vrm1_0,
                              &CapstoneMCRegisterClasses[Capstone::VRM4RegClassID]);

  Inst.addOperand(MCOperand::createReg(Reg));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeVRM8RegisterClass(MCInst &Inst, uint32_t RegNo,
                                            uint64_t Address,
                                            const MCDisassembler *Decoder) {
  if (RegNo >= 32 || RegNo % 8)
    return MCDisassembler::Fail;

  const CapstoneDisassembler *Dis =
      static_cast<const CapstoneDisassembler *>(Decoder);
  const MCRegisterInfo *RI = Dis->getContext().getRegisterInfo();
  MCRegister Reg =
      RI->getMatchingSuperReg(Capstone::V0 + RegNo, Capstone::sub_vrm1_0,
                              &CapstoneMCRegisterClasses[Capstone::VRM8RegClassID]);

  Inst.addOperand(MCOperand::createReg(Reg));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeVMV0RegisterClass(MCInst &Inst, uint32_t RegNo,
                                            uint64_t Address,
                                            const MCDisassembler *Decoder) {
  if (RegNo)
    return MCDisassembler::Fail;

  Inst.addOperand(MCOperand::createReg(Capstone::V0));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeTRRegisterClass(MCInst &Inst, uint32_t RegNo,
                                          uint64_t Address,
                                          const MCDisassembler *Decoder) {
  if (RegNo > 15)
    return MCDisassembler::Fail;

  MCRegister Reg = Capstone::T0 + RegNo;
  Inst.addOperand(MCOperand::createReg(Reg));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeTRM2RegisterClass(MCInst &Inst, uint32_t RegNo,
                                            uint64_t Address,
                                            const MCDisassembler *Decoder) {
  if (RegNo > 15 || RegNo % 2)
    return MCDisassembler::Fail;

  MCRegister Reg = Capstone::T0 + RegNo;
  Inst.addOperand(MCOperand::createReg(Reg));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeTRM4RegisterClass(MCInst &Inst, uint32_t RegNo,
                                            uint64_t Address,
                                            const MCDisassembler *Decoder) {
  if (RegNo > 15 || RegNo % 4)
    return MCDisassembler::Fail;

  MCRegister Reg = Capstone::T0 + RegNo;
  Inst.addOperand(MCOperand::createReg(Reg));
  return MCDisassembler::Success;
}

static DecodeStatus decodeVMaskReg(MCInst &Inst, uint32_t RegNo,
                                   uint64_t Address,
                                   const MCDisassembler *Decoder) {
  if (RegNo >= 2)
    return MCDisassembler::Fail;

  MCRegister Reg = (RegNo == 0) ? Capstone::V0 : Capstone::NoRegister;

  Inst.addOperand(MCOperand::createReg(Reg));
  return MCDisassembler::Success;
}

static DecodeStatus decodeImmThreeOperand(MCInst &Inst,
                                          const MCDisassembler *Decoder) {
  Inst.addOperand(MCOperand::createImm(3));
  return MCDisassembler::Success;
}

static DecodeStatus decodeImmFourOperand(MCInst &Inst,
                                         const MCDisassembler *Decoder) {
  Inst.addOperand(MCOperand::createImm(4));
  return MCDisassembler::Success;
}

template <unsigned N>
static DecodeStatus decodeUImmOperand(MCInst &Inst, uint32_t Imm,
                                      int64_t Address,
                                      const MCDisassembler *Decoder) {
  assert(isUInt<N>(Imm) && "Invalid immediate");
  Inst.addOperand(MCOperand::createImm(Imm));
  return MCDisassembler::Success;
}

template <unsigned Width, unsigned LowerBound>
static DecodeStatus decodeUImmOperandGE(MCInst &Inst, uint32_t Imm,
                                        int64_t Address,
                                        const MCDisassembler *Decoder) {
  assert(isUInt<Width>(Imm) && "Invalid immediate");

  if (Imm < LowerBound)
    return MCDisassembler::Fail;

  Inst.addOperand(MCOperand::createImm(Imm));
  return MCDisassembler::Success;
}

template <unsigned Width, unsigned LowerBound>
static DecodeStatus decodeUImmPlus1OperandGE(MCInst &Inst, uint32_t Imm,
                                             int64_t Address,
                                             const MCDisassembler *Decoder) {
  assert(isUInt<Width>(Imm) && "Invalid immediate");

  if ((Imm + 1) < LowerBound)
    return MCDisassembler::Fail;

  Inst.addOperand(MCOperand::createImm(Imm + 1));
  return MCDisassembler::Success;
}

static DecodeStatus decodeUImmSlistOperand(MCInst &Inst, uint32_t Imm,
                                           int64_t Address,
                                           const MCDisassembler *Decoder) {
  assert(isUInt<3>(Imm) && "Invalid Slist immediate");
  const uint8_t Slist[] = {0, 1, 2, 4, 8, 16, 15, 31};
  Inst.addOperand(MCOperand::createImm(Slist[Imm]));
  return MCDisassembler::Success;
}

static DecodeStatus decodeUImmLog2XLenOperand(MCInst &Inst, uint32_t Imm,
                                              int64_t Address,
                                              const MCDisassembler *Decoder) {
  assert(isUInt<6>(Imm) && "Invalid immediate");

  if (!Decoder->getSubtargetInfo().hasFeature(Capstone::Feature64Bit) &&
      !isUInt<5>(Imm))
    return MCDisassembler::Fail;

  Inst.addOperand(MCOperand::createImm(Imm));
  return MCDisassembler::Success;
}

template <unsigned N>
static DecodeStatus decodeUImmNonZeroOperand(MCInst &Inst, uint32_t Imm,
                                             int64_t Address,
                                             const MCDisassembler *Decoder) {
  if (Imm == 0)
    return MCDisassembler::Fail;
  return decodeUImmOperand<N>(Inst, Imm, Address, Decoder);
}

static DecodeStatus
decodeUImmLog2XLenNonZeroOperand(MCInst &Inst, uint32_t Imm, int64_t Address,
                                 const MCDisassembler *Decoder) {
  if (Imm == 0)
    return MCDisassembler::Fail;
  return decodeUImmLog2XLenOperand(Inst, Imm, Address, Decoder);
}

template <unsigned N>
static DecodeStatus decodeUImmPlus1Operand(MCInst &Inst, uint32_t Imm,
                                           int64_t Address,
                                           const MCDisassembler *Decoder) {
  assert(isUInt<N>(Imm) && "Invalid immediate");
  Inst.addOperand(MCOperand::createImm(Imm + 1));
  return MCDisassembler::Success;
}

static DecodeStatus decodeImmZibiOperand(MCInst &Inst, uint32_t Imm,
                                         int64_t Address,
                                         const MCDisassembler *Decoder) {
  assert(isUInt<5>(Imm) && "Invalid immediate");
  Inst.addOperand(MCOperand::createImm(Imm ? Imm : -1LL));
  return MCDisassembler::Success;
}

template <unsigned N>
static DecodeStatus decodeSImmOperand(MCInst &Inst, uint32_t Imm,
                                      int64_t Address,
                                      const MCDisassembler *Decoder) {
  assert(isUInt<N>(Imm) && "Invalid immediate");
  // Sign-extend the number in the bottom N bits of Imm
  Inst.addOperand(MCOperand::createImm(SignExtend64<N>(Imm)));
  return MCDisassembler::Success;
}

template <unsigned N>
static DecodeStatus decodeSImmNonZeroOperand(MCInst &Inst, uint32_t Imm,
                                             int64_t Address,
                                             const MCDisassembler *Decoder) {
  if (Imm == 0)
    return MCDisassembler::Fail;
  return decodeSImmOperand<N>(Inst, Imm, Address, Decoder);
}

template <unsigned T, unsigned N>
static DecodeStatus decodeSImmOperandAndLslN(MCInst &Inst, uint32_t Imm,
                                             int64_t Address,
                                             const MCDisassembler *Decoder) {
  assert(isUInt<T - N + 1>(Imm) && "Invalid immediate");
  // Sign-extend the number in the bottom T bits of Imm after accounting for
  // the fact that the T bit immediate is stored in T-N bits (the LSB is
  // always zero)
  Inst.addOperand(MCOperand::createImm(SignExtend64<T>(Imm << N)));
  return MCDisassembler::Success;
}

static DecodeStatus decodeCLUIImmOperand(MCInst &Inst, uint32_t Imm,
                                         int64_t Address,
                                         const MCDisassembler *Decoder) {
  assert(isUInt<6>(Imm) && "Invalid immediate");
  if (Imm == 0)
    return MCDisassembler::Fail;
  Imm = SignExtend64<6>(Imm) & 0xfffff;
  Inst.addOperand(MCOperand::createImm(Imm));
  return MCDisassembler::Success;
}

static DecodeStatus decodeFRMArg(MCInst &Inst, uint32_t Imm, int64_t Address,
                                 const MCDisassembler *Decoder) {
  assert(isUInt<3>(Imm) && "Invalid immediate");
  if (!llvm::CapstoneFPRndMode::isValidRoundingMode(Imm))
    return MCDisassembler::Fail;

  Inst.addOperand(MCOperand::createImm(Imm));
  return MCDisassembler::Success;
}

static DecodeStatus decodeRTZArg(MCInst &Inst, uint32_t Imm, int64_t Address,
                                 const MCDisassembler *Decoder) {
  assert(isUInt<3>(Imm) && "Invalid immediate");
  if (Imm != CapstoneFPRndMode::RTZ)
    return MCDisassembler::Fail;

  Inst.addOperand(MCOperand::createImm(Imm));
  return MCDisassembler::Success;
}

static DecodeStatus decodeZcmpRlist(MCInst &Inst, uint32_t Imm,
                                    uint64_t Address,
                                    const MCDisassembler *Decoder) {
  bool IsRVE = Decoder->getSubtargetInfo().hasFeature(Capstone::FeatureStdExtE);
  if (Imm < CapstoneZC::RA || (IsRVE && Imm >= CapstoneZC::RA_S0_S2))
    return MCDisassembler::Fail;
  Inst.addOperand(MCOperand::createImm(Imm));
  return MCDisassembler::Success;
}

static DecodeStatus decodeXqccmpRlistS0(MCInst &Inst, uint32_t Imm,
                                        uint64_t Address,
                                        const MCDisassembler *Decoder) {
  if (Imm < CapstoneZC::RA_S0)
    return MCDisassembler::Fail;
  return decodeZcmpRlist(Inst, Imm, Address, Decoder);
}

#include "CapstoneGenDisassemblerTables.inc"

namespace {

struct DecoderListEntry {
  const uint8_t *Table;
  FeatureBitset ContainedFeatures;
  const char *Desc;

  bool haveContainedFeatures(const FeatureBitset &ActiveFeatures) const {
    return ContainedFeatures.none() ||
           (ContainedFeatures & ActiveFeatures).any();
  }
};

} // end anonymous namespace

static constexpr FeatureBitset XCVFeatureGroup = {
    Capstone::FeatureVendorXCVbitmanip, Capstone::FeatureVendorXCVelw,
    Capstone::FeatureVendorXCVmac,      Capstone::FeatureVendorXCVmem,
    Capstone::FeatureVendorXCValu,      Capstone::FeatureVendorXCVsimd,
    Capstone::FeatureVendorXCVbi};

static constexpr FeatureBitset XRivosFeatureGroup = {
    Capstone::FeatureVendorXRivosVisni,
    Capstone::FeatureVendorXRivosVizip,
};

static constexpr FeatureBitset XqciFeatureGroup = {
    Capstone::FeatureVendorXqcia,   Capstone::FeatureVendorXqciac,
    Capstone::FeatureVendorXqcibi,  Capstone::FeatureVendorXqcibm,
    Capstone::FeatureVendorXqcicli, Capstone::FeatureVendorXqcicm,
    Capstone::FeatureVendorXqcics,  Capstone::FeatureVendorXqcicsr,
    Capstone::FeatureVendorXqciint, Capstone::FeatureVendorXqciio,
    Capstone::FeatureVendorXqcilb,  Capstone::FeatureVendorXqcili,
    Capstone::FeatureVendorXqcilia, Capstone::FeatureVendorXqcilo,
    Capstone::FeatureVendorXqcilsm, Capstone::FeatureVendorXqcisim,
    Capstone::FeatureVendorXqcisls, Capstone::FeatureVendorXqcisync,
};

static constexpr FeatureBitset XSfVectorGroup = {
    Capstone::FeatureVendorXSfvcp,          Capstone::FeatureVendorXSfvqmaccdod,
    Capstone::FeatureVendorXSfvqmaccqoq,    Capstone::FeatureVendorXSfvfwmaccqqq,
    Capstone::FeatureVendorXSfvfnrclipxfqf, Capstone::FeatureVendorXSfmmbase};
static constexpr FeatureBitset XSfSystemGroup = {
    Capstone::FeatureVendorXSiFivecdiscarddlone,
    Capstone::FeatureVendorXSiFivecflushdlone,
};

static constexpr FeatureBitset XMIPSGroup = {
    Capstone::FeatureVendorXMIPSLSP,
    Capstone::FeatureVendorXMIPSCMov,
    Capstone::FeatureVendorXMIPSCBOP,
    Capstone::FeatureVendorXMIPSEXECTL,
};

static constexpr FeatureBitset XTHeadGroup = {
    Capstone::FeatureVendorXTHeadBa,      Capstone::FeatureVendorXTHeadBb,
    Capstone::FeatureVendorXTHeadBs,      Capstone::FeatureVendorXTHeadCondMov,
    Capstone::FeatureVendorXTHeadCmo,     Capstone::FeatureVendorXTHeadFMemIdx,
    Capstone::FeatureVendorXTHeadMac,     Capstone::FeatureVendorXTHeadMemIdx,
    Capstone::FeatureVendorXTHeadMemPair, Capstone::FeatureVendorXTHeadSync,
    Capstone::FeatureVendorXTHeadVdot};

static constexpr FeatureBitset XAndesGroup = {
    Capstone::FeatureVendorXAndesPerf, Capstone::FeatureVendorXAndesBFHCvt,
    Capstone::FeatureVendorXAndesVBFHCvt,
    Capstone::FeatureVendorXAndesVSIntLoad, Capstone::FeatureVendorXAndesVPackFPH,
    Capstone::FeatureVendorXAndesVDot};

static constexpr FeatureBitset XSMTGroup = {Capstone::FeatureVendorXSMTVDot};

static constexpr DecoderListEntry DecoderList32[]{
    // Vendor Extensions
    {DecoderTableXCV32, XCVFeatureGroup, "CORE-V extensions"},
    {DecoderTableXRivos32, XRivosFeatureGroup, "Rivos"},
    {DecoderTableXqci32, XqciFeatureGroup, "Qualcomm uC Extensions"},
    {DecoderTableXVentana32,
     {Capstone::FeatureVendorXVentanaCondOps},
     "XVentanaCondOps"},
    {DecoderTableXTHead32, XTHeadGroup, "T-Head extensions"},
    {DecoderTableXSfvector32, XSfVectorGroup, "SiFive vector extensions"},
    {DecoderTableXSfsystem32, XSfSystemGroup, "SiFive system extensions"},
    {DecoderTableXSfcease32, {Capstone::FeatureVendorXSfcease}, "SiFive sf.cease"},
    {DecoderTableXMIPS32, XMIPSGroup, "Mips extensions"},
    {DecoderTableXAndes32, XAndesGroup, "Andes extensions"},
    {DecoderTableXSMT32, XSMTGroup, "SpacemiT extensions"},
    // Standard Extensions
    {DecoderTable32, {}, "standard 32-bit instructions"},
    {DecoderTableRV32Only32, {}, "RV32-only standard 32-bit instructions"},
    {DecoderTableZfinx32, {}, "Zfinx (Float in Integer)"},
    {DecoderTableZdinxRV32Only32, {}, "RV32-only Zdinx (Double in Integer)"},
};

namespace {
// Define bitwidths for various types used to instantiate the decoder.
template <> constexpr uint32_t InsnBitWidth<uint16_t> = 16;
template <> constexpr uint32_t InsnBitWidth<uint32_t> = 32;
// Use uint64_t to represent 48 bit instructions.
template <> constexpr uint32_t InsnBitWidth<uint64_t> = 48;
} // namespace

DecodeStatus CapstoneDisassembler::getInstruction32(MCInst &MI, uint64_t &Size,
                                                 ArrayRef<uint8_t> Bytes,
                                                 uint64_t Address,
                                                 raw_ostream &CS) const {
  if (Bytes.size() < 4) {
    Size = 0;
    return MCDisassembler::Fail;
  }
  Size = 4;

  uint32_t Insn = support::endian::read32le(Bytes.data());

  for (const DecoderListEntry &Entry : DecoderList32) {
    if (!Entry.haveContainedFeatures(STI.getFeatureBits()))
      continue;

    LLVM_DEBUG(dbgs() << "Trying " << Entry.Desc << " table:\n");
    DecodeStatus Result =
        decodeInstruction(Entry.Table, MI, Insn, Address, this, STI);
    if (Result == MCDisassembler::Fail)
      continue;

    return Result;
  }

  return MCDisassembler::Fail;
}

static constexpr DecoderListEntry DecoderList16[]{
    // Vendor Extensions
    {DecoderTableXqci16, XqciFeatureGroup, "Qualcomm uC 16-bit"},
    {DecoderTableXqccmp16,
     {Capstone::FeatureVendorXqccmp},
     "Xqccmp (Qualcomm 16-bit Push/Pop & Double Move Instructions)"},
    {DecoderTableXwchc16, {Capstone::FeatureVendorXwchc}, "WCH QingKe XW"},
    // Standard Extensions
    // DecoderTableZicfiss16 must be checked before DecoderTable16.
    {DecoderTableZicfiss16, {}, "Zicfiss (Shadow Stack 16-bit)"},
    {DecoderTable16, {}, "standard 16-bit instructions"},
    {DecoderTableRV32Only16, {}, "RV32-only 16-bit instructions"},
    // Zc* instructions incompatible with Zcf or Zcd
    {DecoderTableZcOverlap16,
     {},
     "ZcOverlap (16-bit Instructions overlapping with Zcf/Zcd)"},
};

DecodeStatus CapstoneDisassembler::getInstruction16(MCInst &MI, uint64_t &Size,
                                                 ArrayRef<uint8_t> Bytes,
                                                 uint64_t Address,
                                                 raw_ostream &CS) const {
  if (Bytes.size() < 2) {
    Size = 0;
    return MCDisassembler::Fail;
  }
  Size = 2;

  uint16_t Insn = support::endian::read16le(Bytes.data());

  for (const DecoderListEntry &Entry : DecoderList16) {
    if (!Entry.haveContainedFeatures(STI.getFeatureBits()))
      continue;

    LLVM_DEBUG(dbgs() << "Trying " << Entry.Desc << " table:\n");
    DecodeStatus Result =
        decodeInstruction(Entry.Table, MI, Insn, Address, this, STI);
    if (Result != MCDisassembler::Fail)
      return Result;
  }

  return MCDisassembler::Fail;
}

static constexpr DecoderListEntry DecoderList48[]{
    {DecoderTableXqci48, XqciFeatureGroup, "Qualcomm uC 48bit"},
};

DecodeStatus CapstoneDisassembler::getInstruction48(MCInst &MI, uint64_t &Size,
                                                 ArrayRef<uint8_t> Bytes,
                                                 uint64_t Address,
                                                 raw_ostream &CS) const {
  if (Bytes.size() < 6) {
    Size = 0;
    return MCDisassembler::Fail;
  }
  Size = 6;

  uint64_t Insn = 0;
  for (size_t i = Size; i-- != 0;)
    Insn += (static_cast<uint64_t>(Bytes[i]) << 8 * i);

  for (const DecoderListEntry &Entry : DecoderList48) {
    if (!Entry.haveContainedFeatures(STI.getFeatureBits()))
      continue;

    LLVM_DEBUG(dbgs() << "Trying " << Entry.Desc << " table:\n");
    DecodeStatus Result =
        decodeInstruction(Entry.Table, MI, Insn, Address, this, STI);
    if (Result == MCDisassembler::Fail)
      continue;

    return Result;
  }

  return MCDisassembler::Fail;
}

DecodeStatus CapstoneDisassembler::getInstruction(MCInst &MI, uint64_t &Size,
                                               ArrayRef<uint8_t> Bytes,
                                               uint64_t Address,
                                               raw_ostream &CS) const {
  CommentStream = &CS;
  // It's a 16 bit instruction if bit 0 and 1 are not 0b11.
  if ((Bytes[0] & 0b11) != 0b11)
    return getInstruction16(MI, Size, Bytes, Address, CS);

  // It's a 32 bit instruction if bit 1:0 are 0b11(checked above) and bits 4:2
  // are not 0b111.
  if ((Bytes[0] & 0b1'1100) != 0b1'1100)
    return getInstruction32(MI, Size, Bytes, Address, CS);

  // 48-bit instructions are encoded as 0bxx011111.
  if ((Bytes[0] & 0b11'1111) == 0b01'1111) {
    return getInstruction48(MI, Size, Bytes, Address, CS);
  }

  // 64-bit instructions are encoded as 0x0111111.
  if ((Bytes[0] & 0b111'1111) == 0b011'1111) {
    Size = Bytes.size() >= 8 ? 8 : 0;
    return MCDisassembler::Fail;
  }

  // Remaining cases need to check a second byte.
  if (Bytes.size() < 2) {
    Size = 0;
    return MCDisassembler::Fail;
  }

  // 80-bit through 176-bit instructions are encoded as 0bxnnnxxxx_x1111111.
  // Where the number of bits is (80 + (nnn * 16)) for nnn != 0b111.
  unsigned nnn = (Bytes[1] >> 4) & 0b111;
  if (nnn != 0b111) {
    Size = 10 + (nnn * 2);
    if (Bytes.size() < Size)
      Size = 0;
    return MCDisassembler::Fail;
  }

  // Remaining encodings are reserved for > 176-bit instructions.
  Size = 0;
  return MCDisassembler::Fail;
}
