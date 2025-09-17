//===-- CapstoneRegisterInfo.cpp - Capstone Register Information -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the Capstone implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#include "CapstoneRegisterInfo.h"
#include "Capstone.h"
#include "CapstoneSubtarget.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/Support/ErrorHandling.h"

#define GET_REGINFO_TARGET_DESC
#include "CapstoneGenRegisterInfo.inc"

using namespace llvm;

static cl::opt<bool> DisableCostPerUse("capstone-disable-cost-per-use",
                                       cl::init(false), cl::Hidden);
static cl::opt<bool>
    DisableRegAllocHints("capstone-disable-regalloc-hints", cl::Hidden,
                         cl::init(false),
                         cl::desc("Disable two address hints for register "
                                  "allocation"));

static_assert(Capstone::X1 == Capstone::X0 + 1, "Register list not consecutive");
static_assert(Capstone::X31 == Capstone::X0 + 31, "Register list not consecutive");
static_assert(Capstone::F1_H == Capstone::F0_H + 1, "Register list not consecutive");
static_assert(Capstone::F31_H == Capstone::F0_H + 31,
              "Register list not consecutive");
static_assert(Capstone::F1_F == Capstone::F0_F + 1, "Register list not consecutive");
static_assert(Capstone::F31_F == Capstone::F0_F + 31,
              "Register list not consecutive");
static_assert(Capstone::F1_D == Capstone::F0_D + 1, "Register list not consecutive");
static_assert(Capstone::F31_D == Capstone::F0_D + 31,
              "Register list not consecutive");
static_assert(Capstone::F1_Q == Capstone::F0_Q + 1, "Register list not consecutive");
static_assert(Capstone::F31_Q == Capstone::F0_Q + 31,
              "Register list not consecutive");
static_assert(Capstone::V1 == Capstone::V0 + 1, "Register list not consecutive");
static_assert(Capstone::V31 == Capstone::V0 + 31, "Register list not consecutive");

CapstoneRegisterInfo::CapstoneRegisterInfo(unsigned HwMode)
    : CapstoneGenRegisterInfo(Capstone::X1, /*DwarfFlavour*/0, /*EHFlavor*/0,
                           /*PC*/0, HwMode) {}

const MCPhysReg *
CapstoneRegisterInfo::getIPRACSRegs(const MachineFunction *MF) const {
  return CSR_IPRA_SaveList;
}

const MCPhysReg *
CapstoneRegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {
  auto &Subtarget = MF->getSubtarget<CapstoneSubtarget>();
  if (MF->getFunction().getCallingConv() == CallingConv::GHC)
    return CSR_NoRegs_SaveList;
  if (MF->getFunction().getCallingConv() == CallingConv::PreserveMost)
    return Subtarget.hasStdExtE() ? CSR_RT_MostRegs_RVE_SaveList
                                  : CSR_RT_MostRegs_SaveList;
  if (MF->getFunction().hasFnAttribute("interrupt")) {
    if (Subtarget.hasVInstructions()) {
      if (Subtarget.hasStdExtD())
        return Subtarget.hasStdExtE() ? CSR_XLEN_F64_V_Interrupt_RVE_SaveList
                                      : CSR_XLEN_F64_V_Interrupt_SaveList;
      if (Subtarget.hasStdExtF())
        return Subtarget.hasStdExtE() ? CSR_XLEN_F32_V_Interrupt_RVE_SaveList
                                      : CSR_XLEN_F32_V_Interrupt_SaveList;
      return Subtarget.hasStdExtE() ? CSR_XLEN_V_Interrupt_RVE_SaveList
                                    : CSR_XLEN_V_Interrupt_SaveList;
    }
    if (Subtarget.hasStdExtD())
      return Subtarget.hasStdExtE() ? CSR_XLEN_F64_Interrupt_RVE_SaveList
                                    : CSR_XLEN_F64_Interrupt_SaveList;
    if (Subtarget.hasStdExtF())
      return Subtarget.hasStdExtE() ? CSR_XLEN_F32_Interrupt_RVE_SaveList
                                    : CSR_XLEN_F32_Interrupt_SaveList;
    return Subtarget.hasStdExtE() ? CSR_Interrupt_RVE_SaveList
                                  : CSR_Interrupt_SaveList;
  }

  bool HasVectorCSR =
      MF->getFunction().getCallingConv() == CallingConv::Capstone_VectorCall &&
      Subtarget.hasVInstructions();

  switch (Subtarget.getTargetABI()) {
  default:
    llvm_unreachable("Unrecognized ABI");
  case CapstoneABI::ABI_ILP32E:
  case CapstoneABI::ABI_LP64E:
    return CSR_ILP32E_LP64E_SaveList;
  case CapstoneABI::ABI_ILP32:
  case CapstoneABI::ABI_LP64:
    if (HasVectorCSR)
      return CSR_ILP32_LP64_V_SaveList;
    return CSR_ILP32_LP64_SaveList;
  case CapstoneABI::ABI_ILP32F:
  case CapstoneABI::ABI_LP64F:
    if (HasVectorCSR)
      return CSR_ILP32F_LP64F_V_SaveList;
    return CSR_ILP32F_LP64F_SaveList;
  case CapstoneABI::ABI_ILP32D:
  case CapstoneABI::ABI_LP64D:
    if (HasVectorCSR)
      return CSR_ILP32D_LP64D_V_SaveList;
    return CSR_ILP32D_LP64D_SaveList;
  }
}

BitVector CapstoneRegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  const CapstoneFrameLowering *TFI = getFrameLowering(MF);
  BitVector Reserved(getNumRegs());
  auto &Subtarget = MF.getSubtarget<CapstoneSubtarget>();

  for (size_t Reg = 0; Reg < getNumRegs(); Reg++) {
    // Mark any GPRs requested to be reserved as such
    if (Subtarget.isRegisterReservedByUser(Reg))
      markSuperRegs(Reserved, Reg);

    // Mark all the registers defined as constant in TableGen as reserved.
    if (isConstantPhysReg(Reg))
      markSuperRegs(Reserved, Reg);
  }

  // Use markSuperRegs to ensure any register aliases are also reserved
  markSuperRegs(Reserved, Capstone::X2_H); // sp
  markSuperRegs(Reserved, Capstone::X3_H); // gp
  markSuperRegs(Reserved, Capstone::X4_H); // tp
  if (TFI->hasFP(MF))
    markSuperRegs(Reserved, Capstone::X8_H); // fp
  // Reserve the base register if we need to realign the stack and allocate
  // variable-sized objects at runtime.
  if (TFI->hasBP(MF))
    markSuperRegs(Reserved, CapstoneABI::getBPReg()); // bp

  // Additionally reserve dummy register used to form the register pair
  // beginning with 'x0' for instructions that take register pairs.
  markSuperRegs(Reserved, Capstone::DUMMY_REG_PAIR_WITH_X0);

  // There are only 16 GPRs for RVE.
  if (Subtarget.hasStdExtE())
    for (MCPhysReg Reg = Capstone::X16_H; Reg <= Capstone::X31_H; Reg++)
      markSuperRegs(Reserved, Reg);

  // V registers for code generation. We handle them manually.
  markSuperRegs(Reserved, Capstone::VL);
  markSuperRegs(Reserved, Capstone::VTYPE);
  markSuperRegs(Reserved, Capstone::VXSAT);
  markSuperRegs(Reserved, Capstone::VXRM);

  // Floating point environment registers.
  markSuperRegs(Reserved, Capstone::FRM);
  markSuperRegs(Reserved, Capstone::FFLAGS);

  // SiFive VCIX state registers.
  markSuperRegs(Reserved, Capstone::SF_VCIX_STATE);

  if (MF.getFunction().getCallingConv() == CallingConv::GRAAL) {
    if (Subtarget.hasStdExtE())
      reportFatalUsageError("Graal reserved registers do not exist in RVE");
    markSuperRegs(Reserved, Capstone::X23_H);
    markSuperRegs(Reserved, Capstone::X27_H);
  }

  // Shadow stack pointer.
  markSuperRegs(Reserved, Capstone::SSP);

  assert(checkAllSuperRegsMarked(Reserved));
  return Reserved;
}

bool CapstoneRegisterInfo::isAsmClobberable(const MachineFunction &MF,
                                         MCRegister PhysReg) const {
  return !MF.getSubtarget().isRegisterReservedByUser(PhysReg);
}

const uint32_t *CapstoneRegisterInfo::getNoPreservedMask() const {
  return CSR_NoRegs_RegMask;
}

void CapstoneRegisterInfo::adjustReg(MachineBasicBlock &MBB,
                                  MachineBasicBlock::iterator II,
                                  const DebugLoc &DL, Register DestReg,
                                  Register SrcReg, StackOffset Offset,
                                  MachineInstr::MIFlag Flag,
                                  MaybeAlign RequiredAlign) const {

  if (DestReg == SrcReg && !Offset.getFixed() && !Offset.getScalable())
    return;

  MachineFunction &MF = *MBB.getParent();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const CapstoneSubtarget &ST = MF.getSubtarget<CapstoneSubtarget>();
  const CapstoneInstrInfo *TII = ST.getInstrInfo();

  // Optimize compile time offset case
  if (Offset.getScalable()) {
    if (auto VLEN = ST.getRealVLen()) {
      // 1. Multiply the number of v-slots by the (constant) length of register
      const int64_t VLENB = *VLEN / 8;
      assert(Offset.getScalable() % Capstone::RVVBytesPerBlock == 0 &&
             "Reserve the stack by the multiple of one vector size.");
      const int64_t NumOfVReg = Offset.getScalable() / 8;
      const int64_t FixedOffset = NumOfVReg * VLENB;
      if (!isInt<32>(FixedOffset)) {
        reportFatalUsageError(
            "Frame size outside of the signed 32-bit range not supported");
      }
      Offset = StackOffset::getFixed(FixedOffset + Offset.getFixed());
    }
  }

  bool KillSrcReg = false;

  if (Offset.getScalable()) {
    unsigned ScalableAdjOpc = Capstone::ADD;
    int64_t ScalableValue = Offset.getScalable();
    if (ScalableValue < 0) {
      ScalableValue = -ScalableValue;
      ScalableAdjOpc = Capstone::SUB;
    }
    // Get vlenb and multiply vlen with the number of vector registers.
    Register ScratchReg = DestReg;
    if (DestReg == SrcReg)
      ScratchReg = MRI.createVirtualRegister(&Capstone::GPRRegClass);

    assert(ScalableValue > 0 && "There is no need to get VLEN scaled value.");
    assert(ScalableValue % Capstone::RVVBytesPerBlock == 0 &&
           "Reserve the stack by the multiple of one vector size.");
    assert(isInt<32>(ScalableValue / Capstone::RVVBytesPerBlock) &&
           "Expect the number of vector registers within 32-bits.");
    uint32_t NumOfVReg = ScalableValue / Capstone::RVVBytesPerBlock;
    // Only use vsetvli rather than vlenb if adjusting in the prologue or
    // epilogue, otherwise it may disturb the VTYPE and VL status.
    bool IsPrologueOrEpilogue =
        Flag == MachineInstr::FrameSetup || Flag == MachineInstr::FrameDestroy;
    bool UseVsetvliRatherThanVlenb =
        IsPrologueOrEpilogue && ST.preferVsetvliOverReadVLENB();
    if (UseVsetvliRatherThanVlenb && (NumOfVReg == 1 || NumOfVReg == 2 ||
                                      NumOfVReg == 4 || NumOfVReg == 8)) {
      BuildMI(MBB, II, DL, TII->get(Capstone::PseudoReadVLENBViaVSETVLIX0),
              ScratchReg)
          .addImm(NumOfVReg)
          .setMIFlag(Flag);
      BuildMI(MBB, II, DL, TII->get(ScalableAdjOpc), DestReg)
          .addReg(SrcReg)
          .addReg(ScratchReg, RegState::Kill)
          .setMIFlag(Flag);
    } else {
      if (UseVsetvliRatherThanVlenb)
        BuildMI(MBB, II, DL, TII->get(Capstone::PseudoReadVLENBViaVSETVLIX0),
                ScratchReg)
            .addImm(1)
            .setMIFlag(Flag);
      else
        BuildMI(MBB, II, DL, TII->get(Capstone::PseudoReadVLENB), ScratchReg)
            .setMIFlag(Flag);

      if (ScalableAdjOpc == Capstone::ADD && ST.hasStdExtZba() &&
          (NumOfVReg == 2 || NumOfVReg == 4 || NumOfVReg == 8)) {
        unsigned Opc = NumOfVReg == 2
                           ? Capstone::SH1ADD
                           : (NumOfVReg == 4 ? Capstone::SH2ADD : Capstone::SH3ADD);
        BuildMI(MBB, II, DL, TII->get(Opc), DestReg)
            .addReg(ScratchReg, RegState::Kill)
            .addReg(SrcReg)
            .setMIFlag(Flag);
      } else {
        TII->mulImm(MF, MBB, II, DL, ScratchReg, NumOfVReg, Flag);
        BuildMI(MBB, II, DL, TII->get(ScalableAdjOpc), DestReg)
            .addReg(SrcReg)
            .addReg(ScratchReg, RegState::Kill)
            .setMIFlag(Flag);
      }
    }
    SrcReg = DestReg;
    KillSrcReg = true;
  }

  int64_t Val = Offset.getFixed();
  if (DestReg == SrcReg && Val == 0)
    return;

  const uint64_t Align = RequiredAlign.valueOrOne().value();

  if (isInt<12>(Val)) {
    BuildMI(MBB, II, DL, TII->get(Capstone::ADDI), DestReg)
        .addReg(SrcReg, getKillRegState(KillSrcReg))
        .addImm(Val)
        .setMIFlag(Flag);
    return;
  }

  // Use the QC_E_ADDI instruction from the Xqcilia extension that can take a
  // signed 26-bit immediate.
  if (ST.hasVendorXqcilia() && isInt<26>(Val)) {
    // The one case where using this instruction is sub-optimal is if Val can be
    // materialized with a single compressible LUI and following add/sub is also
    // compressible. Avoid doing this if that is the case.
    int Hi20 = (Val & 0xFFFFF000) >> 12;
    bool IsCompressLUI =
        ((Val & 0xFFF) == 0) && (Hi20 != 0) &&
        (isUInt<5>(Hi20) || (Hi20 >= 0xfffe0 && Hi20 <= 0xfffff));
    bool IsCompressAddSub =
        (SrcReg == DestReg) &&
        ((Val > 0 && Capstone::GPRNoX0RegClass.contains(SrcReg)) ||
         (Val < 0 && Capstone::GPRCRegClass.contains(SrcReg)));

    if (!(IsCompressLUI && IsCompressAddSub)) {
      BuildMI(MBB, II, DL, TII->get(Capstone::QC_E_ADDI), DestReg)
          .addReg(SrcReg, getKillRegState(KillSrcReg))
          .addImm(Val)
          .setMIFlag(Flag);
      return;
    }
  }

  // Try to split the offset across two ADDIs. We need to keep the intermediate
  // result aligned after each ADDI.  We need to determine the maximum value we
  // can put in each ADDI. In the negative direction, we can use -2048 which is
  // always sufficiently aligned. In the positive direction, we need to find the
  // largest 12-bit immediate that is aligned.  Exclude -4096 since it can be
  // created with LUI.
  assert(Align < 2048 && "Required alignment too large");
  int64_t MaxPosAdjStep = 2048 - Align;
  if (Val > -4096 && Val <= (2 * MaxPosAdjStep)) {
    int64_t FirstAdj = Val < 0 ? -2048 : MaxPosAdjStep;
    Val -= FirstAdj;
    BuildMI(MBB, II, DL, TII->get(Capstone::ADDI), DestReg)
        .addReg(SrcReg, getKillRegState(KillSrcReg))
        .addImm(FirstAdj)
        .setMIFlag(Flag);
    BuildMI(MBB, II, DL, TII->get(Capstone::ADDI), DestReg)
        .addReg(DestReg, RegState::Kill)
        .addImm(Val)
        .setMIFlag(Flag);
    return;
  }

  // Use shNadd if doing so lets us materialize a 12 bit immediate with a single
  // instruction.  This saves 1 instruction over the full lui/addi+add fallback
  // path.  We avoid anything which can be done with a single lui as it might
  // be compressible.  Note that the sh1add case is fully covered by the 2x addi
  // case just above and is thus omitted.
  if (ST.hasStdExtZba() && (Val & 0xFFF) != 0) {
    unsigned Opc = 0;
    if (isShiftedInt<12, 3>(Val)) {
      Opc = Capstone::SH3ADD;
      Val = Val >> 3;
    } else if (isShiftedInt<12, 2>(Val)) {
      Opc = Capstone::SH2ADD;
      Val = Val >> 2;
    }
    if (Opc) {
      Register ScratchReg = MRI.createVirtualRegister(&Capstone::GPRRegClass);
      TII->movImm(MBB, II, DL, ScratchReg, Val, Flag);
      BuildMI(MBB, II, DL, TII->get(Opc), DestReg)
          .addReg(ScratchReg, RegState::Kill)
          .addReg(SrcReg, getKillRegState(KillSrcReg))
          .setMIFlag(Flag);
      return;
    }
  }

  unsigned Opc = Capstone::ADD;
  if (Val < 0) {
    Val = -Val;
    Opc = Capstone::SUB;
  }

  Register ScratchReg = MRI.createVirtualRegister(&Capstone::GPRRegClass);
  TII->movImm(MBB, II, DL, ScratchReg, Val, Flag);
  BuildMI(MBB, II, DL, TII->get(Opc), DestReg)
      .addReg(SrcReg, getKillRegState(KillSrcReg))
      .addReg(ScratchReg, RegState::Kill)
      .setMIFlag(Flag);
}

static std::tuple<CapstoneVType::VLMUL, const TargetRegisterClass &, unsigned>
getSpillReloadInfo(unsigned NumRemaining, uint16_t RegEncoding, bool IsSpill) {
  if (NumRemaining >= 8 && RegEncoding % 8 == 0)
    return {CapstoneVType::LMUL_8, Capstone::VRM8RegClass,
            IsSpill ? Capstone::VS8R_V : Capstone::VL8RE8_V};
  if (NumRemaining >= 4 && RegEncoding % 4 == 0)
    return {CapstoneVType::LMUL_4, Capstone::VRM4RegClass,
            IsSpill ? Capstone::VS4R_V : Capstone::VL4RE8_V};
  if (NumRemaining >= 2 && RegEncoding % 2 == 0)
    return {CapstoneVType::LMUL_2, Capstone::VRM2RegClass,
            IsSpill ? Capstone::VS2R_V : Capstone::VL2RE8_V};
  return {CapstoneVType::LMUL_1, Capstone::VRRegClass,
          IsSpill ? Capstone::VS1R_V : Capstone::VL1RE8_V};
}

// Split a VSPILLx_Mx/VSPILLx_Mx pseudo into multiple whole register stores
// separated by LMUL*VLENB bytes.
void CapstoneRegisterInfo::lowerSegmentSpillReload(MachineBasicBlock::iterator II,
                                                bool IsSpill) const {
  DebugLoc DL = II->getDebugLoc();
  MachineBasicBlock &MBB = *II->getParent();
  MachineFunction &MF = *MBB.getParent();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const CapstoneSubtarget &STI = MF.getSubtarget<CapstoneSubtarget>();
  const TargetInstrInfo *TII = STI.getInstrInfo();
  const TargetRegisterInfo *TRI = STI.getRegisterInfo();

  auto ZvlssegInfo = Capstone::isRVVSpillForZvlsseg(II->getOpcode());
  unsigned NF = ZvlssegInfo->first;
  unsigned LMUL = ZvlssegInfo->second;
  unsigned NumRegs = NF * LMUL;
  assert(NumRegs <= 8 && "Invalid NF/LMUL combinations.");

  Register Reg = II->getOperand(0).getReg();
  uint16_t RegEncoding = TRI->getEncodingValue(Reg);
  Register Base = II->getOperand(1).getReg();
  bool IsBaseKill = II->getOperand(1).isKill();
  Register NewBase = MRI.createVirtualRegister(&Capstone::GPRRegClass);

  auto *OldMMO = *(II->memoperands_begin());
  LocationSize OldLoc = OldMMO->getSize();
  assert(OldLoc.isPrecise() && OldLoc.getValue().isKnownMultipleOf(NF));
  TypeSize VRegSize = OldLoc.getValue().divideCoefficientBy(NumRegs);

  Register VLENB = 0;
  unsigned PreHandledNum = 0;
  unsigned I = 0;
  while (I != NumRegs) {
    auto [LMulHandled, RegClass, Opcode] =
        getSpillReloadInfo(NumRegs - I, RegEncoding, IsSpill);
    auto [RegNumHandled, _] = CapstoneVType::decodeVLMUL(LMulHandled);
    bool IsLast = I + RegNumHandled == NumRegs;
    if (PreHandledNum) {
      Register Step;
      // Optimize for constant VLEN.
      if (auto VLEN = STI.getRealVLen()) {
        int64_t Offset = *VLEN / 8 * PreHandledNum;
        Step = MRI.createVirtualRegister(&Capstone::GPRRegClass);
        STI.getInstrInfo()->movImm(MBB, II, DL, Step, Offset);
      } else {
        if (!VLENB) {
          VLENB = MRI.createVirtualRegister(&Capstone::GPRRegClass);
          BuildMI(MBB, II, DL, TII->get(Capstone::PseudoReadVLENB), VLENB);
        }
        uint32_t ShiftAmount = Log2_32(PreHandledNum);
        if (ShiftAmount == 0)
          Step = VLENB;
        else {
          Step = MRI.createVirtualRegister(&Capstone::GPRRegClass);
          BuildMI(MBB, II, DL, TII->get(Capstone::SLLI), Step)
              .addReg(VLENB, getKillRegState(IsLast))
              .addImm(ShiftAmount);
        }
      }

      BuildMI(MBB, II, DL, TII->get(Capstone::ADD), NewBase)
          .addReg(Base, getKillRegState(I != 0 || IsBaseKill))
          .addReg(Step, getKillRegState(Step != VLENB || IsLast));
      Base = NewBase;
    }

    MCRegister ActualReg = findVRegWithEncoding(RegClass, RegEncoding);
    MachineInstrBuilder MIB =
        BuildMI(MBB, II, DL, TII->get(Opcode))
            .addReg(ActualReg, getDefRegState(!IsSpill))
            .addReg(Base, getKillRegState(IsLast))
            .addMemOperand(MF.getMachineMemOperand(OldMMO, OldMMO->getOffset(),
                                                   VRegSize * RegNumHandled));

    // Adding implicit-use of super register to describe we are using part of
    // super register, that prevents machine verifier complaining when part of
    // subreg is undef, see comment in MachineVerifier::checkLiveness for more
    // detail.
    if (IsSpill)
      MIB.addReg(Reg, RegState::Implicit);

    PreHandledNum = RegNumHandled;
    RegEncoding += RegNumHandled;
    I += RegNumHandled;
  }
  II->eraseFromParent();
}

bool CapstoneRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                            int SPAdj, unsigned FIOperandNum,
                                            RegScavenger *RS) const {
  assert(SPAdj == 0 && "Unexpected non-zero SPAdj value");

  MachineInstr &MI = *II;
  MachineFunction &MF = *MI.getParent()->getParent();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  DebugLoc DL = MI.getDebugLoc();

  int FrameIndex = MI.getOperand(FIOperandNum).getIndex();
  Register FrameReg;
  StackOffset Offset =
      getFrameLowering(MF)->getFrameIndexReference(MF, FrameIndex, FrameReg);
  bool IsRVVSpill = Capstone::isRVVSpill(MI);
  if (!IsRVVSpill)
    Offset += StackOffset::getFixed(MI.getOperand(FIOperandNum + 1).getImm());

  if (!isInt<32>(Offset.getFixed())) {
    reportFatalUsageError(
        "Frame offsets outside of the signed 32-bit range not supported");
  }

  if (!IsRVVSpill) {
    int64_t Val = Offset.getFixed();
    int64_t Lo12 = SignExtend64<12>(Val);
    unsigned Opc = MI.getOpcode();

    if (Opc == Capstone::ADDI && !isInt<12>(Val)) {
      // We chose to emit the canonical immediate sequence rather than folding
      // the offset into the using add under the theory that doing so doesn't
      // save dynamic instruction count and some target may fuse the canonical
      // 32 bit immediate sequence.  We still need to clear the portion of the
      // offset encoded in the immediate.
      MI.getOperand(FIOperandNum + 1).ChangeToImmediate(0);
    } else if ((Opc == Capstone::PREFETCH_I || Opc == Capstone::PREFETCH_R ||
                Opc == Capstone::PREFETCH_W) &&
               (Lo12 & 0b11111) != 0) {
      // Prefetch instructions require the offset to be 32 byte aligned.
      MI.getOperand(FIOperandNum + 1).ChangeToImmediate(0);
    } else if (Opc == Capstone::MIPS_PREF && !isUInt<9>(Val)) {
      // MIPS Prefetch instructions require the offset to be 9 bits encoded.
      MI.getOperand(FIOperandNum + 1).ChangeToImmediate(0);
    } else if ((Opc == Capstone::PseudoRV32ZdinxLD ||
                Opc == Capstone::PseudoRV32ZdinxSD) &&
               Lo12 >= 2044) {
      // This instruction will be split into 2 instructions. The second
      // instruction will add 4 to the immediate. If that would overflow 12
      // bits, we can't fold the offset.
      MI.getOperand(FIOperandNum + 1).ChangeToImmediate(0);
    } else {
      // We can encode an add with 12 bit signed immediate in the immediate
      // operand of our user instruction.  As a result, the remaining
      // offset can by construction, at worst, a LUI and a ADD.
      MI.getOperand(FIOperandNum + 1).ChangeToImmediate(Lo12);
      Offset = StackOffset::get((uint64_t)Val - (uint64_t)Lo12,
                                Offset.getScalable());
    }
  }

  if (Offset.getScalable() || Offset.getFixed()) {
    Register DestReg;
    if (MI.getOpcode() == Capstone::ADDI)
      DestReg = MI.getOperand(0).getReg();
    else
      DestReg = MRI.createVirtualRegister(&Capstone::GPRRegClass);
    adjustReg(*II->getParent(), II, DL, DestReg, FrameReg, Offset,
              MachineInstr::NoFlags, std::nullopt);
    MI.getOperand(FIOperandNum).ChangeToRegister(DestReg, /*IsDef*/false,
                                                 /*IsImp*/false,
                                                 /*IsKill*/true);
  } else {
    MI.getOperand(FIOperandNum).ChangeToRegister(FrameReg, /*IsDef*/false,
                                                 /*IsImp*/false,
                                                 /*IsKill*/false);
  }

  // If after materializing the adjustment, we have a pointless ADDI, remove it
  if (MI.getOpcode() == Capstone::ADDI &&
      MI.getOperand(0).getReg() == MI.getOperand(1).getReg() &&
      MI.getOperand(2).getImm() == 0) {
    MI.eraseFromParent();
    return true;
  }

  // Handle spill/fill of synthetic register classes for segment operations to
  // ensure correctness in the edge case one gets spilled.
  switch (MI.getOpcode()) {
  case Capstone::PseudoVSPILL2_M1:
  case Capstone::PseudoVSPILL2_M2:
  case Capstone::PseudoVSPILL2_M4:
  case Capstone::PseudoVSPILL3_M1:
  case Capstone::PseudoVSPILL3_M2:
  case Capstone::PseudoVSPILL4_M1:
  case Capstone::PseudoVSPILL4_M2:
  case Capstone::PseudoVSPILL5_M1:
  case Capstone::PseudoVSPILL6_M1:
  case Capstone::PseudoVSPILL7_M1:
  case Capstone::PseudoVSPILL8_M1:
    lowerSegmentSpillReload(II, /*IsSpill=*/true);
    return true;
  case Capstone::PseudoVRELOAD2_M1:
  case Capstone::PseudoVRELOAD2_M2:
  case Capstone::PseudoVRELOAD2_M4:
  case Capstone::PseudoVRELOAD3_M1:
  case Capstone::PseudoVRELOAD3_M2:
  case Capstone::PseudoVRELOAD4_M1:
  case Capstone::PseudoVRELOAD4_M2:
  case Capstone::PseudoVRELOAD5_M1:
  case Capstone::PseudoVRELOAD6_M1:
  case Capstone::PseudoVRELOAD7_M1:
  case Capstone::PseudoVRELOAD8_M1:
    lowerSegmentSpillReload(II, /*IsSpill=*/false);
    return true;
  }

  return false;
}

bool CapstoneRegisterInfo::requiresVirtualBaseRegisters(
    const MachineFunction &MF) const {
  return true;
}

// Returns true if the instruction's frame index reference would be better
// served by a base register other than FP or SP.
// Used by LocalStackSlotAllocation pass to determine which frame index
// references it should create new base registers for.
bool CapstoneRegisterInfo::needsFrameBaseReg(MachineInstr *MI,
                                          int64_t Offset) const {
  unsigned FIOperandNum = 0;
  for (; !MI->getOperand(FIOperandNum).isFI(); FIOperandNum++)
    assert(FIOperandNum < MI->getNumOperands() &&
           "Instr doesn't have FrameIndex operand");

  // For Capstone, The machine instructions that include a FrameIndex operand
  // are load/store, ADDI instructions.
  unsigned MIFrm = CapstoneII::getFormat(MI->getDesc().TSFlags);
  if (MIFrm != CapstoneII::InstFormatI && MIFrm != CapstoneII::InstFormatS)
    return false;
  // We only generate virtual base registers for loads and stores, so
  // return false for everything else.
  if (!MI->mayLoad() && !MI->mayStore())
    return false;

  const MachineFunction &MF = *MI->getMF();
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  const CapstoneFrameLowering *TFI = getFrameLowering(MF);
  const MachineRegisterInfo &MRI = MF.getRegInfo();

  if (TFI->hasFP(MF) && !shouldRealignStack(MF)) {
    auto &Subtarget = MF.getSubtarget<CapstoneSubtarget>();
    // Estimate the stack size used to store callee saved registers(
    // excludes reserved registers).
    unsigned CalleeSavedSize = 0;
    for (const MCPhysReg *R = MRI.getCalleeSavedRegs(); MCPhysReg Reg = *R;
         ++R) {
      if (Subtarget.isRegisterReservedByUser(Reg))
        continue;

      if (Capstone::GPRRegClass.contains(Reg))
        CalleeSavedSize += getSpillSize(Capstone::GPRRegClass);
      else if (Capstone::FPR64RegClass.contains(Reg))
        CalleeSavedSize += getSpillSize(Capstone::FPR64RegClass);
      else if (Capstone::FPR32RegClass.contains(Reg))
        CalleeSavedSize += getSpillSize(Capstone::FPR32RegClass);
      // Ignore vector registers.
    }

    int64_t MaxFPOffset = Offset - CalleeSavedSize;
    return !isFrameOffsetLegal(MI, Capstone::X8, MaxFPOffset);
  }

  // Assume 128 bytes spill slots size to estimate the maximum possible
  // offset relative to the stack pointer.
  // FIXME: The 128 is copied from ARM. We should run some statistics and pick a
  // real one for Capstone.
  int64_t MaxSPOffset = Offset + 128;
  MaxSPOffset += MFI.getLocalFrameSize();
  return !isFrameOffsetLegal(MI, Capstone::X2, MaxSPOffset);
}

// Determine whether a given base register plus offset immediate is
// encodable to resolve a frame index.
bool CapstoneRegisterInfo::isFrameOffsetLegal(const MachineInstr *MI,
                                           Register BaseReg,
                                           int64_t Offset) const {
  unsigned FIOperandNum = 0;
  while (!MI->getOperand(FIOperandNum).isFI()) {
    FIOperandNum++;
    assert(FIOperandNum < MI->getNumOperands() &&
           "Instr does not have a FrameIndex operand!");
  }

  Offset += getFrameIndexInstrOffset(MI, FIOperandNum);
  return isInt<12>(Offset);
}

// Insert defining instruction(s) for a pointer to FrameIdx before
// insertion point I.
// Return materialized frame pointer.
Register CapstoneRegisterInfo::materializeFrameBaseRegister(MachineBasicBlock *MBB,
                                                         int FrameIdx,
                                                         int64_t Offset) const {
  MachineBasicBlock::iterator MBBI = MBB->begin();
  DebugLoc DL;
  if (MBBI != MBB->end())
    DL = MBBI->getDebugLoc();
  MachineFunction *MF = MBB->getParent();
  MachineRegisterInfo &MFI = MF->getRegInfo();
  const TargetInstrInfo *TII = MF->getSubtarget().getInstrInfo();

  Register BaseReg = MFI.createVirtualRegister(&Capstone::GPRRegClass);
  BuildMI(*MBB, MBBI, DL, TII->get(Capstone::ADDI), BaseReg)
      .addFrameIndex(FrameIdx)
      .addImm(Offset);
  return BaseReg;
}

// Resolve a frame index operand of an instruction to reference the
// indicated base register plus offset instead.
void CapstoneRegisterInfo::resolveFrameIndex(MachineInstr &MI, Register BaseReg,
                                          int64_t Offset) const {
  unsigned FIOperandNum = 0;
  while (!MI.getOperand(FIOperandNum).isFI()) {
    FIOperandNum++;
    assert(FIOperandNum < MI.getNumOperands() &&
           "Instr does not have a FrameIndex operand!");
  }

  Offset += getFrameIndexInstrOffset(&MI, FIOperandNum);
  // FrameIndex Operands are always represented as a
  // register followed by an immediate.
  MI.getOperand(FIOperandNum).ChangeToRegister(BaseReg, false);
  MI.getOperand(FIOperandNum + 1).ChangeToImmediate(Offset);
}

// Get the offset from the referenced frame index in the instruction,
// if there is one.
int64_t CapstoneRegisterInfo::getFrameIndexInstrOffset(const MachineInstr *MI,
                                                    int Idx) const {
  assert((CapstoneII::getFormat(MI->getDesc().TSFlags) == CapstoneII::InstFormatI ||
          CapstoneII::getFormat(MI->getDesc().TSFlags) == CapstoneII::InstFormatS) &&
         "The MI must be I or S format.");
  assert(MI->getOperand(Idx).isFI() && "The Idx'th operand of MI is not a "
                                       "FrameIndex operand");
  return MI->getOperand(Idx + 1).getImm();
}

Register CapstoneRegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  const TargetFrameLowering *TFI = getFrameLowering(MF);
  return TFI->hasFP(MF) ? Capstone::X8 : Capstone::X2;
}

StringRef CapstoneRegisterInfo::getRegAsmName(MCRegister Reg) const {
  if (Reg == Capstone::SF_VCIX_STATE)
    return "sf.vcix_state";
  return TargetRegisterInfo::getRegAsmName(Reg);
}

const uint32_t *
CapstoneRegisterInfo::getCallPreservedMask(const MachineFunction & MF,
                                        CallingConv::ID CC) const {
  auto &Subtarget = MF.getSubtarget<CapstoneSubtarget>();

  if (CC == CallingConv::GHC)
    return CSR_NoRegs_RegMask;
  CapstoneABI::ABI ABI = Subtarget.getTargetABI();
  if (CC == CallingConv::PreserveMost) {
    if (ABI == CapstoneABI::ABI_ILP32E || ABI == CapstoneABI::ABI_LP64E)
      return CSR_RT_MostRegs_RVE_RegMask;
    return CSR_RT_MostRegs_RegMask;
  }
  switch (ABI) {
  default:
    llvm_unreachable("Unrecognized ABI");
  case CapstoneABI::ABI_ILP32E:
  case CapstoneABI::ABI_LP64E:
    return CSR_ILP32E_LP64E_RegMask;
  case CapstoneABI::ABI_ILP32:
  case CapstoneABI::ABI_LP64:
    if (CC == CallingConv::Capstone_VectorCall)
      return CSR_ILP32_LP64_V_RegMask;
    return CSR_ILP32_LP64_RegMask;
  case CapstoneABI::ABI_ILP32F:
  case CapstoneABI::ABI_LP64F:
    if (CC == CallingConv::Capstone_VectorCall)
      return CSR_ILP32F_LP64F_V_RegMask;
    return CSR_ILP32F_LP64F_RegMask;
  case CapstoneABI::ABI_ILP32D:
  case CapstoneABI::ABI_LP64D:
    if (CC == CallingConv::Capstone_VectorCall)
      return CSR_ILP32D_LP64D_V_RegMask;
    return CSR_ILP32D_LP64D_RegMask;
  }
}

const TargetRegisterClass *
CapstoneRegisterInfo::getLargestLegalSuperClass(const TargetRegisterClass *RC,
                                             const MachineFunction &) const {
  if (RC == &Capstone::VMV0RegClass)
    return &Capstone::VRRegClass;
  if (RC == &Capstone::VRNoV0RegClass)
    return &Capstone::VRRegClass;
  if (RC == &Capstone::VRM2NoV0RegClass)
    return &Capstone::VRM2RegClass;
  if (RC == &Capstone::VRM4NoV0RegClass)
    return &Capstone::VRM4RegClass;
  if (RC == &Capstone::VRM8NoV0RegClass)
    return &Capstone::VRM8RegClass;
  return RC;
}

void CapstoneRegisterInfo::getOffsetOpcodes(const StackOffset &Offset,
                                         SmallVectorImpl<uint64_t> &Ops) const {
  // VLENB is the length of a vector register in bytes. We use <vscale x 8 x i8>
  // to represent one vector register. The dwarf offset is
  // VLENB * scalable_offset / 8.
  assert(Offset.getScalable() % 8 == 0 && "Invalid frame offset");

  // Add fixed-sized offset using existing DIExpression interface.
  DIExpression::appendOffset(Ops, Offset.getFixed());

  unsigned VLENB = getDwarfRegNum(Capstone::VLENB, true);
  int64_t VLENBSized = Offset.getScalable() / 8;
  if (VLENBSized > 0) {
    Ops.push_back(dwarf::DW_OP_constu);
    Ops.push_back(VLENBSized);
    Ops.append({dwarf::DW_OP_bregx, VLENB, 0ULL});
    Ops.push_back(dwarf::DW_OP_mul);
    Ops.push_back(dwarf::DW_OP_plus);
  } else if (VLENBSized < 0) {
    Ops.push_back(dwarf::DW_OP_constu);
    Ops.push_back(-VLENBSized);
    Ops.append({dwarf::DW_OP_bregx, VLENB, 0ULL});
    Ops.push_back(dwarf::DW_OP_mul);
    Ops.push_back(dwarf::DW_OP_minus);
  }
}

unsigned
CapstoneRegisterInfo::getRegisterCostTableIndex(const MachineFunction &MF) const {
  return MF.getSubtarget<CapstoneSubtarget>().hasStdExtZca() && !DisableCostPerUse
             ? 1
             : 0;
}

float CapstoneRegisterInfo::getSpillWeightScaleFactor(
    const TargetRegisterClass *RC) const {
  return getRegClassWeight(RC).RegWeight;
}

// Add two address hints to improve chances of being able to use a compressed
// instruction.
bool CapstoneRegisterInfo::getRegAllocationHints(
    Register VirtReg, ArrayRef<MCPhysReg> Order,
    SmallVectorImpl<MCPhysReg> &Hints, const MachineFunction &MF,
    const VirtRegMap *VRM, const LiveRegMatrix *Matrix) const {
  const MachineRegisterInfo *MRI = &MF.getRegInfo();
  auto &Subtarget = MF.getSubtarget<CapstoneSubtarget>();

  bool BaseImplRetVal = TargetRegisterInfo::getRegAllocationHints(
      VirtReg, Order, Hints, MF, VRM, Matrix);

  if (!VRM || DisableRegAllocHints)
    return BaseImplRetVal;

  // Add any two address hints after any copy hints.
  SmallSet<Register, 4> TwoAddrHints;

  auto tryAddHint = [&](const MachineOperand &VRRegMO, const MachineOperand &MO,
                        bool NeedGPRC) -> void {
    Register Reg = MO.getReg();
    Register PhysReg = Reg.isPhysical() ? Reg : Register(VRM->getPhys(Reg));
    // TODO: Support GPRPair subregisters? Need to be careful with even/odd
    // registers. If the virtual register is an odd register of a pair and the
    // physical register is even (or vice versa), we should not add the hint.
    if (PhysReg && (!NeedGPRC || Capstone::GPRCRegClass.contains(PhysReg)) &&
        !MO.getSubReg() && !VRRegMO.getSubReg()) {
      if (!MRI->isReserved(PhysReg) && !is_contained(Hints, PhysReg))
        TwoAddrHints.insert(PhysReg);
    }
  };

  // This is all of the compressible binary instructions. If an instruction
  // needs GPRC register class operands \p NeedGPRC will be set to true.
  auto isCompressible = [&Subtarget](const MachineInstr &MI, bool &NeedGPRC) {
    NeedGPRC = false;
    switch (MI.getOpcode()) {
    default:
      return false;
    case Capstone::AND:
    case Capstone::OR:
    case Capstone::XOR:
    case Capstone::SUB:
    case Capstone::ADDW:
    case Capstone::SUBW:
      NeedGPRC = true;
      return true;
    case Capstone::ANDI: {
      NeedGPRC = true;
      if (!MI.getOperand(2).isImm())
        return false;
      int64_t Imm = MI.getOperand(2).getImm();
      if (isInt<6>(Imm))
        return true;
      // c.zext.b
      return Subtarget.hasStdExtZcb() && Imm == 255;
    }
    case Capstone::SRAI:
    case Capstone::SRLI:
      NeedGPRC = true;
      return true;
    case Capstone::ADD:
    case Capstone::SLLI:
      return true;
    case Capstone::ADDI:
    case Capstone::ADDIW:
      return MI.getOperand(2).isImm() && isInt<6>(MI.getOperand(2).getImm());
    case Capstone::MUL:
    case Capstone::SEXT_B:
    case Capstone::SEXT_H:
    case Capstone::ZEXT_H_RV32:
    case Capstone::ZEXT_H_RV64:
      // c.mul, c.sext.b, c.sext.h, c.zext.h
      NeedGPRC = true;
      return Subtarget.hasStdExtZcb();
    case Capstone::ADD_UW:
      // c.zext.w
      NeedGPRC = true;
      return Subtarget.hasStdExtZcb() && MI.getOperand(2).isReg() &&
             MI.getOperand(2).getReg() == Capstone::X0;
    case Capstone::XORI:
      // c.not
      NeedGPRC = true;
      return Subtarget.hasStdExtZcb() && MI.getOperand(2).isImm() &&
             MI.getOperand(2).getImm() == -1;
    }
  };

  // Returns true if this operand is compressible. For non-registers it always
  // returns true. Immediate range was already checked in isCompressible.
  // For registers, it checks if the register is a GPRC register. reg-reg
  // instructions that require GPRC need all register operands to be GPRC.
  auto isCompressibleOpnd = [&](const MachineOperand &MO) {
    if (!MO.isReg())
      return true;
    Register Reg = MO.getReg();
    Register PhysReg = Reg.isPhysical() ? Reg : Register(VRM->getPhys(Reg));
    return PhysReg && Capstone::GPRCRegClass.contains(PhysReg);
  };

  for (auto &MO : MRI->reg_nodbg_operands(VirtReg)) {
    const MachineInstr &MI = *MO.getParent();
    unsigned OpIdx = MO.getOperandNo();
    bool NeedGPRC;
    if (isCompressible(MI, NeedGPRC)) {
      if (OpIdx == 0 && MI.getOperand(1).isReg()) {
        if (!NeedGPRC || MI.getNumExplicitOperands() < 3 ||
            MI.getOpcode() == Capstone::ADD_UW ||
            isCompressibleOpnd(MI.getOperand(2)))
          tryAddHint(MO, MI.getOperand(1), NeedGPRC);
        if (MI.isCommutable() && MI.getOperand(2).isReg() &&
            (!NeedGPRC || isCompressibleOpnd(MI.getOperand(1))))
          tryAddHint(MO, MI.getOperand(2), NeedGPRC);
      } else if (OpIdx == 1 && (!NeedGPRC || MI.getNumExplicitOperands() < 3 ||
                                isCompressibleOpnd(MI.getOperand(2)))) {
        tryAddHint(MO, MI.getOperand(0), NeedGPRC);
      } else if (MI.isCommutable() && OpIdx == 2 &&
                 (!NeedGPRC || isCompressibleOpnd(MI.getOperand(1)))) {
        tryAddHint(MO, MI.getOperand(0), NeedGPRC);
      }
    }

    // Add a hint if it would allow auipc/lui+addi(w) fusion.  We do this even
    // without the fusions explicitly enabled as the impact is rarely negative
    // and some cores do implement this fusion.
    if ((MI.getOpcode() == Capstone::ADDIW || MI.getOpcode() == Capstone::ADDI) &&
        MI.getOperand(1).isReg()) {
      const MachineBasicBlock &MBB = *MI.getParent();
      MachineBasicBlock::const_iterator I = MI.getIterator();
      // Is the previous instruction a LUI or AUIPC that can be fused?
      if (I != MBB.begin()) {
        I = skipDebugInstructionsBackward(std::prev(I), MBB.begin());
        if ((I->getOpcode() == Capstone::LUI || I->getOpcode() == Capstone::AUIPC) &&
            I->getOperand(0).getReg() == MI.getOperand(1).getReg()) {
          if (OpIdx == 0)
            tryAddHint(MO, MI.getOperand(1), /*NeedGPRC=*/false);
          else
            tryAddHint(MO, MI.getOperand(0), /*NeedGPRC=*/false);
        }
      }
    }
  }

  for (MCPhysReg OrderReg : Order)
    if (TwoAddrHints.count(OrderReg))
      Hints.push_back(OrderReg);

  return BaseImplRetVal;
}

Register
CapstoneRegisterInfo::findVRegWithEncoding(const TargetRegisterClass &RegClass,
                                        uint16_t Encoding) const {
  MCRegister Reg = Capstone::V0 + Encoding;
  if (CapstoneRI::getLMul(RegClass.TSFlags) == CapstoneVType::LMUL_1)
    return Reg;
  return getMatchingSuperReg(Reg, Capstone::sub_vrm1_0, &RegClass);
}
