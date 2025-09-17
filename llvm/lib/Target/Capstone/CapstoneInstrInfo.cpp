//===-- CapstoneInstrInfo.cpp - Capstone Instruction Information -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the Capstone implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "CapstoneInstrInfo.h"
#include "MCTargetDesc/CapstoneBaseInfo.h"
#include "MCTargetDesc/CapstoneMatInt.h"
#include "Capstone.h"
#include "CapstoneMachineFunctionInfo.h"
#include "CapstoneSubtarget.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineCombinerPattern.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineTraceMetrics.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/CodeGen/StackMaps.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

#define GEN_CHECK_COMPRESS_INSTR
#include "CapstoneGenCompressInstEmitter.inc"

#define GET_INSTRINFO_CTOR_DTOR
#define GET_INSTRINFO_NAMED_OPS
#include "CapstoneGenInstrInfo.inc"

#define DEBUG_TYPE "capstone-instr-info"
STATISTIC(NumVRegSpilled,
          "Number of registers within vector register groups spilled");
STATISTIC(NumVRegReloaded,
          "Number of registers within vector register groups reloaded");

static cl::opt<bool> PreferWholeRegisterMove(
    "capstone-prefer-whole-register-move", cl::init(false), cl::Hidden,
    cl::desc("Prefer whole register move for vector registers."));

static cl::opt<MachineTraceStrategy> ForceMachineCombinerStrategy(
    "capstone-force-machine-combiner-strategy", cl::Hidden,
    cl::desc("Force machine combiner to use a specific strategy for machine "
             "trace metrics evaluation."),
    cl::init(MachineTraceStrategy::TS_NumStrategies),
    cl::values(clEnumValN(MachineTraceStrategy::TS_Local, "local",
                          "Local strategy."),
               clEnumValN(MachineTraceStrategy::TS_MinInstrCount, "min-instr",
                          "MinInstrCount strategy.")));

namespace llvm::CapstoneVPseudosTable {

using namespace Capstone;

#define GET_CapstoneVPseudosTable_IMPL
#include "CapstoneGenSearchableTables.inc"

} // namespace llvm::CapstoneVPseudosTable

namespace llvm::Capstone {

#define GET_CapstoneMaskedPseudosTable_IMPL
#include "CapstoneGenSearchableTables.inc"

} // end namespace llvm::Capstone

CapstoneInstrInfo::CapstoneInstrInfo(const CapstoneSubtarget &STI)
    : CapstoneGenInstrInfo(STI, Capstone::ADJCALLSTACKDOWN, Capstone::ADJCALLSTACKUP),
      STI(STI) {}

#define GET_INSTRINFO_HELPERS
#include "CapstoneGenInstrInfo.inc"

MCInst CapstoneInstrInfo::getNop() const {
  if (STI.hasStdExtZca())
    return MCInstBuilder(Capstone::C_NOP);
  return MCInstBuilder(Capstone::ADDI)
      .addReg(Capstone::X0)
      .addReg(Capstone::X0)
      .addImm(0);
}

Register CapstoneInstrInfo::isLoadFromStackSlot(const MachineInstr &MI,
                                             int &FrameIndex) const {
  TypeSize Dummy = TypeSize::getZero();
  return isLoadFromStackSlot(MI, FrameIndex, Dummy);
}

static std::optional<unsigned> getLMULForRVVWholeLoadStore(unsigned Opcode) {
  switch (Opcode) {
  default:
    return std::nullopt;
  case Capstone::VS1R_V:
  case Capstone::VL1RE8_V:
  case Capstone::VL1RE16_V:
  case Capstone::VL1RE32_V:
  case Capstone::VL1RE64_V:
    return 1;
  case Capstone::VS2R_V:
  case Capstone::VL2RE8_V:
  case Capstone::VL2RE16_V:
  case Capstone::VL2RE32_V:
  case Capstone::VL2RE64_V:
    return 2;
  case Capstone::VS4R_V:
  case Capstone::VL4RE8_V:
  case Capstone::VL4RE16_V:
  case Capstone::VL4RE32_V:
  case Capstone::VL4RE64_V:
    return 4;
  case Capstone::VS8R_V:
  case Capstone::VL8RE8_V:
  case Capstone::VL8RE16_V:
  case Capstone::VL8RE32_V:
  case Capstone::VL8RE64_V:
    return 8;
  }
}

Register CapstoneInstrInfo::isLoadFromStackSlot(const MachineInstr &MI,
                                             int &FrameIndex,
                                             TypeSize &MemBytes) const {
  switch (MI.getOpcode()) {
  default:
    return 0;
  case Capstone::LB:
  case Capstone::LBU:
    MemBytes = TypeSize::getFixed(1);
    break;
  case Capstone::LH:
  case Capstone::LH_INX:
  case Capstone::LHU:
  case Capstone::FLH:
    MemBytes = TypeSize::getFixed(2);
    break;
  case Capstone::LW:
  case Capstone::LW_INX:
  case Capstone::FLW:
  case Capstone::LWU:
    MemBytes = TypeSize::getFixed(4);
    break;
  case Capstone::LD:
  case Capstone::LD_RV32:
  case Capstone::FLD:
    MemBytes = TypeSize::getFixed(8);
    break;
  case Capstone::VL1RE8_V:
  case Capstone::VL2RE8_V:
  case Capstone::VL4RE8_V:
  case Capstone::VL8RE8_V:
    if (!MI.getOperand(1).isFI())
      return Register();
    FrameIndex = MI.getOperand(1).getIndex();
    unsigned LMUL = *getLMULForRVVWholeLoadStore(MI.getOpcode());
    MemBytes = TypeSize::getScalable(Capstone::RVVBytesPerBlock * LMUL);
    return MI.getOperand(0).getReg();
  }

  if (MI.getOperand(1).isFI() && MI.getOperand(2).isImm() &&
      MI.getOperand(2).getImm() == 0) {
    FrameIndex = MI.getOperand(1).getIndex();
    return MI.getOperand(0).getReg();
  }

  return 0;
}

Register CapstoneInstrInfo::isStoreToStackSlot(const MachineInstr &MI,
                                            int &FrameIndex) const {
  TypeSize Dummy = TypeSize::getZero();
  return isStoreToStackSlot(MI, FrameIndex, Dummy);
}

Register CapstoneInstrInfo::isStoreToStackSlot(const MachineInstr &MI,
                                            int &FrameIndex,
                                            TypeSize &MemBytes) const {
  switch (MI.getOpcode()) {
  default:
    return 0;
  case Capstone::SB:
    MemBytes = TypeSize::getFixed(1);
    break;
  case Capstone::SH:
  case Capstone::SH_INX:
  case Capstone::FSH:
    MemBytes = TypeSize::getFixed(2);
    break;
  case Capstone::SW:
  case Capstone::SW_INX:
  case Capstone::FSW:
    MemBytes = TypeSize::getFixed(4);
    break;
  case Capstone::SD:
  case Capstone::SD_RV32:
  case Capstone::FSD:
    MemBytes = TypeSize::getFixed(8);
    break;
  case Capstone::VS1R_V:
  case Capstone::VS2R_V:
  case Capstone::VS4R_V:
  case Capstone::VS8R_V:
    if (!MI.getOperand(1).isFI())
      return Register();
    FrameIndex = MI.getOperand(1).getIndex();
    unsigned LMUL = *getLMULForRVVWholeLoadStore(MI.getOpcode());
    MemBytes = TypeSize::getScalable(Capstone::RVVBytesPerBlock * LMUL);
    return MI.getOperand(0).getReg();
  }

  if (MI.getOperand(1).isFI() && MI.getOperand(2).isImm() &&
      MI.getOperand(2).getImm() == 0) {
    FrameIndex = MI.getOperand(1).getIndex();
    return MI.getOperand(0).getReg();
  }

  return 0;
}

bool CapstoneInstrInfo::isReallyTriviallyReMaterializable(
    const MachineInstr &MI) const {
  switch (Capstone::getRVVMCOpcode(MI.getOpcode())) {
  case Capstone::VMV_V_X:
  case Capstone::VFMV_V_F:
  case Capstone::VMV_V_I:
  case Capstone::VMV_S_X:
  case Capstone::VFMV_S_F:
  case Capstone::VID_V:
    return MI.getOperand(1).isUndef();
  default:
    return TargetInstrInfo::isReallyTriviallyReMaterializable(MI);
  }
}

static bool forwardCopyWillClobberTuple(unsigned DstReg, unsigned SrcReg,
                                        unsigned NumRegs) {
  return DstReg > SrcReg && (DstReg - SrcReg) < NumRegs;
}

static bool isConvertibleToVMV_V_V(const CapstoneSubtarget &STI,
                                   const MachineBasicBlock &MBB,
                                   MachineBasicBlock::const_iterator MBBI,
                                   MachineBasicBlock::const_iterator &DefMBBI,
                                   CapstoneVType::VLMUL LMul) {
  if (PreferWholeRegisterMove)
    return false;

  assert(MBBI->getOpcode() == TargetOpcode::COPY &&
         "Unexpected COPY instruction.");
  Register SrcReg = MBBI->getOperand(1).getReg();
  const TargetRegisterInfo *TRI = STI.getRegisterInfo();

  bool FoundDef = false;
  bool FirstVSetVLI = false;
  unsigned FirstSEW = 0;
  while (MBBI != MBB.begin()) {
    --MBBI;
    if (MBBI->isMetaInstruction())
      continue;

    if (CapstoneInstrInfo::isVectorConfigInstr(*MBBI)) {
      // There is a vsetvli between COPY and source define instruction.
      // vy = def_vop ...  (producing instruction)
      // ...
      // vsetvli
      // ...
      // vx = COPY vy
      if (!FoundDef) {
        if (!FirstVSetVLI) {
          FirstVSetVLI = true;
          unsigned FirstVType = MBBI->getOperand(2).getImm();
          CapstoneVType::VLMUL FirstLMul = CapstoneVType::getVLMUL(FirstVType);
          FirstSEW = CapstoneVType::getSEW(FirstVType);
          // The first encountered vsetvli must have the same lmul as the
          // register class of COPY.
          if (FirstLMul != LMul)
            return false;
        }
        // Only permit `vsetvli x0, x0, vtype` between COPY and the source
        // define instruction.
        if (!CapstoneInstrInfo::isVLPreservingConfig(*MBBI))
          return false;
        continue;
      }

      // MBBI is the first vsetvli before the producing instruction.
      unsigned VType = MBBI->getOperand(2).getImm();
      // If there is a vsetvli between COPY and the producing instruction.
      if (FirstVSetVLI) {
        // If SEW is different, return false.
        if (CapstoneVType::getSEW(VType) != FirstSEW)
          return false;
      }

      // If the vsetvli is tail undisturbed, keep the whole register move.
      if (!CapstoneVType::isTailAgnostic(VType))
        return false;

      // The checking is conservative. We only have register classes for
      // LMUL = 1/2/4/8. We should be able to convert vmv1r.v to vmv.v.v
      // for fractional LMUL operations. However, we could not use the vsetvli
      // lmul for widening operations. The result of widening operation is
      // 2 x LMUL.
      return LMul == CapstoneVType::getVLMUL(VType);
    } else if (MBBI->isInlineAsm() || MBBI->isCall()) {
      return false;
    } else if (MBBI->getNumDefs()) {
      // Check all the instructions which will change VL.
      // For example, vleff has implicit def VL.
      if (MBBI->modifiesRegister(Capstone::VL, /*TRI=*/nullptr))
        return false;

      // Only converting whole register copies to vmv.v.v when the defining
      // value appears in the explicit operands.
      for (const MachineOperand &MO : MBBI->explicit_operands()) {
        if (!MO.isReg() || !MO.isDef())
          continue;
        if (!FoundDef && TRI->regsOverlap(MO.getReg(), SrcReg)) {
          // We only permit the source of COPY has the same LMUL as the defined
          // operand.
          // There are cases we need to keep the whole register copy if the LMUL
          // is different.
          // For example,
          // $x0 = PseudoVSETIVLI 4, 73   // vsetivli zero, 4, e16,m2,ta,m
          // $v28m4 = PseudoVWADD_VV_M2 $v26m2, $v8m2
          // # The COPY may be created by vlmul_trunc intrinsic.
          // $v26m2 = COPY renamable $v28m2, implicit killed $v28m4
          //
          // After widening, the valid value will be 4 x e32 elements. If we
          // convert the COPY to vmv.v.v, it will only copy 4 x e16 elements.
          // FIXME: The COPY of subregister of Zvlsseg register will not be able
          // to convert to vmv.v.[v|i] under the constraint.
          if (MO.getReg() != SrcReg)
            return false;

          // In widening reduction instructions with LMUL_1 input vector case,
          // only checking the LMUL is insufficient due to reduction result is
          // always LMUL_1.
          // For example,
          // $x11 = PseudoVSETIVLI 1, 64 // vsetivli a1, 1, e8, m1, ta, mu
          // $v8m1 = PseudoVWREDSUM_VS_M1 $v26, $v27
          // $v26 = COPY killed renamable $v8
          // After widening, The valid value will be 1 x e16 elements. If we
          // convert the COPY to vmv.v.v, it will only copy 1 x e8 elements.
          uint64_t TSFlags = MBBI->getDesc().TSFlags;
          if (CapstoneII::isRVVWideningReduction(TSFlags))
            return false;

          // If the producing instruction does not depend on vsetvli, do not
          // convert COPY to vmv.v.v. For example, VL1R_V or PseudoVRELOAD.
          if (!CapstoneII::hasSEWOp(TSFlags) || !CapstoneII::hasVLOp(TSFlags))
            return false;

          // Found the definition.
          FoundDef = true;
          DefMBBI = MBBI;
          break;
        }
      }
    }
  }

  return false;
}

void CapstoneInstrInfo::copyPhysRegVector(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    const DebugLoc &DL, MCRegister DstReg, MCRegister SrcReg, bool KillSrc,
    const TargetRegisterClass *RegClass) const {
  const CapstoneRegisterInfo *TRI = STI.getRegisterInfo();
  CapstoneVType::VLMUL LMul = CapstoneRI::getLMul(RegClass->TSFlags);
  unsigned NF = CapstoneRI::getNF(RegClass->TSFlags);

  uint16_t SrcEncoding = TRI->getEncodingValue(SrcReg);
  uint16_t DstEncoding = TRI->getEncodingValue(DstReg);
  auto [LMulVal, Fractional] = CapstoneVType::decodeVLMUL(LMul);
  assert(!Fractional && "It is impossible be fractional lmul here.");
  unsigned NumRegs = NF * LMulVal;
  bool ReversedCopy =
      forwardCopyWillClobberTuple(DstEncoding, SrcEncoding, NumRegs);
  if (ReversedCopy) {
    // If the src and dest overlap when copying a tuple, we need to copy the
    // registers in reverse.
    SrcEncoding += NumRegs - 1;
    DstEncoding += NumRegs - 1;
  }

  unsigned I = 0;
  auto GetCopyInfo = [&](uint16_t SrcEncoding, uint16_t DstEncoding)
      -> std::tuple<CapstoneVType::VLMUL, const TargetRegisterClass &, unsigned,
                    unsigned, unsigned> {
    if (ReversedCopy) {
      // For reversed copying, if there are enough aligned registers(8/4/2), we
      // can do a larger copy(LMUL8/4/2).
      // Besides, we have already known that DstEncoding is larger than
      // SrcEncoding in forwardCopyWillClobberTuple, so the difference between
      // DstEncoding and SrcEncoding should be >= LMUL value we try to use to
      // avoid clobbering.
      uint16_t Diff = DstEncoding - SrcEncoding;
      if (I + 8 <= NumRegs && Diff >= 8 && SrcEncoding % 8 == 7 &&
          DstEncoding % 8 == 7)
        return {CapstoneVType::LMUL_8, Capstone::VRM8RegClass, Capstone::VMV8R_V,
                Capstone::PseudoVMV_V_V_M8, Capstone::PseudoVMV_V_I_M8};
      if (I + 4 <= NumRegs && Diff >= 4 && SrcEncoding % 4 == 3 &&
          DstEncoding % 4 == 3)
        return {CapstoneVType::LMUL_4, Capstone::VRM4RegClass, Capstone::VMV4R_V,
                Capstone::PseudoVMV_V_V_M4, Capstone::PseudoVMV_V_I_M4};
      if (I + 2 <= NumRegs && Diff >= 2 && SrcEncoding % 2 == 1 &&
          DstEncoding % 2 == 1)
        return {CapstoneVType::LMUL_2, Capstone::VRM2RegClass, Capstone::VMV2R_V,
                Capstone::PseudoVMV_V_V_M2, Capstone::PseudoVMV_V_I_M2};
      // Or we should do LMUL1 copying.
      return {CapstoneVType::LMUL_1, Capstone::VRRegClass, Capstone::VMV1R_V,
              Capstone::PseudoVMV_V_V_M1, Capstone::PseudoVMV_V_I_M1};
    }

    // For forward copying, if source register encoding and destination register
    // encoding are aligned to 8/4/2, we can do a LMUL8/4/2 copying.
    if (I + 8 <= NumRegs && SrcEncoding % 8 == 0 && DstEncoding % 8 == 0)
      return {CapstoneVType::LMUL_8, Capstone::VRM8RegClass, Capstone::VMV8R_V,
              Capstone::PseudoVMV_V_V_M8, Capstone::PseudoVMV_V_I_M8};
    if (I + 4 <= NumRegs && SrcEncoding % 4 == 0 && DstEncoding % 4 == 0)
      return {CapstoneVType::LMUL_4, Capstone::VRM4RegClass, Capstone::VMV4R_V,
              Capstone::PseudoVMV_V_V_M4, Capstone::PseudoVMV_V_I_M4};
    if (I + 2 <= NumRegs && SrcEncoding % 2 == 0 && DstEncoding % 2 == 0)
      return {CapstoneVType::LMUL_2, Capstone::VRM2RegClass, Capstone::VMV2R_V,
              Capstone::PseudoVMV_V_V_M2, Capstone::PseudoVMV_V_I_M2};
    // Or we should do LMUL1 copying.
    return {CapstoneVType::LMUL_1, Capstone::VRRegClass, Capstone::VMV1R_V,
            Capstone::PseudoVMV_V_V_M1, Capstone::PseudoVMV_V_I_M1};
  };

  while (I != NumRegs) {
    // For non-segment copying, we only do this once as the registers are always
    // aligned.
    // For segment copying, we may do this several times. If the registers are
    // aligned to larger LMUL, we can eliminate some copyings.
    auto [LMulCopied, RegClass, Opc, VVOpc, VIOpc] =
        GetCopyInfo(SrcEncoding, DstEncoding);
    auto [NumCopied, _] = CapstoneVType::decodeVLMUL(LMulCopied);

    MachineBasicBlock::const_iterator DefMBBI;
    if (LMul == LMulCopied &&
        isConvertibleToVMV_V_V(STI, MBB, MBBI, DefMBBI, LMul)) {
      Opc = VVOpc;
      if (DefMBBI->getOpcode() == VIOpc)
        Opc = VIOpc;
    }

    // Emit actual copying.
    // For reversed copying, the encoding should be decreased.
    MCRegister ActualSrcReg = TRI->findVRegWithEncoding(
        RegClass, ReversedCopy ? (SrcEncoding - NumCopied + 1) : SrcEncoding);
    MCRegister ActualDstReg = TRI->findVRegWithEncoding(
        RegClass, ReversedCopy ? (DstEncoding - NumCopied + 1) : DstEncoding);

    auto MIB = BuildMI(MBB, MBBI, DL, get(Opc), ActualDstReg);
    bool UseVMV_V_I = Capstone::getRVVMCOpcode(Opc) == Capstone::VMV_V_I;
    bool UseVMV = UseVMV_V_I || Capstone::getRVVMCOpcode(Opc) == Capstone::VMV_V_V;
    if (UseVMV)
      MIB.addReg(ActualDstReg, RegState::Undef);
    if (UseVMV_V_I)
      MIB = MIB.add(DefMBBI->getOperand(2));
    else
      MIB = MIB.addReg(ActualSrcReg, getKillRegState(KillSrc));
    if (UseVMV) {
      const MCInstrDesc &Desc = DefMBBI->getDesc();
      MIB.add(DefMBBI->getOperand(CapstoneII::getVLOpNum(Desc)));  // AVL
      unsigned Log2SEW =
          DefMBBI->getOperand(CapstoneII::getSEWOpNum(Desc)).getImm();
      MIB.addImm(Log2SEW ? Log2SEW : 3);                        // SEW
      MIB.addImm(0);                                            // tu, mu
      MIB.addReg(Capstone::VL, RegState::Implicit);
      MIB.addReg(Capstone::VTYPE, RegState::Implicit);
    }
    // Add an implicit read of the original source to silence the verifier
    // in the cases where some of the smaller VRs we're copying from might be
    // undef, caused by the fact that the original, larger source VR might not
    // be fully initialized at the time this COPY happens.
    MIB.addReg(SrcReg, RegState::Implicit);

    // If we are copying reversely, we should decrease the encoding.
    SrcEncoding += (ReversedCopy ? -NumCopied : NumCopied);
    DstEncoding += (ReversedCopy ? -NumCopied : NumCopied);
    I += NumCopied;
  }
}

void CapstoneInstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator MBBI,
                                 const DebugLoc &DL, Register DstReg,
                                 Register SrcReg, bool KillSrc,
                                 bool RenamableDest, bool RenamableSrc) const {
  const TargetRegisterInfo *TRI = STI.getRegisterInfo();
  unsigned KillFlag = getKillRegState(KillSrc);

  if (Capstone::GPRRegClass.contains(DstReg, SrcReg)) {
    BuildMI(MBB, MBBI, DL, get(Capstone::ADDI), DstReg)
        .addReg(SrcReg, KillFlag | getRenamableRegState(RenamableSrc))
        .addImm(0);
    return;
  }

  if (Capstone::GPRF16RegClass.contains(DstReg, SrcReg)) {
    BuildMI(MBB, MBBI, DL, get(Capstone::PseudoMV_FPR16INX), DstReg)
        .addReg(SrcReg, KillFlag | getRenamableRegState(RenamableSrc));
    return;
  }

  if (Capstone::GPRF32RegClass.contains(DstReg, SrcReg)) {
    BuildMI(MBB, MBBI, DL, get(Capstone::PseudoMV_FPR32INX), DstReg)
        .addReg(SrcReg, KillFlag | getRenamableRegState(RenamableSrc));
    return;
  }

  if (Capstone::GPRPairRegClass.contains(DstReg, SrcReg)) {
    MCRegister EvenReg = TRI->getSubReg(SrcReg, Capstone::sub_gpr_even);
    MCRegister OddReg = TRI->getSubReg(SrcReg, Capstone::sub_gpr_odd);
    // We need to correct the odd register of X0_Pair.
    if (OddReg == Capstone::DUMMY_REG_PAIR_WITH_X0)
      OddReg = Capstone::X0;
    assert(DstReg != Capstone::X0_Pair && "Cannot write to X0_Pair");

    // Emit an ADDI for both parts of GPRPair.
    BuildMI(MBB, MBBI, DL, get(Capstone::ADDI),
            TRI->getSubReg(DstReg, Capstone::sub_gpr_even))
        .addReg(EvenReg, KillFlag)
        .addImm(0);
    BuildMI(MBB, MBBI, DL, get(Capstone::ADDI),
            TRI->getSubReg(DstReg, Capstone::sub_gpr_odd))
        .addReg(OddReg, KillFlag)
        .addImm(0);
    return;
  }

  // Handle copy from csr
  if (Capstone::VCSRRegClass.contains(SrcReg) &&
      Capstone::GPRRegClass.contains(DstReg)) {
    BuildMI(MBB, MBBI, DL, get(Capstone::CSRRS), DstReg)
        .addImm(CapstoneSysReg::lookupSysRegByName(TRI->getName(SrcReg))->Encoding)
        .addReg(Capstone::X0);
    return;
  }

  if (Capstone::FPR16RegClass.contains(DstReg, SrcReg)) {
    unsigned Opc;
    if (STI.hasStdExtZfh()) {
      Opc = Capstone::FSGNJ_H;
    } else {
      assert(STI.hasStdExtF() &&
             (STI.hasStdExtZfhmin() || STI.hasStdExtZfbfmin()) &&
             "Unexpected extensions");
      // Zfhmin/Zfbfmin doesn't have FSGNJ_H, replace FSGNJ_H with FSGNJ_S.
      DstReg = TRI->getMatchingSuperReg(DstReg, Capstone::sub_16,
                                        &Capstone::FPR32RegClass);
      SrcReg = TRI->getMatchingSuperReg(SrcReg, Capstone::sub_16,
                                        &Capstone::FPR32RegClass);
      Opc = Capstone::FSGNJ_S;
    }
    BuildMI(MBB, MBBI, DL, get(Opc), DstReg)
        .addReg(SrcReg, KillFlag)
        .addReg(SrcReg, KillFlag);
    return;
  }

  if (Capstone::FPR32RegClass.contains(DstReg, SrcReg)) {
    BuildMI(MBB, MBBI, DL, get(Capstone::FSGNJ_S), DstReg)
        .addReg(SrcReg, KillFlag)
        .addReg(SrcReg, KillFlag);
    return;
  }

  if (Capstone::FPR64RegClass.contains(DstReg, SrcReg)) {
    BuildMI(MBB, MBBI, DL, get(Capstone::FSGNJ_D), DstReg)
        .addReg(SrcReg, KillFlag)
        .addReg(SrcReg, KillFlag);
    return;
  }

  if (Capstone::FPR32RegClass.contains(DstReg) &&
      Capstone::GPRRegClass.contains(SrcReg)) {
    BuildMI(MBB, MBBI, DL, get(Capstone::FMV_W_X), DstReg)
        .addReg(SrcReg, KillFlag);
    return;
  }

  if (Capstone::GPRRegClass.contains(DstReg) &&
      Capstone::FPR32RegClass.contains(SrcReg)) {
    BuildMI(MBB, MBBI, DL, get(Capstone::FMV_X_W), DstReg)
        .addReg(SrcReg, KillFlag);
    return;
  }

  if (Capstone::FPR64RegClass.contains(DstReg) &&
      Capstone::GPRRegClass.contains(SrcReg)) {
    assert(STI.getXLen() == 64 && "Unexpected GPR size");
    BuildMI(MBB, MBBI, DL, get(Capstone::FMV_D_X), DstReg)
        .addReg(SrcReg, KillFlag);
    return;
  }

  if (Capstone::GPRRegClass.contains(DstReg) &&
      Capstone::FPR64RegClass.contains(SrcReg)) {
    assert(STI.getXLen() == 64 && "Unexpected GPR size");
    BuildMI(MBB, MBBI, DL, get(Capstone::FMV_X_D), DstReg)
        .addReg(SrcReg, KillFlag);
    return;
  }

  // VR->VR copies.
  const TargetRegisterClass *RegClass =
      TRI->getCommonMinimalPhysRegClass(SrcReg, DstReg);
  if (CapstoneRegisterInfo::isRVVRegClass(RegClass)) {
    copyPhysRegVector(MBB, MBBI, DL, DstReg, SrcReg, KillSrc, RegClass);
    return;
  }

  llvm_unreachable("Impossible reg-to-reg copy");
}

void CapstoneInstrInfo::storeRegToStackSlot(MachineBasicBlock &MBB,
                                         MachineBasicBlock::iterator I,
                                         Register SrcReg, bool IsKill, int FI,
                                         const TargetRegisterClass *RC,
                                         const TargetRegisterInfo *TRI,
                                         Register VReg,
                                         MachineInstr::MIFlag Flags) const {
  MachineFunction *MF = MBB.getParent();
  MachineFrameInfo &MFI = MF->getFrameInfo();

  unsigned Opcode;
  if (Capstone::GPRRegClass.hasSubClassEq(RC)) {
    Opcode = TRI->getRegSizeInBits(Capstone::GPRRegClass) == 32 ?
             Capstone::SW : Capstone::SD;
  } else if (Capstone::GPRF16RegClass.hasSubClassEq(RC)) {
    Opcode = Capstone::SH_INX;
  } else if (Capstone::GPRF32RegClass.hasSubClassEq(RC)) {
    Opcode = Capstone::SW_INX;
  } else if (Capstone::GPRPairRegClass.hasSubClassEq(RC)) {
    Opcode = Capstone::PseudoRV32ZdinxSD;
  } else if (Capstone::FPR16RegClass.hasSubClassEq(RC)) {
    Opcode = Capstone::FSH;
  } else if (Capstone::FPR32RegClass.hasSubClassEq(RC)) {
    Opcode = Capstone::FSW;
  } else if (Capstone::FPR64RegClass.hasSubClassEq(RC)) {
    Opcode = Capstone::FSD;
  } else if (Capstone::VRRegClass.hasSubClassEq(RC)) {
    Opcode = Capstone::VS1R_V;
  } else if (Capstone::VRM2RegClass.hasSubClassEq(RC)) {
    Opcode = Capstone::VS2R_V;
  } else if (Capstone::VRM4RegClass.hasSubClassEq(RC)) {
    Opcode = Capstone::VS4R_V;
  } else if (Capstone::VRM8RegClass.hasSubClassEq(RC)) {
    Opcode = Capstone::VS8R_V;
  } else if (Capstone::VRN2M1RegClass.hasSubClassEq(RC))
    Opcode = Capstone::PseudoVSPILL2_M1;
  else if (Capstone::VRN2M2RegClass.hasSubClassEq(RC))
    Opcode = Capstone::PseudoVSPILL2_M2;
  else if (Capstone::VRN2M4RegClass.hasSubClassEq(RC))
    Opcode = Capstone::PseudoVSPILL2_M4;
  else if (Capstone::VRN3M1RegClass.hasSubClassEq(RC))
    Opcode = Capstone::PseudoVSPILL3_M1;
  else if (Capstone::VRN3M2RegClass.hasSubClassEq(RC))
    Opcode = Capstone::PseudoVSPILL3_M2;
  else if (Capstone::VRN4M1RegClass.hasSubClassEq(RC))
    Opcode = Capstone::PseudoVSPILL4_M1;
  else if (Capstone::VRN4M2RegClass.hasSubClassEq(RC))
    Opcode = Capstone::PseudoVSPILL4_M2;
  else if (Capstone::VRN5M1RegClass.hasSubClassEq(RC))
    Opcode = Capstone::PseudoVSPILL5_M1;
  else if (Capstone::VRN6M1RegClass.hasSubClassEq(RC))
    Opcode = Capstone::PseudoVSPILL6_M1;
  else if (Capstone::VRN7M1RegClass.hasSubClassEq(RC))
    Opcode = Capstone::PseudoVSPILL7_M1;
  else if (Capstone::VRN8M1RegClass.hasSubClassEq(RC))
    Opcode = Capstone::PseudoVSPILL8_M1;
  else
    llvm_unreachable("Can't store this register to stack slot");

  if (CapstoneRegisterInfo::isRVVRegClass(RC)) {
    MachineMemOperand *MMO = MF->getMachineMemOperand(
        MachinePointerInfo::getFixedStack(*MF, FI), MachineMemOperand::MOStore,
        TypeSize::getScalable(MFI.getObjectSize(FI)), MFI.getObjectAlign(FI));

    MFI.setStackID(FI, TargetStackID::ScalableVector);
    BuildMI(MBB, I, DebugLoc(), get(Opcode))
        .addReg(SrcReg, getKillRegState(IsKill))
        .addFrameIndex(FI)
        .addMemOperand(MMO)
        .setMIFlag(Flags);
    NumVRegSpilled += TRI->getRegSizeInBits(*RC) / Capstone::RVVBitsPerBlock;
  } else {
    MachineMemOperand *MMO = MF->getMachineMemOperand(
        MachinePointerInfo::getFixedStack(*MF, FI), MachineMemOperand::MOStore,
        MFI.getObjectSize(FI), MFI.getObjectAlign(FI));

    BuildMI(MBB, I, DebugLoc(), get(Opcode))
        .addReg(SrcReg, getKillRegState(IsKill))
        .addFrameIndex(FI)
        .addImm(0)
        .addMemOperand(MMO)
        .setMIFlag(Flags);
  }
}

void CapstoneInstrInfo::loadRegFromStackSlot(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator I, Register DstReg,
    int FI, const TargetRegisterClass *RC, const TargetRegisterInfo *TRI,
    Register VReg, MachineInstr::MIFlag Flags) const {
  MachineFunction *MF = MBB.getParent();
  MachineFrameInfo &MFI = MF->getFrameInfo();
  DebugLoc DL =
      Flags & MachineInstr::FrameDestroy ? MBB.findDebugLoc(I) : DebugLoc();

  unsigned Opcode;
  if (Capstone::GPRRegClass.hasSubClassEq(RC)) {
    Opcode = TRI->getRegSizeInBits(Capstone::GPRRegClass) == 32 ?
             Capstone::LW : Capstone::LD;
  } else if (Capstone::GPRF16RegClass.hasSubClassEq(RC)) {
    Opcode = Capstone::LH_INX;
  } else if (Capstone::GPRF32RegClass.hasSubClassEq(RC)) {
    Opcode = Capstone::LW_INX;
  } else if (Capstone::GPRPairRegClass.hasSubClassEq(RC)) {
    Opcode = Capstone::PseudoRV32ZdinxLD;
  } else if (Capstone::FPR16RegClass.hasSubClassEq(RC)) {
    Opcode = Capstone::FLH;
  } else if (Capstone::FPR32RegClass.hasSubClassEq(RC)) {
    Opcode = Capstone::FLW;
  } else if (Capstone::FPR64RegClass.hasSubClassEq(RC)) {
    Opcode = Capstone::FLD;
  } else if (Capstone::VRRegClass.hasSubClassEq(RC)) {
    Opcode = Capstone::VL1RE8_V;
  } else if (Capstone::VRM2RegClass.hasSubClassEq(RC)) {
    Opcode = Capstone::VL2RE8_V;
  } else if (Capstone::VRM4RegClass.hasSubClassEq(RC)) {
    Opcode = Capstone::VL4RE8_V;
  } else if (Capstone::VRM8RegClass.hasSubClassEq(RC)) {
    Opcode = Capstone::VL8RE8_V;
  } else if (Capstone::VRN2M1RegClass.hasSubClassEq(RC))
    Opcode = Capstone::PseudoVRELOAD2_M1;
  else if (Capstone::VRN2M2RegClass.hasSubClassEq(RC))
    Opcode = Capstone::PseudoVRELOAD2_M2;
  else if (Capstone::VRN2M4RegClass.hasSubClassEq(RC))
    Opcode = Capstone::PseudoVRELOAD2_M4;
  else if (Capstone::VRN3M1RegClass.hasSubClassEq(RC))
    Opcode = Capstone::PseudoVRELOAD3_M1;
  else if (Capstone::VRN3M2RegClass.hasSubClassEq(RC))
    Opcode = Capstone::PseudoVRELOAD3_M2;
  else if (Capstone::VRN4M1RegClass.hasSubClassEq(RC))
    Opcode = Capstone::PseudoVRELOAD4_M1;
  else if (Capstone::VRN4M2RegClass.hasSubClassEq(RC))
    Opcode = Capstone::PseudoVRELOAD4_M2;
  else if (Capstone::VRN5M1RegClass.hasSubClassEq(RC))
    Opcode = Capstone::PseudoVRELOAD5_M1;
  else if (Capstone::VRN6M1RegClass.hasSubClassEq(RC))
    Opcode = Capstone::PseudoVRELOAD6_M1;
  else if (Capstone::VRN7M1RegClass.hasSubClassEq(RC))
    Opcode = Capstone::PseudoVRELOAD7_M1;
  else if (Capstone::VRN8M1RegClass.hasSubClassEq(RC))
    Opcode = Capstone::PseudoVRELOAD8_M1;
  else
    llvm_unreachable("Can't load this register from stack slot");

  if (CapstoneRegisterInfo::isRVVRegClass(RC)) {
    MachineMemOperand *MMO = MF->getMachineMemOperand(
        MachinePointerInfo::getFixedStack(*MF, FI), MachineMemOperand::MOLoad,
        TypeSize::getScalable(MFI.getObjectSize(FI)), MFI.getObjectAlign(FI));

    MFI.setStackID(FI, TargetStackID::ScalableVector);
    BuildMI(MBB, I, DL, get(Opcode), DstReg)
        .addFrameIndex(FI)
        .addMemOperand(MMO)
        .setMIFlag(Flags);
    NumVRegReloaded += TRI->getRegSizeInBits(*RC) / Capstone::RVVBitsPerBlock;
  } else {
    MachineMemOperand *MMO = MF->getMachineMemOperand(
        MachinePointerInfo::getFixedStack(*MF, FI), MachineMemOperand::MOLoad,
        MFI.getObjectSize(FI), MFI.getObjectAlign(FI));

    BuildMI(MBB, I, DL, get(Opcode), DstReg)
        .addFrameIndex(FI)
        .addImm(0)
        .addMemOperand(MMO)
        .setMIFlag(Flags);
  }
}
std::optional<unsigned> getFoldedOpcode(MachineFunction &MF, MachineInstr &MI,
                                        ArrayRef<unsigned> Ops,
                                        const CapstoneSubtarget &ST) {

  // The below optimizations narrow the load so they are only valid for little
  // endian.
  // TODO: Support big endian by adding an offset into the frame object?
  if (MF.getDataLayout().isBigEndian())
    return std::nullopt;

  // Fold load from stack followed by sext.b/sext.h/sext.w/zext.b/zext.h/zext.w.
  if (Ops.size() != 1 || Ops[0] != 1)
    return std::nullopt;

  switch (MI.getOpcode()) {
  default:
    if (CapstoneInstrInfo::isSEXT_W(MI))
      return Capstone::LW;
    if (CapstoneInstrInfo::isZEXT_W(MI))
      return Capstone::LWU;
    if (CapstoneInstrInfo::isZEXT_B(MI))
      return Capstone::LBU;
    break;
  case Capstone::SEXT_H:
    return Capstone::LH;
  case Capstone::SEXT_B:
    return Capstone::LB;
  case Capstone::ZEXT_H_RV32:
  case Capstone::ZEXT_H_RV64:
    return Capstone::LHU;
  }

  switch (Capstone::getRVVMCOpcode(MI.getOpcode())) {
  default:
    return std::nullopt;
  case Capstone::VMV_X_S: {
    unsigned Log2SEW =
        MI.getOperand(CapstoneII::getSEWOpNum(MI.getDesc())).getImm();
    if (ST.getXLen() < (1U << Log2SEW))
      return std::nullopt;
    switch (Log2SEW) {
    case 3:
      return Capstone::LB;
    case 4:
      return Capstone::LH;
    case 5:
      return Capstone::LW;
    case 6:
      return Capstone::LD;
    default:
      llvm_unreachable("Unexpected SEW");
    }
  }
  case Capstone::VFMV_F_S: {
    unsigned Log2SEW =
        MI.getOperand(CapstoneII::getSEWOpNum(MI.getDesc())).getImm();
    switch (Log2SEW) {
    case 4:
      return Capstone::FLH;
    case 5:
      return Capstone::FLW;
    case 6:
      return Capstone::FLD;
    default:
      llvm_unreachable("Unexpected SEW");
    }
  }
  }
}

// This is the version used during inline spilling
MachineInstr *CapstoneInstrInfo::foldMemoryOperandImpl(
    MachineFunction &MF, MachineInstr &MI, ArrayRef<unsigned> Ops,
    MachineBasicBlock::iterator InsertPt, int FrameIndex, LiveIntervals *LIS,
    VirtRegMap *VRM) const {

  std::optional<unsigned> LoadOpc = getFoldedOpcode(MF, MI, Ops, STI);
  if (!LoadOpc)
    return nullptr;
  Register DstReg = MI.getOperand(0).getReg();
  return BuildMI(*MI.getParent(), InsertPt, MI.getDebugLoc(), get(*LoadOpc),
                 DstReg)
      .addFrameIndex(FrameIndex)
      .addImm(0);
}

void CapstoneInstrInfo::movImm(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator MBBI,
                            const DebugLoc &DL, Register DstReg, uint64_t Val,
                            MachineInstr::MIFlag Flag, bool DstRenamable,
                            bool DstIsDead) const {
  Register SrcReg = Capstone::X0;

  // For RV32, allow a sign or unsigned 32 bit value.
  if (!STI.is64Bit() && !isInt<32>(Val)) {
    // If have a uimm32 it will still fit in a register so we can allow it.
    if (!isUInt<32>(Val))
      report_fatal_error("Should only materialize 32-bit constants for RV32");

    // Sign extend for generateInstSeq.
    Val = SignExtend64<32>(Val);
  }

  CapstoneMatInt::InstSeq Seq = CapstoneMatInt::generateInstSeq(Val, STI);
  assert(!Seq.empty());

  bool SrcRenamable = false;
  unsigned Num = 0;

  for (const CapstoneMatInt::Inst &Inst : Seq) {
    bool LastItem = ++Num == Seq.size();
    unsigned DstRegState = getDeadRegState(DstIsDead && LastItem) |
                           getRenamableRegState(DstRenamable);
    unsigned SrcRegState = getKillRegState(SrcReg != Capstone::X0) |
                           getRenamableRegState(SrcRenamable);
    switch (Inst.getOpndKind()) {
    case CapstoneMatInt::Imm:
      BuildMI(MBB, MBBI, DL, get(Inst.getOpcode()))
          .addReg(DstReg, RegState::Define | DstRegState)
          .addImm(Inst.getImm())
          .setMIFlag(Flag);
      break;
    case CapstoneMatInt::RegX0:
      BuildMI(MBB, MBBI, DL, get(Inst.getOpcode()))
          .addReg(DstReg, RegState::Define | DstRegState)
          .addReg(SrcReg, SrcRegState)
          .addReg(Capstone::X0)
          .setMIFlag(Flag);
      break;
    case CapstoneMatInt::RegReg:
      BuildMI(MBB, MBBI, DL, get(Inst.getOpcode()))
          .addReg(DstReg, RegState::Define | DstRegState)
          .addReg(SrcReg, SrcRegState)
          .addReg(SrcReg, SrcRegState)
          .setMIFlag(Flag);
      break;
    case CapstoneMatInt::RegImm:
      BuildMI(MBB, MBBI, DL, get(Inst.getOpcode()))
          .addReg(DstReg, RegState::Define | DstRegState)
          .addReg(SrcReg, SrcRegState)
          .addImm(Inst.getImm())
          .setMIFlag(Flag);
      break;
    }

    // Only the first instruction has X0 as its source.
    SrcReg = DstReg;
    SrcRenamable = DstRenamable;
  }
}

CapstoneCC::CondCode CapstoneInstrInfo::getCondFromBranchOpc(unsigned Opc) {
  switch (Opc) {
  default:
    return CapstoneCC::COND_INVALID;
  case Capstone::BEQ:
  case Capstone::CV_BEQIMM:
  case Capstone::QC_BEQI:
  case Capstone::QC_E_BEQI:
  case Capstone::NDS_BBC:
  case Capstone::NDS_BEQC:
    return CapstoneCC::COND_EQ;
  case Capstone::BNE:
  case Capstone::QC_BNEI:
  case Capstone::QC_E_BNEI:
  case Capstone::CV_BNEIMM:
  case Capstone::NDS_BBS:
  case Capstone::NDS_BNEC:
    return CapstoneCC::COND_NE;
  case Capstone::BLT:
  case Capstone::QC_BLTI:
  case Capstone::QC_E_BLTI:
    return CapstoneCC::COND_LT;
  case Capstone::BGE:
  case Capstone::QC_BGEI:
  case Capstone::QC_E_BGEI:
    return CapstoneCC::COND_GE;
  case Capstone::BLTU:
  case Capstone::QC_BLTUI:
  case Capstone::QC_E_BLTUI:
    return CapstoneCC::COND_LTU;
  case Capstone::BGEU:
  case Capstone::QC_BGEUI:
  case Capstone::QC_E_BGEUI:
    return CapstoneCC::COND_GEU;
  }
}

bool CapstoneInstrInfo::evaluateCondBranch(CapstoneCC::CondCode CC, int64_t C0,
                                        int64_t C1) {
  switch (CC) {
  default:
    llvm_unreachable("Unexpected CC");
  case CapstoneCC::COND_EQ:
    return C0 == C1;
  case CapstoneCC::COND_NE:
    return C0 != C1;
  case CapstoneCC::COND_LT:
    return C0 < C1;
  case CapstoneCC::COND_GE:
    return C0 >= C1;
  case CapstoneCC::COND_LTU:
    return (uint64_t)C0 < (uint64_t)C1;
  case CapstoneCC::COND_GEU:
    return (uint64_t)C0 >= (uint64_t)C1;
  }
}

// The contents of values added to Cond are not examined outside of
// CapstoneInstrInfo, giving us flexibility in what to push to it. For Capstone, we
// push BranchOpcode, Reg1, Reg2.
static void parseCondBranch(MachineInstr &LastInst, MachineBasicBlock *&Target,
                            SmallVectorImpl<MachineOperand> &Cond) {
  // Block ends with fall-through condbranch.
  assert(LastInst.getDesc().isConditionalBranch() &&
         "Unknown conditional branch");
  Target = LastInst.getOperand(2).getMBB();
  Cond.push_back(MachineOperand::CreateImm(LastInst.getOpcode()));
  Cond.push_back(LastInst.getOperand(0));
  Cond.push_back(LastInst.getOperand(1));
}

unsigned CapstoneCC::getBrCond(CapstoneCC::CondCode CC, unsigned SelectOpc) {
  switch (SelectOpc) {
  default:
    switch (CC) {
    default:
      llvm_unreachable("Unexpected condition code!");
    case CapstoneCC::COND_EQ:
      return Capstone::BEQ;
    case CapstoneCC::COND_NE:
      return Capstone::BNE;
    case CapstoneCC::COND_LT:
      return Capstone::BLT;
    case CapstoneCC::COND_GE:
      return Capstone::BGE;
    case CapstoneCC::COND_LTU:
      return Capstone::BLTU;
    case CapstoneCC::COND_GEU:
      return Capstone::BGEU;
    }
    break;
  case Capstone::Select_GPR_Using_CC_SImm5_CV:
    switch (CC) {
    default:
      llvm_unreachable("Unexpected condition code!");
    case CapstoneCC::COND_EQ:
      return Capstone::CV_BEQIMM;
    case CapstoneCC::COND_NE:
      return Capstone::CV_BNEIMM;
    }
    break;
  case Capstone::Select_GPRNoX0_Using_CC_SImm5NonZero_QC:
    switch (CC) {
    default:
      llvm_unreachable("Unexpected condition code!");
    case CapstoneCC::COND_EQ:
      return Capstone::QC_BEQI;
    case CapstoneCC::COND_NE:
      return Capstone::QC_BNEI;
    case CapstoneCC::COND_LT:
      return Capstone::QC_BLTI;
    case CapstoneCC::COND_GE:
      return Capstone::QC_BGEI;
    }
    break;
  case Capstone::Select_GPRNoX0_Using_CC_UImm5NonZero_QC:
    switch (CC) {
    default:
      llvm_unreachable("Unexpected condition code!");
    case CapstoneCC::COND_LTU:
      return Capstone::QC_BLTUI;
    case CapstoneCC::COND_GEU:
      return Capstone::QC_BGEUI;
    }
    break;
  case Capstone::Select_GPRNoX0_Using_CC_SImm16NonZero_QC:
    switch (CC) {
    default:
      llvm_unreachable("Unexpected condition code!");
    case CapstoneCC::COND_EQ:
      return Capstone::QC_E_BEQI;
    case CapstoneCC::COND_NE:
      return Capstone::QC_E_BNEI;
    case CapstoneCC::COND_LT:
      return Capstone::QC_E_BLTI;
    case CapstoneCC::COND_GE:
      return Capstone::QC_E_BGEI;
    }
    break;
  case Capstone::Select_GPRNoX0_Using_CC_UImm16NonZero_QC:
    switch (CC) {
    default:
      llvm_unreachable("Unexpected condition code!");
    case CapstoneCC::COND_LTU:
      return Capstone::QC_E_BLTUI;
    case CapstoneCC::COND_GEU:
      return Capstone::QC_E_BGEUI;
    }
    break;
  case Capstone::Select_GPR_Using_CC_UImmLog2XLen_NDS:
    switch (CC) {
    default:
      llvm_unreachable("Unexpected condition code!");
    case CapstoneCC::COND_EQ:
      return Capstone::NDS_BBC;
    case CapstoneCC::COND_NE:
      return Capstone::NDS_BBS;
    }
    break;
  case Capstone::Select_GPR_Using_CC_UImm7_NDS:
    switch (CC) {
    default:
      llvm_unreachable("Unexpected condition code!");
    case CapstoneCC::COND_EQ:
      return Capstone::NDS_BEQC;
    case CapstoneCC::COND_NE:
      return Capstone::NDS_BNEC;
    }
    break;
  }
}

CapstoneCC::CondCode CapstoneCC::getOppositeBranchCondition(CapstoneCC::CondCode CC) {
  switch (CC) {
  default:
    llvm_unreachable("Unrecognized conditional branch");
  case CapstoneCC::COND_EQ:
    return CapstoneCC::COND_NE;
  case CapstoneCC::COND_NE:
    return CapstoneCC::COND_EQ;
  case CapstoneCC::COND_LT:
    return CapstoneCC::COND_GE;
  case CapstoneCC::COND_GE:
    return CapstoneCC::COND_LT;
  case CapstoneCC::COND_LTU:
    return CapstoneCC::COND_GEU;
  case CapstoneCC::COND_GEU:
    return CapstoneCC::COND_LTU;
  }
}

bool CapstoneInstrInfo::analyzeBranch(MachineBasicBlock &MBB,
                                   MachineBasicBlock *&TBB,
                                   MachineBasicBlock *&FBB,
                                   SmallVectorImpl<MachineOperand> &Cond,
                                   bool AllowModify) const {
  TBB = FBB = nullptr;
  Cond.clear();

  // If the block has no terminators, it just falls into the block after it.
  MachineBasicBlock::iterator I = MBB.getLastNonDebugInstr();
  if (I == MBB.end() || !isUnpredicatedTerminator(*I))
    return false;

  // Count the number of terminators and find the first unconditional or
  // indirect branch.
  MachineBasicBlock::iterator FirstUncondOrIndirectBr = MBB.end();
  int NumTerminators = 0;
  for (auto J = I.getReverse(); J != MBB.rend() && isUnpredicatedTerminator(*J);
       J++) {
    NumTerminators++;
    if (J->getDesc().isUnconditionalBranch() ||
        J->getDesc().isIndirectBranch()) {
      FirstUncondOrIndirectBr = J.getReverse();
    }
  }

  // If AllowModify is true, we can erase any terminators after
  // FirstUncondOrIndirectBR.
  if (AllowModify && FirstUncondOrIndirectBr != MBB.end()) {
    while (std::next(FirstUncondOrIndirectBr) != MBB.end()) {
      std::next(FirstUncondOrIndirectBr)->eraseFromParent();
      NumTerminators--;
    }
    I = FirstUncondOrIndirectBr;
  }

  // We can't handle blocks that end in an indirect branch.
  if (I->getDesc().isIndirectBranch())
    return true;

  // We can't handle Generic branch opcodes from Global ISel.
  if (I->isPreISelOpcode())
    return true;

  // We can't handle blocks with more than 2 terminators.
  if (NumTerminators > 2)
    return true;

  // Handle a single unconditional branch.
  if (NumTerminators == 1 && I->getDesc().isUnconditionalBranch()) {
    TBB = getBranchDestBlock(*I);
    return false;
  }

  // Handle a single conditional branch.
  if (NumTerminators == 1 && I->getDesc().isConditionalBranch()) {
    parseCondBranch(*I, TBB, Cond);
    return false;
  }

  // Handle a conditional branch followed by an unconditional branch.
  if (NumTerminators == 2 && std::prev(I)->getDesc().isConditionalBranch() &&
      I->getDesc().isUnconditionalBranch()) {
    parseCondBranch(*std::prev(I), TBB, Cond);
    FBB = getBranchDestBlock(*I);
    return false;
  }

  // Otherwise, we can't handle this.
  return true;
}

unsigned CapstoneInstrInfo::removeBranch(MachineBasicBlock &MBB,
                                      int *BytesRemoved) const {
  if (BytesRemoved)
    *BytesRemoved = 0;
  MachineBasicBlock::iterator I = MBB.getLastNonDebugInstr();
  if (I == MBB.end())
    return 0;

  if (!I->getDesc().isUnconditionalBranch() &&
      !I->getDesc().isConditionalBranch())
    return 0;

  // Remove the branch.
  if (BytesRemoved)
    *BytesRemoved += getInstSizeInBytes(*I);
  I->eraseFromParent();

  I = MBB.end();

  if (I == MBB.begin())
    return 1;
  --I;
  if (!I->getDesc().isConditionalBranch())
    return 1;

  // Remove the branch.
  if (BytesRemoved)
    *BytesRemoved += getInstSizeInBytes(*I);
  I->eraseFromParent();
  return 2;
}

// Inserts a branch into the end of the specific MachineBasicBlock, returning
// the number of instructions inserted.
unsigned CapstoneInstrInfo::insertBranch(
    MachineBasicBlock &MBB, MachineBasicBlock *TBB, MachineBasicBlock *FBB,
    ArrayRef<MachineOperand> Cond, const DebugLoc &DL, int *BytesAdded) const {
  if (BytesAdded)
    *BytesAdded = 0;

  // Shouldn't be a fall through.
  assert(TBB && "insertBranch must not be told to insert a fallthrough");
  assert((Cond.size() == 3 || Cond.size() == 0) &&
         "Capstone branch conditions have two components!");

  // Unconditional branch.
  if (Cond.empty()) {
    MachineInstr &MI = *BuildMI(&MBB, DL, get(Capstone::PseudoBR)).addMBB(TBB);
    if (BytesAdded)
      *BytesAdded += getInstSizeInBytes(MI);
    return 1;
  }

  // Either a one or two-way conditional branch.
  MachineInstr &CondMI = *BuildMI(&MBB, DL, get(Cond[0].getImm()))
                              .add(Cond[1])
                              .add(Cond[2])
                              .addMBB(TBB);
  if (BytesAdded)
    *BytesAdded += getInstSizeInBytes(CondMI);

  // One-way conditional branch.
  if (!FBB)
    return 1;

  // Two-way conditional branch.
  MachineInstr &MI = *BuildMI(&MBB, DL, get(Capstone::PseudoBR)).addMBB(FBB);
  if (BytesAdded)
    *BytesAdded += getInstSizeInBytes(MI);
  return 2;
}

void CapstoneInstrInfo::insertIndirectBranch(MachineBasicBlock &MBB,
                                          MachineBasicBlock &DestBB,
                                          MachineBasicBlock &RestoreBB,
                                          const DebugLoc &DL, int64_t BrOffset,
                                          RegScavenger *RS) const {
  assert(RS && "RegScavenger required for long branching");
  assert(MBB.empty() &&
         "new block should be inserted for expanding unconditional branch");
  assert(MBB.pred_size() == 1);
  assert(RestoreBB.empty() &&
         "restore block should be inserted for restoring clobbered registers");

  MachineFunction *MF = MBB.getParent();
  MachineRegisterInfo &MRI = MF->getRegInfo();
  CapstoneMachineFunctionInfo *RVFI = MF->getInfo<CapstoneMachineFunctionInfo>();
  const TargetRegisterInfo *TRI = MF->getSubtarget().getRegisterInfo();

  if (!isInt<32>(BrOffset))
    report_fatal_error(
        "Branch offsets outside of the signed 32-bit range not supported");

  // FIXME: A virtual register must be used initially, as the register
  // scavenger won't work with empty blocks (SIInstrInfo::insertIndirectBranch
  // uses the same workaround).
  Register ScratchReg = MRI.createVirtualRegister(&Capstone::GPRJALRRegClass);
  auto II = MBB.end();
  // We may also update the jump target to RestoreBB later.
  MachineInstr &MI = *BuildMI(MBB, II, DL, get(Capstone::PseudoJump))
                          .addReg(ScratchReg, RegState::Define | RegState::Dead)
                          .addMBB(&DestBB, CapstoneII::MO_CALL);

  RS->enterBasicBlockEnd(MBB);
  Register TmpGPR =
      RS->scavengeRegisterBackwards(Capstone::GPRRegClass, MI.getIterator(),
                                    /*RestoreAfter=*/false, /*SpAdj=*/0,
                                    /*AllowSpill=*/false);
  if (TmpGPR != Capstone::NoRegister)
    RS->setRegUsed(TmpGPR);
  else {
    // The case when there is no scavenged register needs special handling.

    // Pick s11(or s1 for rve) because it doesn't make a difference.
    TmpGPR = STI.hasStdExtE() ? Capstone::X9 : Capstone::X27;

    int FrameIndex = RVFI->getBranchRelaxationScratchFrameIndex();
    if (FrameIndex == -1)
      report_fatal_error("underestimated function size");

    storeRegToStackSlot(MBB, MI, TmpGPR, /*IsKill=*/true, FrameIndex,
                        &Capstone::GPRRegClass, TRI, Register());
    TRI->eliminateFrameIndex(std::prev(MI.getIterator()),
                             /*SpAdj=*/0, /*FIOperandNum=*/1);

    MI.getOperand(1).setMBB(&RestoreBB);

    loadRegFromStackSlot(RestoreBB, RestoreBB.end(), TmpGPR, FrameIndex,
                         &Capstone::GPRRegClass, TRI, Register());
    TRI->eliminateFrameIndex(RestoreBB.back(),
                             /*SpAdj=*/0, /*FIOperandNum=*/1);
  }

  MRI.replaceRegWith(ScratchReg, TmpGPR);
  MRI.clearVirtRegs();
}

bool CapstoneInstrInfo::reverseBranchCondition(
    SmallVectorImpl<MachineOperand> &Cond) const {
  assert((Cond.size() == 3) && "Invalid branch condition!");
  switch (Cond[0].getImm()) {
  default:
    llvm_unreachable("Unknown conditional branch!");
  case Capstone::BEQ:
    Cond[0].setImm(Capstone::BNE);
    break;
  case Capstone::BNE:
    Cond[0].setImm(Capstone::BEQ);
    break;
  case Capstone::BLT:
    Cond[0].setImm(Capstone::BGE);
    break;
  case Capstone::BGE:
    Cond[0].setImm(Capstone::BLT);
    break;
  case Capstone::BLTU:
    Cond[0].setImm(Capstone::BGEU);
    break;
  case Capstone::BGEU:
    Cond[0].setImm(Capstone::BLTU);
    break;
  case Capstone::CV_BEQIMM:
    Cond[0].setImm(Capstone::CV_BNEIMM);
    break;
  case Capstone::CV_BNEIMM:
    Cond[0].setImm(Capstone::CV_BEQIMM);
    break;
  case Capstone::QC_BEQI:
    Cond[0].setImm(Capstone::QC_BNEI);
    break;
  case Capstone::QC_BNEI:
    Cond[0].setImm(Capstone::QC_BEQI);
    break;
  case Capstone::QC_BGEI:
    Cond[0].setImm(Capstone::QC_BLTI);
    break;
  case Capstone::QC_BLTI:
    Cond[0].setImm(Capstone::QC_BGEI);
    break;
  case Capstone::QC_BGEUI:
    Cond[0].setImm(Capstone::QC_BLTUI);
    break;
  case Capstone::QC_BLTUI:
    Cond[0].setImm(Capstone::QC_BGEUI);
    break;
  case Capstone::QC_E_BEQI:
    Cond[0].setImm(Capstone::QC_E_BNEI);
    break;
  case Capstone::QC_E_BNEI:
    Cond[0].setImm(Capstone::QC_E_BEQI);
    break;
  case Capstone::QC_E_BGEI:
    Cond[0].setImm(Capstone::QC_E_BLTI);
    break;
  case Capstone::QC_E_BLTI:
    Cond[0].setImm(Capstone::QC_E_BGEI);
    break;
  case Capstone::QC_E_BGEUI:
    Cond[0].setImm(Capstone::QC_E_BLTUI);
    break;
  case Capstone::QC_E_BLTUI:
    Cond[0].setImm(Capstone::QC_E_BGEUI);
    break;
  case Capstone::NDS_BBC:
    Cond[0].setImm(Capstone::NDS_BBS);
    break;
  case Capstone::NDS_BBS:
    Cond[0].setImm(Capstone::NDS_BBC);
    break;
  case Capstone::NDS_BEQC:
    Cond[0].setImm(Capstone::NDS_BNEC);
    break;
  case Capstone::NDS_BNEC:
    Cond[0].setImm(Capstone::NDS_BEQC);
    break;
  }

  return false;
}

// Return true if the instruction is a load immediate instruction (i.e.
// ADDI x0, imm).
static bool isLoadImm(const MachineInstr *MI, int64_t &Imm) {
  if (MI->getOpcode() == Capstone::ADDI && MI->getOperand(1).isReg() &&
      MI->getOperand(1).getReg() == Capstone::X0) {
    Imm = MI->getOperand(2).getImm();
    return true;
  }
  return false;
}

bool CapstoneInstrInfo::isFromLoadImm(const MachineRegisterInfo &MRI,
                                   const MachineOperand &Op, int64_t &Imm) {
  // Either a load from immediate instruction or X0.
  if (!Op.isReg())
    return false;

  Register Reg = Op.getReg();
  if (Reg == Capstone::X0) {
    Imm = 0;
    return true;
  }
  return Reg.isVirtual() && isLoadImm(MRI.getVRegDef(Reg), Imm);
}

bool CapstoneInstrInfo::optimizeCondBranch(MachineInstr &MI) const {
  bool IsSigned = false;
  bool IsEquality = false;
  switch (MI.getOpcode()) {
  default:
    return false;
  case Capstone::BEQ:
  case Capstone::BNE:
    IsEquality = true;
    break;
  case Capstone::BGE:
  case Capstone::BLT:
    IsSigned = true;
    break;
  case Capstone::BGEU:
  case Capstone::BLTU:
    break;
  }

  MachineBasicBlock *MBB = MI.getParent();
  MachineRegisterInfo &MRI = MBB->getParent()->getRegInfo();

  const MachineOperand &LHS = MI.getOperand(0);
  const MachineOperand &RHS = MI.getOperand(1);
  MachineBasicBlock *TBB = MI.getOperand(2).getMBB();

  CapstoneCC::CondCode CC = getCondFromBranchOpc(MI.getOpcode());
  assert(CC != CapstoneCC::COND_INVALID);

  // Canonicalize conditional branches which can be constant folded into
  // beqz or bnez.  We can't modify the CFG here.
  int64_t C0, C1;
  if (isFromLoadImm(MRI, LHS, C0) && isFromLoadImm(MRI, RHS, C1)) {
    unsigned NewOpc = evaluateCondBranch(CC, C0, C1) ? Capstone::BEQ : Capstone::BNE;
    // Build the new branch and remove the old one.
    BuildMI(*MBB, MI, MI.getDebugLoc(), get(NewOpc))
        .addReg(Capstone::X0)
        .addReg(Capstone::X0)
        .addMBB(TBB);
    MI.eraseFromParent();
    return true;
  }

  if (IsEquality)
    return false;

  // For two constants C0 and C1 from
  // ```
  // li Y, C0
  // li Z, C1
  // ```
  // 1. if C1 = C0 + 1
  // we can turn:
  //  (a) blt Y, X -> bge X, Z
  //  (b) bge Y, X -> blt X, Z
  //
  // 2. if C1 = C0 - 1
  // we can turn:
  //  (a) blt X, Y -> bge Z, X
  //  (b) bge X, Y -> blt Z, X
  //
  // To make sure this optimization is really beneficial, we only
  // optimize for cases where Y had only one use (i.e. only used by the branch).
  // Try to find the register for constant Z; return
  // invalid register otherwise.
  auto searchConst = [&](int64_t C1) -> Register {
    MachineBasicBlock::reverse_iterator II(&MI), E = MBB->rend();
    auto DefC1 = std::find_if(++II, E, [&](const MachineInstr &I) -> bool {
      int64_t Imm;
      return isLoadImm(&I, Imm) && Imm == C1 &&
             I.getOperand(0).getReg().isVirtual();
    });
    if (DefC1 != E)
      return DefC1->getOperand(0).getReg();

    return Register();
  };

  unsigned NewOpc = CapstoneCC::getBrCond(getOppositeBranchCondition(CC));

  // Might be case 1.
  // Don't change 0 to 1 since we can use x0.
  // For unsigned cases changing -1U to 0 would be incorrect.
  // The incorrect case for signed would be INT_MAX, but isFromLoadImm can't
  // return that.
  if (isFromLoadImm(MRI, LHS, C0) && C0 != 0 && LHS.getReg().isVirtual() &&
      MRI.hasOneUse(LHS.getReg()) && (IsSigned || C0 != -1)) {
    assert(isInt<12>(C0) && "Unexpected immediate");
    if (Register RegZ = searchConst(C0 + 1)) {
      BuildMI(*MBB, MI, MI.getDebugLoc(), get(NewOpc))
          .add(RHS)
          .addReg(RegZ)
          .addMBB(TBB);
      // We might extend the live range of Z, clear its kill flag to
      // account for this.
      MRI.clearKillFlags(RegZ);
      MI.eraseFromParent();
      return true;
    }
  }

  // Might be case 2.
  // For signed cases we don't want to change 0 since we can use x0.
  // For unsigned cases changing 0 to -1U would be incorrect.
  // The incorrect case for signed would be INT_MIN, but isFromLoadImm can't
  // return that.
  if (isFromLoadImm(MRI, RHS, C0) && C0 != 0 && RHS.getReg().isVirtual() &&
      MRI.hasOneUse(RHS.getReg())) {
    assert(isInt<12>(C0) && "Unexpected immediate");
    if (Register RegZ = searchConst(C0 - 1)) {
      BuildMI(*MBB, MI, MI.getDebugLoc(), get(NewOpc))
          .addReg(RegZ)
          .add(LHS)
          .addMBB(TBB);
      // We might extend the live range of Z, clear its kill flag to
      // account for this.
      MRI.clearKillFlags(RegZ);
      MI.eraseFromParent();
      return true;
    }
  }

  return false;
}

MachineBasicBlock *
CapstoneInstrInfo::getBranchDestBlock(const MachineInstr &MI) const {
  assert(MI.getDesc().isBranch() && "Unexpected opcode!");
  // The branch target is always the last operand.
  int NumOp = MI.getNumExplicitOperands();
  return MI.getOperand(NumOp - 1).getMBB();
}

bool CapstoneInstrInfo::isBranchOffsetInRange(unsigned BranchOp,
                                           int64_t BrOffset) const {
  unsigned XLen = STI.getXLen();
  // Ideally we could determine the supported branch offset from the
  // CapstoneII::FormMask, but this can't be used for Pseudo instructions like
  // PseudoBR.
  switch (BranchOp) {
  default:
    llvm_unreachable("Unexpected opcode!");
  case Capstone::NDS_BBC:
  case Capstone::NDS_BBS:
  case Capstone::NDS_BEQC:
  case Capstone::NDS_BNEC:
    return isInt<11>(BrOffset);
  case Capstone::BEQ:
  case Capstone::BNE:
  case Capstone::BLT:
  case Capstone::BGE:
  case Capstone::BLTU:
  case Capstone::BGEU:
  case Capstone::CV_BEQIMM:
  case Capstone::CV_BNEIMM:
  case Capstone::QC_BEQI:
  case Capstone::QC_BNEI:
  case Capstone::QC_BGEI:
  case Capstone::QC_BLTI:
  case Capstone::QC_BLTUI:
  case Capstone::QC_BGEUI:
  case Capstone::QC_E_BEQI:
  case Capstone::QC_E_BNEI:
  case Capstone::QC_E_BGEI:
  case Capstone::QC_E_BLTI:
  case Capstone::QC_E_BLTUI:
  case Capstone::QC_E_BGEUI:
    return isInt<13>(BrOffset);
  case Capstone::JAL:
  case Capstone::PseudoBR:
    return isInt<21>(BrOffset);
  case Capstone::PseudoJump:
    return isInt<32>(SignExtend64(BrOffset + 0x800, XLen));
  }
}

// If the operation has a predicated pseudo instruction, return the pseudo
// instruction opcode. Otherwise, return Capstone::INSTRUCTION_LIST_END.
// TODO: Support more operations.
unsigned getPredicatedOpcode(unsigned Opcode) {
  switch (Opcode) {
  case Capstone::ADD:   return Capstone::PseudoCCADD;   break;
  case Capstone::SUB:   return Capstone::PseudoCCSUB;   break;
  case Capstone::SLL:   return Capstone::PseudoCCSLL;   break;
  case Capstone::SRL:   return Capstone::PseudoCCSRL;   break;
  case Capstone::SRA:   return Capstone::PseudoCCSRA;   break;
  case Capstone::AND:   return Capstone::PseudoCCAND;   break;
  case Capstone::OR:    return Capstone::PseudoCCOR;    break;
  case Capstone::XOR:   return Capstone::PseudoCCXOR;   break;

  case Capstone::ADDI:  return Capstone::PseudoCCADDI;  break;
  case Capstone::SLLI:  return Capstone::PseudoCCSLLI;  break;
  case Capstone::SRLI:  return Capstone::PseudoCCSRLI;  break;
  case Capstone::SRAI:  return Capstone::PseudoCCSRAI;  break;
  case Capstone::ANDI:  return Capstone::PseudoCCANDI;  break;
  case Capstone::ORI:   return Capstone::PseudoCCORI;   break;
  case Capstone::XORI:  return Capstone::PseudoCCXORI;  break;

  case Capstone::ADDW:  return Capstone::PseudoCCADDW;  break;
  case Capstone::SUBW:  return Capstone::PseudoCCSUBW;  break;
  case Capstone::SLLW:  return Capstone::PseudoCCSLLW;  break;
  case Capstone::SRLW:  return Capstone::PseudoCCSRLW;  break;
  case Capstone::SRAW:  return Capstone::PseudoCCSRAW;  break;

  case Capstone::ADDIW: return Capstone::PseudoCCADDIW; break;
  case Capstone::SLLIW: return Capstone::PseudoCCSLLIW; break;
  case Capstone::SRLIW: return Capstone::PseudoCCSRLIW; break;
  case Capstone::SRAIW: return Capstone::PseudoCCSRAIW; break;

  case Capstone::ANDN:  return Capstone::PseudoCCANDN;  break;
  case Capstone::ORN:   return Capstone::PseudoCCORN;   break;
  case Capstone::XNOR:  return Capstone::PseudoCCXNOR;  break;

  case Capstone::NDS_BFOS:  return Capstone::PseudoCCNDS_BFOS;  break;
  case Capstone::NDS_BFOZ:  return Capstone::PseudoCCNDS_BFOZ;  break;
  }

  return Capstone::INSTRUCTION_LIST_END;
}

/// Identify instructions that can be folded into a CCMOV instruction, and
/// return the defining instruction.
static MachineInstr *canFoldAsPredicatedOp(Register Reg,
                                           const MachineRegisterInfo &MRI,
                                           const TargetInstrInfo *TII) {
  if (!Reg.isVirtual())
    return nullptr;
  if (!MRI.hasOneNonDBGUse(Reg))
    return nullptr;
  MachineInstr *MI = MRI.getVRegDef(Reg);
  if (!MI)
    return nullptr;
  // Check if MI can be predicated and folded into the CCMOV.
  if (getPredicatedOpcode(MI->getOpcode()) == Capstone::INSTRUCTION_LIST_END)
    return nullptr;
  // Don't predicate li idiom.
  if (MI->getOpcode() == Capstone::ADDI && MI->getOperand(1).isReg() &&
      MI->getOperand(1).getReg() == Capstone::X0)
    return nullptr;
  // Check if MI has any other defs or physreg uses.
  for (const MachineOperand &MO : llvm::drop_begin(MI->operands())) {
    // Reject frame index operands, PEI can't handle the predicated pseudos.
    if (MO.isFI() || MO.isCPI() || MO.isJTI())
      return nullptr;
    if (!MO.isReg())
      continue;
    // MI can't have any tied operands, that would conflict with predication.
    if (MO.isTied())
      return nullptr;
    if (MO.isDef())
      return nullptr;
    // Allow constant physregs.
    if (MO.getReg().isPhysical() && !MRI.isConstantPhysReg(MO.getReg()))
      return nullptr;
  }
  bool DontMoveAcrossStores = true;
  if (!MI->isSafeToMove(DontMoveAcrossStores))
    return nullptr;
  return MI;
}

bool CapstoneInstrInfo::analyzeSelect(const MachineInstr &MI,
                                   SmallVectorImpl<MachineOperand> &Cond,
                                   unsigned &TrueOp, unsigned &FalseOp,
                                   bool &Optimizable) const {
  assert(MI.getOpcode() == Capstone::PseudoCCMOVGPR &&
         "Unknown select instruction");
  // CCMOV operands:
  // 0: Def.
  // 1: LHS of compare.
  // 2: RHS of compare.
  // 3: Condition code.
  // 4: False use.
  // 5: True use.
  TrueOp = 5;
  FalseOp = 4;
  Cond.push_back(MI.getOperand(1));
  Cond.push_back(MI.getOperand(2));
  Cond.push_back(MI.getOperand(3));
  // We can only fold when we support short forward branch opt.
  Optimizable = STI.hasShortForwardBranchOpt();
  return false;
}

MachineInstr *
CapstoneInstrInfo::optimizeSelect(MachineInstr &MI,
                               SmallPtrSetImpl<MachineInstr *> &SeenMIs,
                               bool PreferFalse) const {
  assert(MI.getOpcode() == Capstone::PseudoCCMOVGPR &&
         "Unknown select instruction");
  if (!STI.hasShortForwardBranchOpt())
    return nullptr;

  MachineRegisterInfo &MRI = MI.getParent()->getParent()->getRegInfo();
  MachineInstr *DefMI =
      canFoldAsPredicatedOp(MI.getOperand(5).getReg(), MRI, this);
  bool Invert = !DefMI;
  if (!DefMI)
    DefMI = canFoldAsPredicatedOp(MI.getOperand(4).getReg(), MRI, this);
  if (!DefMI)
    return nullptr;

  // Find new register class to use.
  MachineOperand FalseReg = MI.getOperand(Invert ? 5 : 4);
  Register DestReg = MI.getOperand(0).getReg();
  const TargetRegisterClass *PreviousClass = MRI.getRegClass(FalseReg.getReg());
  if (!MRI.constrainRegClass(DestReg, PreviousClass))
    return nullptr;

  unsigned PredOpc = getPredicatedOpcode(DefMI->getOpcode());
  assert(PredOpc != Capstone::INSTRUCTION_LIST_END && "Unexpected opcode!");

  // Create a new predicated version of DefMI.
  MachineInstrBuilder NewMI =
      BuildMI(*MI.getParent(), MI, MI.getDebugLoc(), get(PredOpc), DestReg);

  // Copy the condition portion.
  NewMI.add(MI.getOperand(1));
  NewMI.add(MI.getOperand(2));

  // Add condition code, inverting if necessary.
  auto CC = static_cast<CapstoneCC::CondCode>(MI.getOperand(3).getImm());
  if (Invert)
    CC = CapstoneCC::getOppositeBranchCondition(CC);
  NewMI.addImm(CC);

  // Copy the false register.
  NewMI.add(FalseReg);

  // Copy all the DefMI operands.
  const MCInstrDesc &DefDesc = DefMI->getDesc();
  for (unsigned i = 1, e = DefDesc.getNumOperands(); i != e; ++i)
    NewMI.add(DefMI->getOperand(i));

  // Update SeenMIs set: register newly created MI and erase removed DefMI.
  SeenMIs.insert(NewMI);
  SeenMIs.erase(DefMI);

  // If MI is inside a loop, and DefMI is outside the loop, then kill flags on
  // DefMI would be invalid when transferred inside the loop.  Checking for a
  // loop is expensive, but at least remove kill flags if they are in different
  // BBs.
  if (DefMI->getParent() != MI.getParent())
    NewMI->clearKillInfo();

  // The caller will erase MI, but not DefMI.
  DefMI->eraseFromParent();
  return NewMI;
}

unsigned CapstoneInstrInfo::getInstSizeInBytes(const MachineInstr &MI) const {
  if (MI.isMetaInstruction())
    return 0;

  unsigned Opcode = MI.getOpcode();

  if (Opcode == TargetOpcode::INLINEASM ||
      Opcode == TargetOpcode::INLINEASM_BR) {
    const MachineFunction &MF = *MI.getParent()->getParent();
    return getInlineAsmLength(MI.getOperand(0).getSymbolName(),
                              *MF.getTarget().getMCAsmInfo());
  }

  if (!MI.memoperands_empty()) {
    MachineMemOperand *MMO = *(MI.memoperands_begin());
    if (STI.hasStdExtZihintntl() && MMO->isNonTemporal()) {
      if (STI.hasStdExtZca()) {
        if (isCompressibleInst(MI, STI))
          return 4; // c.ntl.all + c.load/c.store
        return 6;   // c.ntl.all + load/store
      }
      return 8; // ntl.all + load/store
    }
  }

  if (Opcode == TargetOpcode::BUNDLE)
    return getInstBundleLength(MI);

  if (MI.getParent() && MI.getParent()->getParent()) {
    if (isCompressibleInst(MI, STI))
      return 2;
  }

  switch (Opcode) {
  case Capstone::PseudoMV_FPR16INX:
  case Capstone::PseudoMV_FPR32INX:
    // MV is always compressible to either c.mv or c.li rd, 0.
    return STI.hasStdExtZca() ? 2 : 4;
  case TargetOpcode::STACKMAP:
    // The upper bound for a stackmap intrinsic is the full length of its shadow
    return StackMapOpers(&MI).getNumPatchBytes();
  case TargetOpcode::PATCHPOINT:
    // The size of the patchpoint intrinsic is the number of bytes requested
    return PatchPointOpers(&MI).getNumPatchBytes();
  case TargetOpcode::STATEPOINT: {
    // The size of the statepoint intrinsic is the number of bytes requested
    unsigned NumBytes = StatepointOpers(&MI).getNumPatchBytes();
    // No patch bytes means at most a PseudoCall is emitted
    return std::max(NumBytes, 8U);
  }
  case TargetOpcode::PATCHABLE_FUNCTION_ENTER:
  case TargetOpcode::PATCHABLE_FUNCTION_EXIT:
  case TargetOpcode::PATCHABLE_TAIL_CALL: {
    const MachineFunction &MF = *MI.getParent()->getParent();
    const Function &F = MF.getFunction();
    if (Opcode == TargetOpcode::PATCHABLE_FUNCTION_ENTER &&
        F.hasFnAttribute("patchable-function-entry")) {
      unsigned Num;
      if (F.getFnAttribute("patchable-function-entry")
              .getValueAsString()
              .getAsInteger(10, Num))
        return get(Opcode).getSize();

      // Number of C.NOP or NOP
      return (STI.hasStdExtZca() ? 2 : 4) * Num;
    }
    // XRay uses C.JAL + 21 or 33 C.NOP for each sled in RV32 and RV64,
    // respectively.
    return STI.is64Bit() ? 68 : 44;
  }
  default:
    return get(Opcode).getSize();
  }
}

unsigned CapstoneInstrInfo::getInstBundleLength(const MachineInstr &MI) const {
  unsigned Size = 0;
  MachineBasicBlock::const_instr_iterator I = MI.getIterator();
  MachineBasicBlock::const_instr_iterator E = MI.getParent()->instr_end();
  while (++I != E && I->isInsideBundle()) {
    assert(!I->isBundle() && "No nested bundle!");
    Size += getInstSizeInBytes(*I);
  }
  return Size;
}

bool CapstoneInstrInfo::isAsCheapAsAMove(const MachineInstr &MI) const {
  const unsigned Opcode = MI.getOpcode();
  switch (Opcode) {
  default:
    break;
  case Capstone::FSGNJ_D:
  case Capstone::FSGNJ_S:
  case Capstone::FSGNJ_H:
  case Capstone::FSGNJ_D_INX:
  case Capstone::FSGNJ_D_IN32X:
  case Capstone::FSGNJ_S_INX:
  case Capstone::FSGNJ_H_INX:
    // The canonical floating-point move is fsgnj rd, rs, rs.
    return MI.getOperand(1).isReg() && MI.getOperand(2).isReg() &&
           MI.getOperand(1).getReg() == MI.getOperand(2).getReg();
  case Capstone::ADDI:
  case Capstone::ORI:
  case Capstone::XORI:
    return (MI.getOperand(1).isReg() &&
            MI.getOperand(1).getReg() == Capstone::X0) ||
           (MI.getOperand(2).isImm() && MI.getOperand(2).getImm() == 0);
  }
  return MI.isAsCheapAsAMove();
}

std::optional<DestSourcePair>
CapstoneInstrInfo::isCopyInstrImpl(const MachineInstr &MI) const {
  if (MI.isMoveReg())
    return DestSourcePair{MI.getOperand(0), MI.getOperand(1)};
  switch (MI.getOpcode()) {
  default:
    break;
  case Capstone::ADD:
  case Capstone::OR:
  case Capstone::XOR:
    if (MI.getOperand(1).isReg() && MI.getOperand(1).getReg() == Capstone::X0 &&
        MI.getOperand(2).isReg())
      return DestSourcePair{MI.getOperand(0), MI.getOperand(2)};
    if (MI.getOperand(2).isReg() && MI.getOperand(2).getReg() == Capstone::X0 &&
        MI.getOperand(1).isReg())
      return DestSourcePair{MI.getOperand(0), MI.getOperand(1)};
    break;
  case Capstone::ADDI:
    // Operand 1 can be a frameindex but callers expect registers
    if (MI.getOperand(1).isReg() && MI.getOperand(2).isImm() &&
        MI.getOperand(2).getImm() == 0)
      return DestSourcePair{MI.getOperand(0), MI.getOperand(1)};
    break;
  case Capstone::SUB:
    if (MI.getOperand(2).isReg() && MI.getOperand(2).getReg() == Capstone::X0 &&
        MI.getOperand(1).isReg())
      return DestSourcePair{MI.getOperand(0), MI.getOperand(1)};
    break;
  case Capstone::SH1ADD:
  case Capstone::SH1ADD_UW:
  case Capstone::SH2ADD:
  case Capstone::SH2ADD_UW:
  case Capstone::SH3ADD:
  case Capstone::SH3ADD_UW:
    if (MI.getOperand(1).isReg() && MI.getOperand(1).getReg() == Capstone::X0 &&
        MI.getOperand(2).isReg())
      return DestSourcePair{MI.getOperand(0), MI.getOperand(2)};
    break;
  case Capstone::FSGNJ_D:
  case Capstone::FSGNJ_S:
  case Capstone::FSGNJ_H:
  case Capstone::FSGNJ_D_INX:
  case Capstone::FSGNJ_D_IN32X:
  case Capstone::FSGNJ_S_INX:
  case Capstone::FSGNJ_H_INX:
    // The canonical floating-point move is fsgnj rd, rs, rs.
    if (MI.getOperand(1).isReg() && MI.getOperand(2).isReg() &&
        MI.getOperand(1).getReg() == MI.getOperand(2).getReg())
      return DestSourcePair{MI.getOperand(0), MI.getOperand(1)};
    break;
  }
  return std::nullopt;
}

MachineTraceStrategy CapstoneInstrInfo::getMachineCombinerTraceStrategy() const {
  if (ForceMachineCombinerStrategy.getNumOccurrences() == 0) {
    // The option is unused. Choose Local strategy only for in-order cores. When
    // scheduling model is unspecified, use MinInstrCount strategy as more
    // generic one.
    const auto &SchedModel = STI.getSchedModel();
    return (!SchedModel.hasInstrSchedModel() || SchedModel.isOutOfOrder())
               ? MachineTraceStrategy::TS_MinInstrCount
               : MachineTraceStrategy::TS_Local;
  }
  // The strategy was forced by the option.
  return ForceMachineCombinerStrategy;
}

void CapstoneInstrInfo::finalizeInsInstrs(
    MachineInstr &Root, unsigned &Pattern,
    SmallVectorImpl<MachineInstr *> &InsInstrs) const {
  int16_t FrmOpIdx =
      Capstone::getNamedOperandIdx(Root.getOpcode(), Capstone::OpName::frm);
  if (FrmOpIdx < 0) {
    assert(all_of(InsInstrs,
                  [](MachineInstr *MI) {
                    return Capstone::getNamedOperandIdx(MI->getOpcode(),
                                                     Capstone::OpName::frm) < 0;
                  }) &&
           "New instructions require FRM whereas the old one does not have it");
    return;
  }

  const MachineOperand &FRM = Root.getOperand(FrmOpIdx);
  MachineFunction &MF = *Root.getMF();

  for (auto *NewMI : InsInstrs) {
    // We'd already added the FRM operand.
    if (static_cast<unsigned>(Capstone::getNamedOperandIdx(
            NewMI->getOpcode(), Capstone::OpName::frm)) != NewMI->getNumOperands())
      continue;
    MachineInstrBuilder MIB(MF, NewMI);
    MIB.add(FRM);
    if (FRM.getImm() == CapstoneFPRndMode::DYN)
      MIB.addUse(Capstone::FRM, RegState::Implicit);
  }
}

static bool isFADD(unsigned Opc) {
  switch (Opc) {
  default:
    return false;
  case Capstone::FADD_H:
  case Capstone::FADD_S:
  case Capstone::FADD_D:
    return true;
  }
}

static bool isFSUB(unsigned Opc) {
  switch (Opc) {
  default:
    return false;
  case Capstone::FSUB_H:
  case Capstone::FSUB_S:
  case Capstone::FSUB_D:
    return true;
  }
}

static bool isFMUL(unsigned Opc) {
  switch (Opc) {
  default:
    return false;
  case Capstone::FMUL_H:
  case Capstone::FMUL_S:
  case Capstone::FMUL_D:
    return true;
  }
}

bool CapstoneInstrInfo::isVectorAssociativeAndCommutative(const MachineInstr &Inst,
                                                       bool Invert) const {
#define OPCODE_LMUL_CASE(OPC)                                                  \
  case Capstone::OPC##_M1:                                                        \
  case Capstone::OPC##_M2:                                                        \
  case Capstone::OPC##_M4:                                                        \
  case Capstone::OPC##_M8:                                                        \
  case Capstone::OPC##_MF2:                                                       \
  case Capstone::OPC##_MF4:                                                       \
  case Capstone::OPC##_MF8

#define OPCODE_LMUL_MASK_CASE(OPC)                                             \
  case Capstone::OPC##_M1_MASK:                                                   \
  case Capstone::OPC##_M2_MASK:                                                   \
  case Capstone::OPC##_M4_MASK:                                                   \
  case Capstone::OPC##_M8_MASK:                                                   \
  case Capstone::OPC##_MF2_MASK:                                                  \
  case Capstone::OPC##_MF4_MASK:                                                  \
  case Capstone::OPC##_MF8_MASK

  unsigned Opcode = Inst.getOpcode();
  if (Invert) {
    if (auto InvOpcode = getInverseOpcode(Opcode))
      Opcode = *InvOpcode;
    else
      return false;
  }

  // clang-format off
  switch (Opcode) {
  default:
    return false;
  OPCODE_LMUL_CASE(PseudoVADD_VV):
  OPCODE_LMUL_MASK_CASE(PseudoVADD_VV):
  OPCODE_LMUL_CASE(PseudoVMUL_VV):
  OPCODE_LMUL_MASK_CASE(PseudoVMUL_VV):
    return true;
  }
  // clang-format on

#undef OPCODE_LMUL_MASK_CASE
#undef OPCODE_LMUL_CASE
}

bool CapstoneInstrInfo::areRVVInstsReassociable(const MachineInstr &Root,
                                             const MachineInstr &Prev) const {
  if (!areOpcodesEqualOrInverse(Root.getOpcode(), Prev.getOpcode()))
    return false;

  assert(Root.getMF() == Prev.getMF());
  const MachineRegisterInfo *MRI = &Root.getMF()->getRegInfo();
  const TargetRegisterInfo *TRI = MRI->getTargetRegisterInfo();

  // Make sure vtype operands are also the same.
  const MCInstrDesc &Desc = get(Root.getOpcode());
  const uint64_t TSFlags = Desc.TSFlags;

  auto checkImmOperand = [&](unsigned OpIdx) {
    return Root.getOperand(OpIdx).getImm() == Prev.getOperand(OpIdx).getImm();
  };

  auto checkRegOperand = [&](unsigned OpIdx) {
    return Root.getOperand(OpIdx).getReg() == Prev.getOperand(OpIdx).getReg();
  };

  // PassThru
  // TODO: Potentially we can loosen the condition to consider Root to be
  // associable with Prev if Root has NoReg as passthru. In which case we
  // also need to loosen the condition on vector policies between these.
  if (!checkRegOperand(1))
    return false;

  // SEW
  if (CapstoneII::hasSEWOp(TSFlags) &&
      !checkImmOperand(CapstoneII::getSEWOpNum(Desc)))
    return false;

  // Mask
  if (CapstoneII::usesMaskPolicy(TSFlags)) {
    const MachineBasicBlock *MBB = Root.getParent();
    const MachineBasicBlock::const_reverse_iterator It1(&Root);
    const MachineBasicBlock::const_reverse_iterator It2(&Prev);
    Register MI1VReg;

    bool SeenMI2 = false;
    for (auto End = MBB->rend(), It = It1; It != End; ++It) {
      if (It == It2) {
        SeenMI2 = true;
        if (!MI1VReg.isValid())
          // There is no V0 def between Root and Prev; they're sharing the
          // same V0.
          break;
      }

      if (It->modifiesRegister(Capstone::V0, TRI)) {
        Register SrcReg = It->getOperand(1).getReg();
        // If it's not VReg it'll be more difficult to track its defs, so
        // bailing out here just to be safe.
        if (!SrcReg.isVirtual())
          return false;

        if (!MI1VReg.isValid()) {
          // This is the V0 def for Root.
          MI1VReg = SrcReg;
          continue;
        }

        // Some random mask updates.
        if (!SeenMI2)
          continue;

        // This is the V0 def for Prev; check if it's the same as that of
        // Root.
        if (MI1VReg != SrcReg)
          return false;
        else
          break;
      }
    }

    // If we haven't encountered Prev, it's likely that this function was
    // called in a wrong way (e.g. Root is before Prev).
    assert(SeenMI2 && "Prev is expected to appear before Root");
  }

  // Tail / Mask policies
  if (CapstoneII::hasVecPolicyOp(TSFlags) &&
      !checkImmOperand(CapstoneII::getVecPolicyOpNum(Desc)))
    return false;

  // VL
  if (CapstoneII::hasVLOp(TSFlags)) {
    unsigned OpIdx = CapstoneII::getVLOpNum(Desc);
    const MachineOperand &Op1 = Root.getOperand(OpIdx);
    const MachineOperand &Op2 = Prev.getOperand(OpIdx);
    if (Op1.getType() != Op2.getType())
      return false;
    switch (Op1.getType()) {
    case MachineOperand::MO_Register:
      if (Op1.getReg() != Op2.getReg())
        return false;
      break;
    case MachineOperand::MO_Immediate:
      if (Op1.getImm() != Op2.getImm())
        return false;
      break;
    default:
      llvm_unreachable("Unrecognized VL operand type");
    }
  }

  // Rounding modes
  if (CapstoneII::hasRoundModeOp(TSFlags) &&
      !checkImmOperand(CapstoneII::getVLOpNum(Desc) - 1))
    return false;

  return true;
}

// Most of our RVV pseudos have passthru operand, so the real operands
// start from index = 2.
bool CapstoneInstrInfo::hasReassociableVectorSibling(const MachineInstr &Inst,
                                                  bool &Commuted) const {
  const MachineBasicBlock *MBB = Inst.getParent();
  const MachineRegisterInfo &MRI = MBB->getParent()->getRegInfo();
  assert(CapstoneII::isFirstDefTiedToFirstUse(get(Inst.getOpcode())) &&
         "Expect the present of passthrough operand.");
  MachineInstr *MI1 = MRI.getUniqueVRegDef(Inst.getOperand(2).getReg());
  MachineInstr *MI2 = MRI.getUniqueVRegDef(Inst.getOperand(3).getReg());

  // If only one operand has the same or inverse opcode and it's the second
  // source operand, the operands must be commuted.
  Commuted = !areRVVInstsReassociable(Inst, *MI1) &&
             areRVVInstsReassociable(Inst, *MI2);
  if (Commuted)
    std::swap(MI1, MI2);

  return areRVVInstsReassociable(Inst, *MI1) &&
         (isVectorAssociativeAndCommutative(*MI1) ||
          isVectorAssociativeAndCommutative(*MI1, /* Invert */ true)) &&
         hasReassociableOperands(*MI1, MBB) &&
         MRI.hasOneNonDBGUse(MI1->getOperand(0).getReg());
}

bool CapstoneInstrInfo::hasReassociableOperands(
    const MachineInstr &Inst, const MachineBasicBlock *MBB) const {
  if (!isVectorAssociativeAndCommutative(Inst) &&
      !isVectorAssociativeAndCommutative(Inst, /*Invert=*/true))
    return TargetInstrInfo::hasReassociableOperands(Inst, MBB);

  const MachineOperand &Op1 = Inst.getOperand(2);
  const MachineOperand &Op2 = Inst.getOperand(3);
  const MachineRegisterInfo &MRI = MBB->getParent()->getRegInfo();

  // We need virtual register definitions for the operands that we will
  // reassociate.
  MachineInstr *MI1 = nullptr;
  MachineInstr *MI2 = nullptr;
  if (Op1.isReg() && Op1.getReg().isVirtual())
    MI1 = MRI.getUniqueVRegDef(Op1.getReg());
  if (Op2.isReg() && Op2.getReg().isVirtual())
    MI2 = MRI.getUniqueVRegDef(Op2.getReg());

  // And at least one operand must be defined in MBB.
  return MI1 && MI2 && (MI1->getParent() == MBB || MI2->getParent() == MBB);
}

void CapstoneInstrInfo::getReassociateOperandIndices(
    const MachineInstr &Root, unsigned Pattern,
    std::array<unsigned, 5> &OperandIndices) const {
  TargetInstrInfo::getReassociateOperandIndices(Root, Pattern, OperandIndices);
  if (Capstone::getRVVMCOpcode(Root.getOpcode())) {
    // Skip the passthrough operand, so increment all indices by one.
    for (unsigned I = 0; I < 5; ++I)
      ++OperandIndices[I];
  }
}

bool CapstoneInstrInfo::hasReassociableSibling(const MachineInstr &Inst,
                                            bool &Commuted) const {
  if (isVectorAssociativeAndCommutative(Inst) ||
      isVectorAssociativeAndCommutative(Inst, /*Invert=*/true))
    return hasReassociableVectorSibling(Inst, Commuted);

  if (!TargetInstrInfo::hasReassociableSibling(Inst, Commuted))
    return false;

  const MachineRegisterInfo &MRI = Inst.getMF()->getRegInfo();
  unsigned OperandIdx = Commuted ? 2 : 1;
  const MachineInstr &Sibling =
      *MRI.getVRegDef(Inst.getOperand(OperandIdx).getReg());

  int16_t InstFrmOpIdx =
      Capstone::getNamedOperandIdx(Inst.getOpcode(), Capstone::OpName::frm);
  int16_t SiblingFrmOpIdx =
      Capstone::getNamedOperandIdx(Sibling.getOpcode(), Capstone::OpName::frm);

  return (InstFrmOpIdx < 0 && SiblingFrmOpIdx < 0) ||
         Capstone::hasEqualFRM(Inst, Sibling);
}

bool CapstoneInstrInfo::isAssociativeAndCommutative(const MachineInstr &Inst,
                                                 bool Invert) const {
  if (isVectorAssociativeAndCommutative(Inst, Invert))
    return true;

  unsigned Opc = Inst.getOpcode();
  if (Invert) {
    auto InverseOpcode = getInverseOpcode(Opc);
    if (!InverseOpcode)
      return false;
    Opc = *InverseOpcode;
  }

  if (isFADD(Opc) || isFMUL(Opc))
    return Inst.getFlag(MachineInstr::MIFlag::FmReassoc) &&
           Inst.getFlag(MachineInstr::MIFlag::FmNsz);

  switch (Opc) {
  default:
    return false;
  case Capstone::ADD:
  case Capstone::ADDW:
  case Capstone::AND:
  case Capstone::OR:
  case Capstone::XOR:
  // From Capstone ISA spec, if both the high and low bits of the same product
  // are required, then the recommended code sequence is:
  //
  // MULH[[S]U] rdh, rs1, rs2
  // MUL        rdl, rs1, rs2
  // (source register specifiers must be in same order and rdh cannot be the
  //  same as rs1 or rs2)
  //
  // Microarchitectures can then fuse these into a single multiply operation
  // instead of performing two separate multiplies.
  // MachineCombiner may reassociate MUL operands and lose the fusion
  // opportunity.
  case Capstone::MUL:
  case Capstone::MULW:
  case Capstone::MIN:
  case Capstone::MINU:
  case Capstone::MAX:
  case Capstone::MAXU:
  case Capstone::FMIN_H:
  case Capstone::FMIN_S:
  case Capstone::FMIN_D:
  case Capstone::FMAX_H:
  case Capstone::FMAX_S:
  case Capstone::FMAX_D:
    return true;
  }

  return false;
}

std::optional<unsigned>
CapstoneInstrInfo::getInverseOpcode(unsigned Opcode) const {
#define RVV_OPC_LMUL_CASE(OPC, INV)                                            \
  case Capstone::OPC##_M1:                                                        \
    return Capstone::INV##_M1;                                                    \
  case Capstone::OPC##_M2:                                                        \
    return Capstone::INV##_M2;                                                    \
  case Capstone::OPC##_M4:                                                        \
    return Capstone::INV##_M4;                                                    \
  case Capstone::OPC##_M8:                                                        \
    return Capstone::INV##_M8;                                                    \
  case Capstone::OPC##_MF2:                                                       \
    return Capstone::INV##_MF2;                                                   \
  case Capstone::OPC##_MF4:                                                       \
    return Capstone::INV##_MF4;                                                   \
  case Capstone::OPC##_MF8:                                                       \
    return Capstone::INV##_MF8

#define RVV_OPC_LMUL_MASK_CASE(OPC, INV)                                       \
  case Capstone::OPC##_M1_MASK:                                                   \
    return Capstone::INV##_M1_MASK;                                               \
  case Capstone::OPC##_M2_MASK:                                                   \
    return Capstone::INV##_M2_MASK;                                               \
  case Capstone::OPC##_M4_MASK:                                                   \
    return Capstone::INV##_M4_MASK;                                               \
  case Capstone::OPC##_M8_MASK:                                                   \
    return Capstone::INV##_M8_MASK;                                               \
  case Capstone::OPC##_MF2_MASK:                                                  \
    return Capstone::INV##_MF2_MASK;                                              \
  case Capstone::OPC##_MF4_MASK:                                                  \
    return Capstone::INV##_MF4_MASK;                                              \
  case Capstone::OPC##_MF8_MASK:                                                  \
    return Capstone::INV##_MF8_MASK

  switch (Opcode) {
  default:
    return std::nullopt;
  case Capstone::FADD_H:
    return Capstone::FSUB_H;
  case Capstone::FADD_S:
    return Capstone::FSUB_S;
  case Capstone::FADD_D:
    return Capstone::FSUB_D;
  case Capstone::FSUB_H:
    return Capstone::FADD_H;
  case Capstone::FSUB_S:
    return Capstone::FADD_S;
  case Capstone::FSUB_D:
    return Capstone::FADD_D;
  case Capstone::ADD:
    return Capstone::SUB;
  case Capstone::SUB:
    return Capstone::ADD;
  case Capstone::ADDW:
    return Capstone::SUBW;
  case Capstone::SUBW:
    return Capstone::ADDW;
    // clang-format off
  RVV_OPC_LMUL_CASE(PseudoVADD_VV, PseudoVSUB_VV);
  RVV_OPC_LMUL_MASK_CASE(PseudoVADD_VV, PseudoVSUB_VV);
  RVV_OPC_LMUL_CASE(PseudoVSUB_VV, PseudoVADD_VV);
  RVV_OPC_LMUL_MASK_CASE(PseudoVSUB_VV, PseudoVADD_VV);
    // clang-format on
  }

#undef RVV_OPC_LMUL_MASK_CASE
#undef RVV_OPC_LMUL_CASE
}

static bool canCombineFPFusedMultiply(const MachineInstr &Root,
                                      const MachineOperand &MO,
                                      bool DoRegPressureReduce) {
  if (!MO.isReg() || !MO.getReg().isVirtual())
    return false;
  const MachineRegisterInfo &MRI = Root.getMF()->getRegInfo();
  MachineInstr *MI = MRI.getVRegDef(MO.getReg());
  if (!MI || !isFMUL(MI->getOpcode()))
    return false;

  if (!Root.getFlag(MachineInstr::MIFlag::FmContract) ||
      !MI->getFlag(MachineInstr::MIFlag::FmContract))
    return false;

  // Try combining even if fmul has more than one use as it eliminates
  // dependency between fadd(fsub) and fmul. However, it can extend liveranges
  // for fmul operands, so reject the transformation in register pressure
  // reduction mode.
  if (DoRegPressureReduce && !MRI.hasOneNonDBGUse(MI->getOperand(0).getReg()))
    return false;

  // Do not combine instructions from different basic blocks.
  if (Root.getParent() != MI->getParent())
    return false;
  return Capstone::hasEqualFRM(Root, *MI);
}

static bool getFPFusedMultiplyPatterns(MachineInstr &Root,
                                       SmallVectorImpl<unsigned> &Patterns,
                                       bool DoRegPressureReduce) {
  unsigned Opc = Root.getOpcode();
  bool IsFAdd = isFADD(Opc);
  if (!IsFAdd && !isFSUB(Opc))
    return false;
  bool Added = false;
  if (canCombineFPFusedMultiply(Root, Root.getOperand(1),
                                DoRegPressureReduce)) {
    Patterns.push_back(IsFAdd ? CapstoneMachineCombinerPattern::FMADD_AX
                              : CapstoneMachineCombinerPattern::FMSUB);
    Added = true;
  }
  if (canCombineFPFusedMultiply(Root, Root.getOperand(2),
                                DoRegPressureReduce)) {
    Patterns.push_back(IsFAdd ? CapstoneMachineCombinerPattern::FMADD_XA
                              : CapstoneMachineCombinerPattern::FNMSUB);
    Added = true;
  }
  return Added;
}

static bool getFPPatterns(MachineInstr &Root,
                          SmallVectorImpl<unsigned> &Patterns,
                          bool DoRegPressureReduce) {
  return getFPFusedMultiplyPatterns(Root, Patterns, DoRegPressureReduce);
}

/// Utility routine that checks if \param MO is defined by an
/// \param CombineOpc instruction in the basic block \param MBB
static const MachineInstr *canCombine(const MachineBasicBlock &MBB,
                                      const MachineOperand &MO,
                                      unsigned CombineOpc) {
  const MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();
  const MachineInstr *MI = nullptr;

  if (MO.isReg() && MO.getReg().isVirtual())
    MI = MRI.getUniqueVRegDef(MO.getReg());
  // And it needs to be in the trace (otherwise, it won't have a depth).
  if (!MI || MI->getParent() != &MBB || MI->getOpcode() != CombineOpc)
    return nullptr;
  // Must only used by the user we combine with.
  if (!MRI.hasOneNonDBGUse(MI->getOperand(0).getReg()))
    return nullptr;

  return MI;
}

/// Utility routine that checks if \param MO is defined by a SLLI in \param
/// MBB that can be combined by splitting across 2 SHXADD instructions. The
/// first SHXADD shift amount is given by \param OuterShiftAmt.
static bool canCombineShiftIntoShXAdd(const MachineBasicBlock &MBB,
                                      const MachineOperand &MO,
                                      unsigned OuterShiftAmt) {
  const MachineInstr *ShiftMI = canCombine(MBB, MO, Capstone::SLLI);
  if (!ShiftMI)
    return false;

  unsigned InnerShiftAmt = ShiftMI->getOperand(2).getImm();
  if (InnerShiftAmt < OuterShiftAmt || (InnerShiftAmt - OuterShiftAmt) > 3)
    return false;

  return true;
}

// Returns the shift amount from a SHXADD instruction. Returns 0 if the
// instruction is not a SHXADD.
static unsigned getSHXADDShiftAmount(unsigned Opc) {
  switch (Opc) {
  default:
    return 0;
  case Capstone::SH1ADD:
    return 1;
  case Capstone::SH2ADD:
    return 2;
  case Capstone::SH3ADD:
    return 3;
  }
}

// Returns the shift amount from a SHXADD.UW instruction. Returns 0 if the
// instruction is not a SHXADD.UW.
static unsigned getSHXADDUWShiftAmount(unsigned Opc) {
  switch (Opc) {
  default:
    return 0;
  case Capstone::SH1ADD_UW:
    return 1;
  case Capstone::SH2ADD_UW:
    return 2;
  case Capstone::SH3ADD_UW:
    return 3;
  }
}

// Look for opportunities to combine (sh3add Z, (add X, (slli Y, 5))) into
// (sh3add (sh2add Y, Z), X).
static bool getSHXADDPatterns(const MachineInstr &Root,
                              SmallVectorImpl<unsigned> &Patterns) {
  unsigned ShiftAmt = getSHXADDShiftAmount(Root.getOpcode());
  if (!ShiftAmt)
    return false;

  const MachineBasicBlock &MBB = *Root.getParent();

  const MachineInstr *AddMI = canCombine(MBB, Root.getOperand(2), Capstone::ADD);
  if (!AddMI)
    return false;

  bool Found = false;
  if (canCombineShiftIntoShXAdd(MBB, AddMI->getOperand(1), ShiftAmt)) {
    Patterns.push_back(CapstoneMachineCombinerPattern::SHXADD_ADD_SLLI_OP1);
    Found = true;
  }
  if (canCombineShiftIntoShXAdd(MBB, AddMI->getOperand(2), ShiftAmt)) {
    Patterns.push_back(CapstoneMachineCombinerPattern::SHXADD_ADD_SLLI_OP2);
    Found = true;
  }

  return Found;
}

CombinerObjective CapstoneInstrInfo::getCombinerObjective(unsigned Pattern) const {
  switch (Pattern) {
  case CapstoneMachineCombinerPattern::FMADD_AX:
  case CapstoneMachineCombinerPattern::FMADD_XA:
  case CapstoneMachineCombinerPattern::FMSUB:
  case CapstoneMachineCombinerPattern::FNMSUB:
    return CombinerObjective::MustReduceDepth;
  default:
    return TargetInstrInfo::getCombinerObjective(Pattern);
  }
}

bool CapstoneInstrInfo::getMachineCombinerPatterns(
    MachineInstr &Root, SmallVectorImpl<unsigned> &Patterns,
    bool DoRegPressureReduce) const {

  if (getFPPatterns(Root, Patterns, DoRegPressureReduce))
    return true;

  if (getSHXADDPatterns(Root, Patterns))
    return true;

  return TargetInstrInfo::getMachineCombinerPatterns(Root, Patterns,
                                                     DoRegPressureReduce);
}

static unsigned getFPFusedMultiplyOpcode(unsigned RootOpc, unsigned Pattern) {
  switch (RootOpc) {
  default:
    llvm_unreachable("Unexpected opcode");
  case Capstone::FADD_H:
    return Capstone::FMADD_H;
  case Capstone::FADD_S:
    return Capstone::FMADD_S;
  case Capstone::FADD_D:
    return Capstone::FMADD_D;
  case Capstone::FSUB_H:
    return Pattern == CapstoneMachineCombinerPattern::FMSUB ? Capstone::FMSUB_H
                                                         : Capstone::FNMSUB_H;
  case Capstone::FSUB_S:
    return Pattern == CapstoneMachineCombinerPattern::FMSUB ? Capstone::FMSUB_S
                                                         : Capstone::FNMSUB_S;
  case Capstone::FSUB_D:
    return Pattern == CapstoneMachineCombinerPattern::FMSUB ? Capstone::FMSUB_D
                                                         : Capstone::FNMSUB_D;
  }
}

static unsigned getAddendOperandIdx(unsigned Pattern) {
  switch (Pattern) {
  default:
    llvm_unreachable("Unexpected pattern");
  case CapstoneMachineCombinerPattern::FMADD_AX:
  case CapstoneMachineCombinerPattern::FMSUB:
    return 2;
  case CapstoneMachineCombinerPattern::FMADD_XA:
  case CapstoneMachineCombinerPattern::FNMSUB:
    return 1;
  }
}

static void combineFPFusedMultiply(MachineInstr &Root, MachineInstr &Prev,
                                   unsigned Pattern,
                                   SmallVectorImpl<MachineInstr *> &InsInstrs,
                                   SmallVectorImpl<MachineInstr *> &DelInstrs) {
  MachineFunction *MF = Root.getMF();
  MachineRegisterInfo &MRI = MF->getRegInfo();
  const TargetInstrInfo *TII = MF->getSubtarget().getInstrInfo();

  MachineOperand &Mul1 = Prev.getOperand(1);
  MachineOperand &Mul2 = Prev.getOperand(2);
  MachineOperand &Dst = Root.getOperand(0);
  MachineOperand &Addend = Root.getOperand(getAddendOperandIdx(Pattern));

  Register DstReg = Dst.getReg();
  unsigned FusedOpc = getFPFusedMultiplyOpcode(Root.getOpcode(), Pattern);
  uint32_t IntersectedFlags = Root.getFlags() & Prev.getFlags();
  DebugLoc MergedLoc =
      DILocation::getMergedLocation(Root.getDebugLoc(), Prev.getDebugLoc());

  bool Mul1IsKill = Mul1.isKill();
  bool Mul2IsKill = Mul2.isKill();
  bool AddendIsKill = Addend.isKill();

  // We need to clear kill flags since we may be extending the live range past
  // a kill. If the mul had kill flags, we can preserve those since we know
  // where the previous range stopped.
  MRI.clearKillFlags(Mul1.getReg());
  MRI.clearKillFlags(Mul2.getReg());

  MachineInstrBuilder MIB =
      BuildMI(*MF, MergedLoc, TII->get(FusedOpc), DstReg)
          .addReg(Mul1.getReg(), getKillRegState(Mul1IsKill))
          .addReg(Mul2.getReg(), getKillRegState(Mul2IsKill))
          .addReg(Addend.getReg(), getKillRegState(AddendIsKill))
          .setMIFlags(IntersectedFlags);

  InsInstrs.push_back(MIB);
  if (MRI.hasOneNonDBGUse(Prev.getOperand(0).getReg()))
    DelInstrs.push_back(&Prev);
  DelInstrs.push_back(&Root);
}

// Combine patterns like (sh3add Z, (add X, (slli Y, 5))) to
// (sh3add (sh2add Y, Z), X) if the shift amount can be split across two
// shXadd instructions. The outer shXadd keeps its original opcode.
static void
genShXAddAddShift(MachineInstr &Root, unsigned AddOpIdx,
                  SmallVectorImpl<MachineInstr *> &InsInstrs,
                  SmallVectorImpl<MachineInstr *> &DelInstrs,
                  DenseMap<Register, unsigned> &InstrIdxForVirtReg) {
  MachineFunction *MF = Root.getMF();
  MachineRegisterInfo &MRI = MF->getRegInfo();
  const TargetInstrInfo *TII = MF->getSubtarget().getInstrInfo();

  unsigned OuterShiftAmt = getSHXADDShiftAmount(Root.getOpcode());
  assert(OuterShiftAmt != 0 && "Unexpected opcode");

  MachineInstr *AddMI = MRI.getUniqueVRegDef(Root.getOperand(2).getReg());
  MachineInstr *ShiftMI =
      MRI.getUniqueVRegDef(AddMI->getOperand(AddOpIdx).getReg());

  unsigned InnerShiftAmt = ShiftMI->getOperand(2).getImm();
  assert(InnerShiftAmt >= OuterShiftAmt && "Unexpected shift amount");

  unsigned InnerOpc;
  switch (InnerShiftAmt - OuterShiftAmt) {
  default:
    llvm_unreachable("Unexpected shift amount");
  case 0:
    InnerOpc = Capstone::ADD;
    break;
  case 1:
    InnerOpc = Capstone::SH1ADD;
    break;
  case 2:
    InnerOpc = Capstone::SH2ADD;
    break;
  case 3:
    InnerOpc = Capstone::SH3ADD;
    break;
  }

  const MachineOperand &X = AddMI->getOperand(3 - AddOpIdx);
  const MachineOperand &Y = ShiftMI->getOperand(1);
  const MachineOperand &Z = Root.getOperand(1);

  Register NewVR = MRI.createVirtualRegister(&Capstone::GPRRegClass);

  auto MIB1 = BuildMI(*MF, MIMetadata(Root), TII->get(InnerOpc), NewVR)
                  .addReg(Y.getReg(), getKillRegState(Y.isKill()))
                  .addReg(Z.getReg(), getKillRegState(Z.isKill()));
  auto MIB2 = BuildMI(*MF, MIMetadata(Root), TII->get(Root.getOpcode()),
                      Root.getOperand(0).getReg())
                  .addReg(NewVR, RegState::Kill)
                  .addReg(X.getReg(), getKillRegState(X.isKill()));

  InstrIdxForVirtReg.insert(std::make_pair(NewVR, 0));
  InsInstrs.push_back(MIB1);
  InsInstrs.push_back(MIB2);
  DelInstrs.push_back(ShiftMI);
  DelInstrs.push_back(AddMI);
  DelInstrs.push_back(&Root);
}

void CapstoneInstrInfo::genAlternativeCodeSequence(
    MachineInstr &Root, unsigned Pattern,
    SmallVectorImpl<MachineInstr *> &InsInstrs,
    SmallVectorImpl<MachineInstr *> &DelInstrs,
    DenseMap<Register, unsigned> &InstrIdxForVirtReg) const {
  MachineRegisterInfo &MRI = Root.getMF()->getRegInfo();
  switch (Pattern) {
  default:
    TargetInstrInfo::genAlternativeCodeSequence(Root, Pattern, InsInstrs,
                                                DelInstrs, InstrIdxForVirtReg);
    return;
  case CapstoneMachineCombinerPattern::FMADD_AX:
  case CapstoneMachineCombinerPattern::FMSUB: {
    MachineInstr &Prev = *MRI.getVRegDef(Root.getOperand(1).getReg());
    combineFPFusedMultiply(Root, Prev, Pattern, InsInstrs, DelInstrs);
    return;
  }
  case CapstoneMachineCombinerPattern::FMADD_XA:
  case CapstoneMachineCombinerPattern::FNMSUB: {
    MachineInstr &Prev = *MRI.getVRegDef(Root.getOperand(2).getReg());
    combineFPFusedMultiply(Root, Prev, Pattern, InsInstrs, DelInstrs);
    return;
  }
  case CapstoneMachineCombinerPattern::SHXADD_ADD_SLLI_OP1:
    genShXAddAddShift(Root, 1, InsInstrs, DelInstrs, InstrIdxForVirtReg);
    return;
  case CapstoneMachineCombinerPattern::SHXADD_ADD_SLLI_OP2:
    genShXAddAddShift(Root, 2, InsInstrs, DelInstrs, InstrIdxForVirtReg);
    return;
  }
}

bool CapstoneInstrInfo::verifyInstruction(const MachineInstr &MI,
                                       StringRef &ErrInfo) const {
  MCInstrDesc const &Desc = MI.getDesc();

  for (const auto &[Index, Operand] : enumerate(Desc.operands())) {
    unsigned OpType = Operand.OperandType;
    if (OpType >= CapstoneOp::OPERAND_FIRST_Capstone_IMM &&
        OpType <= CapstoneOp::OPERAND_LAST_Capstone_IMM) {
      const MachineOperand &MO = MI.getOperand(Index);
      if (MO.isReg()) {
        ErrInfo = "Expected a non-register operand.";
        return false;
      }
      if (MO.isImm()) {
        int64_t Imm = MO.getImm();
        bool Ok;
        switch (OpType) {
        default:
          llvm_unreachable("Unexpected operand type");

          // clang-format off
#define CASE_OPERAND_UIMM(NUM)                                                 \
  case CapstoneOp::OPERAND_UIMM##NUM:                                             \
    Ok = isUInt<NUM>(Imm);                                                     \
    break;
#define CASE_OPERAND_SIMM(NUM)                                                 \
  case CapstoneOp::OPERAND_SIMM##NUM:                                             \
    Ok = isInt<NUM>(Imm);                                                      \
    break;
        CASE_OPERAND_UIMM(1)
        CASE_OPERAND_UIMM(2)
        CASE_OPERAND_UIMM(3)
        CASE_OPERAND_UIMM(4)
        CASE_OPERAND_UIMM(5)
        CASE_OPERAND_UIMM(6)
        CASE_OPERAND_UIMM(7)
        CASE_OPERAND_UIMM(8)
        CASE_OPERAND_UIMM(9)
        CASE_OPERAND_UIMM(10)
        CASE_OPERAND_UIMM(12)
        CASE_OPERAND_UIMM(16)
        CASE_OPERAND_UIMM(20)
        CASE_OPERAND_UIMM(32)
        CASE_OPERAND_UIMM(48)
        CASE_OPERAND_UIMM(64)
          // clang-format on
        case CapstoneOp::OPERAND_UIMM2_LSB0:
          Ok = isShiftedUInt<1, 1>(Imm);
          break;
        case CapstoneOp::OPERAND_UIMM5_LSB0:
          Ok = isShiftedUInt<4, 1>(Imm);
          break;
        case CapstoneOp::OPERAND_UIMM5_NONZERO:
          Ok = isUInt<5>(Imm) && (Imm != 0);
          break;
        case CapstoneOp::OPERAND_UIMM5_GT3:
          Ok = isUInt<5>(Imm) && (Imm > 3);
          break;
        case CapstoneOp::OPERAND_UIMM5_PLUS1:
          Ok = (isUInt<5>(Imm) && (Imm != 0)) || (Imm == 32);
          break;
        case CapstoneOp::OPERAND_UIMM6_LSB0:
          Ok = isShiftedUInt<5, 1>(Imm);
          break;
        case CapstoneOp::OPERAND_UIMM7_LSB00:
          Ok = isShiftedUInt<5, 2>(Imm);
          break;
        case CapstoneOp::OPERAND_UIMM7_LSB000:
          Ok = isShiftedUInt<4, 3>(Imm);
          break;
        case CapstoneOp::OPERAND_UIMM8_LSB00:
          Ok = isShiftedUInt<6, 2>(Imm);
          break;
        case CapstoneOp::OPERAND_UIMM8_LSB000:
          Ok = isShiftedUInt<5, 3>(Imm);
          break;
        case CapstoneOp::OPERAND_UIMM8_GE32:
          Ok = isUInt<8>(Imm) && Imm >= 32;
          break;
        case CapstoneOp::OPERAND_UIMM9_LSB000:
          Ok = isShiftedUInt<6, 3>(Imm);
          break;
        case CapstoneOp::OPERAND_SIMM10_LSB0000_NONZERO:
          Ok = isShiftedInt<6, 4>(Imm) && (Imm != 0);
          break;
        case CapstoneOp::OPERAND_UIMM10_LSB00_NONZERO:
          Ok = isShiftedUInt<8, 2>(Imm) && (Imm != 0);
          break;
        case CapstoneOp::OPERAND_UIMM16_NONZERO:
          Ok = isUInt<16>(Imm) && (Imm != 0);
          break;
        case CapstoneOp::OPERAND_THREE:
          Ok = Imm == 3;
          break;
        case CapstoneOp::OPERAND_FOUR:
          Ok = Imm == 4;
          break;
          // clang-format off
        CASE_OPERAND_SIMM(5)
        CASE_OPERAND_SIMM(6)
        CASE_OPERAND_SIMM(11)
        CASE_OPERAND_SIMM(12)
        CASE_OPERAND_SIMM(26)
        // clang-format on
        case CapstoneOp::OPERAND_SIMM5_PLUS1:
          Ok = (isInt<5>(Imm) && Imm != -16) || Imm == 16;
          break;
        case CapstoneOp::OPERAND_SIMM5_NONZERO:
          Ok = isInt<5>(Imm) && (Imm != 0);
          break;
        case CapstoneOp::OPERAND_SIMM6_NONZERO:
          Ok = Imm != 0 && isInt<6>(Imm);
          break;
        case CapstoneOp::OPERAND_VTYPEI10:
          Ok = isUInt<10>(Imm);
          break;
        case CapstoneOp::OPERAND_VTYPEI11:
          Ok = isUInt<11>(Imm);
          break;
        case CapstoneOp::OPERAND_SIMM12_LSB00000:
          Ok = isShiftedInt<7, 5>(Imm);
          break;
        case CapstoneOp::OPERAND_SIMM16_NONZERO:
          Ok = isInt<16>(Imm) && (Imm != 0);
          break;
        case CapstoneOp::OPERAND_SIMM20_LI:
          Ok = isInt<20>(Imm);
          break;
        case CapstoneOp::OPERAND_BARE_SIMM32:
          Ok = isInt<32>(Imm);
          break;
        case CapstoneOp::OPERAND_UIMMLOG2XLEN:
          Ok = STI.is64Bit() ? isUInt<6>(Imm) : isUInt<5>(Imm);
          break;
        case CapstoneOp::OPERAND_UIMMLOG2XLEN_NONZERO:
          Ok = STI.is64Bit() ? isUInt<6>(Imm) : isUInt<5>(Imm);
          Ok = Ok && Imm != 0;
          break;
        case CapstoneOp::OPERAND_CLUI_IMM:
          Ok = (isUInt<5>(Imm) && Imm != 0) ||
               (Imm >= 0xfffe0 && Imm <= 0xfffff);
          break;
        case CapstoneOp::OPERAND_RVKRNUM:
          Ok = Imm >= 0 && Imm <= 10;
          break;
        case CapstoneOp::OPERAND_RVKRNUM_0_7:
          Ok = Imm >= 0 && Imm <= 7;
          break;
        case CapstoneOp::OPERAND_RVKRNUM_1_10:
          Ok = Imm >= 1 && Imm <= 10;
          break;
        case CapstoneOp::OPERAND_RVKRNUM_2_14:
          Ok = Imm >= 2 && Imm <= 14;
          break;
        case CapstoneOp::OPERAND_RLIST:
          Ok = Imm >= CapstoneZC::RA && Imm <= CapstoneZC::RA_S0_S11;
          break;
        case CapstoneOp::OPERAND_RLIST_S0:
          Ok = Imm >= CapstoneZC::RA_S0 && Imm <= CapstoneZC::RA_S0_S11;
          break;
        case CapstoneOp::OPERAND_STACKADJ:
          Ok = Imm >= 0 && Imm <= 48 && Imm % 16 == 0;
          break;
        case CapstoneOp::OPERAND_FRMARG:
          Ok = CapstoneFPRndMode::isValidRoundingMode(Imm);
          break;
        case CapstoneOp::OPERAND_RTZARG:
          Ok = Imm == CapstoneFPRndMode::RTZ;
          break;
        case CapstoneOp::OPERAND_COND_CODE:
          Ok = Imm >= 0 && Imm < CapstoneCC::COND_INVALID;
          break;
        case CapstoneOp::OPERAND_VEC_POLICY:
          Ok = (Imm &
                (CapstoneVType::TAIL_AGNOSTIC | CapstoneVType::MASK_AGNOSTIC)) == Imm;
          break;
        case CapstoneOp::OPERAND_SEW:
          Ok = (isUInt<5>(Imm) && CapstoneVType::isValidSEW(1 << Imm));
          break;
        case CapstoneOp::OPERAND_SEW_MASK:
          Ok = Imm == 0;
          break;
        case CapstoneOp::OPERAND_VEC_RM:
          assert(CapstoneII::hasRoundModeOp(Desc.TSFlags));
          if (CapstoneII::usesVXRM(Desc.TSFlags))
            Ok = isUInt<2>(Imm);
          else
            Ok = CapstoneFPRndMode::isValidRoundingMode(Imm);
          break;
        }
        if (!Ok) {
          ErrInfo = "Invalid immediate";
          return false;
        }
      }
    }
  }

  const uint64_t TSFlags = Desc.TSFlags;
  if (CapstoneII::hasVLOp(TSFlags)) {
    const MachineOperand &Op = MI.getOperand(CapstoneII::getVLOpNum(Desc));
    if (!Op.isImm() && !Op.isReg())  {
      ErrInfo = "Invalid operand type for VL operand";
      return false;
    }
    if (Op.isReg() && Op.getReg() != Capstone::NoRegister) {
      const MachineRegisterInfo &MRI = MI.getParent()->getParent()->getRegInfo();
      auto *RC = MRI.getRegClass(Op.getReg());
      if (!Capstone::GPRRegClass.hasSubClassEq(RC)) {
        ErrInfo = "Invalid register class for VL operand";
        return false;
      }
    }
    if (!CapstoneII::hasSEWOp(TSFlags)) {
      ErrInfo = "VL operand w/o SEW operand?";
      return false;
    }
  }
  if (CapstoneII::hasSEWOp(TSFlags)) {
    unsigned OpIdx = CapstoneII::getSEWOpNum(Desc);
    if (!MI.getOperand(OpIdx).isImm()) {
      ErrInfo = "SEW value expected to be an immediate";
      return false;
    }
    uint64_t Log2SEW = MI.getOperand(OpIdx).getImm();
    if (Log2SEW > 31) {
      ErrInfo = "Unexpected SEW value";
      return false;
    }
    unsigned SEW = Log2SEW ? 1 << Log2SEW : 8;
    if (!CapstoneVType::isValidSEW(SEW)) {
      ErrInfo = "Unexpected SEW value";
      return false;
    }
  }
  if (CapstoneII::hasVecPolicyOp(TSFlags)) {
    unsigned OpIdx = CapstoneII::getVecPolicyOpNum(Desc);
    if (!MI.getOperand(OpIdx).isImm()) {
      ErrInfo = "Policy operand expected to be an immediate";
      return false;
    }
    uint64_t Policy = MI.getOperand(OpIdx).getImm();
    if (Policy > (CapstoneVType::TAIL_AGNOSTIC | CapstoneVType::MASK_AGNOSTIC)) {
      ErrInfo = "Invalid Policy Value";
      return false;
    }
    if (!CapstoneII::hasVLOp(TSFlags)) {
      ErrInfo = "policy operand w/o VL operand?";
      return false;
    }

    // VecPolicy operands can only exist on instructions with passthru/merge
    // arguments. Note that not all arguments with passthru have vec policy
    // operands- some instructions have implicit policies.
    unsigned UseOpIdx;
    if (!MI.isRegTiedToUseOperand(0, &UseOpIdx)) {
      ErrInfo = "policy operand w/o tied operand?";
      return false;
    }
  }

  if (int Idx = CapstoneII::getFRMOpNum(Desc);
      Idx >= 0 && MI.getOperand(Idx).getImm() == CapstoneFPRndMode::DYN &&
      !MI.readsRegister(Capstone::FRM, /*TRI=*/nullptr)) {
    ErrInfo = "dynamic rounding mode should read FRM";
    return false;
  }

  return true;
}

bool CapstoneInstrInfo::canFoldIntoAddrMode(const MachineInstr &MemI, Register Reg,
                                         const MachineInstr &AddrI,
                                         ExtAddrMode &AM) const {
  switch (MemI.getOpcode()) {
  default:
    return false;
  case Capstone::LB:
  case Capstone::LBU:
  case Capstone::LH:
  case Capstone::LH_INX:
  case Capstone::LHU:
  case Capstone::LW:
  case Capstone::LW_INX:
  case Capstone::LWU:
  case Capstone::LD:
  case Capstone::LD_RV32:
  case Capstone::FLH:
  case Capstone::FLW:
  case Capstone::FLD:
  case Capstone::SB:
  case Capstone::SH:
  case Capstone::SH_INX:
  case Capstone::SW:
  case Capstone::SW_INX:
  case Capstone::SD:
  case Capstone::SD_RV32:
  case Capstone::FSH:
  case Capstone::FSW:
  case Capstone::FSD:
    break;
  }

  if (MemI.getOperand(0).getReg() == Reg)
    return false;

  if (AddrI.getOpcode() != Capstone::ADDI || !AddrI.getOperand(1).isReg() ||
      !AddrI.getOperand(2).isImm())
    return false;

  int64_t OldOffset = MemI.getOperand(2).getImm();
  int64_t Disp = AddrI.getOperand(2).getImm();
  int64_t NewOffset = OldOffset + Disp;
  if (!STI.is64Bit())
    NewOffset = SignExtend64<32>(NewOffset);

  if (!isInt<12>(NewOffset))
    return false;

  AM.BaseReg = AddrI.getOperand(1).getReg();
  AM.ScaledReg = 0;
  AM.Scale = 0;
  AM.Displacement = NewOffset;
  AM.Form = ExtAddrMode::Formula::Basic;
  return true;
}

MachineInstr *CapstoneInstrInfo::emitLdStWithAddr(MachineInstr &MemI,
                                               const ExtAddrMode &AM) const {

  const DebugLoc &DL = MemI.getDebugLoc();
  MachineBasicBlock &MBB = *MemI.getParent();

  assert(AM.ScaledReg == 0 && AM.Scale == 0 &&
         "Addressing mode not supported for folding");

  return BuildMI(MBB, MemI, DL, get(MemI.getOpcode()))
      .addReg(MemI.getOperand(0).getReg(),
              MemI.mayLoad() ? RegState::Define : 0)
      .addReg(AM.BaseReg)
      .addImm(AM.Displacement)
      .setMemRefs(MemI.memoperands())
      .setMIFlags(MemI.getFlags());
}

// TODO: At the moment, MIPS introduced paring of instructions operating with
// word or double word. This should be extended with more instructions when more
// vendors support load/store pairing.
bool CapstoneInstrInfo::isPairableLdStInstOpc(unsigned Opc) {
  switch (Opc) {
  default:
    return false;
  case Capstone::SW:
  case Capstone::SD:
  case Capstone::LD:
  case Capstone::LW:
    return true;
  }
}

bool CapstoneInstrInfo::isLdStSafeToPair(const MachineInstr &LdSt,
                                      const TargetRegisterInfo *TRI) {
  // If this is a volatile load/store, don't mess with it.
  if (LdSt.hasOrderedMemoryRef() || LdSt.getNumExplicitOperands() != 3)
    return false;

  if (LdSt.getOperand(1).isFI())
    return true;

  assert(LdSt.getOperand(1).isReg() && "Expected a reg operand.");
  // Can't cluster if the instruction modifies the base register
  // or it is update form. e.g. ld x5,8(x5)
  if (LdSt.modifiesRegister(LdSt.getOperand(1).getReg(), TRI))
    return false;

  if (!LdSt.getOperand(2).isImm())
    return false;

  return true;
}

bool CapstoneInstrInfo::getMemOperandsWithOffsetWidth(
    const MachineInstr &LdSt, SmallVectorImpl<const MachineOperand *> &BaseOps,
    int64_t &Offset, bool &OffsetIsScalable, LocationSize &Width,
    const TargetRegisterInfo *TRI) const {
  if (!LdSt.mayLoadOrStore())
    return false;

  // Conservatively, only handle scalar loads/stores for now.
  switch (LdSt.getOpcode()) {
  case Capstone::LB:
  case Capstone::LBU:
  case Capstone::SB:
  case Capstone::LH:
  case Capstone::LH_INX:
  case Capstone::LHU:
  case Capstone::FLH:
  case Capstone::SH:
  case Capstone::SH_INX:
  case Capstone::FSH:
  case Capstone::LW:
  case Capstone::LW_INX:
  case Capstone::LWU:
  case Capstone::FLW:
  case Capstone::SW:
  case Capstone::SW_INX:
  case Capstone::FSW:
  case Capstone::LD:
  case Capstone::LD_RV32:
  case Capstone::FLD:
  case Capstone::SD:
  case Capstone::SD_RV32:
  case Capstone::FSD:
    break;
  default:
    return false;
  }
  const MachineOperand *BaseOp;
  OffsetIsScalable = false;
  if (!getMemOperandWithOffsetWidth(LdSt, BaseOp, Offset, Width, TRI))
    return false;
  BaseOps.push_back(BaseOp);
  return true;
}

// TODO: This was copied from SIInstrInfo. Could it be lifted to a common
// helper?
static bool memOpsHaveSameBasePtr(const MachineInstr &MI1,
                                  ArrayRef<const MachineOperand *> BaseOps1,
                                  const MachineInstr &MI2,
                                  ArrayRef<const MachineOperand *> BaseOps2) {
  // Only examine the first "base" operand of each instruction, on the
  // assumption that it represents the real base address of the memory access.
  // Other operands are typically offsets or indices from this base address.
  if (BaseOps1.front()->isIdenticalTo(*BaseOps2.front()))
    return true;

  if (!MI1.hasOneMemOperand() || !MI2.hasOneMemOperand())
    return false;

  auto MO1 = *MI1.memoperands_begin();
  auto MO2 = *MI2.memoperands_begin();
  if (MO1->getAddrSpace() != MO2->getAddrSpace())
    return false;

  auto Base1 = MO1->getValue();
  auto Base2 = MO2->getValue();
  if (!Base1 || !Base2)
    return false;
  Base1 = getUnderlyingObject(Base1);
  Base2 = getUnderlyingObject(Base2);

  if (isa<UndefValue>(Base1) || isa<UndefValue>(Base2))
    return false;

  return Base1 == Base2;
}

bool CapstoneInstrInfo::shouldClusterMemOps(
    ArrayRef<const MachineOperand *> BaseOps1, int64_t Offset1,
    bool OffsetIsScalable1, ArrayRef<const MachineOperand *> BaseOps2,
    int64_t Offset2, bool OffsetIsScalable2, unsigned ClusterSize,
    unsigned NumBytes) const {
  // If the mem ops (to be clustered) do not have the same base ptr, then they
  // should not be clustered
  if (!BaseOps1.empty() && !BaseOps2.empty()) {
    const MachineInstr &FirstLdSt = *BaseOps1.front()->getParent();
    const MachineInstr &SecondLdSt = *BaseOps2.front()->getParent();
    if (!memOpsHaveSameBasePtr(FirstLdSt, BaseOps1, SecondLdSt, BaseOps2))
      return false;
  } else if (!BaseOps1.empty() || !BaseOps2.empty()) {
    // If only one base op is empty, they do not have the same base ptr
    return false;
  }

  unsigned CacheLineSize =
      BaseOps1.front()->getParent()->getMF()->getSubtarget().getCacheLineSize();
  // Assume a cache line size of 64 bytes if no size is set in CapstoneSubtarget.
  CacheLineSize = CacheLineSize ? CacheLineSize : 64;
  // Cluster if the memory operations are on the same or a neighbouring cache
  // line, but limit the maximum ClusterSize to avoid creating too much
  // additional register pressure.
  return ClusterSize <= 4 && std::abs(Offset1 - Offset2) < CacheLineSize;
}

// Set BaseReg (the base register operand), Offset (the byte offset being
// accessed) and the access Width of the passed instruction that reads/writes
// memory. Returns false if the instruction does not read/write memory or the
// BaseReg/Offset/Width can't be determined. Is not guaranteed to always
// recognise base operands and offsets in all cases.
// TODO: Add an IsScalable bool ref argument (like the equivalent AArch64
// function) and set it as appropriate.
bool CapstoneInstrInfo::getMemOperandWithOffsetWidth(
    const MachineInstr &LdSt, const MachineOperand *&BaseReg, int64_t &Offset,
    LocationSize &Width, const TargetRegisterInfo *TRI) const {
  if (!LdSt.mayLoadOrStore())
    return false;

  // Here we assume the standard Capstone ISA, which uses a base+offset
  // addressing mode. You'll need to relax these conditions to support custom
  // load/store instructions.
  if (LdSt.getNumExplicitOperands() != 3)
    return false;
  if ((!LdSt.getOperand(1).isReg() && !LdSt.getOperand(1).isFI()) ||
      !LdSt.getOperand(2).isImm())
    return false;

  if (!LdSt.hasOneMemOperand())
    return false;

  Width = (*LdSt.memoperands_begin())->getSize();
  BaseReg = &LdSt.getOperand(1);
  Offset = LdSt.getOperand(2).getImm();
  return true;
}

bool CapstoneInstrInfo::areMemAccessesTriviallyDisjoint(
    const MachineInstr &MIa, const MachineInstr &MIb) const {
  assert(MIa.mayLoadOrStore() && "MIa must be a load or store.");
  assert(MIb.mayLoadOrStore() && "MIb must be a load or store.");

  if (MIa.hasUnmodeledSideEffects() || MIb.hasUnmodeledSideEffects() ||
      MIa.hasOrderedMemoryRef() || MIb.hasOrderedMemoryRef())
    return false;

  // Retrieve the base register, offset from the base register and width. Width
  // is the size of memory that is being loaded/stored (e.g. 1, 2, 4).  If
  // base registers are identical, and the offset of a lower memory access +
  // the width doesn't overlap the offset of a higher memory access,
  // then the memory accesses are different.
  const TargetRegisterInfo *TRI = STI.getRegisterInfo();
  const MachineOperand *BaseOpA = nullptr, *BaseOpB = nullptr;
  int64_t OffsetA = 0, OffsetB = 0;
  LocationSize WidthA = LocationSize::precise(0),
               WidthB = LocationSize::precise(0);
  if (getMemOperandWithOffsetWidth(MIa, BaseOpA, OffsetA, WidthA, TRI) &&
      getMemOperandWithOffsetWidth(MIb, BaseOpB, OffsetB, WidthB, TRI)) {
    if (BaseOpA->isIdenticalTo(*BaseOpB)) {
      int LowOffset = std::min(OffsetA, OffsetB);
      int HighOffset = std::max(OffsetA, OffsetB);
      LocationSize LowWidth = (LowOffset == OffsetA) ? WidthA : WidthB;
      if (LowWidth.hasValue() &&
          LowOffset + (int)LowWidth.getValue() <= HighOffset)
        return true;
    }
  }
  return false;
}

std::pair<unsigned, unsigned>
CapstoneInstrInfo::decomposeMachineOperandsTargetFlags(unsigned TF) const {
  const unsigned Mask = CapstoneII::MO_DIRECT_FLAG_MASK;
  return std::make_pair(TF & Mask, TF & ~Mask);
}

ArrayRef<std::pair<unsigned, const char *>>
CapstoneInstrInfo::getSerializableDirectMachineOperandTargetFlags() const {
  using namespace CapstoneII;
  static const std::pair<unsigned, const char *> TargetFlags[] = {
      {MO_CALL, "capstone-call"},
      {MO_LO, "capstone-lo"},
      {MO_HI, "capstone-hi"},
      {MO_PCREL_LO, "capstone-pcrel-lo"},
      {MO_PCREL_HI, "capstone-pcrel-hi"},
      {MO_GOT_HI, "capstone-got-hi"},
      {MO_TPREL_LO, "capstone-tprel-lo"},
      {MO_TPREL_HI, "capstone-tprel-hi"},
      {MO_TPREL_ADD, "capstone-tprel-add"},
      {MO_TLS_GOT_HI, "capstone-tls-got-hi"},
      {MO_TLS_GD_HI, "capstone-tls-gd-hi"},
      {MO_TLSDESC_HI, "capstone-tlsdesc-hi"},
      {MO_TLSDESC_LOAD_LO, "capstone-tlsdesc-load-lo"},
      {MO_TLSDESC_ADD_LO, "capstone-tlsdesc-add-lo"},
      {MO_TLSDESC_CALL, "capstone-tlsdesc-call"}};
  return ArrayRef(TargetFlags);
}
bool CapstoneInstrInfo::isFunctionSafeToOutlineFrom(
    MachineFunction &MF, bool OutlineFromLinkOnceODRs) const {
  const Function &F = MF.getFunction();

  // Can F be deduplicated by the linker? If it can, don't outline from it.
  if (!OutlineFromLinkOnceODRs && F.hasLinkOnceODRLinkage())
    return false;

  // Don't outline from functions with section markings; the program could
  // expect that all the code is in the named section.
  if (F.hasSection())
    return false;

  // It's safe to outline from MF.
  return true;
}

bool CapstoneInstrInfo::isMBBSafeToOutlineFrom(MachineBasicBlock &MBB,
                                            unsigned &Flags) const {
  // More accurate safety checking is done in getOutliningCandidateInfo.
  return TargetInstrInfo::isMBBSafeToOutlineFrom(MBB, Flags);
}

// Enum values indicating how an outlined call should be constructed.
enum MachineOutlinerConstructionID {
  MachineOutlinerTailCall,
  MachineOutlinerDefault
};

bool CapstoneInstrInfo::shouldOutlineFromFunctionByDefault(
    MachineFunction &MF) const {
  return MF.getFunction().hasMinSize();
}

static bool isCandidatePatchable(const MachineBasicBlock &MBB) {
  const MachineFunction *MF = MBB.getParent();
  const Function &F = MF->getFunction();
  return F.getFnAttribute("fentry-call").getValueAsBool() ||
         F.hasFnAttribute("patchable-function-entry");
}

static bool isMIReadsReg(const MachineInstr &MI, const TargetRegisterInfo *TRI,
                         MCRegister RegNo) {
  return MI.readsRegister(RegNo, TRI) ||
         MI.getDesc().hasImplicitUseOfPhysReg(RegNo);
}

static bool isMIModifiesReg(const MachineInstr &MI,
                            const TargetRegisterInfo *TRI, MCRegister RegNo) {
  return MI.modifiesRegister(RegNo, TRI) ||
         MI.getDesc().hasImplicitDefOfPhysReg(RegNo);
}

static bool cannotInsertTailCall(const MachineBasicBlock &MBB) {
  if (!MBB.back().isReturn())
    return true;
  if (isCandidatePatchable(MBB))
    return true;

  // If the candidate reads the pre-set register
  // that can be used for expanding PseudoTAIL instruction,
  // then we cannot insert tail call.
  const TargetSubtargetInfo &STI = MBB.getParent()->getSubtarget();
  MCRegister TailExpandUseRegNo =
      CapstoneII::getTailExpandUseRegNo(STI.getFeatureBits());
  for (const MachineInstr &MI : MBB) {
    if (isMIReadsReg(MI, STI.getRegisterInfo(), TailExpandUseRegNo))
      return true;
    if (isMIModifiesReg(MI, STI.getRegisterInfo(), TailExpandUseRegNo))
      break;
  }
  return false;
}

static bool analyzeCandidate(outliner::Candidate &C) {
  // If last instruction is return then we can rely on
  // the verification already performed in the getOutliningTypeImpl.
  if (C.back().isReturn()) {
    assert(!cannotInsertTailCall(*C.getMBB()) &&
           "The candidate who uses return instruction must be outlined "
           "using tail call");
    return false;
  }

  // Filter out candidates where the X5 register (t0) can't be used to setup
  // the function call.
  const TargetRegisterInfo *TRI = C.getMF()->getSubtarget().getRegisterInfo();
  if (llvm::any_of(C, [TRI](const MachineInstr &MI) {
        return isMIModifiesReg(MI, TRI, Capstone::X5);
      }))
    return true;

  return !C.isAvailableAcrossAndOutOfSeq(Capstone::X5, *TRI);
}

std::optional<std::unique_ptr<outliner::OutlinedFunction>>
CapstoneInstrInfo::getOutliningCandidateInfo(
    const MachineModuleInfo &MMI,
    std::vector<outliner::Candidate> &RepeatedSequenceLocs,
    unsigned MinRepeats) const {

  // Analyze each candidate and erase the ones that are not viable.
  llvm::erase_if(RepeatedSequenceLocs, analyzeCandidate);

  // If the sequence doesn't have enough candidates left, then we're done.
  if (RepeatedSequenceLocs.size() < MinRepeats)
    return std::nullopt;

  // Each RepeatedSequenceLoc is identical.
  outliner::Candidate &Candidate = RepeatedSequenceLocs[0];
  unsigned InstrSizeCExt =
      Candidate.getMF()->getSubtarget<CapstoneSubtarget>().hasStdExtZca() ? 2 : 4;
  unsigned CallOverhead = 0, FrameOverhead = 0;

  MachineOutlinerConstructionID MOCI = MachineOutlinerDefault;
  if (Candidate.back().isReturn()) {
    MOCI = MachineOutlinerTailCall;
    // tail call = auipc + jalr in the worst case without linker relaxation.
    // FIXME: This code suggests the JALR can be compressed - how?
    CallOverhead = 4 + InstrSizeCExt;
    // Using tail call we move ret instruction from caller to callee.
    FrameOverhead = 0;
  } else {
    // call t0, function = 8 bytes.
    CallOverhead = 8;
    // jr t0 = 4 bytes, 2 bytes if compressed instructions are enabled.
    FrameOverhead = InstrSizeCExt;
  }

  for (auto &C : RepeatedSequenceLocs)
    C.setCallInfo(MOCI, CallOverhead);

  unsigned SequenceSize = 0;
  for (auto &MI : Candidate)
    SequenceSize += getInstSizeInBytes(MI);

  return std::make_unique<outliner::OutlinedFunction>(
      RepeatedSequenceLocs, SequenceSize, FrameOverhead, MOCI);
}

outliner::InstrType
CapstoneInstrInfo::getOutliningTypeImpl(const MachineModuleInfo &MMI,
                                     MachineBasicBlock::iterator &MBBI,
                                     unsigned Flags) const {
  MachineInstr &MI = *MBBI;
  MachineBasicBlock *MBB = MI.getParent();
  const TargetRegisterInfo *TRI =
      MBB->getParent()->getSubtarget().getRegisterInfo();
  const auto &F = MI.getMF()->getFunction();

  // We can manually strip out CFI instructions later.
  if (MI.isCFIInstruction())
    // If current function has exception handling code, we can't outline &
    // strip these CFI instructions since it may break .eh_frame section
    // needed in unwinding.
    return F.needsUnwindTableEntry() ? outliner::InstrType::Illegal
                                     : outliner::InstrType::Invisible;

  if (cannotInsertTailCall(*MBB) &&
      (MI.isReturn() || isMIModifiesReg(MI, TRI, Capstone::X5)))
    return outliner::InstrType::Illegal;

  // Make sure the operands don't reference something unsafe.
  for (const auto &MO : MI.operands()) {

    // pcrel-hi and pcrel-lo can't put in separate sections, filter that out
    // if any possible.
    if (MO.getTargetFlags() == CapstoneII::MO_PCREL_LO &&
        (MI.getMF()->getTarget().getFunctionSections() || F.hasComdat() ||
         F.hasSection() || F.getSectionPrefix()))
      return outliner::InstrType::Illegal;
  }

  if (isLPAD(MI))
    return outliner::InstrType::Illegal;

  return outliner::InstrType::Legal;
}

void CapstoneInstrInfo::buildOutlinedFrame(
    MachineBasicBlock &MBB, MachineFunction &MF,
    const outliner::OutlinedFunction &OF) const {

  // Strip out any CFI instructions
  bool Changed = true;
  while (Changed) {
    Changed = false;
    auto I = MBB.begin();
    auto E = MBB.end();
    for (; I != E; ++I) {
      if (I->isCFIInstruction()) {
        I->removeFromParent();
        Changed = true;
        break;
      }
    }
  }

  if (OF.FrameConstructionID == MachineOutlinerTailCall)
    return;

  MBB.addLiveIn(Capstone::X5);

  // Add in a return instruction to the end of the outlined frame.
  MBB.insert(MBB.end(), BuildMI(MF, DebugLoc(), get(Capstone::JALR))
      .addReg(Capstone::X0, RegState::Define)
      .addReg(Capstone::X5)
      .addImm(0));
}

MachineBasicBlock::iterator CapstoneInstrInfo::insertOutlinedCall(
    Module &M, MachineBasicBlock &MBB, MachineBasicBlock::iterator &It,
    MachineFunction &MF, outliner::Candidate &C) const {

  if (C.CallConstructionID == MachineOutlinerTailCall) {
    It = MBB.insert(It, BuildMI(MF, DebugLoc(), get(Capstone::PseudoTAIL))
                            .addGlobalAddress(M.getNamedValue(MF.getName()),
                                              /*Offset=*/0, CapstoneII::MO_CALL));
    return It;
  }

  // Add in a call instruction to the outlined function at the given location.
  It = MBB.insert(It,
                  BuildMI(MF, DebugLoc(), get(Capstone::PseudoCALLReg), Capstone::X5)
                      .addGlobalAddress(M.getNamedValue(MF.getName()), 0,
                                        CapstoneII::MO_CALL));
  return It;
}

std::optional<RegImmPair> CapstoneInstrInfo::isAddImmediate(const MachineInstr &MI,
                                                         Register Reg) const {
  // TODO: Handle cases where Reg is a super- or sub-register of the
  // destination register.
  const MachineOperand &Op0 = MI.getOperand(0);
  if (!Op0.isReg() || Reg != Op0.getReg())
    return std::nullopt;

  // Don't consider ADDIW as a candidate because the caller may not be aware
  // of its sign extension behaviour.
  if (MI.getOpcode() == Capstone::ADDI && MI.getOperand(1).isReg() &&
      MI.getOperand(2).isImm())
    return RegImmPair{MI.getOperand(1).getReg(), MI.getOperand(2).getImm()};

  return std::nullopt;
}

// MIR printer helper function to annotate Operands with a comment.
std::string CapstoneInstrInfo::createMIROperandComment(
    const MachineInstr &MI, const MachineOperand &Op, unsigned OpIdx,
    const TargetRegisterInfo *TRI) const {
  // Print a generic comment for this operand if there is one.
  std::string GenericComment =
      TargetInstrInfo::createMIROperandComment(MI, Op, OpIdx, TRI);
  if (!GenericComment.empty())
    return GenericComment;

  // If not, we must have an immediate operand.
  if (!Op.isImm())
    return std::string();

  const MCInstrDesc &Desc = MI.getDesc();
  if (OpIdx >= Desc.getNumOperands())
    return std::string();

  std::string Comment;
  raw_string_ostream OS(Comment);

  const MCOperandInfo &OpInfo = Desc.operands()[OpIdx];

  // Print the full VType operand of vsetvli/vsetivli instructions, and the SEW
  // operand of vector codegen pseudos.
  switch (OpInfo.OperandType) {
  case CapstoneOp::OPERAND_VTYPEI10:
  case CapstoneOp::OPERAND_VTYPEI11: {
    unsigned Imm = Op.getImm();
    CapstoneVType::printVType(Imm, OS);
    break;
  }
  case CapstoneOp::OPERAND_SEW:
  case CapstoneOp::OPERAND_SEW_MASK: {
    unsigned Log2SEW = Op.getImm();
    unsigned SEW = Log2SEW ? 1 << Log2SEW : 8;
    assert(CapstoneVType::isValidSEW(SEW) && "Unexpected SEW");
    OS << "e" << SEW;
    break;
  }
  case CapstoneOp::OPERAND_VEC_POLICY:
    unsigned Policy = Op.getImm();
    assert(Policy <= (CapstoneVType::TAIL_AGNOSTIC | CapstoneVType::MASK_AGNOSTIC) &&
           "Invalid Policy Value");
    OS << (Policy & CapstoneVType::TAIL_AGNOSTIC ? "ta" : "tu") << ", "
       << (Policy & CapstoneVType::MASK_AGNOSTIC ? "ma" : "mu");
    break;
  }

  return Comment;
}

// clang-format off
#define CASE_RVV_OPCODE_UNMASK_LMUL(OP, LMUL)                                 \
  Capstone::Pseudo##OP##_##LMUL

#define CASE_RVV_OPCODE_MASK_LMUL(OP, LMUL)                                   \
  Capstone::Pseudo##OP##_##LMUL##_MASK

#define CASE_RVV_OPCODE_LMUL(OP, LMUL)                                        \
  CASE_RVV_OPCODE_UNMASK_LMUL(OP, LMUL):                                      \
  case CASE_RVV_OPCODE_MASK_LMUL(OP, LMUL)

#define CASE_RVV_OPCODE_UNMASK_WIDEN(OP)                                      \
  CASE_RVV_OPCODE_UNMASK_LMUL(OP, MF8):                                       \
  case CASE_RVV_OPCODE_UNMASK_LMUL(OP, MF4):                                  \
  case CASE_RVV_OPCODE_UNMASK_LMUL(OP, MF2):                                  \
  case CASE_RVV_OPCODE_UNMASK_LMUL(OP, M1):                                   \
  case CASE_RVV_OPCODE_UNMASK_LMUL(OP, M2):                                   \
  case CASE_RVV_OPCODE_UNMASK_LMUL(OP, M4)

#define CASE_RVV_OPCODE_UNMASK(OP)                                            \
  CASE_RVV_OPCODE_UNMASK_WIDEN(OP):                                           \
  case CASE_RVV_OPCODE_UNMASK_LMUL(OP, M8)

#define CASE_RVV_OPCODE_MASK_WIDEN(OP)                                        \
  CASE_RVV_OPCODE_MASK_LMUL(OP, MF8):                                         \
  case CASE_RVV_OPCODE_MASK_LMUL(OP, MF4):                                    \
  case CASE_RVV_OPCODE_MASK_LMUL(OP, MF2):                                    \
  case CASE_RVV_OPCODE_MASK_LMUL(OP, M1):                                     \
  case CASE_RVV_OPCODE_MASK_LMUL(OP, M2):                                     \
  case CASE_RVV_OPCODE_MASK_LMUL(OP, M4)

#define CASE_RVV_OPCODE_MASK(OP)                                              \
  CASE_RVV_OPCODE_MASK_WIDEN(OP):                                             \
  case CASE_RVV_OPCODE_MASK_LMUL(OP, M8)

#define CASE_RVV_OPCODE_WIDEN(OP)                                             \
  CASE_RVV_OPCODE_UNMASK_WIDEN(OP):                                           \
  case CASE_RVV_OPCODE_MASK_WIDEN(OP)

#define CASE_RVV_OPCODE(OP)                                                   \
  CASE_RVV_OPCODE_UNMASK(OP):                                                 \
  case CASE_RVV_OPCODE_MASK(OP)
// clang-format on

// clang-format off
#define CASE_VMA_OPCODE_COMMON(OP, TYPE, LMUL)                                 \
  Capstone::PseudoV##OP##_##TYPE##_##LMUL

#define CASE_VMA_OPCODE_LMULS(OP, TYPE)                                        \
  CASE_VMA_OPCODE_COMMON(OP, TYPE, MF8):                                       \
  case CASE_VMA_OPCODE_COMMON(OP, TYPE, MF4):                                  \
  case CASE_VMA_OPCODE_COMMON(OP, TYPE, MF2):                                  \
  case CASE_VMA_OPCODE_COMMON(OP, TYPE, M1):                                   \
  case CASE_VMA_OPCODE_COMMON(OP, TYPE, M2):                                   \
  case CASE_VMA_OPCODE_COMMON(OP, TYPE, M4):                                   \
  case CASE_VMA_OPCODE_COMMON(OP, TYPE, M8)

// VFMA instructions are SEW specific.
#define CASE_VFMA_OPCODE_COMMON(OP, TYPE, LMUL, SEW)                           \
  Capstone::PseudoV##OP##_##TYPE##_##LMUL##_##SEW

#define CASE_VFMA_OPCODE_LMULS_M1(OP, TYPE, SEW)                               \
  CASE_VFMA_OPCODE_COMMON(OP, TYPE, M1, SEW):                                  \
  case CASE_VFMA_OPCODE_COMMON(OP, TYPE, M2, SEW):                             \
  case CASE_VFMA_OPCODE_COMMON(OP, TYPE, M4, SEW):                             \
  case CASE_VFMA_OPCODE_COMMON(OP, TYPE, M8, SEW)

#define CASE_VFMA_OPCODE_LMULS_MF2(OP, TYPE, SEW)                              \
  CASE_VFMA_OPCODE_COMMON(OP, TYPE, MF2, SEW):                                 \
  case CASE_VFMA_OPCODE_LMULS_M1(OP, TYPE, SEW)

#define CASE_VFMA_OPCODE_LMULS_MF4(OP, TYPE, SEW)                              \
  CASE_VFMA_OPCODE_COMMON(OP, TYPE, MF4, SEW):                                 \
  case CASE_VFMA_OPCODE_LMULS_MF2(OP, TYPE, SEW)

#define CASE_VFMA_OPCODE_VV(OP)                                                \
  CASE_VFMA_OPCODE_LMULS_MF4(OP, VV, E16):                                     \
  case CASE_VFMA_OPCODE_LMULS_MF2(OP, VV, E32):                                \
  case CASE_VFMA_OPCODE_LMULS_M1(OP, VV, E64)

#define CASE_VFMA_SPLATS(OP)                                                   \
  CASE_VFMA_OPCODE_LMULS_MF4(OP, VFPR16, E16):                                 \
  case CASE_VFMA_OPCODE_LMULS_MF2(OP, VFPR32, E32):                            \
  case CASE_VFMA_OPCODE_LMULS_M1(OP, VFPR64, E64)
// clang-format on

bool CapstoneInstrInfo::findCommutedOpIndices(const MachineInstr &MI,
                                           unsigned &SrcOpIdx1,
                                           unsigned &SrcOpIdx2) const {
  const MCInstrDesc &Desc = MI.getDesc();
  if (!Desc.isCommutable())
    return false;

  switch (MI.getOpcode()) {
  case Capstone::TH_MVEQZ:
  case Capstone::TH_MVNEZ:
    // We can't commute operands if operand 2 (i.e., rs1 in
    // mveqz/mvnez rd,rs1,rs2) is the zero-register (as it is
    // not valid as the in/out-operand 1).
    if (MI.getOperand(2).getReg() == Capstone::X0)
      return false;
    // Operands 1 and 2 are commutable, if we switch the opcode.
    return fixCommutedOpIndices(SrcOpIdx1, SrcOpIdx2, 1, 2);
  case Capstone::TH_MULA:
  case Capstone::TH_MULAW:
  case Capstone::TH_MULAH:
  case Capstone::TH_MULS:
  case Capstone::TH_MULSW:
  case Capstone::TH_MULSH:
    // Operands 2 and 3 are commutable.
    return fixCommutedOpIndices(SrcOpIdx1, SrcOpIdx2, 2, 3);
  case Capstone::PseudoCCMOVGPRNoX0:
  case Capstone::PseudoCCMOVGPR:
    // Operands 4 and 5 are commutable.
    return fixCommutedOpIndices(SrcOpIdx1, SrcOpIdx2, 4, 5);
  case CASE_RVV_OPCODE(VADD_VV):
  case CASE_RVV_OPCODE(VAND_VV):
  case CASE_RVV_OPCODE(VOR_VV):
  case CASE_RVV_OPCODE(VXOR_VV):
  case CASE_RVV_OPCODE_MASK(VMSEQ_VV):
  case CASE_RVV_OPCODE_MASK(VMSNE_VV):
  case CASE_RVV_OPCODE(VMIN_VV):
  case CASE_RVV_OPCODE(VMINU_VV):
  case CASE_RVV_OPCODE(VMAX_VV):
  case CASE_RVV_OPCODE(VMAXU_VV):
  case CASE_RVV_OPCODE(VMUL_VV):
  case CASE_RVV_OPCODE(VMULH_VV):
  case CASE_RVV_OPCODE(VMULHU_VV):
  case CASE_RVV_OPCODE_WIDEN(VWADD_VV):
  case CASE_RVV_OPCODE_WIDEN(VWADDU_VV):
  case CASE_RVV_OPCODE_WIDEN(VWMUL_VV):
  case CASE_RVV_OPCODE_WIDEN(VWMULU_VV):
  case CASE_RVV_OPCODE_WIDEN(VWMACC_VV):
  case CASE_RVV_OPCODE_WIDEN(VWMACCU_VV):
  case CASE_RVV_OPCODE_UNMASK(VADC_VVM):
  case CASE_RVV_OPCODE(VSADD_VV):
  case CASE_RVV_OPCODE(VSADDU_VV):
  case CASE_RVV_OPCODE(VAADD_VV):
  case CASE_RVV_OPCODE(VAADDU_VV):
  case CASE_RVV_OPCODE(VSMUL_VV):
    // Operands 2 and 3 are commutable.
    return fixCommutedOpIndices(SrcOpIdx1, SrcOpIdx2, 2, 3);
  case CASE_VFMA_SPLATS(FMADD):
  case CASE_VFMA_SPLATS(FMSUB):
  case CASE_VFMA_SPLATS(FMACC):
  case CASE_VFMA_SPLATS(FMSAC):
  case CASE_VFMA_SPLATS(FNMADD):
  case CASE_VFMA_SPLATS(FNMSUB):
  case CASE_VFMA_SPLATS(FNMACC):
  case CASE_VFMA_SPLATS(FNMSAC):
  case CASE_VFMA_OPCODE_VV(FMACC):
  case CASE_VFMA_OPCODE_VV(FMSAC):
  case CASE_VFMA_OPCODE_VV(FNMACC):
  case CASE_VFMA_OPCODE_VV(FNMSAC):
  case CASE_VMA_OPCODE_LMULS(MADD, VX):
  case CASE_VMA_OPCODE_LMULS(NMSUB, VX):
  case CASE_VMA_OPCODE_LMULS(MACC, VX):
  case CASE_VMA_OPCODE_LMULS(NMSAC, VX):
  case CASE_VMA_OPCODE_LMULS(MACC, VV):
  case CASE_VMA_OPCODE_LMULS(NMSAC, VV): {
    // If the tail policy is undisturbed we can't commute.
    assert(CapstoneII::hasVecPolicyOp(MI.getDesc().TSFlags));
    if ((MI.getOperand(CapstoneII::getVecPolicyOpNum(MI.getDesc())).getImm() &
         1) == 0)
      return false;

    // For these instructions we can only swap operand 1 and operand 3 by
    // changing the opcode.
    unsigned CommutableOpIdx1 = 1;
    unsigned CommutableOpIdx2 = 3;
    if (!fixCommutedOpIndices(SrcOpIdx1, SrcOpIdx2, CommutableOpIdx1,
                              CommutableOpIdx2))
      return false;
    return true;
  }
  case CASE_VFMA_OPCODE_VV(FMADD):
  case CASE_VFMA_OPCODE_VV(FMSUB):
  case CASE_VFMA_OPCODE_VV(FNMADD):
  case CASE_VFMA_OPCODE_VV(FNMSUB):
  case CASE_VMA_OPCODE_LMULS(MADD, VV):
  case CASE_VMA_OPCODE_LMULS(NMSUB, VV): {
    // If the tail policy is undisturbed we can't commute.
    assert(CapstoneII::hasVecPolicyOp(MI.getDesc().TSFlags));
    if ((MI.getOperand(CapstoneII::getVecPolicyOpNum(MI.getDesc())).getImm() &
         1) == 0)
      return false;

    // For these instructions we have more freedom. We can commute with the
    // other multiplicand or with the addend/subtrahend/minuend.

    // Any fixed operand must be from source 1, 2 or 3.
    if (SrcOpIdx1 != CommuteAnyOperandIndex && SrcOpIdx1 > 3)
      return false;
    if (SrcOpIdx2 != CommuteAnyOperandIndex && SrcOpIdx2 > 3)
      return false;

    // It both ops are fixed one must be the tied source.
    if (SrcOpIdx1 != CommuteAnyOperandIndex &&
        SrcOpIdx2 != CommuteAnyOperandIndex && SrcOpIdx1 != 1 && SrcOpIdx2 != 1)
      return false;

    // Look for two different register operands assumed to be commutable
    // regardless of the FMA opcode. The FMA opcode is adjusted later if
    // needed.
    if (SrcOpIdx1 == CommuteAnyOperandIndex ||
        SrcOpIdx2 == CommuteAnyOperandIndex) {
      // At least one of operands to be commuted is not specified and
      // this method is free to choose appropriate commutable operands.
      unsigned CommutableOpIdx1 = SrcOpIdx1;
      if (SrcOpIdx1 == SrcOpIdx2) {
        // Both of operands are not fixed. Set one of commutable
        // operands to the tied source.
        CommutableOpIdx1 = 1;
      } else if (SrcOpIdx1 == CommuteAnyOperandIndex) {
        // Only one of the operands is not fixed.
        CommutableOpIdx1 = SrcOpIdx2;
      }

      // CommutableOpIdx1 is well defined now. Let's choose another commutable
      // operand and assign its index to CommutableOpIdx2.
      unsigned CommutableOpIdx2;
      if (CommutableOpIdx1 != 1) {
        // If we haven't already used the tied source, we must use it now.
        CommutableOpIdx2 = 1;
      } else {
        Register Op1Reg = MI.getOperand(CommutableOpIdx1).getReg();

        // The commuted operands should have different registers.
        // Otherwise, the commute transformation does not change anything and
        // is useless. We use this as a hint to make our decision.
        if (Op1Reg != MI.getOperand(2).getReg())
          CommutableOpIdx2 = 2;
        else
          CommutableOpIdx2 = 3;
      }

      // Assign the found pair of commutable indices to SrcOpIdx1 and
      // SrcOpIdx2 to return those values.
      if (!fixCommutedOpIndices(SrcOpIdx1, SrcOpIdx2, CommutableOpIdx1,
                                CommutableOpIdx2))
        return false;
    }

    return true;
  }
  }

  return TargetInstrInfo::findCommutedOpIndices(MI, SrcOpIdx1, SrcOpIdx2);
}

// clang-format off
#define CASE_VMA_CHANGE_OPCODE_COMMON(OLDOP, NEWOP, TYPE, LMUL)                \
  case Capstone::PseudoV##OLDOP##_##TYPE##_##LMUL:                                \
    Opc = Capstone::PseudoV##NEWOP##_##TYPE##_##LMUL;                             \
    break;

#define CASE_VMA_CHANGE_OPCODE_LMULS(OLDOP, NEWOP, TYPE)                       \
  CASE_VMA_CHANGE_OPCODE_COMMON(OLDOP, NEWOP, TYPE, MF8)                       \
  CASE_VMA_CHANGE_OPCODE_COMMON(OLDOP, NEWOP, TYPE, MF4)                       \
  CASE_VMA_CHANGE_OPCODE_COMMON(OLDOP, NEWOP, TYPE, MF2)                       \
  CASE_VMA_CHANGE_OPCODE_COMMON(OLDOP, NEWOP, TYPE, M1)                        \
  CASE_VMA_CHANGE_OPCODE_COMMON(OLDOP, NEWOP, TYPE, M2)                        \
  CASE_VMA_CHANGE_OPCODE_COMMON(OLDOP, NEWOP, TYPE, M4)                        \
  CASE_VMA_CHANGE_OPCODE_COMMON(OLDOP, NEWOP, TYPE, M8)

// VFMA depends on SEW.
#define CASE_VFMA_CHANGE_OPCODE_COMMON(OLDOP, NEWOP, TYPE, LMUL, SEW)          \
  case Capstone::PseudoV##OLDOP##_##TYPE##_##LMUL##_##SEW:                        \
    Opc = Capstone::PseudoV##NEWOP##_##TYPE##_##LMUL##_##SEW;                     \
    break;

#define CASE_VFMA_CHANGE_OPCODE_LMULS_M1(OLDOP, NEWOP, TYPE, SEW)              \
  CASE_VFMA_CHANGE_OPCODE_COMMON(OLDOP, NEWOP, TYPE, M1, SEW)                  \
  CASE_VFMA_CHANGE_OPCODE_COMMON(OLDOP, NEWOP, TYPE, M2, SEW)                  \
  CASE_VFMA_CHANGE_OPCODE_COMMON(OLDOP, NEWOP, TYPE, M4, SEW)                  \
  CASE_VFMA_CHANGE_OPCODE_COMMON(OLDOP, NEWOP, TYPE, M8, SEW)

#define CASE_VFMA_CHANGE_OPCODE_LMULS_MF2(OLDOP, NEWOP, TYPE, SEW)             \
  CASE_VFMA_CHANGE_OPCODE_COMMON(OLDOP, NEWOP, TYPE, MF2, SEW)                 \
  CASE_VFMA_CHANGE_OPCODE_LMULS_M1(OLDOP, NEWOP, TYPE, SEW)

#define CASE_VFMA_CHANGE_OPCODE_LMULS_MF4(OLDOP, NEWOP, TYPE, SEW)             \
  CASE_VFMA_CHANGE_OPCODE_COMMON(OLDOP, NEWOP, TYPE, MF4, SEW)                 \
  CASE_VFMA_CHANGE_OPCODE_LMULS_MF2(OLDOP, NEWOP, TYPE, SEW)

#define CASE_VFMA_CHANGE_OPCODE_VV(OLDOP, NEWOP)                               \
  CASE_VFMA_CHANGE_OPCODE_LMULS_MF4(OLDOP, NEWOP, VV, E16)                     \
  CASE_VFMA_CHANGE_OPCODE_LMULS_MF2(OLDOP, NEWOP, VV, E32)                     \
  CASE_VFMA_CHANGE_OPCODE_LMULS_M1(OLDOP, NEWOP, VV, E64)

#define CASE_VFMA_CHANGE_OPCODE_SPLATS(OLDOP, NEWOP)                           \
  CASE_VFMA_CHANGE_OPCODE_LMULS_MF4(OLDOP, NEWOP, VFPR16, E16)                 \
  CASE_VFMA_CHANGE_OPCODE_LMULS_MF2(OLDOP, NEWOP, VFPR32, E32)                 \
  CASE_VFMA_CHANGE_OPCODE_LMULS_M1(OLDOP, NEWOP, VFPR64, E64)
// clang-format on

MachineInstr *CapstoneInstrInfo::commuteInstructionImpl(MachineInstr &MI,
                                                     bool NewMI,
                                                     unsigned OpIdx1,
                                                     unsigned OpIdx2) const {
  auto cloneIfNew = [NewMI](MachineInstr &MI) -> MachineInstr & {
    if (NewMI)
      return *MI.getParent()->getParent()->CloneMachineInstr(&MI);
    return MI;
  };

  switch (MI.getOpcode()) {
  case Capstone::TH_MVEQZ:
  case Capstone::TH_MVNEZ: {
    auto &WorkingMI = cloneIfNew(MI);
    WorkingMI.setDesc(get(MI.getOpcode() == Capstone::TH_MVEQZ ? Capstone::TH_MVNEZ
                                                            : Capstone::TH_MVEQZ));
    return TargetInstrInfo::commuteInstructionImpl(WorkingMI, false, OpIdx1,
                                                   OpIdx2);
  }
  case Capstone::PseudoCCMOVGPRNoX0:
  case Capstone::PseudoCCMOVGPR: {
    // CCMOV can be commuted by inverting the condition.
    auto CC = static_cast<CapstoneCC::CondCode>(MI.getOperand(3).getImm());
    CC = CapstoneCC::getOppositeBranchCondition(CC);
    auto &WorkingMI = cloneIfNew(MI);
    WorkingMI.getOperand(3).setImm(CC);
    return TargetInstrInfo::commuteInstructionImpl(WorkingMI, /*NewMI*/ false,
                                                   OpIdx1, OpIdx2);
  }
  case CASE_VFMA_SPLATS(FMACC):
  case CASE_VFMA_SPLATS(FMADD):
  case CASE_VFMA_SPLATS(FMSAC):
  case CASE_VFMA_SPLATS(FMSUB):
  case CASE_VFMA_SPLATS(FNMACC):
  case CASE_VFMA_SPLATS(FNMADD):
  case CASE_VFMA_SPLATS(FNMSAC):
  case CASE_VFMA_SPLATS(FNMSUB):
  case CASE_VFMA_OPCODE_VV(FMACC):
  case CASE_VFMA_OPCODE_VV(FMSAC):
  case CASE_VFMA_OPCODE_VV(FNMACC):
  case CASE_VFMA_OPCODE_VV(FNMSAC):
  case CASE_VMA_OPCODE_LMULS(MADD, VX):
  case CASE_VMA_OPCODE_LMULS(NMSUB, VX):
  case CASE_VMA_OPCODE_LMULS(MACC, VX):
  case CASE_VMA_OPCODE_LMULS(NMSAC, VX):
  case CASE_VMA_OPCODE_LMULS(MACC, VV):
  case CASE_VMA_OPCODE_LMULS(NMSAC, VV): {
    // It only make sense to toggle these between clobbering the
    // addend/subtrahend/minuend one of the multiplicands.
    assert((OpIdx1 == 1 || OpIdx2 == 1) && "Unexpected opcode index");
    assert((OpIdx1 == 3 || OpIdx2 == 3) && "Unexpected opcode index");
    unsigned Opc;
    switch (MI.getOpcode()) {
      default:
        llvm_unreachable("Unexpected opcode");
      CASE_VFMA_CHANGE_OPCODE_SPLATS(FMACC, FMADD)
      CASE_VFMA_CHANGE_OPCODE_SPLATS(FMADD, FMACC)
      CASE_VFMA_CHANGE_OPCODE_SPLATS(FMSAC, FMSUB)
      CASE_VFMA_CHANGE_OPCODE_SPLATS(FMSUB, FMSAC)
      CASE_VFMA_CHANGE_OPCODE_SPLATS(FNMACC, FNMADD)
      CASE_VFMA_CHANGE_OPCODE_SPLATS(FNMADD, FNMACC)
      CASE_VFMA_CHANGE_OPCODE_SPLATS(FNMSAC, FNMSUB)
      CASE_VFMA_CHANGE_OPCODE_SPLATS(FNMSUB, FNMSAC)
      CASE_VFMA_CHANGE_OPCODE_VV(FMACC, FMADD)
      CASE_VFMA_CHANGE_OPCODE_VV(FMSAC, FMSUB)
      CASE_VFMA_CHANGE_OPCODE_VV(FNMACC, FNMADD)
      CASE_VFMA_CHANGE_OPCODE_VV(FNMSAC, FNMSUB)
      CASE_VMA_CHANGE_OPCODE_LMULS(MACC, MADD, VX)
      CASE_VMA_CHANGE_OPCODE_LMULS(MADD, MACC, VX)
      CASE_VMA_CHANGE_OPCODE_LMULS(NMSAC, NMSUB, VX)
      CASE_VMA_CHANGE_OPCODE_LMULS(NMSUB, NMSAC, VX)
      CASE_VMA_CHANGE_OPCODE_LMULS(MACC, MADD, VV)
      CASE_VMA_CHANGE_OPCODE_LMULS(NMSAC, NMSUB, VV)
    }

    auto &WorkingMI = cloneIfNew(MI);
    WorkingMI.setDesc(get(Opc));
    return TargetInstrInfo::commuteInstructionImpl(WorkingMI, /*NewMI=*/false,
                                                   OpIdx1, OpIdx2);
  }
  case CASE_VFMA_OPCODE_VV(FMADD):
  case CASE_VFMA_OPCODE_VV(FMSUB):
  case CASE_VFMA_OPCODE_VV(FNMADD):
  case CASE_VFMA_OPCODE_VV(FNMSUB):
  case CASE_VMA_OPCODE_LMULS(MADD, VV):
  case CASE_VMA_OPCODE_LMULS(NMSUB, VV): {
    assert((OpIdx1 == 1 || OpIdx2 == 1) && "Unexpected opcode index");
    // If one of the operands, is the addend we need to change opcode.
    // Otherwise we're just swapping 2 of the multiplicands.
    if (OpIdx1 == 3 || OpIdx2 == 3) {
      unsigned Opc;
      switch (MI.getOpcode()) {
        default:
          llvm_unreachable("Unexpected opcode");
        CASE_VFMA_CHANGE_OPCODE_VV(FMADD, FMACC)
        CASE_VFMA_CHANGE_OPCODE_VV(FMSUB, FMSAC)
        CASE_VFMA_CHANGE_OPCODE_VV(FNMADD, FNMACC)
        CASE_VFMA_CHANGE_OPCODE_VV(FNMSUB, FNMSAC)
        CASE_VMA_CHANGE_OPCODE_LMULS(MADD, MACC, VV)
        CASE_VMA_CHANGE_OPCODE_LMULS(NMSUB, NMSAC, VV)
      }

      auto &WorkingMI = cloneIfNew(MI);
      WorkingMI.setDesc(get(Opc));
      return TargetInstrInfo::commuteInstructionImpl(WorkingMI, /*NewMI=*/false,
                                                     OpIdx1, OpIdx2);
    }
    // Let the default code handle it.
    break;
  }
  }

  return TargetInstrInfo::commuteInstructionImpl(MI, NewMI, OpIdx1, OpIdx2);
}

#undef CASE_VMA_CHANGE_OPCODE_COMMON
#undef CASE_VMA_CHANGE_OPCODE_LMULS
#undef CASE_VFMA_CHANGE_OPCODE_COMMON
#undef CASE_VFMA_CHANGE_OPCODE_LMULS_M1
#undef CASE_VFMA_CHANGE_OPCODE_LMULS_MF2
#undef CASE_VFMA_CHANGE_OPCODE_LMULS_MF4
#undef CASE_VFMA_CHANGE_OPCODE_VV
#undef CASE_VFMA_CHANGE_OPCODE_SPLATS

#undef CASE_RVV_OPCODE_UNMASK_LMUL
#undef CASE_RVV_OPCODE_MASK_LMUL
#undef CASE_RVV_OPCODE_LMUL
#undef CASE_RVV_OPCODE_UNMASK_WIDEN
#undef CASE_RVV_OPCODE_UNMASK
#undef CASE_RVV_OPCODE_MASK_WIDEN
#undef CASE_RVV_OPCODE_MASK
#undef CASE_RVV_OPCODE_WIDEN
#undef CASE_RVV_OPCODE

#undef CASE_VMA_OPCODE_COMMON
#undef CASE_VMA_OPCODE_LMULS
#undef CASE_VFMA_OPCODE_COMMON
#undef CASE_VFMA_OPCODE_LMULS_M1
#undef CASE_VFMA_OPCODE_LMULS_MF2
#undef CASE_VFMA_OPCODE_LMULS_MF4
#undef CASE_VFMA_OPCODE_VV
#undef CASE_VFMA_SPLATS

bool CapstoneInstrInfo::simplifyInstruction(MachineInstr &MI) const {
  switch (MI.getOpcode()) {
  default:
    break;
  case Capstone::ADD:
  case Capstone::OR:
  case Capstone::XOR:
    // Normalize (so we hit the next if clause).
    // add/[x]or rd, zero, rs => add/[x]or rd, rs, zero
    if (MI.getOperand(1).getReg() == Capstone::X0)
      commuteInstruction(MI);
    // add/[x]or rd, rs, zero => addi rd, rs, 0
    if (MI.getOperand(2).getReg() == Capstone::X0) {
      MI.getOperand(2).ChangeToImmediate(0);
      MI.setDesc(get(Capstone::ADDI));
      return true;
    }
    // xor rd, rs, rs => addi rd, zero, 0
    if (MI.getOpcode() == Capstone::XOR &&
        MI.getOperand(1).getReg() == MI.getOperand(2).getReg()) {
      MI.getOperand(1).setReg(Capstone::X0);
      MI.getOperand(2).ChangeToImmediate(0);
      MI.setDesc(get(Capstone::ADDI));
      return true;
    }
    break;
  case Capstone::ORI:
  case Capstone::XORI:
    // [x]ori rd, zero, N => addi rd, zero, N
    if (MI.getOperand(1).getReg() == Capstone::X0) {
      MI.setDesc(get(Capstone::ADDI));
      return true;
    }
    break;
  case Capstone::SUB:
    // sub rd, rs, zero => addi rd, rs, 0
    if (MI.getOperand(2).getReg() == Capstone::X0) {
      MI.getOperand(2).ChangeToImmediate(0);
      MI.setDesc(get(Capstone::ADDI));
      return true;
    }
    break;
  case Capstone::SUBW:
    // subw rd, rs, zero => addiw rd, rs, 0
    if (MI.getOperand(2).getReg() == Capstone::X0) {
      MI.getOperand(2).ChangeToImmediate(0);
      MI.setDesc(get(Capstone::ADDIW));
      return true;
    }
    break;
  case Capstone::ADDW:
    // Normalize (so we hit the next if clause).
    // addw rd, zero, rs => addw rd, rs, zero
    if (MI.getOperand(1).getReg() == Capstone::X0)
      commuteInstruction(MI);
    // addw rd, rs, zero => addiw rd, rs, 0
    if (MI.getOperand(2).getReg() == Capstone::X0) {
      MI.getOperand(2).ChangeToImmediate(0);
      MI.setDesc(get(Capstone::ADDIW));
      return true;
    }
    break;
  case Capstone::SH1ADD:
  case Capstone::SH1ADD_UW:
  case Capstone::SH2ADD:
  case Capstone::SH2ADD_UW:
  case Capstone::SH3ADD:
  case Capstone::SH3ADD_UW:
    // shNadd[.uw] rd, zero, rs => addi rd, rs, 0
    if (MI.getOperand(1).getReg() == Capstone::X0) {
      MI.removeOperand(1);
      MI.addOperand(MachineOperand::CreateImm(0));
      MI.setDesc(get(Capstone::ADDI));
      return true;
    }
    // shNadd[.uw] rd, rs, zero => slli[.uw] rd, rs, N
    if (MI.getOperand(2).getReg() == Capstone::X0) {
      MI.removeOperand(2);
      unsigned Opc = MI.getOpcode();
      if (Opc == Capstone::SH1ADD_UW || Opc == Capstone::SH2ADD_UW ||
          Opc == Capstone::SH3ADD_UW) {
        MI.addOperand(MachineOperand::CreateImm(getSHXADDUWShiftAmount(Opc)));
        MI.setDesc(get(Capstone::SLLI_UW));
        return true;
      }
      MI.addOperand(MachineOperand::CreateImm(getSHXADDShiftAmount(Opc)));
      MI.setDesc(get(Capstone::SLLI));
      return true;
    }
    break;
  case Capstone::AND:
  case Capstone::MUL:
  case Capstone::MULH:
  case Capstone::MULHSU:
  case Capstone::MULHU:
  case Capstone::MULW:
    // and rd, zero, rs => addi rd, zero, 0
    // mul* rd, zero, rs => addi rd, zero, 0
    // and rd, rs, zero => addi rd, zero, 0
    // mul* rd, rs, zero => addi rd, zero, 0
    if (MI.getOperand(1).getReg() == Capstone::X0 ||
        MI.getOperand(2).getReg() == Capstone::X0) {
      MI.getOperand(1).setReg(Capstone::X0);
      MI.getOperand(2).ChangeToImmediate(0);
      MI.setDesc(get(Capstone::ADDI));
      return true;
    }
    break;
  case Capstone::ANDI:
    // andi rd, zero, C => addi rd, zero, 0
    if (MI.getOperand(1).getReg() == Capstone::X0) {
      MI.getOperand(2).setImm(0);
      MI.setDesc(get(Capstone::ADDI));
      return true;
    }
    break;
  case Capstone::SLL:
  case Capstone::SRL:
  case Capstone::SRA:
    // shift rd, zero, rs => addi rd, zero, 0
    if (MI.getOperand(1).getReg() == Capstone::X0) {
      MI.getOperand(2).ChangeToImmediate(0);
      MI.setDesc(get(Capstone::ADDI));
      return true;
    }
    // shift rd, rs, zero => addi rd, rs, 0
    if (MI.getOperand(2).getReg() == Capstone::X0) {
      MI.getOperand(2).ChangeToImmediate(0);
      MI.setDesc(get(Capstone::ADDI));
      return true;
    }
    break;
  case Capstone::SLLW:
  case Capstone::SRLW:
  case Capstone::SRAW:
    // shiftw rd, zero, rs => addi rd, zero, 0
    if (MI.getOperand(1).getReg() == Capstone::X0) {
      MI.getOperand(2).ChangeToImmediate(0);
      MI.setDesc(get(Capstone::ADDI));
      return true;
    }
    break;
  case Capstone::SLLI:
  case Capstone::SRLI:
  case Capstone::SRAI:
  case Capstone::SLLIW:
  case Capstone::SRLIW:
  case Capstone::SRAIW:
  case Capstone::SLLI_UW:
    // shiftimm rd, zero, N => addi rd, zero, 0
    if (MI.getOperand(1).getReg() == Capstone::X0) {
      MI.getOperand(2).setImm(0);
      MI.setDesc(get(Capstone::ADDI));
      return true;
    }
    break;
  case Capstone::SLTU:
  case Capstone::ADD_UW:
    // sltu rd, zero, zero => addi rd, zero, 0
    // add.uw rd, zero, zero => addi rd, zero, 0
    if (MI.getOperand(1).getReg() == Capstone::X0 &&
        MI.getOperand(2).getReg() == Capstone::X0) {
      MI.getOperand(2).ChangeToImmediate(0);
      MI.setDesc(get(Capstone::ADDI));
      return true;
    }
    // add.uw rd, zero, rs => addi rd, rs, 0
    if (MI.getOpcode() == Capstone::ADD_UW &&
        MI.getOperand(1).getReg() == Capstone::X0) {
      MI.removeOperand(1);
      MI.addOperand(MachineOperand::CreateImm(0));
      MI.setDesc(get(Capstone::ADDI));
    }
    break;
  case Capstone::SLTIU:
    // sltiu rd, zero, NZC => addi rd, zero, 1
    // sltiu rd, zero, 0 => addi rd, zero, 0
    if (MI.getOperand(1).getReg() == Capstone::X0) {
      MI.getOperand(2).setImm(MI.getOperand(2).getImm() != 0);
      MI.setDesc(get(Capstone::ADDI));
      return true;
    }
    break;
  case Capstone::SEXT_H:
  case Capstone::SEXT_B:
  case Capstone::ZEXT_H_RV32:
  case Capstone::ZEXT_H_RV64:
    // sext.[hb] rd, zero => addi rd, zero, 0
    // zext.h rd, zero => addi rd, zero, 0
    if (MI.getOperand(1).getReg() == Capstone::X0) {
      MI.addOperand(MachineOperand::CreateImm(0));
      MI.setDesc(get(Capstone::ADDI));
      return true;
    }
    break;
  case Capstone::MIN:
  case Capstone::MINU:
  case Capstone::MAX:
  case Capstone::MAXU:
    // min|max rd, rs, rs => addi rd, rs, 0
    if (MI.getOperand(1).getReg() == MI.getOperand(2).getReg()) {
      MI.getOperand(2).ChangeToImmediate(0);
      MI.setDesc(get(Capstone::ADDI));
      return true;
    }
    break;
  case Capstone::BEQ:
  case Capstone::BNE:
    // b{eq,ne} zero, rs, imm => b{eq,ne} rs, zero, imm
    if (MI.getOperand(0).getReg() == Capstone::X0) {
      MachineOperand MO0 = MI.getOperand(0);
      MI.removeOperand(0);
      MI.insert(MI.operands_begin() + 1, {MO0});
    }
    break;
  case Capstone::BLTU:
    // bltu zero, rs, imm => bne rs, zero, imm
    if (MI.getOperand(0).getReg() == Capstone::X0) {
      MachineOperand MO0 = MI.getOperand(0);
      MI.removeOperand(0);
      MI.insert(MI.operands_begin() + 1, {MO0});
      MI.setDesc(get(Capstone::BNE));
    }
    break;
  case Capstone::BGEU:
    // bgeu zero, rs, imm => beq rs, zero, imm
    if (MI.getOperand(0).getReg() == Capstone::X0) {
      MachineOperand MO0 = MI.getOperand(0);
      MI.removeOperand(0);
      MI.insert(MI.operands_begin() + 1, {MO0});
      MI.setDesc(get(Capstone::BEQ));
    }
    break;
  }
  return false;
}

// clang-format off
#define CASE_WIDEOP_OPCODE_COMMON(OP, LMUL)                                    \
  Capstone::PseudoV##OP##_##LMUL##_TIED

#define CASE_WIDEOP_OPCODE_LMULS(OP)                                           \
  CASE_WIDEOP_OPCODE_COMMON(OP, MF8):                                          \
  case CASE_WIDEOP_OPCODE_COMMON(OP, MF4):                                     \
  case CASE_WIDEOP_OPCODE_COMMON(OP, MF2):                                     \
  case CASE_WIDEOP_OPCODE_COMMON(OP, M1):                                      \
  case CASE_WIDEOP_OPCODE_COMMON(OP, M2):                                      \
  case CASE_WIDEOP_OPCODE_COMMON(OP, M4)

#define CASE_WIDEOP_CHANGE_OPCODE_COMMON(OP, LMUL)                             \
  case Capstone::PseudoV##OP##_##LMUL##_TIED:                                     \
    NewOpc = Capstone::PseudoV##OP##_##LMUL;                                      \
    break;

#define CASE_WIDEOP_CHANGE_OPCODE_LMULS(OP)                                    \
  CASE_WIDEOP_CHANGE_OPCODE_COMMON(OP, MF8)                                    \
  CASE_WIDEOP_CHANGE_OPCODE_COMMON(OP, MF4)                                    \
  CASE_WIDEOP_CHANGE_OPCODE_COMMON(OP, MF2)                                    \
  CASE_WIDEOP_CHANGE_OPCODE_COMMON(OP, M1)                                     \
  CASE_WIDEOP_CHANGE_OPCODE_COMMON(OP, M2)                                     \
  CASE_WIDEOP_CHANGE_OPCODE_COMMON(OP, M4)

// FP Widening Ops may by SEW aware. Create SEW aware cases for these cases.
#define CASE_FP_WIDEOP_OPCODE_COMMON(OP, LMUL, SEW)                            \
  Capstone::PseudoV##OP##_##LMUL##_##SEW##_TIED

#define CASE_FP_WIDEOP_OPCODE_LMULS(OP)                                        \
  CASE_FP_WIDEOP_OPCODE_COMMON(OP, MF4, E16):                                  \
  case CASE_FP_WIDEOP_OPCODE_COMMON(OP, MF2, E16):                             \
  case CASE_FP_WIDEOP_OPCODE_COMMON(OP, MF2, E32):                             \
  case CASE_FP_WIDEOP_OPCODE_COMMON(OP, M1, E16):                              \
  case CASE_FP_WIDEOP_OPCODE_COMMON(OP, M1, E32):                              \
  case CASE_FP_WIDEOP_OPCODE_COMMON(OP, M2, E16):                              \
  case CASE_FP_WIDEOP_OPCODE_COMMON(OP, M2, E32):                              \
  case CASE_FP_WIDEOP_OPCODE_COMMON(OP, M4, E16):                              \
  case CASE_FP_WIDEOP_OPCODE_COMMON(OP, M4, E32)                               \

#define CASE_FP_WIDEOP_CHANGE_OPCODE_COMMON(OP, LMUL, SEW)                     \
  case Capstone::PseudoV##OP##_##LMUL##_##SEW##_TIED:                             \
    NewOpc = Capstone::PseudoV##OP##_##LMUL##_##SEW;                              \
    break;

#define CASE_FP_WIDEOP_CHANGE_OPCODE_LMULS(OP)                                 \
  CASE_FP_WIDEOP_CHANGE_OPCODE_COMMON(OP, MF4, E16)                            \
  CASE_FP_WIDEOP_CHANGE_OPCODE_COMMON(OP, MF2, E16)                            \
  CASE_FP_WIDEOP_CHANGE_OPCODE_COMMON(OP, MF2, E32)                            \
  CASE_FP_WIDEOP_CHANGE_OPCODE_COMMON(OP, M1, E16)                             \
  CASE_FP_WIDEOP_CHANGE_OPCODE_COMMON(OP, M1, E32)                             \
  CASE_FP_WIDEOP_CHANGE_OPCODE_COMMON(OP, M2, E16)                             \
  CASE_FP_WIDEOP_CHANGE_OPCODE_COMMON(OP, M2, E32)                             \
  CASE_FP_WIDEOP_CHANGE_OPCODE_COMMON(OP, M4, E16)                             \
  CASE_FP_WIDEOP_CHANGE_OPCODE_COMMON(OP, M4, E32)                             \
// clang-format on

MachineInstr *CapstoneInstrInfo::convertToThreeAddress(MachineInstr &MI,
                                                    LiveVariables *LV,
                                                    LiveIntervals *LIS) const {
  MachineInstrBuilder MIB;
  switch (MI.getOpcode()) {
  default:
    return nullptr;
  case CASE_FP_WIDEOP_OPCODE_LMULS(FWADD_WV):
  case CASE_FP_WIDEOP_OPCODE_LMULS(FWSUB_WV): {
    assert(CapstoneII::hasVecPolicyOp(MI.getDesc().TSFlags) &&
           MI.getNumExplicitOperands() == 7 &&
           "Expect 7 explicit operands rd, rs2, rs1, rm, vl, sew, policy");
    // If the tail policy is undisturbed we can't convert.
    if ((MI.getOperand(CapstoneII::getVecPolicyOpNum(MI.getDesc())).getImm() &
         1) == 0)
      return nullptr;
    // clang-format off
    unsigned NewOpc;
    switch (MI.getOpcode()) {
    default:
      llvm_unreachable("Unexpected opcode");
    CASE_FP_WIDEOP_CHANGE_OPCODE_LMULS(FWADD_WV)
    CASE_FP_WIDEOP_CHANGE_OPCODE_LMULS(FWSUB_WV)
    }
    // clang-format on

    MachineBasicBlock &MBB = *MI.getParent();
    MIB = BuildMI(MBB, MI, MI.getDebugLoc(), get(NewOpc))
              .add(MI.getOperand(0))
              .addReg(MI.getOperand(0).getReg(), RegState::Undef)
              .add(MI.getOperand(1))
              .add(MI.getOperand(2))
              .add(MI.getOperand(3))
              .add(MI.getOperand(4))
              .add(MI.getOperand(5))
              .add(MI.getOperand(6));
    break;
  }
  case CASE_WIDEOP_OPCODE_LMULS(WADD_WV):
  case CASE_WIDEOP_OPCODE_LMULS(WADDU_WV):
  case CASE_WIDEOP_OPCODE_LMULS(WSUB_WV):
  case CASE_WIDEOP_OPCODE_LMULS(WSUBU_WV): {
    // If the tail policy is undisturbed we can't convert.
    assert(CapstoneII::hasVecPolicyOp(MI.getDesc().TSFlags) &&
           MI.getNumExplicitOperands() == 6);
    if ((MI.getOperand(CapstoneII::getVecPolicyOpNum(MI.getDesc())).getImm() &
         1) == 0)
      return nullptr;

    // clang-format off
    unsigned NewOpc;
    switch (MI.getOpcode()) {
    default:
      llvm_unreachable("Unexpected opcode");
    CASE_WIDEOP_CHANGE_OPCODE_LMULS(WADD_WV)
    CASE_WIDEOP_CHANGE_OPCODE_LMULS(WADDU_WV)
    CASE_WIDEOP_CHANGE_OPCODE_LMULS(WSUB_WV)
    CASE_WIDEOP_CHANGE_OPCODE_LMULS(WSUBU_WV)
    }
    // clang-format on

    MachineBasicBlock &MBB = *MI.getParent();
    MIB = BuildMI(MBB, MI, MI.getDebugLoc(), get(NewOpc))
              .add(MI.getOperand(0))
              .addReg(MI.getOperand(0).getReg(), RegState::Undef)
              .add(MI.getOperand(1))
              .add(MI.getOperand(2))
              .add(MI.getOperand(3))
              .add(MI.getOperand(4))
              .add(MI.getOperand(5));
    break;
  }
  }
  MIB.copyImplicitOps(MI);

  if (LV) {
    unsigned NumOps = MI.getNumOperands();
    for (unsigned I = 1; I < NumOps; ++I) {
      MachineOperand &Op = MI.getOperand(I);
      if (Op.isReg() && Op.isKill())
        LV->replaceKillInstruction(Op.getReg(), MI, *MIB);
    }
  }

  if (LIS) {
    SlotIndex Idx = LIS->ReplaceMachineInstrInMaps(MI, *MIB);

    if (MI.getOperand(0).isEarlyClobber()) {
      // Use operand 1 was tied to early-clobber def operand 0, so its live
      // interval could have ended at an early-clobber slot. Now they are not
      // tied we need to update it to the normal register slot.
      LiveInterval &LI = LIS->getInterval(MI.getOperand(1).getReg());
      LiveRange::Segment *S = LI.getSegmentContaining(Idx);
      if (S->end == Idx.getRegSlot(true))
        S->end = Idx.getRegSlot();
    }
  }

  return MIB;
}

#undef CASE_WIDEOP_OPCODE_COMMON
#undef CASE_WIDEOP_OPCODE_LMULS
#undef CASE_WIDEOP_CHANGE_OPCODE_COMMON
#undef CASE_WIDEOP_CHANGE_OPCODE_LMULS
#undef CASE_FP_WIDEOP_OPCODE_COMMON
#undef CASE_FP_WIDEOP_OPCODE_LMULS
#undef CASE_FP_WIDEOP_CHANGE_OPCODE_COMMON
#undef CASE_FP_WIDEOP_CHANGE_OPCODE_LMULS

void CapstoneInstrInfo::mulImm(MachineFunction &MF, MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator II, const DebugLoc &DL,
                            Register DestReg, uint32_t Amount,
                            MachineInstr::MIFlag Flag) const {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  if (llvm::has_single_bit<uint32_t>(Amount)) {
    uint32_t ShiftAmount = Log2_32(Amount);
    if (ShiftAmount == 0)
      return;
    BuildMI(MBB, II, DL, get(Capstone::SLLI), DestReg)
        .addReg(DestReg, RegState::Kill)
        .addImm(ShiftAmount)
        .setMIFlag(Flag);
  } else if (STI.hasShlAdd(3) &&
             ((Amount % 3 == 0 && isPowerOf2_64(Amount / 3)) ||
              (Amount % 5 == 0 && isPowerOf2_64(Amount / 5)) ||
              (Amount % 9 == 0 && isPowerOf2_64(Amount / 9)))) {
    // We can use Zba SHXADD+SLLI instructions for multiply in some cases.
    unsigned Opc;
    uint32_t ShiftAmount;
    if (Amount % 9 == 0) {
      Opc = Capstone::SH3ADD;
      ShiftAmount = Log2_64(Amount / 9);
    } else if (Amount % 5 == 0) {
      Opc = Capstone::SH2ADD;
      ShiftAmount = Log2_64(Amount / 5);
    } else if (Amount % 3 == 0) {
      Opc = Capstone::SH1ADD;
      ShiftAmount = Log2_64(Amount / 3);
    } else {
      llvm_unreachable("implied by if-clause");
    }
    if (ShiftAmount)
      BuildMI(MBB, II, DL, get(Capstone::SLLI), DestReg)
          .addReg(DestReg, RegState::Kill)
          .addImm(ShiftAmount)
          .setMIFlag(Flag);
    BuildMI(MBB, II, DL, get(Opc), DestReg)
        .addReg(DestReg, RegState::Kill)
        .addReg(DestReg)
        .setMIFlag(Flag);
  } else if (llvm::has_single_bit<uint32_t>(Amount - 1)) {
    Register ScaledRegister = MRI.createVirtualRegister(&Capstone::GPRRegClass);
    uint32_t ShiftAmount = Log2_32(Amount - 1);
    BuildMI(MBB, II, DL, get(Capstone::SLLI), ScaledRegister)
        .addReg(DestReg)
        .addImm(ShiftAmount)
        .setMIFlag(Flag);
    BuildMI(MBB, II, DL, get(Capstone::ADD), DestReg)
        .addReg(ScaledRegister, RegState::Kill)
        .addReg(DestReg, RegState::Kill)
        .setMIFlag(Flag);
  } else if (llvm::has_single_bit<uint32_t>(Amount + 1)) {
    Register ScaledRegister = MRI.createVirtualRegister(&Capstone::GPRRegClass);
    uint32_t ShiftAmount = Log2_32(Amount + 1);
    BuildMI(MBB, II, DL, get(Capstone::SLLI), ScaledRegister)
        .addReg(DestReg)
        .addImm(ShiftAmount)
        .setMIFlag(Flag);
    BuildMI(MBB, II, DL, get(Capstone::SUB), DestReg)
        .addReg(ScaledRegister, RegState::Kill)
        .addReg(DestReg, RegState::Kill)
        .setMIFlag(Flag);
  } else if (STI.hasStdExtZmmul()) {
    Register N = MRI.createVirtualRegister(&Capstone::GPRRegClass);
    movImm(MBB, II, DL, N, Amount, Flag);
    BuildMI(MBB, II, DL, get(Capstone::MUL), DestReg)
        .addReg(DestReg, RegState::Kill)
        .addReg(N, RegState::Kill)
        .setMIFlag(Flag);
  } else {
    Register Acc;
    uint32_t PrevShiftAmount = 0;
    for (uint32_t ShiftAmount = 0; Amount >> ShiftAmount; ShiftAmount++) {
      if (Amount & (1U << ShiftAmount)) {
        if (ShiftAmount)
          BuildMI(MBB, II, DL, get(Capstone::SLLI), DestReg)
              .addReg(DestReg, RegState::Kill)
              .addImm(ShiftAmount - PrevShiftAmount)
              .setMIFlag(Flag);
        if (Amount >> (ShiftAmount + 1)) {
          // If we don't have an accmulator yet, create it and copy DestReg.
          if (!Acc) {
            Acc = MRI.createVirtualRegister(&Capstone::GPRRegClass);
            BuildMI(MBB, II, DL, get(TargetOpcode::COPY), Acc)
                .addReg(DestReg)
                .setMIFlag(Flag);
          } else {
            BuildMI(MBB, II, DL, get(Capstone::ADD), Acc)
                .addReg(Acc, RegState::Kill)
                .addReg(DestReg)
                .setMIFlag(Flag);
          }
        }
        PrevShiftAmount = ShiftAmount;
      }
    }
    assert(Acc && "Expected valid accumulator");
    BuildMI(MBB, II, DL, get(Capstone::ADD), DestReg)
        .addReg(DestReg, RegState::Kill)
        .addReg(Acc, RegState::Kill)
        .setMIFlag(Flag);
  }
}

ArrayRef<std::pair<MachineMemOperand::Flags, const char *>>
CapstoneInstrInfo::getSerializableMachineMemOperandTargetFlags() const {
  static const std::pair<MachineMemOperand::Flags, const char *> TargetFlags[] =
      {{MONontemporalBit0, "capstone-nontemporal-domain-bit-0"},
       {MONontemporalBit1, "capstone-nontemporal-domain-bit-1"}};
  return ArrayRef(TargetFlags);
}

unsigned CapstoneInstrInfo::getTailDuplicateSize(CodeGenOptLevel OptLevel) const {
  return OptLevel >= CodeGenOptLevel::Aggressive
             ? STI.getTailDupAggressiveThreshold()
             : 2;
}

bool Capstone::isRVVSpill(const MachineInstr &MI) {
  // RVV lacks any support for immediate addressing for stack addresses, so be
  // conservative.
  unsigned Opcode = MI.getOpcode();
  if (!CapstoneVPseudosTable::getPseudoInfo(Opcode) &&
      !getLMULForRVVWholeLoadStore(Opcode) && !isRVVSpillForZvlsseg(Opcode))
    return false;
  return true;
}

std::optional<std::pair<unsigned, unsigned>>
Capstone::isRVVSpillForZvlsseg(unsigned Opcode) {
  switch (Opcode) {
  default:
    return std::nullopt;
  case Capstone::PseudoVSPILL2_M1:
  case Capstone::PseudoVRELOAD2_M1:
    return std::make_pair(2u, 1u);
  case Capstone::PseudoVSPILL2_M2:
  case Capstone::PseudoVRELOAD2_M2:
    return std::make_pair(2u, 2u);
  case Capstone::PseudoVSPILL2_M4:
  case Capstone::PseudoVRELOAD2_M4:
    return std::make_pair(2u, 4u);
  case Capstone::PseudoVSPILL3_M1:
  case Capstone::PseudoVRELOAD3_M1:
    return std::make_pair(3u, 1u);
  case Capstone::PseudoVSPILL3_M2:
  case Capstone::PseudoVRELOAD3_M2:
    return std::make_pair(3u, 2u);
  case Capstone::PseudoVSPILL4_M1:
  case Capstone::PseudoVRELOAD4_M1:
    return std::make_pair(4u, 1u);
  case Capstone::PseudoVSPILL4_M2:
  case Capstone::PseudoVRELOAD4_M2:
    return std::make_pair(4u, 2u);
  case Capstone::PseudoVSPILL5_M1:
  case Capstone::PseudoVRELOAD5_M1:
    return std::make_pair(5u, 1u);
  case Capstone::PseudoVSPILL6_M1:
  case Capstone::PseudoVRELOAD6_M1:
    return std::make_pair(6u, 1u);
  case Capstone::PseudoVSPILL7_M1:
  case Capstone::PseudoVRELOAD7_M1:
    return std::make_pair(7u, 1u);
  case Capstone::PseudoVSPILL8_M1:
  case Capstone::PseudoVRELOAD8_M1:
    return std::make_pair(8u, 1u);
  }
}

bool Capstone::hasEqualFRM(const MachineInstr &MI1, const MachineInstr &MI2) {
  int16_t MI1FrmOpIdx =
      Capstone::getNamedOperandIdx(MI1.getOpcode(), Capstone::OpName::frm);
  int16_t MI2FrmOpIdx =
      Capstone::getNamedOperandIdx(MI2.getOpcode(), Capstone::OpName::frm);
  if (MI1FrmOpIdx < 0 || MI2FrmOpIdx < 0)
    return false;
  MachineOperand FrmOp1 = MI1.getOperand(MI1FrmOpIdx);
  MachineOperand FrmOp2 = MI2.getOperand(MI2FrmOpIdx);
  return FrmOp1.getImm() == FrmOp2.getImm();
}

std::optional<unsigned>
Capstone::getVectorLowDemandedScalarBits(unsigned Opcode, unsigned Log2SEW) {
  switch (Opcode) {
  default:
    return std::nullopt;

  // 11.6. Vector Single-Width Shift Instructions
  case Capstone::VSLL_VX:
  case Capstone::VSRL_VX:
  case Capstone::VSRA_VX:
  // 12.4. Vector Single-Width Scaling Shift Instructions
  case Capstone::VSSRL_VX:
  case Capstone::VSSRA_VX:
  // Zvbb
  case Capstone::VROL_VX:
  case Capstone::VROR_VX:
    // Only the low lg2(SEW) bits of the shift-amount value are used.
    return Log2SEW;

  // 11.7 Vector Narrowing Integer Right Shift Instructions
  case Capstone::VNSRL_WX:
  case Capstone::VNSRA_WX:
  // 12.5. Vector Narrowing Fixed-Point Clip Instructions
  case Capstone::VNCLIPU_WX:
  case Capstone::VNCLIP_WX:
  // Zvbb
  case Capstone::VWSLL_VX:
    // Only the low lg2(2*SEW) bits of the shift-amount value are used.
    return Log2SEW + 1;

  // 11.1. Vector Single-Width Integer Add and Subtract
  case Capstone::VADD_VX:
  case Capstone::VSUB_VX:
  case Capstone::VRSUB_VX:
  // 11.2. Vector Widening Integer Add/Subtract
  case Capstone::VWADDU_VX:
  case Capstone::VWSUBU_VX:
  case Capstone::VWADD_VX:
  case Capstone::VWSUB_VX:
  case Capstone::VWADDU_WX:
  case Capstone::VWSUBU_WX:
  case Capstone::VWADD_WX:
  case Capstone::VWSUB_WX:
  // 11.4. Vector Integer Add-with-Carry / Subtract-with-Borrow Instructions
  case Capstone::VADC_VXM:
  case Capstone::VADC_VIM:
  case Capstone::VMADC_VXM:
  case Capstone::VMADC_VIM:
  case Capstone::VMADC_VX:
  case Capstone::VSBC_VXM:
  case Capstone::VMSBC_VXM:
  case Capstone::VMSBC_VX:
  // 11.5 Vector Bitwise Logical Instructions
  case Capstone::VAND_VX:
  case Capstone::VOR_VX:
  case Capstone::VXOR_VX:
  // 11.8. Vector Integer Compare Instructions
  case Capstone::VMSEQ_VX:
  case Capstone::VMSNE_VX:
  case Capstone::VMSLTU_VX:
  case Capstone::VMSLT_VX:
  case Capstone::VMSLEU_VX:
  case Capstone::VMSLE_VX:
  case Capstone::VMSGTU_VX:
  case Capstone::VMSGT_VX:
  // 11.9. Vector Integer Min/Max Instructions
  case Capstone::VMINU_VX:
  case Capstone::VMIN_VX:
  case Capstone::VMAXU_VX:
  case Capstone::VMAX_VX:
  // 11.10. Vector Single-Width Integer Multiply Instructions
  case Capstone::VMUL_VX:
  case Capstone::VMULH_VX:
  case Capstone::VMULHU_VX:
  case Capstone::VMULHSU_VX:
  // 11.11. Vector Integer Divide Instructions
  case Capstone::VDIVU_VX:
  case Capstone::VDIV_VX:
  case Capstone::VREMU_VX:
  case Capstone::VREM_VX:
  // 11.12. Vector Widening Integer Multiply Instructions
  case Capstone::VWMUL_VX:
  case Capstone::VWMULU_VX:
  case Capstone::VWMULSU_VX:
  // 11.13. Vector Single-Width Integer Multiply-Add Instructions
  case Capstone::VMACC_VX:
  case Capstone::VNMSAC_VX:
  case Capstone::VMADD_VX:
  case Capstone::VNMSUB_VX:
  // 11.14. Vector Widening Integer Multiply-Add Instructions
  case Capstone::VWMACCU_VX:
  case Capstone::VWMACC_VX:
  case Capstone::VWMACCSU_VX:
  case Capstone::VWMACCUS_VX:
  // 11.15. Vector Integer Merge Instructions
  case Capstone::VMERGE_VXM:
  // 11.16. Vector Integer Move Instructions
  case Capstone::VMV_V_X:
  // 12.1. Vector Single-Width Saturating Add and Subtract
  case Capstone::VSADDU_VX:
  case Capstone::VSADD_VX:
  case Capstone::VSSUBU_VX:
  case Capstone::VSSUB_VX:
  // 12.2. Vector Single-Width Averaging Add and Subtract
  case Capstone::VAADDU_VX:
  case Capstone::VAADD_VX:
  case Capstone::VASUBU_VX:
  case Capstone::VASUB_VX:
  // 12.3. Vector Single-Width Fractional Multiply with Rounding and Saturation
  case Capstone::VSMUL_VX:
  // 16.1. Integer Scalar Move Instructions
  case Capstone::VMV_S_X:
  // Zvbb
  case Capstone::VANDN_VX:
    return 1U << Log2SEW;
  }
}

unsigned Capstone::getRVVMCOpcode(unsigned RVVPseudoOpcode) {
  const CapstoneVPseudosTable::PseudoInfo *RVV =
      CapstoneVPseudosTable::getPseudoInfo(RVVPseudoOpcode);
  if (!RVV)
    return 0;
  return RVV->BaseInstr;
}

unsigned Capstone::getDestLog2EEW(const MCInstrDesc &Desc, unsigned Log2SEW) {
  unsigned DestEEW =
      (Desc.TSFlags & CapstoneII::DestEEWMask) >> CapstoneII::DestEEWShift;
  // EEW = 1
  if (DestEEW == 0)
    return 0;
  // EEW = SEW * n
  unsigned Scaled = Log2SEW + (DestEEW - 1);
  assert(Scaled >= 3 && Scaled <= 6);
  return Scaled;
}

static std::optional<int64_t> getEffectiveImm(const MachineOperand &MO) {
  assert(MO.isImm() || MO.getReg().isVirtual());
  if (MO.isImm())
    return MO.getImm();
  const MachineInstr *Def =
      MO.getParent()->getMF()->getRegInfo().getVRegDef(MO.getReg());
  int64_t Imm;
  if (isLoadImm(Def, Imm))
    return Imm;
  return std::nullopt;
}

/// Given two VL operands, do we know that LHS <= RHS? Must be used in SSA form.
bool Capstone::isVLKnownLE(const MachineOperand &LHS, const MachineOperand &RHS) {
  assert((LHS.isImm() || LHS.getParent()->getMF()->getRegInfo().isSSA()) &&
         (RHS.isImm() || RHS.getParent()->getMF()->getRegInfo().isSSA()));
  if (LHS.isReg() && RHS.isReg() && LHS.getReg().isVirtual() &&
      LHS.getReg() == RHS.getReg())
    return true;
  if (RHS.isImm() && RHS.getImm() == Capstone::VLMaxSentinel)
    return true;
  if (LHS.isImm() && LHS.getImm() == 0)
    return true;
  if (LHS.isImm() && LHS.getImm() == Capstone::VLMaxSentinel)
    return false;
  std::optional<int64_t> LHSImm = getEffectiveImm(LHS),
                         RHSImm = getEffectiveImm(RHS);
  if (!LHSImm || !RHSImm)
    return false;
  return LHSImm <= RHSImm;
}

namespace {
class CapstonePipelinerLoopInfo : public TargetInstrInfo::PipelinerLoopInfo {
  const MachineInstr *LHS;
  const MachineInstr *RHS;
  SmallVector<MachineOperand, 3> Cond;

public:
  CapstonePipelinerLoopInfo(const MachineInstr *LHS, const MachineInstr *RHS,
                         const SmallVectorImpl<MachineOperand> &Cond)
      : LHS(LHS), RHS(RHS), Cond(Cond.begin(), Cond.end()) {}

  bool shouldIgnoreForPipelining(const MachineInstr *MI) const override {
    // Make the instructions for loop control be placed in stage 0.
    // The predecessors of LHS/RHS are considered by the caller.
    if (LHS && MI == LHS)
      return true;
    if (RHS && MI == RHS)
      return true;
    return false;
  }

  std::optional<bool> createTripCountGreaterCondition(
      int TC, MachineBasicBlock &MBB,
      SmallVectorImpl<MachineOperand> &CondParam) override {
    // A branch instruction will be inserted as "if (Cond) goto epilogue".
    // Cond is normalized for such use.
    // The predecessors of the branch are assumed to have already been inserted.
    CondParam = Cond;
    return {};
  }

  void setPreheader(MachineBasicBlock *NewPreheader) override {}

  void adjustTripCount(int TripCountAdjust) override {}
};
} // namespace

std::unique_ptr<TargetInstrInfo::PipelinerLoopInfo>
CapstoneInstrInfo::analyzeLoopForPipelining(MachineBasicBlock *LoopBB) const {
  MachineBasicBlock *TBB = nullptr, *FBB = nullptr;
  SmallVector<MachineOperand, 4> Cond;
  if (analyzeBranch(*LoopBB, TBB, FBB, Cond, /*AllowModify=*/false))
    return nullptr;

  // Infinite loops are not supported
  if (TBB == LoopBB && FBB == LoopBB)
    return nullptr;

  // Must be conditional branch
  if (FBB == nullptr)
    return nullptr;

  assert((TBB == LoopBB || FBB == LoopBB) &&
         "The Loop must be a single-basic-block loop");

  // Normalization for createTripCountGreaterCondition()
  if (TBB == LoopBB)
    reverseBranchCondition(Cond);

  const MachineRegisterInfo &MRI = LoopBB->getParent()->getRegInfo();
  auto FindRegDef = [&MRI](MachineOperand &Op) -> const MachineInstr * {
    if (!Op.isReg())
      return nullptr;
    Register Reg = Op.getReg();
    if (!Reg.isVirtual())
      return nullptr;
    return MRI.getVRegDef(Reg);
  };

  const MachineInstr *LHS = FindRegDef(Cond[1]);
  const MachineInstr *RHS = FindRegDef(Cond[2]);
  if (LHS && LHS->isPHI())
    return nullptr;
  if (RHS && RHS->isPHI())
    return nullptr;

  return std::make_unique<CapstonePipelinerLoopInfo>(LHS, RHS, Cond);
}

// FIXME: We should remove this if we have a default generic scheduling model.
bool CapstoneInstrInfo::isHighLatencyDef(int Opc) const {
  unsigned RVVMCOpcode = Capstone::getRVVMCOpcode(Opc);
  Opc = RVVMCOpcode ? RVVMCOpcode : Opc;
  switch (Opc) {
  default:
    return false;
  // Integer div/rem.
  case Capstone::DIV:
  case Capstone::DIVW:
  case Capstone::DIVU:
  case Capstone::DIVUW:
  case Capstone::REM:
  case Capstone::REMW:
  case Capstone::REMU:
  case Capstone::REMUW:
  // Floating-point div/sqrt.
  case Capstone::FDIV_H:
  case Capstone::FDIV_S:
  case Capstone::FDIV_D:
  case Capstone::FDIV_H_INX:
  case Capstone::FDIV_S_INX:
  case Capstone::FDIV_D_INX:
  case Capstone::FDIV_D_IN32X:
  case Capstone::FSQRT_H:
  case Capstone::FSQRT_S:
  case Capstone::FSQRT_D:
  case Capstone::FSQRT_H_INX:
  case Capstone::FSQRT_S_INX:
  case Capstone::FSQRT_D_INX:
  case Capstone::FSQRT_D_IN32X:
  // Vector integer div/rem
  case Capstone::VDIV_VV:
  case Capstone::VDIV_VX:
  case Capstone::VDIVU_VV:
  case Capstone::VDIVU_VX:
  case Capstone::VREM_VV:
  case Capstone::VREM_VX:
  case Capstone::VREMU_VV:
  case Capstone::VREMU_VX:
  // Vector floating-point div/sqrt.
  case Capstone::VFDIV_VV:
  case Capstone::VFDIV_VF:
  case Capstone::VFRDIV_VF:
  case Capstone::VFSQRT_V:
  case Capstone::VFRSQRT7_V:
    return true;
  }
}
