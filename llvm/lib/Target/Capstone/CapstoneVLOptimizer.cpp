//===-------------- CapstoneVLOptimizer.cpp - VL Optimizer -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// This pass reduces the VL where possible at the MI level, before VSETVLI
// instructions are inserted.
//
// The purpose of this optimization is to make the VL argument, for instructions
// that have a VL argument, as small as possible.
//
// This is split into a sparse dataflow analysis where we determine what VL is
// demanded by each instruction first, and then afterwards try to reduce the VL
// of each instruction if it demands less than its VL operand.
//
// The analysis is explained in more detail in the 2025 EuroLLVM Developers'
// Meeting talk "Accidental Dataflow Analysis: Extending the Capstone VL
// Optimizer", which is available on YouTube at
// https://www.youtube.com/watch?v=Mfb5fRSdJAc
//
// The slides for the talk are available at
// https://llvm.org/devmtg/2025-04/slides/technical_talk/lau_accidental_dataflow.pdf
//
//===---------------------------------------------------------------------===//

#include "Capstone.h"
#include "CapstoneSubtarget.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "capstone-vl-optimizer"
#define PASS_NAME "Capstone VL Optimizer"

namespace {

/// Wrapper around MachineOperand that defaults to immediate 0.
struct DemandedVL {
  MachineOperand VL;
  DemandedVL() : VL(MachineOperand::CreateImm(0)) {}
  DemandedVL(MachineOperand VL) : VL(VL) {}
  static DemandedVL vlmax() {
    return DemandedVL(MachineOperand::CreateImm(Capstone::VLMaxSentinel));
  }
  bool operator!=(const DemandedVL &Other) const {
    return !VL.isIdenticalTo(Other.VL);
  }

  DemandedVL max(const DemandedVL &X) const {
    if (Capstone::isVLKnownLE(VL, X.VL))
      return X;
    if (Capstone::isVLKnownLE(X.VL, VL))
      return *this;
    return DemandedVL::vlmax();
  }
};

class CapstoneVLOptimizer : public MachineFunctionPass {
  const MachineRegisterInfo *MRI;
  const MachineDominatorTree *MDT;
  const TargetInstrInfo *TII;

public:
  static char ID;

  CapstoneVLOptimizer() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<MachineDominatorTreeWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override { return PASS_NAME; }

private:
  DemandedVL getMinimumVLForUser(const MachineOperand &UserOp) const;
  /// Returns true if the users of \p MI have compatible EEWs and SEWs.
  bool checkUsers(const MachineInstr &MI) const;
  bool tryReduceVL(MachineInstr &MI) const;
  bool isCandidate(const MachineInstr &MI) const;
  void transfer(const MachineInstr &MI);

  /// For a given instruction, records what elements of it are demanded by
  /// downstream users.
  DenseMap<const MachineInstr *, DemandedVL> DemandedVLs;
  SetVector<const MachineInstr *> Worklist;

  /// \returns all vector virtual registers that \p MI uses.
  auto virtual_vec_uses(const MachineInstr &MI) const {
    return make_filter_range(MI.uses(), [this](const MachineOperand &MO) {
      return MO.isReg() && MO.getReg().isVirtual() &&
             CapstoneRegisterInfo::isRVVRegClass(MRI->getRegClass(MO.getReg()));
    });
  }
};

/// Represents the EMUL and EEW of a MachineOperand.
struct OperandInfo {
  // Represent as 1,2,4,8, ... and fractional indicator. This is because
  // EMUL can take on values that don't map to CapstoneVType::VLMUL values exactly.
  // For example, a mask operand can have an EMUL less than MF8.
  // If nullopt, then EMUL isn't used (i.e. only a single scalar is read).
  std::optional<std::pair<unsigned, bool>> EMUL;

  unsigned Log2EEW;

  OperandInfo(CapstoneVType::VLMUL EMUL, unsigned Log2EEW)
      : EMUL(CapstoneVType::decodeVLMUL(EMUL)), Log2EEW(Log2EEW) {}

  OperandInfo(std::pair<unsigned, bool> EMUL, unsigned Log2EEW)
      : EMUL(EMUL), Log2EEW(Log2EEW) {}

  OperandInfo(unsigned Log2EEW) : Log2EEW(Log2EEW) {}

  OperandInfo() = delete;

  /// Return true if the EMUL and EEW produced by \p Def is compatible with the
  /// EMUL and EEW used by \p User.
  static bool areCompatible(const OperandInfo &Def, const OperandInfo &User) {
    if (Def.Log2EEW != User.Log2EEW)
      return false;
    if (User.EMUL && Def.EMUL != User.EMUL)
      return false;
    return true;
  }

  void print(raw_ostream &OS) const {
    if (EMUL) {
      OS << "EMUL: m";
      if (EMUL->second)
        OS << "f";
      OS << EMUL->first;
    } else
      OS << "EMUL: none\n";
    OS << ", EEW: " << (1 << Log2EEW);
  }
};

} // end anonymous namespace

char CapstoneVLOptimizer::ID = 0;
INITIALIZE_PASS_BEGIN(CapstoneVLOptimizer, DEBUG_TYPE, PASS_NAME, false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_END(CapstoneVLOptimizer, DEBUG_TYPE, PASS_NAME, false, false)

FunctionPass *llvm::createCapstoneVLOptimizerPass() {
  return new CapstoneVLOptimizer();
}

LLVM_ATTRIBUTE_UNUSED
static raw_ostream &operator<<(raw_ostream &OS, const OperandInfo &OI) {
  OI.print(OS);
  return OS;
}

LLVM_ATTRIBUTE_UNUSED
static raw_ostream &operator<<(raw_ostream &OS,
                               const std::optional<OperandInfo> &OI) {
  if (OI)
    OI->print(OS);
  else
    OS << "nullopt";
  return OS;
}

/// Return EMUL = (EEW / SEW) * LMUL where EEW comes from Log2EEW and LMUL and
/// SEW are from the TSFlags of MI.
static std::pair<unsigned, bool>
getEMULEqualsEEWDivSEWTimesLMUL(unsigned Log2EEW, const MachineInstr &MI) {
  CapstoneVType::VLMUL MIVLMUL = CapstoneII::getLMul(MI.getDesc().TSFlags);
  auto [MILMUL, MILMULIsFractional] = CapstoneVType::decodeVLMUL(MIVLMUL);
  unsigned MILog2SEW =
      MI.getOperand(CapstoneII::getSEWOpNum(MI.getDesc())).getImm();

  // Mask instructions will have 0 as the SEW operand. But the LMUL of these
  // instructions is calculated is as if the SEW operand was 3 (e8).
  if (MILog2SEW == 0)
    MILog2SEW = 3;

  unsigned MISEW = 1 << MILog2SEW;

  unsigned EEW = 1 << Log2EEW;
  // Calculate (EEW/SEW)*LMUL preserving fractions less than 1. Use GCD
  // to put fraction in simplest form.
  unsigned Num = EEW, Denom = MISEW;
  int GCD = MILMULIsFractional ? std::gcd(Num, Denom * MILMUL)
                               : std::gcd(Num * MILMUL, Denom);
  Num = MILMULIsFractional ? Num / GCD : Num * MILMUL / GCD;
  Denom = MILMULIsFractional ? Denom * MILMUL / GCD : Denom / GCD;
  return std::make_pair(Num > Denom ? Num : Denom, Denom > Num);
}

/// Dest has EEW=SEW. Source EEW=SEW/Factor (i.e. F2 => EEW/2).
/// SEW comes from TSFlags of MI.
static unsigned getIntegerExtensionOperandEEW(unsigned Factor,
                                              const MachineInstr &MI,
                                              const MachineOperand &MO) {
  unsigned MILog2SEW =
      MI.getOperand(CapstoneII::getSEWOpNum(MI.getDesc())).getImm();

  if (MO.getOperandNo() == 0)
    return MILog2SEW;

  unsigned MISEW = 1 << MILog2SEW;
  unsigned EEW = MISEW / Factor;
  unsigned Log2EEW = Log2_32(EEW);

  return Log2EEW;
}

#define VSEG_CASES(Prefix, EEW)                                                \
  Capstone::Prefix##SEG2E##EEW##_V:                                               \
  case Capstone::Prefix##SEG3E##EEW##_V:                                          \
  case Capstone::Prefix##SEG4E##EEW##_V:                                          \
  case Capstone::Prefix##SEG5E##EEW##_V:                                          \
  case Capstone::Prefix##SEG6E##EEW##_V:                                          \
  case Capstone::Prefix##SEG7E##EEW##_V:                                          \
  case Capstone::Prefix##SEG8E##EEW##_V
#define VSSEG_CASES(EEW)    VSEG_CASES(VS, EEW)
#define VSSSEG_CASES(EEW)   VSEG_CASES(VSS, EEW)
#define VSUXSEG_CASES(EEW)  VSEG_CASES(VSUX, I##EEW)
#define VSOXSEG_CASES(EEW)  VSEG_CASES(VSOX, I##EEW)

static std::optional<unsigned> getOperandLog2EEW(const MachineOperand &MO) {
  const MachineInstr &MI = *MO.getParent();
  const MCInstrDesc &Desc = MI.getDesc();
  const CapstoneVPseudosTable::PseudoInfo *RVV =
      CapstoneVPseudosTable::getPseudoInfo(MI.getOpcode());
  assert(RVV && "Could not find MI in PseudoTable");

  // MI has a SEW associated with it. The RVV specification defines
  // the EEW of each operand and definition in relation to MI.SEW.
  unsigned MILog2SEW = MI.getOperand(CapstoneII::getSEWOpNum(Desc)).getImm();

  const bool HasPassthru = CapstoneII::isFirstDefTiedToFirstUse(Desc);
  const bool IsTied = CapstoneII::isTiedPseudo(Desc.TSFlags);

  bool IsMODef = MO.getOperandNo() == 0 ||
                 (HasPassthru && MO.getOperandNo() == MI.getNumExplicitDefs());

  // All mask operands have EEW=1
  const MCOperandInfo &Info = Desc.operands()[MO.getOperandNo()];
  if (Info.OperandType == MCOI::OPERAND_REGISTER &&
      Info.RegClass == Capstone::VMV0RegClassID)
    return 0;

  // switch against BaseInstr to reduce number of cases that need to be
  // considered.
  switch (RVV->BaseInstr) {

  // 6. Configuration-Setting Instructions
  // Configuration setting instructions do not read or write vector registers
  case Capstone::VSETIVLI:
  case Capstone::VSETVL:
  case Capstone::VSETVLI:
    llvm_unreachable("Configuration setting instructions do not read or write "
                     "vector registers");

  // Vector Loads and Stores
  // Vector Unit-Stride Instructions
  // Vector Strided Instructions
  /// Dest EEW encoded in the instruction
  case Capstone::VLM_V:
  case Capstone::VSM_V:
    return 0;
  case Capstone::VLE8_V:
  case Capstone::VSE8_V:
  case Capstone::VLSE8_V:
  case Capstone::VSSE8_V:
  case VSSEG_CASES(8):
  case VSSSEG_CASES(8):
    return 3;
  case Capstone::VLE16_V:
  case Capstone::VSE16_V:
  case Capstone::VLSE16_V:
  case Capstone::VSSE16_V:
  case VSSEG_CASES(16):
  case VSSSEG_CASES(16):
    return 4;
  case Capstone::VLE32_V:
  case Capstone::VSE32_V:
  case Capstone::VLSE32_V:
  case Capstone::VSSE32_V:
  case VSSEG_CASES(32):
  case VSSSEG_CASES(32):
    return 5;
  case Capstone::VLE64_V:
  case Capstone::VSE64_V:
  case Capstone::VLSE64_V:
  case Capstone::VSSE64_V:
  case VSSEG_CASES(64):
  case VSSSEG_CASES(64):
    return 6;

  // Vector Indexed Instructions
  // vs(o|u)xei<eew>.v
  // Dest/Data (operand 0) EEW=SEW.  Source EEW=<eew>.
  case Capstone::VLUXEI8_V:
  case Capstone::VLOXEI8_V:
  case Capstone::VSUXEI8_V:
  case Capstone::VSOXEI8_V:
  case VSUXSEG_CASES(8):
  case VSOXSEG_CASES(8): {
    if (MO.getOperandNo() == 0)
      return MILog2SEW;
    return 3;
  }
  case Capstone::VLUXEI16_V:
  case Capstone::VLOXEI16_V:
  case Capstone::VSUXEI16_V:
  case Capstone::VSOXEI16_V:
  case VSUXSEG_CASES(16):
  case VSOXSEG_CASES(16): {
    if (MO.getOperandNo() == 0)
      return MILog2SEW;
    return 4;
  }
  case Capstone::VLUXEI32_V:
  case Capstone::VLOXEI32_V:
  case Capstone::VSUXEI32_V:
  case Capstone::VSOXEI32_V:
  case VSUXSEG_CASES(32):
  case VSOXSEG_CASES(32): {
    if (MO.getOperandNo() == 0)
      return MILog2SEW;
    return 5;
  }
  case Capstone::VLUXEI64_V:
  case Capstone::VLOXEI64_V:
  case Capstone::VSUXEI64_V:
  case Capstone::VSOXEI64_V:
  case VSUXSEG_CASES(64):
  case VSOXSEG_CASES(64): {
    if (MO.getOperandNo() == 0)
      return MILog2SEW;
    return 6;
  }

  // Vector Integer Arithmetic Instructions
  // Vector Single-Width Integer Add and Subtract
  case Capstone::VADD_VI:
  case Capstone::VADD_VV:
  case Capstone::VADD_VX:
  case Capstone::VSUB_VV:
  case Capstone::VSUB_VX:
  case Capstone::VRSUB_VI:
  case Capstone::VRSUB_VX:
  // Vector Bitwise Logical Instructions
  // Vector Single-Width Shift Instructions
  // EEW=SEW.
  case Capstone::VAND_VI:
  case Capstone::VAND_VV:
  case Capstone::VAND_VX:
  case Capstone::VOR_VI:
  case Capstone::VOR_VV:
  case Capstone::VOR_VX:
  case Capstone::VXOR_VI:
  case Capstone::VXOR_VV:
  case Capstone::VXOR_VX:
  case Capstone::VSLL_VI:
  case Capstone::VSLL_VV:
  case Capstone::VSLL_VX:
  case Capstone::VSRL_VI:
  case Capstone::VSRL_VV:
  case Capstone::VSRL_VX:
  case Capstone::VSRA_VI:
  case Capstone::VSRA_VV:
  case Capstone::VSRA_VX:
  // Vector Integer Min/Max Instructions
  // EEW=SEW.
  case Capstone::VMINU_VV:
  case Capstone::VMINU_VX:
  case Capstone::VMIN_VV:
  case Capstone::VMIN_VX:
  case Capstone::VMAXU_VV:
  case Capstone::VMAXU_VX:
  case Capstone::VMAX_VV:
  case Capstone::VMAX_VX:
  // Vector Single-Width Integer Multiply Instructions
  // Source and Dest EEW=SEW.
  case Capstone::VMUL_VV:
  case Capstone::VMUL_VX:
  case Capstone::VMULH_VV:
  case Capstone::VMULH_VX:
  case Capstone::VMULHU_VV:
  case Capstone::VMULHU_VX:
  case Capstone::VMULHSU_VV:
  case Capstone::VMULHSU_VX:
  // Vector Integer Divide Instructions
  // EEW=SEW.
  case Capstone::VDIVU_VV:
  case Capstone::VDIVU_VX:
  case Capstone::VDIV_VV:
  case Capstone::VDIV_VX:
  case Capstone::VREMU_VV:
  case Capstone::VREMU_VX:
  case Capstone::VREM_VV:
  case Capstone::VREM_VX:
  // Vector Single-Width Integer Multiply-Add Instructions
  // EEW=SEW.
  case Capstone::VMACC_VV:
  case Capstone::VMACC_VX:
  case Capstone::VNMSAC_VV:
  case Capstone::VNMSAC_VX:
  case Capstone::VMADD_VV:
  case Capstone::VMADD_VX:
  case Capstone::VNMSUB_VV:
  case Capstone::VNMSUB_VX:
  // Vector Integer Merge Instructions
  // Vector Integer Add-with-Carry / Subtract-with-Borrow Instructions
  // EEW=SEW, except the mask operand has EEW=1. Mask operand is handled
  // before this switch.
  case Capstone::VMERGE_VIM:
  case Capstone::VMERGE_VVM:
  case Capstone::VMERGE_VXM:
  case Capstone::VADC_VIM:
  case Capstone::VADC_VVM:
  case Capstone::VADC_VXM:
  case Capstone::VSBC_VVM:
  case Capstone::VSBC_VXM:
  // Vector Integer Move Instructions
  // Vector Fixed-Point Arithmetic Instructions
  // Vector Single-Width Saturating Add and Subtract
  // Vector Single-Width Averaging Add and Subtract
  // EEW=SEW.
  case Capstone::VMV_V_I:
  case Capstone::VMV_V_V:
  case Capstone::VMV_V_X:
  case Capstone::VSADDU_VI:
  case Capstone::VSADDU_VV:
  case Capstone::VSADDU_VX:
  case Capstone::VSADD_VI:
  case Capstone::VSADD_VV:
  case Capstone::VSADD_VX:
  case Capstone::VSSUBU_VV:
  case Capstone::VSSUBU_VX:
  case Capstone::VSSUB_VV:
  case Capstone::VSSUB_VX:
  case Capstone::VAADDU_VV:
  case Capstone::VAADDU_VX:
  case Capstone::VAADD_VV:
  case Capstone::VAADD_VX:
  case Capstone::VASUBU_VV:
  case Capstone::VASUBU_VX:
  case Capstone::VASUB_VV:
  case Capstone::VASUB_VX:
  // Vector Single-Width Fractional Multiply with Rounding and Saturation
  // EEW=SEW. The instruction produces 2*SEW product internally but
  // saturates to fit into SEW bits.
  case Capstone::VSMUL_VV:
  case Capstone::VSMUL_VX:
  // Vector Single-Width Scaling Shift Instructions
  // EEW=SEW.
  case Capstone::VSSRL_VI:
  case Capstone::VSSRL_VV:
  case Capstone::VSSRL_VX:
  case Capstone::VSSRA_VI:
  case Capstone::VSSRA_VV:
  case Capstone::VSSRA_VX:
  // Vector Permutation Instructions
  // Integer Scalar Move Instructions
  // Floating-Point Scalar Move Instructions
  // EEW=SEW.
  case Capstone::VMV_X_S:
  case Capstone::VMV_S_X:
  case Capstone::VFMV_F_S:
  case Capstone::VFMV_S_F:
  // Vector Slide Instructions
  // EEW=SEW.
  case Capstone::VSLIDEUP_VI:
  case Capstone::VSLIDEUP_VX:
  case Capstone::VSLIDEDOWN_VI:
  case Capstone::VSLIDEDOWN_VX:
  case Capstone::VSLIDE1UP_VX:
  case Capstone::VFSLIDE1UP_VF:
  case Capstone::VSLIDE1DOWN_VX:
  case Capstone::VFSLIDE1DOWN_VF:
  // Vector Register Gather Instructions
  // EEW=SEW. For mask operand, EEW=1.
  case Capstone::VRGATHER_VI:
  case Capstone::VRGATHER_VV:
  case Capstone::VRGATHER_VX:
  // Vector Element Index Instruction
  case Capstone::VID_V:
  // Vector Single-Width Floating-Point Add/Subtract Instructions
  case Capstone::VFADD_VF:
  case Capstone::VFADD_VV:
  case Capstone::VFSUB_VF:
  case Capstone::VFSUB_VV:
  case Capstone::VFRSUB_VF:
  // Vector Single-Width Floating-Point Multiply/Divide Instructions
  case Capstone::VFMUL_VF:
  case Capstone::VFMUL_VV:
  case Capstone::VFDIV_VF:
  case Capstone::VFDIV_VV:
  case Capstone::VFRDIV_VF:
  // Vector Single-Width Floating-Point Fused Multiply-Add Instructions
  case Capstone::VFMACC_VV:
  case Capstone::VFMACC_VF:
  case Capstone::VFNMACC_VV:
  case Capstone::VFNMACC_VF:
  case Capstone::VFMSAC_VV:
  case Capstone::VFMSAC_VF:
  case Capstone::VFNMSAC_VV:
  case Capstone::VFNMSAC_VF:
  case Capstone::VFMADD_VV:
  case Capstone::VFMADD_VF:
  case Capstone::VFNMADD_VV:
  case Capstone::VFNMADD_VF:
  case Capstone::VFMSUB_VV:
  case Capstone::VFMSUB_VF:
  case Capstone::VFNMSUB_VV:
  case Capstone::VFNMSUB_VF:
  // Vector Floating-Point Square-Root Instruction
  case Capstone::VFSQRT_V:
  // Vector Floating-Point Reciprocal Square-Root Estimate Instruction
  case Capstone::VFRSQRT7_V:
  // Vector Floating-Point Reciprocal Estimate Instruction
  case Capstone::VFREC7_V:
  // Vector Floating-Point MIN/MAX Instructions
  case Capstone::VFMIN_VF:
  case Capstone::VFMIN_VV:
  case Capstone::VFMAX_VF:
  case Capstone::VFMAX_VV:
  // Vector Floating-Point Sign-Injection Instructions
  case Capstone::VFSGNJ_VF:
  case Capstone::VFSGNJ_VV:
  case Capstone::VFSGNJN_VV:
  case Capstone::VFSGNJN_VF:
  case Capstone::VFSGNJX_VF:
  case Capstone::VFSGNJX_VV:
  // Vector Floating-Point Classify Instruction
  case Capstone::VFCLASS_V:
  // Vector Floating-Point Move Instruction
  case Capstone::VFMV_V_F:
  // Single-Width Floating-Point/Integer Type-Convert Instructions
  case Capstone::VFCVT_XU_F_V:
  case Capstone::VFCVT_X_F_V:
  case Capstone::VFCVT_RTZ_XU_F_V:
  case Capstone::VFCVT_RTZ_X_F_V:
  case Capstone::VFCVT_F_XU_V:
  case Capstone::VFCVT_F_X_V:
  // Vector Floating-Point Merge Instruction
  case Capstone::VFMERGE_VFM:
  // Vector count population in mask vcpop.m
  // vfirst find-first-set mask bit
  case Capstone::VCPOP_M:
  case Capstone::VFIRST_M:
  // Vector Bit-manipulation Instructions (Zvbb)
  // Vector And-Not
  case Capstone::VANDN_VV:
  case Capstone::VANDN_VX:
  // Vector Reverse Bits in Elements
  case Capstone::VBREV_V:
  // Vector Reverse Bits in Bytes
  case Capstone::VBREV8_V:
  // Vector Reverse Bytes
  case Capstone::VREV8_V:
  // Vector Count Leading Zeros
  case Capstone::VCLZ_V:
  // Vector Count Trailing Zeros
  case Capstone::VCTZ_V:
  // Vector Population Count
  case Capstone::VCPOP_V:
  // Vector Rotate Left
  case Capstone::VROL_VV:
  case Capstone::VROL_VX:
  // Vector Rotate Right
  case Capstone::VROR_VI:
  case Capstone::VROR_VV:
  case Capstone::VROR_VX:
  // Vector Carry-less Multiplication Instructions (Zvbc)
  // Vector Carry-less Multiply
  case Capstone::VCLMUL_VV:
  case Capstone::VCLMUL_VX:
  // Vector Carry-less Multiply Return High Half
  case Capstone::VCLMULH_VV:
  case Capstone::VCLMULH_VX:
    return MILog2SEW;

  // Vector Widening Shift Left Logical (Zvbb)
  case Capstone::VWSLL_VI:
  case Capstone::VWSLL_VX:
  case Capstone::VWSLL_VV:
  // Vector Widening Integer Add/Subtract
  // Def uses EEW=2*SEW . Operands use EEW=SEW.
  case Capstone::VWADDU_VV:
  case Capstone::VWADDU_VX:
  case Capstone::VWSUBU_VV:
  case Capstone::VWSUBU_VX:
  case Capstone::VWADD_VV:
  case Capstone::VWADD_VX:
  case Capstone::VWSUB_VV:
  case Capstone::VWSUB_VX:
  // Vector Widening Integer Multiply Instructions
  // Destination EEW=2*SEW. Source EEW=SEW.
  case Capstone::VWMUL_VV:
  case Capstone::VWMUL_VX:
  case Capstone::VWMULSU_VV:
  case Capstone::VWMULSU_VX:
  case Capstone::VWMULU_VV:
  case Capstone::VWMULU_VX:
  // Vector Widening Integer Multiply-Add Instructions
  // Destination EEW=2*SEW. Source EEW=SEW.
  // A SEW-bit*SEW-bit multiply of the sources forms a 2*SEW-bit value, which
  // is then added to the 2*SEW-bit Dest. These instructions never have a
  // passthru operand.
  case Capstone::VWMACCU_VV:
  case Capstone::VWMACCU_VX:
  case Capstone::VWMACC_VV:
  case Capstone::VWMACC_VX:
  case Capstone::VWMACCSU_VV:
  case Capstone::VWMACCSU_VX:
  case Capstone::VWMACCUS_VX:
  // Vector Widening Floating-Point Fused Multiply-Add Instructions
  case Capstone::VFWMACC_VF:
  case Capstone::VFWMACC_VV:
  case Capstone::VFWNMACC_VF:
  case Capstone::VFWNMACC_VV:
  case Capstone::VFWMSAC_VF:
  case Capstone::VFWMSAC_VV:
  case Capstone::VFWNMSAC_VF:
  case Capstone::VFWNMSAC_VV:
  case Capstone::VFWMACCBF16_VV:
  case Capstone::VFWMACCBF16_VF:
  // Vector Widening Floating-Point Add/Subtract Instructions
  // Dest EEW=2*SEW. Source EEW=SEW.
  case Capstone::VFWADD_VV:
  case Capstone::VFWADD_VF:
  case Capstone::VFWSUB_VV:
  case Capstone::VFWSUB_VF:
  // Vector Widening Floating-Point Multiply
  case Capstone::VFWMUL_VF:
  case Capstone::VFWMUL_VV:
  // Widening Floating-Point/Integer Type-Convert Instructions
  case Capstone::VFWCVT_XU_F_V:
  case Capstone::VFWCVT_X_F_V:
  case Capstone::VFWCVT_RTZ_XU_F_V:
  case Capstone::VFWCVT_RTZ_X_F_V:
  case Capstone::VFWCVT_F_XU_V:
  case Capstone::VFWCVT_F_X_V:
  case Capstone::VFWCVT_F_F_V:
  case Capstone::VFWCVTBF16_F_F_V:
    return IsMODef ? MILog2SEW + 1 : MILog2SEW;

  // Def and Op1 uses EEW=2*SEW. Op2 uses EEW=SEW.
  case Capstone::VWADDU_WV:
  case Capstone::VWADDU_WX:
  case Capstone::VWSUBU_WV:
  case Capstone::VWSUBU_WX:
  case Capstone::VWADD_WV:
  case Capstone::VWADD_WX:
  case Capstone::VWSUB_WV:
  case Capstone::VWSUB_WX:
  // Vector Widening Floating-Point Add/Subtract Instructions
  case Capstone::VFWADD_WF:
  case Capstone::VFWADD_WV:
  case Capstone::VFWSUB_WF:
  case Capstone::VFWSUB_WV: {
    bool IsOp1 = (HasPassthru && !IsTied) ? MO.getOperandNo() == 2
                                          : MO.getOperandNo() == 1;
    bool TwoTimes = IsMODef || IsOp1;
    return TwoTimes ? MILog2SEW + 1 : MILog2SEW;
  }

  // Vector Integer Extension
  case Capstone::VZEXT_VF2:
  case Capstone::VSEXT_VF2:
    return getIntegerExtensionOperandEEW(2, MI, MO);
  case Capstone::VZEXT_VF4:
  case Capstone::VSEXT_VF4:
    return getIntegerExtensionOperandEEW(4, MI, MO);
  case Capstone::VZEXT_VF8:
  case Capstone::VSEXT_VF8:
    return getIntegerExtensionOperandEEW(8, MI, MO);

  // Vector Narrowing Integer Right Shift Instructions
  // Destination EEW=SEW, Op 1 has EEW=2*SEW. Op2 has EEW=SEW
  case Capstone::VNSRL_WX:
  case Capstone::VNSRL_WI:
  case Capstone::VNSRL_WV:
  case Capstone::VNSRA_WI:
  case Capstone::VNSRA_WV:
  case Capstone::VNSRA_WX:
  // Vector Narrowing Fixed-Point Clip Instructions
  // Destination and Op1 EEW=SEW. Op2 EEW=2*SEW.
  case Capstone::VNCLIPU_WI:
  case Capstone::VNCLIPU_WV:
  case Capstone::VNCLIPU_WX:
  case Capstone::VNCLIP_WI:
  case Capstone::VNCLIP_WV:
  case Capstone::VNCLIP_WX:
  // Narrowing Floating-Point/Integer Type-Convert Instructions
  case Capstone::VFNCVT_XU_F_W:
  case Capstone::VFNCVT_X_F_W:
  case Capstone::VFNCVT_RTZ_XU_F_W:
  case Capstone::VFNCVT_RTZ_X_F_W:
  case Capstone::VFNCVT_F_XU_W:
  case Capstone::VFNCVT_F_X_W:
  case Capstone::VFNCVT_F_F_W:
  case Capstone::VFNCVT_ROD_F_F_W:
  case Capstone::VFNCVTBF16_F_F_W: {
    assert(!IsTied);
    bool IsOp1 = HasPassthru ? MO.getOperandNo() == 2 : MO.getOperandNo() == 1;
    bool TwoTimes = IsOp1;
    return TwoTimes ? MILog2SEW + 1 : MILog2SEW;
  }

  // Vector Mask Instructions
  // Vector Mask-Register Logical Instructions
  // vmsbf.m set-before-first mask bit
  // vmsif.m set-including-first mask bit
  // vmsof.m set-only-first mask bit
  // EEW=1
  // We handle the cases when operand is a v0 mask operand above the switch,
  // but these instructions may use non-v0 mask operands and need to be handled
  // specifically.
  case Capstone::VMAND_MM:
  case Capstone::VMNAND_MM:
  case Capstone::VMANDN_MM:
  case Capstone::VMXOR_MM:
  case Capstone::VMOR_MM:
  case Capstone::VMNOR_MM:
  case Capstone::VMORN_MM:
  case Capstone::VMXNOR_MM:
  case Capstone::VMSBF_M:
  case Capstone::VMSIF_M:
  case Capstone::VMSOF_M: {
    return MILog2SEW;
  }

  // Vector Compress Instruction
  // EEW=SEW, except the mask operand has EEW=1. Mask operand is not handled
  // before this switch.
  case Capstone::VCOMPRESS_VM:
    return MO.getOperandNo() == 3 ? 0 : MILog2SEW;

  // Vector Iota Instruction
  // EEW=SEW, except the mask operand has EEW=1. Mask operand is not handled
  // before this switch.
  case Capstone::VIOTA_M: {
    if (IsMODef || MO.getOperandNo() == 1)
      return MILog2SEW;
    return 0;
  }

  // Vector Integer Compare Instructions
  // Dest EEW=1. Source EEW=SEW.
  case Capstone::VMSEQ_VI:
  case Capstone::VMSEQ_VV:
  case Capstone::VMSEQ_VX:
  case Capstone::VMSNE_VI:
  case Capstone::VMSNE_VV:
  case Capstone::VMSNE_VX:
  case Capstone::VMSLTU_VV:
  case Capstone::VMSLTU_VX:
  case Capstone::VMSLT_VV:
  case Capstone::VMSLT_VX:
  case Capstone::VMSLEU_VV:
  case Capstone::VMSLEU_VI:
  case Capstone::VMSLEU_VX:
  case Capstone::VMSLE_VV:
  case Capstone::VMSLE_VI:
  case Capstone::VMSLE_VX:
  case Capstone::VMSGTU_VI:
  case Capstone::VMSGTU_VX:
  case Capstone::VMSGT_VI:
  case Capstone::VMSGT_VX:
  // Vector Integer Add-with-Carry / Subtract-with-Borrow Instructions
  // Dest EEW=1. Source EEW=SEW. Mask source operand handled above this switch.
  case Capstone::VMADC_VIM:
  case Capstone::VMADC_VVM:
  case Capstone::VMADC_VXM:
  case Capstone::VMSBC_VVM:
  case Capstone::VMSBC_VXM:
  // Dest EEW=1. Source EEW=SEW.
  case Capstone::VMADC_VV:
  case Capstone::VMADC_VI:
  case Capstone::VMADC_VX:
  case Capstone::VMSBC_VV:
  case Capstone::VMSBC_VX:
  // 13.13. Vector Floating-Point Compare Instructions
  // Dest EEW=1. Source EEW=SEW
  case Capstone::VMFEQ_VF:
  case Capstone::VMFEQ_VV:
  case Capstone::VMFNE_VF:
  case Capstone::VMFNE_VV:
  case Capstone::VMFLT_VF:
  case Capstone::VMFLT_VV:
  case Capstone::VMFLE_VF:
  case Capstone::VMFLE_VV:
  case Capstone::VMFGT_VF:
  case Capstone::VMFGE_VF: {
    if (IsMODef)
      return 0;
    return MILog2SEW;
  }

  // Vector Reduction Operations
  // Vector Single-Width Integer Reduction Instructions
  case Capstone::VREDAND_VS:
  case Capstone::VREDMAX_VS:
  case Capstone::VREDMAXU_VS:
  case Capstone::VREDMIN_VS:
  case Capstone::VREDMINU_VS:
  case Capstone::VREDOR_VS:
  case Capstone::VREDSUM_VS:
  case Capstone::VREDXOR_VS:
  // Vector Single-Width Floating-Point Reduction Instructions
  case Capstone::VFREDMAX_VS:
  case Capstone::VFREDMIN_VS:
  case Capstone::VFREDOSUM_VS:
  case Capstone::VFREDUSUM_VS: {
    return MILog2SEW;
  }

  // Vector Widening Integer Reduction Instructions
  // The Dest and VS1 read only element 0 for the vector register. Return
  // 2*EEW for these. VS2 has EEW=SEW and EMUL=LMUL.
  case Capstone::VWREDSUM_VS:
  case Capstone::VWREDSUMU_VS:
  // Vector Widening Floating-Point Reduction Instructions
  case Capstone::VFWREDOSUM_VS:
  case Capstone::VFWREDUSUM_VS: {
    bool TwoTimes = IsMODef || MO.getOperandNo() == 3;
    return TwoTimes ? MILog2SEW + 1 : MILog2SEW;
  }

  // Vector Register Gather with 16-bit Index Elements Instruction
  // Dest and source data EEW=SEW. Index vector EEW=16.
  case Capstone::VRGATHEREI16_VV: {
    if (MO.getOperandNo() == 2)
      return 4;
    return MILog2SEW;
  }

  default:
    return std::nullopt;
  }
}

static std::optional<OperandInfo> getOperandInfo(const MachineOperand &MO) {
  const MachineInstr &MI = *MO.getParent();
  const CapstoneVPseudosTable::PseudoInfo *RVV =
      CapstoneVPseudosTable::getPseudoInfo(MI.getOpcode());
  assert(RVV && "Could not find MI in PseudoTable");

  std::optional<unsigned> Log2EEW = getOperandLog2EEW(MO);
  if (!Log2EEW)
    return std::nullopt;

  switch (RVV->BaseInstr) {
  // Vector Reduction Operations
  // Vector Single-Width Integer Reduction Instructions
  // Vector Widening Integer Reduction Instructions
  // Vector Widening Floating-Point Reduction Instructions
  // The Dest and VS1 only read element 0 of the vector register. Return just
  // the EEW for these.
  case Capstone::VREDAND_VS:
  case Capstone::VREDMAX_VS:
  case Capstone::VREDMAXU_VS:
  case Capstone::VREDMIN_VS:
  case Capstone::VREDMINU_VS:
  case Capstone::VREDOR_VS:
  case Capstone::VREDSUM_VS:
  case Capstone::VREDXOR_VS:
  case Capstone::VWREDSUM_VS:
  case Capstone::VWREDSUMU_VS:
  case Capstone::VFWREDOSUM_VS:
  case Capstone::VFWREDUSUM_VS:
    if (MO.getOperandNo() != 2)
      return OperandInfo(*Log2EEW);
    break;
  };

  // All others have EMUL=EEW/SEW*LMUL
  return OperandInfo(getEMULEqualsEEWDivSEWTimesLMUL(*Log2EEW, MI), *Log2EEW);
}

static bool isTupleInsertInstr(const MachineInstr &MI);

/// Return true if this optimization should consider MI for VL reduction. This
/// white-list approach simplifies this optimization for instructions that may
/// have more complex semantics with relation to how it uses VL.
static bool isSupportedInstr(const MachineInstr &MI) {
  if (MI.isPHI() || MI.isFullCopy() || isTupleInsertInstr(MI))
    return true;

  const CapstoneVPseudosTable::PseudoInfo *RVV =
      CapstoneVPseudosTable::getPseudoInfo(MI.getOpcode());

  if (!RVV)
    return false;

  switch (RVV->BaseInstr) {
  // Vector Unit-Stride Instructions
  // Vector Strided Instructions
  case Capstone::VLM_V:
  case Capstone::VLE8_V:
  case Capstone::VLSE8_V:
  case Capstone::VLE16_V:
  case Capstone::VLSE16_V:
  case Capstone::VLE32_V:
  case Capstone::VLSE32_V:
  case Capstone::VLE64_V:
  case Capstone::VLSE64_V:
  // Vector Indexed Instructions
  case Capstone::VLUXEI8_V:
  case Capstone::VLOXEI8_V:
  case Capstone::VLUXEI16_V:
  case Capstone::VLOXEI16_V:
  case Capstone::VLUXEI32_V:
  case Capstone::VLOXEI32_V:
  case Capstone::VLUXEI64_V:
  case Capstone::VLOXEI64_V:
  // Vector Single-Width Integer Add and Subtract
  case Capstone::VADD_VI:
  case Capstone::VADD_VV:
  case Capstone::VADD_VX:
  case Capstone::VSUB_VV:
  case Capstone::VSUB_VX:
  case Capstone::VRSUB_VI:
  case Capstone::VRSUB_VX:
  // Vector Bitwise Logical Instructions
  // Vector Single-Width Shift Instructions
  case Capstone::VAND_VI:
  case Capstone::VAND_VV:
  case Capstone::VAND_VX:
  case Capstone::VOR_VI:
  case Capstone::VOR_VV:
  case Capstone::VOR_VX:
  case Capstone::VXOR_VI:
  case Capstone::VXOR_VV:
  case Capstone::VXOR_VX:
  case Capstone::VSLL_VI:
  case Capstone::VSLL_VV:
  case Capstone::VSLL_VX:
  case Capstone::VSRL_VI:
  case Capstone::VSRL_VV:
  case Capstone::VSRL_VX:
  case Capstone::VSRA_VI:
  case Capstone::VSRA_VV:
  case Capstone::VSRA_VX:
  // Vector Widening Integer Add/Subtract
  case Capstone::VWADDU_VV:
  case Capstone::VWADDU_VX:
  case Capstone::VWSUBU_VV:
  case Capstone::VWSUBU_VX:
  case Capstone::VWADD_VV:
  case Capstone::VWADD_VX:
  case Capstone::VWSUB_VV:
  case Capstone::VWSUB_VX:
  case Capstone::VWADDU_WV:
  case Capstone::VWADDU_WX:
  case Capstone::VWSUBU_WV:
  case Capstone::VWSUBU_WX:
  case Capstone::VWADD_WV:
  case Capstone::VWADD_WX:
  case Capstone::VWSUB_WV:
  case Capstone::VWSUB_WX:
  // Vector Integer Extension
  case Capstone::VZEXT_VF2:
  case Capstone::VSEXT_VF2:
  case Capstone::VZEXT_VF4:
  case Capstone::VSEXT_VF4:
  case Capstone::VZEXT_VF8:
  case Capstone::VSEXT_VF8:
  // Vector Narrowing Integer Right Shift Instructions
  case Capstone::VNSRL_WX:
  case Capstone::VNSRL_WI:
  case Capstone::VNSRL_WV:
  case Capstone::VNSRA_WI:
  case Capstone::VNSRA_WV:
  case Capstone::VNSRA_WX:
  // Vector Integer Compare Instructions
  case Capstone::VMSEQ_VI:
  case Capstone::VMSEQ_VV:
  case Capstone::VMSEQ_VX:
  case Capstone::VMSNE_VI:
  case Capstone::VMSNE_VV:
  case Capstone::VMSNE_VX:
  case Capstone::VMSLTU_VV:
  case Capstone::VMSLTU_VX:
  case Capstone::VMSLT_VV:
  case Capstone::VMSLT_VX:
  case Capstone::VMSLEU_VV:
  case Capstone::VMSLEU_VI:
  case Capstone::VMSLEU_VX:
  case Capstone::VMSLE_VV:
  case Capstone::VMSLE_VI:
  case Capstone::VMSLE_VX:
  case Capstone::VMSGTU_VI:
  case Capstone::VMSGTU_VX:
  case Capstone::VMSGT_VI:
  case Capstone::VMSGT_VX:
  // Vector Integer Min/Max Instructions
  case Capstone::VMINU_VV:
  case Capstone::VMINU_VX:
  case Capstone::VMIN_VV:
  case Capstone::VMIN_VX:
  case Capstone::VMAXU_VV:
  case Capstone::VMAXU_VX:
  case Capstone::VMAX_VV:
  case Capstone::VMAX_VX:
  // Vector Single-Width Integer Multiply Instructions
  case Capstone::VMUL_VV:
  case Capstone::VMUL_VX:
  case Capstone::VMULH_VV:
  case Capstone::VMULH_VX:
  case Capstone::VMULHU_VV:
  case Capstone::VMULHU_VX:
  case Capstone::VMULHSU_VV:
  case Capstone::VMULHSU_VX:
  // Vector Integer Divide Instructions
  case Capstone::VDIVU_VV:
  case Capstone::VDIVU_VX:
  case Capstone::VDIV_VV:
  case Capstone::VDIV_VX:
  case Capstone::VREMU_VV:
  case Capstone::VREMU_VX:
  case Capstone::VREM_VV:
  case Capstone::VREM_VX:
  // Vector Widening Integer Multiply Instructions
  case Capstone::VWMUL_VV:
  case Capstone::VWMUL_VX:
  case Capstone::VWMULSU_VV:
  case Capstone::VWMULSU_VX:
  case Capstone::VWMULU_VV:
  case Capstone::VWMULU_VX:
  // Vector Single-Width Integer Multiply-Add Instructions
  case Capstone::VMACC_VV:
  case Capstone::VMACC_VX:
  case Capstone::VNMSAC_VV:
  case Capstone::VNMSAC_VX:
  case Capstone::VMADD_VV:
  case Capstone::VMADD_VX:
  case Capstone::VNMSUB_VV:
  case Capstone::VNMSUB_VX:
  // Vector Integer Merge Instructions
  case Capstone::VMERGE_VIM:
  case Capstone::VMERGE_VVM:
  case Capstone::VMERGE_VXM:
  // Vector Integer Add-with-Carry / Subtract-with-Borrow Instructions
  case Capstone::VADC_VIM:
  case Capstone::VADC_VVM:
  case Capstone::VADC_VXM:
  case Capstone::VMADC_VIM:
  case Capstone::VMADC_VVM:
  case Capstone::VMADC_VXM:
  case Capstone::VSBC_VVM:
  case Capstone::VSBC_VXM:
  case Capstone::VMSBC_VVM:
  case Capstone::VMSBC_VXM:
  case Capstone::VMADC_VV:
  case Capstone::VMADC_VI:
  case Capstone::VMADC_VX:
  case Capstone::VMSBC_VV:
  case Capstone::VMSBC_VX:
  // Vector Widening Integer Multiply-Add Instructions
  case Capstone::VWMACCU_VV:
  case Capstone::VWMACCU_VX:
  case Capstone::VWMACC_VV:
  case Capstone::VWMACC_VX:
  case Capstone::VWMACCSU_VV:
  case Capstone::VWMACCSU_VX:
  case Capstone::VWMACCUS_VX:
  // Vector Integer Move Instructions
  case Capstone::VMV_V_I:
  case Capstone::VMV_V_X:
  case Capstone::VMV_V_V:
  // Vector Single-Width Saturating Add and Subtract
  case Capstone::VSADDU_VV:
  case Capstone::VSADDU_VX:
  case Capstone::VSADDU_VI:
  case Capstone::VSADD_VV:
  case Capstone::VSADD_VX:
  case Capstone::VSADD_VI:
  case Capstone::VSSUBU_VV:
  case Capstone::VSSUBU_VX:
  case Capstone::VSSUB_VV:
  case Capstone::VSSUB_VX:
  // Vector Single-Width Averaging Add and Subtract
  case Capstone::VAADDU_VV:
  case Capstone::VAADDU_VX:
  case Capstone::VAADD_VV:
  case Capstone::VAADD_VX:
  case Capstone::VASUBU_VV:
  case Capstone::VASUBU_VX:
  case Capstone::VASUB_VV:
  case Capstone::VASUB_VX:
  // Vector Single-Width Fractional Multiply with Rounding and Saturation
  case Capstone::VSMUL_VV:
  case Capstone::VSMUL_VX:
  // Vector Single-Width Scaling Shift Instructions
  case Capstone::VSSRL_VV:
  case Capstone::VSSRL_VX:
  case Capstone::VSSRL_VI:
  case Capstone::VSSRA_VV:
  case Capstone::VSSRA_VX:
  case Capstone::VSSRA_VI:
  // Vector Narrowing Fixed-Point Clip Instructions
  case Capstone::VNCLIPU_WV:
  case Capstone::VNCLIPU_WX:
  case Capstone::VNCLIPU_WI:
  case Capstone::VNCLIP_WV:
  case Capstone::VNCLIP_WX:
  case Capstone::VNCLIP_WI:
  // Vector Bit-manipulation Instructions (Zvbb)
  // Vector And-Not
  case Capstone::VANDN_VV:
  case Capstone::VANDN_VX:
  // Vector Reverse Bits in Elements
  case Capstone::VBREV_V:
  // Vector Reverse Bits in Bytes
  case Capstone::VBREV8_V:
  // Vector Reverse Bytes
  case Capstone::VREV8_V:
  // Vector Count Leading Zeros
  case Capstone::VCLZ_V:
  // Vector Count Trailing Zeros
  case Capstone::VCTZ_V:
  // Vector Population Count
  case Capstone::VCPOP_V:
  // Vector Rotate Left
  case Capstone::VROL_VV:
  case Capstone::VROL_VX:
  // Vector Rotate Right
  case Capstone::VROR_VI:
  case Capstone::VROR_VV:
  case Capstone::VROR_VX:
  // Vector Widening Shift Left Logical
  case Capstone::VWSLL_VI:
  case Capstone::VWSLL_VX:
  case Capstone::VWSLL_VV:
  // Vector Carry-less Multiplication Instructions (Zvbc)
  // Vector Carry-less Multiply
  case Capstone::VCLMUL_VV:
  case Capstone::VCLMUL_VX:
  // Vector Carry-less Multiply Return High Half
  case Capstone::VCLMULH_VV:
  case Capstone::VCLMULH_VX:
  // Vector Mask Instructions
  // Vector Mask-Register Logical Instructions
  // vmsbf.m set-before-first mask bit
  // vmsif.m set-including-first mask bit
  // vmsof.m set-only-first mask bit
  // Vector Iota Instruction
  // Vector Element Index Instruction
  case Capstone::VMAND_MM:
  case Capstone::VMNAND_MM:
  case Capstone::VMANDN_MM:
  case Capstone::VMXOR_MM:
  case Capstone::VMOR_MM:
  case Capstone::VMNOR_MM:
  case Capstone::VMORN_MM:
  case Capstone::VMXNOR_MM:
  case Capstone::VMSBF_M:
  case Capstone::VMSIF_M:
  case Capstone::VMSOF_M:
  case Capstone::VIOTA_M:
  case Capstone::VID_V:
  // Vector Slide Instructions
  case Capstone::VSLIDEUP_VX:
  case Capstone::VSLIDEUP_VI:
  case Capstone::VSLIDEDOWN_VX:
  case Capstone::VSLIDEDOWN_VI:
  case Capstone::VSLIDE1UP_VX:
  case Capstone::VFSLIDE1UP_VF:
  // Vector Register Gather Instructions
  case Capstone::VRGATHER_VI:
  case Capstone::VRGATHER_VV:
  case Capstone::VRGATHER_VX:
  case Capstone::VRGATHEREI16_VV:
  // Vector Single-Width Floating-Point Add/Subtract Instructions
  case Capstone::VFADD_VF:
  case Capstone::VFADD_VV:
  case Capstone::VFSUB_VF:
  case Capstone::VFSUB_VV:
  case Capstone::VFRSUB_VF:
  // Vector Widening Floating-Point Add/Subtract Instructions
  case Capstone::VFWADD_VV:
  case Capstone::VFWADD_VF:
  case Capstone::VFWSUB_VV:
  case Capstone::VFWSUB_VF:
  case Capstone::VFWADD_WF:
  case Capstone::VFWADD_WV:
  case Capstone::VFWSUB_WF:
  case Capstone::VFWSUB_WV:
  // Vector Single-Width Floating-Point Multiply/Divide Instructions
  case Capstone::VFMUL_VF:
  case Capstone::VFMUL_VV:
  case Capstone::VFDIV_VF:
  case Capstone::VFDIV_VV:
  case Capstone::VFRDIV_VF:
  // Vector Widening Floating-Point Multiply
  case Capstone::VFWMUL_VF:
  case Capstone::VFWMUL_VV:
  // Vector Single-Width Floating-Point Fused Multiply-Add Instructions
  case Capstone::VFMACC_VV:
  case Capstone::VFMACC_VF:
  case Capstone::VFNMACC_VV:
  case Capstone::VFNMACC_VF:
  case Capstone::VFMSAC_VV:
  case Capstone::VFMSAC_VF:
  case Capstone::VFNMSAC_VV:
  case Capstone::VFNMSAC_VF:
  case Capstone::VFMADD_VV:
  case Capstone::VFMADD_VF:
  case Capstone::VFNMADD_VV:
  case Capstone::VFNMADD_VF:
  case Capstone::VFMSUB_VV:
  case Capstone::VFMSUB_VF:
  case Capstone::VFNMSUB_VV:
  case Capstone::VFNMSUB_VF:
  // Vector Widening Floating-Point Fused Multiply-Add Instructions
  case Capstone::VFWMACC_VV:
  case Capstone::VFWMACC_VF:
  case Capstone::VFWNMACC_VV:
  case Capstone::VFWNMACC_VF:
  case Capstone::VFWMSAC_VV:
  case Capstone::VFWMSAC_VF:
  case Capstone::VFWNMSAC_VV:
  case Capstone::VFWNMSAC_VF:
  case Capstone::VFWMACCBF16_VV:
  case Capstone::VFWMACCBF16_VF:
  // Vector Floating-Point Square-Root Instruction
  case Capstone::VFSQRT_V:
  // Vector Floating-Point Reciprocal Square-Root Estimate Instruction
  case Capstone::VFRSQRT7_V:
  // Vector Floating-Point Reciprocal Estimate Instruction
  case Capstone::VFREC7_V:
  // Vector Floating-Point MIN/MAX Instructions
  case Capstone::VFMIN_VF:
  case Capstone::VFMIN_VV:
  case Capstone::VFMAX_VF:
  case Capstone::VFMAX_VV:
  // Vector Floating-Point Sign-Injection Instructions
  case Capstone::VFSGNJ_VF:
  case Capstone::VFSGNJ_VV:
  case Capstone::VFSGNJN_VV:
  case Capstone::VFSGNJN_VF:
  case Capstone::VFSGNJX_VF:
  case Capstone::VFSGNJX_VV:
  // Vector Floating-Point Compare Instructions
  case Capstone::VMFEQ_VF:
  case Capstone::VMFEQ_VV:
  case Capstone::VMFNE_VF:
  case Capstone::VMFNE_VV:
  case Capstone::VMFLT_VF:
  case Capstone::VMFLT_VV:
  case Capstone::VMFLE_VF:
  case Capstone::VMFLE_VV:
  case Capstone::VMFGT_VF:
  case Capstone::VMFGE_VF:
  // Vector Floating-Point Classify Instruction
  case Capstone::VFCLASS_V:
  // Vector Floating-Point Merge Instruction
  case Capstone::VFMERGE_VFM:
  // Vector Floating-Point Move Instruction
  case Capstone::VFMV_V_F:
  // Single-Width Floating-Point/Integer Type-Convert Instructions
  case Capstone::VFCVT_XU_F_V:
  case Capstone::VFCVT_X_F_V:
  case Capstone::VFCVT_RTZ_XU_F_V:
  case Capstone::VFCVT_RTZ_X_F_V:
  case Capstone::VFCVT_F_XU_V:
  case Capstone::VFCVT_F_X_V:
  // Widening Floating-Point/Integer Type-Convert Instructions
  case Capstone::VFWCVT_XU_F_V:
  case Capstone::VFWCVT_X_F_V:
  case Capstone::VFWCVT_RTZ_XU_F_V:
  case Capstone::VFWCVT_RTZ_X_F_V:
  case Capstone::VFWCVT_F_XU_V:
  case Capstone::VFWCVT_F_X_V:
  case Capstone::VFWCVT_F_F_V:
  case Capstone::VFWCVTBF16_F_F_V:
  // Narrowing Floating-Point/Integer Type-Convert Instructions
  case Capstone::VFNCVT_XU_F_W:
  case Capstone::VFNCVT_X_F_W:
  case Capstone::VFNCVT_RTZ_XU_F_W:
  case Capstone::VFNCVT_RTZ_X_F_W:
  case Capstone::VFNCVT_F_XU_W:
  case Capstone::VFNCVT_F_X_W:
  case Capstone::VFNCVT_F_F_W:
  case Capstone::VFNCVT_ROD_F_F_W:
  case Capstone::VFNCVTBF16_F_F_W:
    return true;
  }

  return false;
}

/// Return true if MO is a vector operand but is used as a scalar operand.
static bool isVectorOpUsedAsScalarOp(const MachineOperand &MO) {
  const MachineInstr *MI = MO.getParent();
  const CapstoneVPseudosTable::PseudoInfo *RVV =
      CapstoneVPseudosTable::getPseudoInfo(MI->getOpcode());

  if (!RVV)
    return false;

  switch (RVV->BaseInstr) {
  // Reductions only use vs1[0] of vs1
  case Capstone::VREDAND_VS:
  case Capstone::VREDMAX_VS:
  case Capstone::VREDMAXU_VS:
  case Capstone::VREDMIN_VS:
  case Capstone::VREDMINU_VS:
  case Capstone::VREDOR_VS:
  case Capstone::VREDSUM_VS:
  case Capstone::VREDXOR_VS:
  case Capstone::VWREDSUM_VS:
  case Capstone::VWREDSUMU_VS:
  case Capstone::VFREDMAX_VS:
  case Capstone::VFREDMIN_VS:
  case Capstone::VFREDOSUM_VS:
  case Capstone::VFREDUSUM_VS:
  case Capstone::VFWREDOSUM_VS:
  case Capstone::VFWREDUSUM_VS:
    return MO.getOperandNo() == 3;
  case Capstone::VMV_X_S:
  case Capstone::VFMV_F_S:
    return MO.getOperandNo() == 1;
  default:
    return false;
  }
}

bool CapstoneVLOptimizer::isCandidate(const MachineInstr &MI) const {
  const MCInstrDesc &Desc = MI.getDesc();
  if (!CapstoneII::hasVLOp(Desc.TSFlags) || !CapstoneII::hasSEWOp(Desc.TSFlags))
    return false;

  if (MI.getNumExplicitDefs() != 1)
    return false;

  // Some instructions have implicit defs e.g. $vxsat. If they might be read
  // later then we can't reduce VL.
  if (!MI.allImplicitDefsAreDead()) {
    LLVM_DEBUG(dbgs() << "Not a candidate because has non-dead implicit def\n");
    return false;
  }

  if (MI.mayRaiseFPException()) {
    LLVM_DEBUG(dbgs() << "Not a candidate because may raise FP exception\n");
    return false;
  }

  for (const MachineMemOperand *MMO : MI.memoperands()) {
    if (MMO->isVolatile()) {
      LLVM_DEBUG(dbgs() << "Not a candidate because contains volatile MMO\n");
      return false;
    }
  }

  // Some instructions that produce vectors have semantics that make it more
  // difficult to determine whether the VL can be reduced. For example, some
  // instructions, such as reductions, may write lanes past VL to a scalar
  // register. Other instructions, such as some loads or stores, may write
  // lower lanes using data from higher lanes. There may be other complex
  // semantics not mentioned here that make it hard to determine whether
  // the VL can be optimized. As a result, a white-list of supported
  // instructions is used. Over time, more instructions can be supported
  // upon careful examination of their semantics under the logic in this
  // optimization.
  // TODO: Use a better approach than a white-list, such as adding
  // properties to instructions using something like TSFlags.
  if (!isSupportedInstr(MI)) {
    LLVM_DEBUG(dbgs() << "Not a candidate due to unsupported instruction: "
                      << MI);
    return false;
  }

  assert(!CapstoneII::elementsDependOnVL(
             TII->get(Capstone::getRVVMCOpcode(MI.getOpcode())).TSFlags) &&
         "Instruction shouldn't be supported if elements depend on VL");

  assert(CapstoneRI::isVRegClass(
             MRI->getRegClass(MI.getOperand(0).getReg())->TSFlags) &&
         "All supported instructions produce a vector register result");

  LLVM_DEBUG(dbgs() << "Found a candidate for VL reduction: " << MI << "\n");
  return true;
}

DemandedVL
CapstoneVLOptimizer::getMinimumVLForUser(const MachineOperand &UserOp) const {
  const MachineInstr &UserMI = *UserOp.getParent();
  const MCInstrDesc &Desc = UserMI.getDesc();

  if (UserMI.isPHI() || UserMI.isFullCopy() || isTupleInsertInstr(UserMI))
    return DemandedVLs.lookup(&UserMI);

  if (!CapstoneII::hasVLOp(Desc.TSFlags) || !CapstoneII::hasSEWOp(Desc.TSFlags)) {
    LLVM_DEBUG(dbgs() << "  Abort due to lack of VL, assume that"
                         " use VLMAX\n");
    return DemandedVL::vlmax();
  }

  if (CapstoneII::readsPastVL(
          TII->get(Capstone::getRVVMCOpcode(UserMI.getOpcode())).TSFlags)) {
    LLVM_DEBUG(dbgs() << "  Abort because used by unsafe instruction\n");
    return DemandedVL::vlmax();
  }

  unsigned VLOpNum = CapstoneII::getVLOpNum(Desc);
  const MachineOperand &VLOp = UserMI.getOperand(VLOpNum);
  // Looking for an immediate or a register VL that isn't X0.
  assert((!VLOp.isReg() || VLOp.getReg() != Capstone::X0) &&
         "Did not expect X0 VL");

  // If the user is a passthru it will read the elements past VL, so
  // abort if any of the elements past VL are demanded.
  if (UserOp.isTied()) {
    assert(UserOp.getOperandNo() == UserMI.getNumExplicitDefs() &&
           CapstoneII::isFirstDefTiedToFirstUse(UserMI.getDesc()));
    if (!Capstone::isVLKnownLE(DemandedVLs.lookup(&UserMI).VL, VLOp)) {
      LLVM_DEBUG(dbgs() << "  Abort because user is passthru in "
                           "instruction with demanded tail\n");
      return DemandedVL::vlmax();
    }
  }

  // Instructions like reductions may use a vector register as a scalar
  // register. In this case, we should treat it as only reading the first lane.
  if (isVectorOpUsedAsScalarOp(UserOp)) {
    LLVM_DEBUG(dbgs() << "    Used this operand as a scalar operand\n");
    return MachineOperand::CreateImm(1);
  }

  // If we know the demanded VL of UserMI, then we can reduce the VL it
  // requires.
  if (Capstone::isVLKnownLE(DemandedVLs.lookup(&UserMI).VL, VLOp))
    return DemandedVLs.lookup(&UserMI);

  return VLOp;
}

/// Return true if MI is an instruction used for assembling registers
/// for segmented store instructions, namely, CapstoneISD::TUPLE_INSERT.
/// Currently it's lowered to INSERT_SUBREG.
static bool isTupleInsertInstr(const MachineInstr &MI) {
  if (!MI.isInsertSubreg())
    return false;

  const MachineRegisterInfo &MRI = MI.getMF()->getRegInfo();
  const TargetRegisterClass *DstRC = MRI.getRegClass(MI.getOperand(0).getReg());
  const TargetRegisterInfo *TRI = MRI.getTargetRegisterInfo();
  if (!CapstoneRI::isVRegClass(DstRC->TSFlags))
    return false;
  unsigned NF = CapstoneRI::getNF(DstRC->TSFlags);
  if (NF < 2)
    return false;

  // Check whether INSERT_SUBREG has the correct subreg index for tuple inserts.
  auto VLMul = CapstoneRI::getLMul(DstRC->TSFlags);
  unsigned SubRegIdx = MI.getOperand(3).getImm();
  [[maybe_unused]] auto [LMul, IsFractional] = CapstoneVType::decodeVLMUL(VLMul);
  assert(!IsFractional && "unexpected LMUL for tuple register classes");
  return TRI->getSubRegIdxSize(SubRegIdx) == Capstone::RVVBitsPerBlock * LMul;
}

static bool isSegmentedStoreInstr(const MachineInstr &MI) {
  switch (Capstone::getRVVMCOpcode(MI.getOpcode())) {
  case VSSEG_CASES(8):
  case VSSSEG_CASES(8):
  case VSUXSEG_CASES(8):
  case VSOXSEG_CASES(8):
  case VSSEG_CASES(16):
  case VSSSEG_CASES(16):
  case VSUXSEG_CASES(16):
  case VSOXSEG_CASES(16):
  case VSSEG_CASES(32):
  case VSSSEG_CASES(32):
  case VSUXSEG_CASES(32):
  case VSOXSEG_CASES(32):
  case VSSEG_CASES(64):
  case VSSSEG_CASES(64):
  case VSUXSEG_CASES(64):
  case VSOXSEG_CASES(64):
    return true;
  default:
    return false;
  }
}

bool CapstoneVLOptimizer::checkUsers(const MachineInstr &MI) const {
  if (MI.isPHI() || MI.isFullCopy() || isTupleInsertInstr(MI))
    return true;

  SmallSetVector<MachineOperand *, 8> OpWorklist;
  SmallPtrSet<const MachineInstr *, 4> PHISeen;
  for (auto &UserOp : MRI->use_operands(MI.getOperand(0).getReg()))
    OpWorklist.insert(&UserOp);

  while (!OpWorklist.empty()) {
    MachineOperand &UserOp = *OpWorklist.pop_back_val();
    const MachineInstr &UserMI = *UserOp.getParent();
    LLVM_DEBUG(dbgs() << "  Checking user: " << UserMI << "\n");

    if (UserMI.isFullCopy() && UserMI.getOperand(0).getReg().isVirtual()) {
      LLVM_DEBUG(dbgs() << "    Peeking through uses of COPY\n");
      OpWorklist.insert_range(llvm::make_pointer_range(
          MRI->use_operands(UserMI.getOperand(0).getReg())));
      continue;
    }

    if (isTupleInsertInstr(UserMI)) {
      LLVM_DEBUG(dbgs().indent(4) << "Peeking through uses of INSERT_SUBREG\n");
      for (MachineOperand &UseOp :
           MRI->use_operands(UserMI.getOperand(0).getReg())) {
        const MachineInstr &CandidateMI = *UseOp.getParent();
        // We should not propagate the VL if the user is not a segmented store
        // or another INSERT_SUBREG, since VL just works differently
        // between segmented operations (per-field) v.s. other RVV ops (on the
        // whole register group).
        if (!isTupleInsertInstr(CandidateMI) &&
            !isSegmentedStoreInstr(CandidateMI))
          return false;
        OpWorklist.insert(&UseOp);
      }
      continue;
    }

    if (UserMI.isPHI()) {
      // Don't follow PHI cycles
      if (!PHISeen.insert(&UserMI).second)
        continue;
      LLVM_DEBUG(dbgs() << "    Peeking through uses of PHI\n");
      OpWorklist.insert_range(llvm::make_pointer_range(
          MRI->use_operands(UserMI.getOperand(0).getReg())));
      continue;
    }

    if (!CapstoneII::hasSEWOp(UserMI.getDesc().TSFlags)) {
      LLVM_DEBUG(dbgs() << "    Abort due to lack of SEW operand\n");
      return false;
    }

    std::optional<OperandInfo> ConsumerInfo = getOperandInfo(UserOp);
    std::optional<OperandInfo> ProducerInfo = getOperandInfo(MI.getOperand(0));
    if (!ConsumerInfo || !ProducerInfo) {
      LLVM_DEBUG(dbgs() << "    Abort due to unknown operand information.\n");
      LLVM_DEBUG(dbgs() << "      ConsumerInfo is: " << ConsumerInfo << "\n");
      LLVM_DEBUG(dbgs() << "      ProducerInfo is: " << ProducerInfo << "\n");
      return false;
    }

    if (!OperandInfo::areCompatible(*ProducerInfo, *ConsumerInfo)) {
      LLVM_DEBUG(
          dbgs()
          << "    Abort due to incompatible information for EMUL or EEW.\n");
      LLVM_DEBUG(dbgs() << "      ConsumerInfo is: " << ConsumerInfo << "\n");
      LLVM_DEBUG(dbgs() << "      ProducerInfo is: " << ProducerInfo << "\n");
      return false;
    }
  }

  return true;
}

bool CapstoneVLOptimizer::tryReduceVL(MachineInstr &MI) const {
  LLVM_DEBUG(dbgs() << "Trying to reduce VL for " << MI);

  unsigned VLOpNum = CapstoneII::getVLOpNum(MI.getDesc());
  MachineOperand &VLOp = MI.getOperand(VLOpNum);

  // If the VL is 1, then there is no need to reduce it. This is an
  // optimization, not needed to preserve correctness.
  if (VLOp.isImm() && VLOp.getImm() == 1) {
    LLVM_DEBUG(dbgs() << "  Abort due to VL == 1, no point in reducing.\n");
    return false;
  }

  auto *CommonVL = &DemandedVLs.at(&MI).VL;

  assert((CommonVL->isImm() || CommonVL->getReg().isVirtual()) &&
         "Expected VL to be an Imm or virtual Reg");

  // If the VL is defined by a vleff that doesn't dominate MI, try using the
  // vleff's AVL. It will be greater than or equal to the output VL.
  if (CommonVL->isReg()) {
    const MachineInstr *VLMI = MRI->getVRegDef(CommonVL->getReg());
    if (CapstoneInstrInfo::isFaultOnlyFirstLoad(*VLMI) &&
        !MDT->dominates(VLMI, &MI))
      CommonVL = &VLMI->getOperand(CapstoneII::getVLOpNum(VLMI->getDesc()));
  }

  if (!Capstone::isVLKnownLE(*CommonVL, VLOp)) {
    LLVM_DEBUG(dbgs() << "  Abort due to CommonVL not <= VLOp.\n");
    return false;
  }

  if (CommonVL->isIdenticalTo(VLOp)) {
    LLVM_DEBUG(
        dbgs() << "  Abort due to CommonVL == VLOp, no point in reducing.\n");
    return false;
  }

  if (CommonVL->isImm()) {
    LLVM_DEBUG(dbgs() << "  Reduce VL from " << VLOp << " to "
                      << CommonVL->getImm() << " for " << MI << "\n");
    VLOp.ChangeToImmediate(CommonVL->getImm());
    return true;
  }
  const MachineInstr *VLMI = MRI->getVRegDef(CommonVL->getReg());
  if (!MDT->dominates(VLMI, &MI)) {
    LLVM_DEBUG(dbgs() << "  Abort due to VL not dominating.\n");
    return false;
  }
  LLVM_DEBUG(
      dbgs() << "  Reduce VL from " << VLOp << " to "
             << printReg(CommonVL->getReg(), MRI->getTargetRegisterInfo())
             << " for " << MI << "\n");

  // All our checks passed. We can reduce VL.
  VLOp.ChangeToRegister(CommonVL->getReg(), false);
  return true;
}

static bool isPhysical(const MachineOperand &MO) {
  return MO.isReg() && MO.getReg().isPhysical();
}

/// Look through \p MI's operands and propagate what it demands to its uses.
void CapstoneVLOptimizer::transfer(const MachineInstr &MI) {
  if (!isSupportedInstr(MI) || !checkUsers(MI) || any_of(MI.defs(), isPhysical))
    DemandedVLs[&MI] = DemandedVL::vlmax();

  for (const MachineOperand &MO : virtual_vec_uses(MI)) {
    const MachineInstr *Def = MRI->getVRegDef(MO.getReg());
    DemandedVL Prev = DemandedVLs[Def];
    DemandedVLs[Def] = DemandedVLs[Def].max(getMinimumVLForUser(MO));
    if (DemandedVLs[Def] != Prev)
      Worklist.insert(Def);
  }
}

bool CapstoneVLOptimizer::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  MRI = &MF.getRegInfo();
  MDT = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();

  const CapstoneSubtarget &ST = MF.getSubtarget<CapstoneSubtarget>();
  if (!ST.hasVInstructions())
    return false;

  TII = ST.getInstrInfo();

  assert(DemandedVLs.empty());

  // For each instruction that defines a vector, propagate the VL it
  // uses to its inputs.
  for (MachineBasicBlock *MBB : post_order(&MF)) {
    assert(MDT->isReachableFromEntry(MBB));
    for (MachineInstr &MI : reverse(*MBB))
      Worklist.insert(&MI);
  }

  while (!Worklist.empty()) {
    const MachineInstr *MI = Worklist.front();
    Worklist.remove(MI);
    transfer(*MI);
  }

  // Then go through and see if we can reduce the VL of any instructions to
  // only what's demanded.
  bool MadeChange = false;
  for (MachineBasicBlock &MBB : MF) {
    // Avoid unreachable blocks as they have degenerate dominance
    if (!MDT->isReachableFromEntry(&MBB))
      continue;

    for (auto &MI : reverse(MBB)) {
      if (!isCandidate(MI))
        continue;
      if (!tryReduceVL(MI))
        continue;
      MadeChange = true;
    }
  }

  DemandedVLs.clear();
  return MadeChange;
}
