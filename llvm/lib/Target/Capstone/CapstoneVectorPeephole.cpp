//===- CapstoneVectorPeephole.cpp - MI Vector Pseudo Peepholes ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass performs various vector pseudo peephole optimisations after
// instruction selection.
//
// Currently it converts vmerge.vvm to vmv.v.v
// PseudoVMERGE_VVM %false, %false, %true, %allonesmask, %vl, %sew
// ->
// PseudoVMV_V_V %false, %true, %vl, %sew
//
// And masked pseudos to unmasked pseudos
// PseudoVADD_V_V_MASK %passthru, %a, %b, %allonesmask, %vl, sew, policy
// ->
// PseudoVADD_V_V %passthru %a, %b, %vl, sew, policy
//
// It also converts AVLs to VLMAX where possible
// %vl = VLENB * something
// PseudoVADD_V_V %passthru, %a, %b, %vl, sew, policy
// ->
// PseudoVADD_V_V %passthru, %a, %b, -1, sew, policy
//
//===----------------------------------------------------------------------===//

#include "Capstone.h"
#include "CapstoneSubtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"

using namespace llvm;

#define DEBUG_TYPE "capstone-vector-peephole"

namespace {

class CapstoneVectorPeephole : public MachineFunctionPass {
public:
  static char ID;
  const TargetInstrInfo *TII;
  MachineRegisterInfo *MRI;
  const TargetRegisterInfo *TRI;
  const CapstoneSubtarget *ST;
  CapstoneVectorPeephole() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;
  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().setIsSSA();
  }

  StringRef getPassName() const override {
    return "Capstone Vector Peephole Optimization";
  }

private:
  bool tryToReduceVL(MachineInstr &MI) const;
  bool convertToVLMAX(MachineInstr &MI) const;
  bool convertToWholeRegister(MachineInstr &MI) const;
  bool convertToUnmasked(MachineInstr &MI) const;
  bool convertAllOnesVMergeToVMv(MachineInstr &MI) const;
  bool convertSameMaskVMergeToVMv(MachineInstr &MI);
  bool foldUndefPassthruVMV_V_V(MachineInstr &MI);
  bool foldVMV_V_V(MachineInstr &MI);
  bool foldVMergeToMask(MachineInstr &MI) const;

  bool hasSameEEW(const MachineInstr &User, const MachineInstr &Src) const;
  bool isAllOnesMask(const MachineInstr *MaskDef) const;
  std::optional<unsigned> getConstant(const MachineOperand &VL) const;
  bool ensureDominates(const MachineOperand &Use, MachineInstr &Src) const;
  bool isKnownSameDefs(Register A, Register B) const;
};

} // namespace

char CapstoneVectorPeephole::ID = 0;

INITIALIZE_PASS(CapstoneVectorPeephole, DEBUG_TYPE, "Capstone Fold Masks", false,
                false)

/// Given \p User that has an input operand with EEW=SEW, which uses the dest
/// operand of \p Src with an unknown EEW, return true if their EEWs match.
bool CapstoneVectorPeephole::hasSameEEW(const MachineInstr &User,
                                     const MachineInstr &Src) const {
  unsigned UserLog2SEW =
      User.getOperand(CapstoneII::getSEWOpNum(User.getDesc())).getImm();
  unsigned SrcLog2SEW =
      Src.getOperand(CapstoneII::getSEWOpNum(Src.getDesc())).getImm();
  unsigned SrcLog2EEW = Capstone::getDestLog2EEW(
      TII->get(Capstone::getRVVMCOpcode(Src.getOpcode())), SrcLog2SEW);
  return SrcLog2EEW == UserLog2SEW;
}

// Attempt to reduce the VL of an instruction whose sole use is feeding a
// instruction with a narrower VL.  This currently works backwards from the
// user instruction (which might have a smaller VL).
bool CapstoneVectorPeephole::tryToReduceVL(MachineInstr &MI) const {
  // Note that the goal here is a bit multifaceted.
  // 1) For store's reducing the VL of the value being stored may help to
  //    reduce VL toggles.  This is somewhat of an artifact of the fact we
  //    promote arithmetic instructions but VL predicate stores.
  // 2) For vmv.v.v reducing VL eagerly on the source instruction allows us
  //    to share code with the foldVMV_V_V transform below.
  //
  // Note that to the best of our knowledge, reducing VL is generally not
  // a significant win on real hardware unless we can also reduce LMUL which
  // this code doesn't try to do.
  //
  // TODO: We can handle a bunch more instructions here, and probably
  // recurse backwards through operands too.
  SmallVector<unsigned, 2> SrcIndices = {0};
  switch (Capstone::getRVVMCOpcode(MI.getOpcode())) {
  default:
    return false;
  case Capstone::VSE8_V:
  case Capstone::VSE16_V:
  case Capstone::VSE32_V:
  case Capstone::VSE64_V:
    break;
  case Capstone::VMV_V_V:
    SrcIndices[0] = 2;
    break;
  case Capstone::VMERGE_VVM:
    SrcIndices.assign({2, 3});
    break;
  case Capstone::VREDSUM_VS:
  case Capstone::VREDMAXU_VS:
  case Capstone::VREDMAX_VS:
  case Capstone::VREDMINU_VS:
  case Capstone::VREDMIN_VS:
  case Capstone::VREDAND_VS:
  case Capstone::VREDOR_VS:
  case Capstone::VREDXOR_VS:
  case Capstone::VWREDSUM_VS:
  case Capstone::VWREDSUMU_VS:
  case Capstone::VFREDUSUM_VS:
  case Capstone::VFREDOSUM_VS:
  case Capstone::VFREDMAX_VS:
  case Capstone::VFREDMIN_VS:
  case Capstone::VFWREDUSUM_VS:
  case Capstone::VFWREDOSUM_VS:
    SrcIndices[0] = 2;
    break;
  }

  MachineOperand &VL = MI.getOperand(CapstoneII::getVLOpNum(MI.getDesc()));
  if (VL.isImm() && VL.getImm() == Capstone::VLMaxSentinel)
    return false;

  bool Changed = false;
  for (unsigned SrcIdx : SrcIndices) {
    Register SrcReg = MI.getOperand(SrcIdx).getReg();
    // Note: one *use*, not one *user*.
    if (!MRI->hasOneUse(SrcReg))
      continue;

    MachineInstr *Src = MRI->getVRegDef(SrcReg);
    if (!Src || Src->hasUnmodeledSideEffects() ||
        Src->getParent() != MI.getParent() || Src->getNumDefs() != 1 ||
        !CapstoneII::hasVLOp(Src->getDesc().TSFlags) ||
        !CapstoneII::hasSEWOp(Src->getDesc().TSFlags))
      continue;

    // Src's dest needs to have the same EEW as MI's input.
    if (!hasSameEEW(MI, *Src))
      continue;

    bool ElementsDependOnVL = CapstoneII::elementsDependOnVL(
        TII->get(Capstone::getRVVMCOpcode(Src->getOpcode())).TSFlags);
    if (ElementsDependOnVL || Src->mayRaiseFPException())
      continue;

    MachineOperand &SrcVL =
        Src->getOperand(CapstoneII::getVLOpNum(Src->getDesc()));
    if (VL.isIdenticalTo(SrcVL) || !Capstone::isVLKnownLE(VL, SrcVL))
      continue;

    if (!ensureDominates(VL, *Src))
      continue;

    if (VL.isImm())
      SrcVL.ChangeToImmediate(VL.getImm());
    else if (VL.isReg())
      SrcVL.ChangeToRegister(VL.getReg(), false);

    Changed = true;
  }

  // TODO: For instructions with a passthru, we could clear the passthru
  // and tail policy since we've just proven the tail is not demanded.
  return Changed;
}

/// Check if an operand is an immediate or a materialized ADDI $x0, imm.
std::optional<unsigned>
CapstoneVectorPeephole::getConstant(const MachineOperand &VL) const {
  if (VL.isImm())
    return VL.getImm();

  MachineInstr *Def = MRI->getVRegDef(VL.getReg());
  if (!Def || Def->getOpcode() != Capstone::ADDI ||
      Def->getOperand(1).getReg() != Capstone::X0)
    return std::nullopt;
  return Def->getOperand(2).getImm();
}

/// Convert AVLs that are known to be VLMAX to the VLMAX sentinel.
bool CapstoneVectorPeephole::convertToVLMAX(MachineInstr &MI) const {
  if (!CapstoneII::hasVLOp(MI.getDesc().TSFlags) ||
      !CapstoneII::hasSEWOp(MI.getDesc().TSFlags))
    return false;

  auto LMUL = CapstoneVType::decodeVLMUL(CapstoneII::getLMul(MI.getDesc().TSFlags));
  // Fixed-point value, denominator=8
  unsigned LMULFixed = LMUL.second ? (8 / LMUL.first) : 8 * LMUL.first;
  unsigned Log2SEW = MI.getOperand(CapstoneII::getSEWOpNum(MI.getDesc())).getImm();
  // A Log2SEW of 0 is an operation on mask registers only
  unsigned SEW = Log2SEW ? 1 << Log2SEW : 8;
  assert(CapstoneVType::isValidSEW(SEW) && "Unexpected SEW");
  assert(8 * LMULFixed / SEW > 0);

  // If the exact VLEN is known then we know VLMAX, check if the AVL == VLMAX.
  MachineOperand &VL = MI.getOperand(CapstoneII::getVLOpNum(MI.getDesc()));
  if (auto VLen = ST->getRealVLen(), AVL = getConstant(VL);
      VLen && AVL && (*VLen * LMULFixed) / SEW == *AVL * 8) {
    VL.ChangeToImmediate(Capstone::VLMaxSentinel);
    return true;
  }

  // If an AVL is a VLENB that's possibly scaled to be equal to VLMAX, convert
  // it to the VLMAX sentinel value.
  if (!VL.isReg())
    return false;
  MachineInstr *Def = MRI->getVRegDef(VL.getReg());
  if (!Def)
    return false;

  // Fixed-point value, denominator=8
  uint64_t ScaleFixed = 8;
  // Check if the VLENB was potentially scaled with slli/srli
  if (Def->getOpcode() == Capstone::SLLI) {
    assert(Def->getOperand(2).getImm() < 64);
    ScaleFixed <<= Def->getOperand(2).getImm();
    Def = MRI->getVRegDef(Def->getOperand(1).getReg());
  } else if (Def->getOpcode() == Capstone::SRLI) {
    assert(Def->getOperand(2).getImm() < 64);
    ScaleFixed >>= Def->getOperand(2).getImm();
    Def = MRI->getVRegDef(Def->getOperand(1).getReg());
  }

  if (!Def || Def->getOpcode() != Capstone::PseudoReadVLENB)
    return false;

  // AVL = (VLENB * Scale)
  //
  // VLMAX = (VLENB * 8 * LMUL) / SEW
  //
  // AVL == VLMAX
  // -> VLENB * Scale == (VLENB * 8 * LMUL) / SEW
  // -> Scale == (8 * LMUL) / SEW
  if (ScaleFixed != 8 * LMULFixed / SEW)
    return false;

  VL.ChangeToImmediate(Capstone::VLMaxSentinel);

  return true;
}

bool CapstoneVectorPeephole::isAllOnesMask(const MachineInstr *MaskDef) const {
  while (MaskDef->isCopy() && MaskDef->getOperand(1).getReg().isVirtual())
    MaskDef = MRI->getVRegDef(MaskDef->getOperand(1).getReg());

  // TODO: Check that the VMSET is the expected bitwidth? The pseudo has
  // undefined behaviour if it's the wrong bitwidth, so we could choose to
  // assume that it's all-ones? Same applies to its VL.
  switch (MaskDef->getOpcode()) {
  case Capstone::PseudoVMSET_M_B1:
  case Capstone::PseudoVMSET_M_B2:
  case Capstone::PseudoVMSET_M_B4:
  case Capstone::PseudoVMSET_M_B8:
  case Capstone::PseudoVMSET_M_B16:
  case Capstone::PseudoVMSET_M_B32:
  case Capstone::PseudoVMSET_M_B64:
    return true;
  default:
    return false;
  }
}

/// Convert unit strided unmasked loads and stores to whole-register equivalents
/// to avoid the dependency on $vl and $vtype.
///
/// %x = PseudoVLE8_V_M1 %passthru, %ptr, %vlmax, policy
/// PseudoVSE8_V_M1 %v, %ptr, %vlmax
///
/// ->
///
/// %x = VL1RE8_V %ptr
/// VS1R_V %v, %ptr
bool CapstoneVectorPeephole::convertToWholeRegister(MachineInstr &MI) const {
#define CASE_WHOLE_REGISTER_LMUL_SEW(lmul, sew)                                \
  case Capstone::PseudoVLE##sew##_V_M##lmul:                                      \
    NewOpc = Capstone::VL##lmul##RE##sew##_V;                                     \
    break;                                                                     \
  case Capstone::PseudoVSE##sew##_V_M##lmul:                                      \
    NewOpc = Capstone::VS##lmul##R_V;                                             \
    break;
#define CASE_WHOLE_REGISTER_LMUL(lmul)                                         \
  CASE_WHOLE_REGISTER_LMUL_SEW(lmul, 8)                                        \
  CASE_WHOLE_REGISTER_LMUL_SEW(lmul, 16)                                       \
  CASE_WHOLE_REGISTER_LMUL_SEW(lmul, 32)                                       \
  CASE_WHOLE_REGISTER_LMUL_SEW(lmul, 64)

  unsigned NewOpc;
  switch (MI.getOpcode()) {
    CASE_WHOLE_REGISTER_LMUL(1)
    CASE_WHOLE_REGISTER_LMUL(2)
    CASE_WHOLE_REGISTER_LMUL(4)
    CASE_WHOLE_REGISTER_LMUL(8)
  default:
    return false;
  }

  MachineOperand &VLOp = MI.getOperand(CapstoneII::getVLOpNum(MI.getDesc()));
  if (!VLOp.isImm() || VLOp.getImm() != Capstone::VLMaxSentinel)
    return false;

  // Whole register instructions aren't pseudos so they don't have
  // policy/SEW/AVL ops, and they don't have passthrus.
  if (CapstoneII::hasVecPolicyOp(MI.getDesc().TSFlags))
    MI.removeOperand(CapstoneII::getVecPolicyOpNum(MI.getDesc()));
  MI.removeOperand(CapstoneII::getSEWOpNum(MI.getDesc()));
  MI.removeOperand(CapstoneII::getVLOpNum(MI.getDesc()));
  if (CapstoneII::isFirstDefTiedToFirstUse(MI.getDesc()))
    MI.removeOperand(1);

  MI.setDesc(TII->get(NewOpc));

  return true;
}

static unsigned getVMV_V_VOpcodeForVMERGE_VVM(const MachineInstr &MI) {
#define CASE_VMERGE_TO_VMV(lmul)                                               \
  case Capstone::PseudoVMERGE_VVM_##lmul:                                         \
    return Capstone::PseudoVMV_V_V_##lmul;
  switch (MI.getOpcode()) {
  default:
    return 0;
    CASE_VMERGE_TO_VMV(MF8)
    CASE_VMERGE_TO_VMV(MF4)
    CASE_VMERGE_TO_VMV(MF2)
    CASE_VMERGE_TO_VMV(M1)
    CASE_VMERGE_TO_VMV(M2)
    CASE_VMERGE_TO_VMV(M4)
    CASE_VMERGE_TO_VMV(M8)
  }
}

/// Convert a PseudoVMERGE_VVM with an all ones mask to a PseudoVMV_V_V.
///
/// %x = PseudoVMERGE_VVM %passthru, %false, %true, %allones, sew, vl
/// ->
/// %x = PseudoVMV_V_V %passthru, %true, vl, sew, tu_mu
bool CapstoneVectorPeephole::convertAllOnesVMergeToVMv(MachineInstr &MI) const {
  unsigned NewOpc = getVMV_V_VOpcodeForVMERGE_VVM(MI);
  if (!NewOpc)
    return false;
  if (!isAllOnesMask(MRI->getVRegDef(MI.getOperand(4).getReg())))
    return false;

  MI.setDesc(TII->get(NewOpc));
  MI.removeOperand(2); // False operand
  MI.removeOperand(3); // Mask operand
  MI.addOperand(
      MachineOperand::CreateImm(CapstoneVType::TAIL_UNDISTURBED_MASK_UNDISTURBED));

  // vmv.v.v doesn't have a mask operand, so we may be able to inflate the
  // register class for the destination and passthru operands e.g. VRNoV0 -> VR
  MRI->recomputeRegClass(MI.getOperand(0).getReg());
  if (MI.getOperand(1).getReg() != Capstone::NoRegister)
    MRI->recomputeRegClass(MI.getOperand(1).getReg());
  return true;
}

bool CapstoneVectorPeephole::isKnownSameDefs(Register A, Register B) const {
  if (A.isPhysical() || B.isPhysical())
    return false;

  auto LookThruVirtRegCopies = [this](Register Reg) {
    while (MachineInstr *Def = MRI->getUniqueVRegDef(Reg)) {
      if (!Def->isFullCopy())
        break;
      Register Src = Def->getOperand(1).getReg();
      if (!Src.isVirtual())
        break;
      Reg = Src;
    }
    return Reg;
  };

  return LookThruVirtRegCopies(A) == LookThruVirtRegCopies(B);
}

/// If a PseudoVMERGE_VVM's true operand is a masked pseudo and both have the
/// same mask, and the masked pseudo's passthru is the same as the false
/// operand, we can convert the PseudoVMERGE_VVM to a PseudoVMV_V_V.
///
/// %true = PseudoVADD_VV_M1_MASK %false, %x, %y, %mask, vl1, sew, policy
/// %x = PseudoVMERGE_VVM %passthru, %false, %true, %mask, vl2, sew
/// ->
/// %true = PseudoVADD_VV_M1_MASK %false, %x, %y, %mask, vl1, sew, policy
/// %x = PseudoVMV_V_V %passthru, %true, vl2, sew, tu_mu
bool CapstoneVectorPeephole::convertSameMaskVMergeToVMv(MachineInstr &MI) {
  unsigned NewOpc = getVMV_V_VOpcodeForVMERGE_VVM(MI);
  if (!NewOpc)
    return false;
  MachineInstr *True = MRI->getVRegDef(MI.getOperand(3).getReg());

  if (!True || True->getParent() != MI.getParent())
    return false;

  auto *TrueMaskedInfo = Capstone::getMaskedPseudoInfo(True->getOpcode());
  if (!TrueMaskedInfo || !hasSameEEW(MI, *True))
    return false;

  const MachineOperand &TrueMask =
      True->getOperand(TrueMaskedInfo->MaskOpIdx + True->getNumExplicitDefs());
  const MachineOperand &MIMask = MI.getOperand(4);
  if (!isKnownSameDefs(TrueMask.getReg(), MIMask.getReg()))
    return false;

  // Masked off lanes past TrueVL will come from False, and converting to vmv
  // will lose these lanes unless MIVL <= TrueVL.
  // TODO: We could relax this for False == Passthru and True policy == TU
  const MachineOperand &MIVL = MI.getOperand(CapstoneII::getVLOpNum(MI.getDesc()));
  const MachineOperand &TrueVL =
      True->getOperand(CapstoneII::getVLOpNum(True->getDesc()));
  if (!Capstone::isVLKnownLE(MIVL, TrueVL))
    return false;

  // True's passthru needs to be equivalent to False
  Register TruePassthruReg = True->getOperand(1).getReg();
  Register FalseReg = MI.getOperand(2).getReg();
  if (TruePassthruReg != FalseReg) {
    // If True's passthru is undef see if we can change it to False
    if (TruePassthruReg != Capstone::NoRegister ||
        !MRI->hasOneUse(MI.getOperand(3).getReg()) ||
        !ensureDominates(MI.getOperand(2), *True))
      return false;
    True->getOperand(1).setReg(MI.getOperand(2).getReg());
    // If True is masked then its passthru needs to be in VRNoV0.
    MRI->constrainRegClass(True->getOperand(1).getReg(),
                           TII->getRegClass(True->getDesc(), 1, TRI));
  }

  MI.setDesc(TII->get(NewOpc));
  MI.removeOperand(2); // False operand
  MI.removeOperand(3); // Mask operand
  MI.addOperand(
      MachineOperand::CreateImm(CapstoneVType::TAIL_UNDISTURBED_MASK_UNDISTURBED));

  // vmv.v.v doesn't have a mask operand, so we may be able to inflate the
  // register class for the destination and passthru operands e.g. VRNoV0 -> VR
  MRI->recomputeRegClass(MI.getOperand(0).getReg());
  if (MI.getOperand(1).getReg() != Capstone::NoRegister)
    MRI->recomputeRegClass(MI.getOperand(1).getReg());
  return true;
}

bool CapstoneVectorPeephole::convertToUnmasked(MachineInstr &MI) const {
  const Capstone::CapstoneMaskedPseudoInfo *I =
      Capstone::getMaskedPseudoInfo(MI.getOpcode());
  if (!I)
    return false;

  if (!isAllOnesMask(MRI->getVRegDef(
          MI.getOperand(I->MaskOpIdx + MI.getNumExplicitDefs()).getReg())))
    return false;

  // There are two classes of pseudos in the table - compares and
  // everything else.  See the comment on CapstoneMaskedPseudo for details.
  const unsigned Opc = I->UnmaskedPseudo;
  const MCInstrDesc &MCID = TII->get(Opc);
  [[maybe_unused]] const bool HasPolicyOp =
      CapstoneII::hasVecPolicyOp(MCID.TSFlags);
  const bool HasPassthru = CapstoneII::isFirstDefTiedToFirstUse(MCID);
  const MCInstrDesc &MaskedMCID = TII->get(MI.getOpcode());
  assert((CapstoneII::hasVecPolicyOp(MaskedMCID.TSFlags) ||
          !CapstoneII::hasVecPolicyOp(MCID.TSFlags)) &&
         "Unmasked pseudo has policy but masked pseudo doesn't?");
  assert(HasPolicyOp == HasPassthru && "Unexpected pseudo structure");
  assert(!(HasPassthru && !CapstoneII::isFirstDefTiedToFirstUse(MaskedMCID)) &&
         "Unmasked with passthru but masked with no passthru?");
  (void)HasPolicyOp;

  MI.setDesc(MCID);

  // Drop the policy operand if unmasked doesn't need it.
  if (CapstoneII::hasVecPolicyOp(MaskedMCID.TSFlags) &&
      !CapstoneII::hasVecPolicyOp(MCID.TSFlags))
    MI.removeOperand(CapstoneII::getVecPolicyOpNum(MaskedMCID));

  // TODO: Increment all MaskOpIdxs in tablegen by num of explicit defs?
  unsigned MaskOpIdx = I->MaskOpIdx + MI.getNumExplicitDefs();
  MI.removeOperand(MaskOpIdx);

  // The unmasked pseudo will no longer be constrained to the vrnov0 reg class,
  // so try and relax it to vr.
  MRI->recomputeRegClass(MI.getOperand(0).getReg());

  // If the original masked pseudo had a passthru, relax it or remove it.
  if (CapstoneII::isFirstDefTiedToFirstUse(MaskedMCID)) {
    unsigned PassthruOpIdx = MI.getNumExplicitDefs();
    if (HasPassthru) {
      if (MI.getOperand(PassthruOpIdx).getReg() != Capstone::NoRegister)
        MRI->recomputeRegClass(MI.getOperand(PassthruOpIdx).getReg());
    } else
      MI.removeOperand(PassthruOpIdx);
  }

  return true;
}

/// Check if it's safe to move From down to To, checking that no physical
/// registers are clobbered.
static bool isSafeToMove(const MachineInstr &From, const MachineInstr &To) {
  assert(From.getParent() == To.getParent());
  SmallVector<Register> PhysUses, PhysDefs;
  for (const MachineOperand &MO : From.all_uses())
    if (MO.getReg().isPhysical())
      PhysUses.push_back(MO.getReg());
  for (const MachineOperand &MO : From.all_defs())
    if (MO.getReg().isPhysical())
      PhysDefs.push_back(MO.getReg());
  bool SawStore = false;
  for (auto II = std::next(From.getIterator()); II != To.getIterator(); II++) {
    for (Register PhysReg : PhysUses)
      if (II->definesRegister(PhysReg, nullptr))
        return false;
    for (Register PhysReg : PhysDefs)
      if (II->definesRegister(PhysReg, nullptr) ||
          II->readsRegister(PhysReg, nullptr))
        return false;
    if (II->mayStore()) {
      SawStore = true;
      break;
    }
  }
  return From.isSafeToMove(SawStore);
}

/// Given A and B are in the same MBB, returns true if A comes before B.
static bool dominates(MachineBasicBlock::const_iterator A,
                      MachineBasicBlock::const_iterator B) {
  assert(A->getParent() == B->getParent());
  const MachineBasicBlock *MBB = A->getParent();
  auto MBBEnd = MBB->end();
  if (B == MBBEnd)
    return true;

  MachineBasicBlock::const_iterator I = MBB->begin();
  for (; &*I != A && &*I != B; ++I)
    ;

  return &*I == A;
}

/// If the register in \p MO doesn't dominate \p Src, try to move \p Src so it
/// does. Returns false if doesn't dominate and we can't move. \p MO must be in
/// the same basic block as \Src.
bool CapstoneVectorPeephole::ensureDominates(const MachineOperand &MO,
                                          MachineInstr &Src) const {
  assert(MO.getParent()->getParent() == Src.getParent());
  if (!MO.isReg() || MO.getReg() == Capstone::NoRegister)
    return true;

  MachineInstr *Def = MRI->getVRegDef(MO.getReg());
  if (Def->getParent() == Src.getParent() && !dominates(Def, Src)) {
    if (!isSafeToMove(Src, *Def->getNextNode()))
      return false;
    Src.moveBefore(Def->getNextNode());
  }

  return true;
}

/// If a PseudoVMV_V_V's passthru is undef then we can replace it with its input
bool CapstoneVectorPeephole::foldUndefPassthruVMV_V_V(MachineInstr &MI) {
  if (Capstone::getRVVMCOpcode(MI.getOpcode()) != Capstone::VMV_V_V)
    return false;
  if (MI.getOperand(1).getReg() != Capstone::NoRegister)
    return false;

  // If the input was a pseudo with a policy operand, we can give it a tail
  // agnostic policy if MI's undef tail subsumes the input's.
  MachineInstr *Src = MRI->getVRegDef(MI.getOperand(2).getReg());
  if (Src && !Src->hasUnmodeledSideEffects() &&
      MRI->hasOneUse(MI.getOperand(2).getReg()) &&
      CapstoneII::hasVLOp(Src->getDesc().TSFlags) &&
      CapstoneII::hasVecPolicyOp(Src->getDesc().TSFlags) && hasSameEEW(MI, *Src)) {
    const MachineOperand &MIVL = MI.getOperand(3);
    const MachineOperand &SrcVL =
        Src->getOperand(CapstoneII::getVLOpNum(Src->getDesc()));

    MachineOperand &SrcPolicy =
        Src->getOperand(CapstoneII::getVecPolicyOpNum(Src->getDesc()));

    if (Capstone::isVLKnownLE(MIVL, SrcVL))
      SrcPolicy.setImm(SrcPolicy.getImm() | CapstoneVType::TAIL_AGNOSTIC);
  }

  MRI->constrainRegClass(MI.getOperand(2).getReg(),
                         MRI->getRegClass(MI.getOperand(0).getReg()));
  MRI->replaceRegWith(MI.getOperand(0).getReg(), MI.getOperand(2).getReg());
  MRI->clearKillFlags(MI.getOperand(2).getReg());
  MI.eraseFromParent();
  return true;
}

/// If a PseudoVMV_V_V is the only user of its input, fold its passthru and VL
/// into it.
///
/// %x = PseudoVADD_V_V_M1 %passthru, %a, %b, %vl1, sew, policy
/// %y = PseudoVMV_V_V_M1 %passthru, %x, %vl2, sew, policy
///    (where %vl1 <= %vl2, see related tryToReduceVL)
///
/// ->
///
/// %y = PseudoVADD_V_V_M1 %passthru, %a, %b, vl1, sew, policy
bool CapstoneVectorPeephole::foldVMV_V_V(MachineInstr &MI) {
  if (Capstone::getRVVMCOpcode(MI.getOpcode()) != Capstone::VMV_V_V)
    return false;

  MachineOperand &Passthru = MI.getOperand(1);

  if (!MRI->hasOneUse(MI.getOperand(2).getReg()))
    return false;

  MachineInstr *Src = MRI->getVRegDef(MI.getOperand(2).getReg());
  if (!Src || Src->hasUnmodeledSideEffects() ||
      Src->getParent() != MI.getParent() ||
      !CapstoneII::isFirstDefTiedToFirstUse(Src->getDesc()) ||
      !CapstoneII::hasVLOp(Src->getDesc().TSFlags))
    return false;

  // Src's dest needs to have the same EEW as MI's input.
  if (!hasSameEEW(MI, *Src))
    return false;

  // Src needs to have the same passthru as VMV_V_V
  MachineOperand &SrcPassthru = Src->getOperand(Src->getNumExplicitDefs());
  if (SrcPassthru.getReg() != Capstone::NoRegister &&
      SrcPassthru.getReg() != Passthru.getReg())
    return false;

  // Src VL will have already been reduced if legal (see tryToReduceVL),
  // so we don't need to handle a smaller source VL here.  However, the
  // user's VL may be larger
  MachineOperand &SrcVL = Src->getOperand(CapstoneII::getVLOpNum(Src->getDesc()));
  if (!Capstone::isVLKnownLE(SrcVL, MI.getOperand(3)))
    return false;

  // If the new passthru doesn't dominate Src, try to move Src so it does.
  if (!ensureDominates(Passthru, *Src))
    return false;

  if (SrcPassthru.getReg() != Passthru.getReg()) {
    SrcPassthru.setReg(Passthru.getReg());
    // If Src is masked then its passthru needs to be in VRNoV0.
    if (Passthru.getReg() != Capstone::NoRegister)
      MRI->constrainRegClass(
          Passthru.getReg(),
          TII->getRegClass(Src->getDesc(), SrcPassthru.getOperandNo(), TRI));
  }

  if (CapstoneII::hasVecPolicyOp(Src->getDesc().TSFlags)) {
    // If MI was tail agnostic and the VL didn't increase, preserve it.
    int64_t Policy = CapstoneVType::TAIL_UNDISTURBED_MASK_UNDISTURBED;
    if ((MI.getOperand(5).getImm() & CapstoneVType::TAIL_AGNOSTIC) &&
        Capstone::isVLKnownLE(MI.getOperand(3), SrcVL))
      Policy |= CapstoneVType::TAIL_AGNOSTIC;
    Src->getOperand(CapstoneII::getVecPolicyOpNum(Src->getDesc())).setImm(Policy);
  }

  MRI->constrainRegClass(Src->getOperand(0).getReg(),
                         MRI->getRegClass(MI.getOperand(0).getReg()));
  MRI->replaceRegWith(MI.getOperand(0).getReg(), Src->getOperand(0).getReg());
  MI.eraseFromParent();

  return true;
}

/// Try to fold away VMERGE_VVM instructions into their operands:
///
/// %true = PseudoVADD_VV ...
/// %x = PseudoVMERGE_VVM_M1 %false, %false, %true, %mask
/// ->
/// %x = PseudoVADD_VV_M1_MASK %false, ..., %mask
///
/// We can only fold if vmerge's passthru operand, vmerge's false operand and
/// %true's passthru operand (if it has one) are the same. This is because we
/// have to consolidate them into one passthru operand in the result.
///
/// If %true is masked, then we can use its mask instead of vmerge's if vmerge's
/// mask is all ones.
///
/// The resulting VL is the minimum of the two VLs.
///
/// The resulting policy is the effective policy the vmerge would have had,
/// i.e. whether or not it's passthru operand was implicit-def.
bool CapstoneVectorPeephole::foldVMergeToMask(MachineInstr &MI) const {
  if (Capstone::getRVVMCOpcode(MI.getOpcode()) != Capstone::VMERGE_VVM)
    return false;

  Register PassthruReg = MI.getOperand(1).getReg();
  Register FalseReg = MI.getOperand(2).getReg();
  Register TrueReg = MI.getOperand(3).getReg();
  if (!TrueReg.isVirtual() || !MRI->hasOneUse(TrueReg))
    return false;
  MachineInstr &True = *MRI->getUniqueVRegDef(TrueReg);
  if (True.getParent() != MI.getParent())
    return false;
  const MachineOperand &MaskOp = MI.getOperand(4);
  MachineInstr *Mask = MRI->getUniqueVRegDef(MaskOp.getReg());
  assert(Mask);

  const Capstone::CapstoneMaskedPseudoInfo *Info =
      Capstone::lookupMaskedIntrinsicByUnmasked(True.getOpcode());
  if (!Info)
    return false;

  // If the EEW of True is different from vmerge's SEW, then we can't fold.
  if (!hasSameEEW(MI, True))
    return false;

  // We require that either passthru and false are the same, or that passthru
  // is undefined.
  if (PassthruReg && !isKnownSameDefs(PassthruReg, FalseReg))
    return false;

  std::optional<std::pair<unsigned, unsigned>> NeedsCommute;

  // If True has a passthru operand then it needs to be the same as vmerge's
  // False, since False will be used for the result's passthru operand.
  Register TruePassthru = True.getOperand(True.getNumExplicitDefs()).getReg();
  if (CapstoneII::isFirstDefTiedToFirstUse(True.getDesc()) && TruePassthru &&
      !isKnownSameDefs(TruePassthru, FalseReg)) {
    // If True's passthru != False, check if it uses False in another operand
    // and try to commute it.
    int OtherIdx = True.findRegisterUseOperandIdx(FalseReg, TRI);
    if (OtherIdx == -1)
      return false;
    unsigned OpIdx1 = OtherIdx;
    unsigned OpIdx2 = True.getNumExplicitDefs();
    if (!TII->findCommutedOpIndices(True, OpIdx1, OpIdx2))
      return false;
    NeedsCommute = {OpIdx1, OpIdx2};
  }

  // Make sure it doesn't raise any observable fp exceptions, since changing the
  // active elements will affect how fflags is set.
  if (True.hasUnmodeledSideEffects() || True.mayRaiseFPException())
    return false;

  const MachineOperand &VMergeVL =
      MI.getOperand(CapstoneII::getVLOpNum(MI.getDesc()));
  const MachineOperand &TrueVL =
      True.getOperand(CapstoneII::getVLOpNum(True.getDesc()));

  MachineOperand MinVL = MachineOperand::CreateImm(0);
  if (Capstone::isVLKnownLE(TrueVL, VMergeVL))
    MinVL = TrueVL;
  else if (Capstone::isVLKnownLE(VMergeVL, TrueVL))
    MinVL = VMergeVL;
  else
    return false;

  unsigned RVVTSFlags =
      TII->get(Capstone::getRVVMCOpcode(True.getOpcode())).TSFlags;
  if (CapstoneII::elementsDependOnVL(RVVTSFlags) && !TrueVL.isIdenticalTo(MinVL))
    return false;
  if (CapstoneII::elementsDependOnMask(RVVTSFlags) && !isAllOnesMask(Mask))
    return false;

  // Use a tumu policy, relaxing it to tail agnostic provided that the passthru
  // operand is undefined.
  //
  // However, if the VL became smaller than what the vmerge had originally, then
  // elements past VL that were previously in the vmerge's body will have moved
  // to the tail. In that case we always need to use tail undisturbed to
  // preserve them.
  uint64_t Policy = CapstoneVType::TAIL_UNDISTURBED_MASK_UNDISTURBED;
  if (!PassthruReg && Capstone::isVLKnownLE(VMergeVL, MinVL))
    Policy |= CapstoneVType::TAIL_AGNOSTIC;

  assert(CapstoneII::hasVecPolicyOp(True.getDesc().TSFlags) &&
         "Foldable unmasked pseudo should have a policy op already");

  // Make sure the mask dominates True, otherwise move down True so it does.
  // VL will always dominate since if it's a register they need to be the same.
  if (!ensureDominates(MaskOp, True))
    return false;

  if (NeedsCommute) {
    auto [OpIdx1, OpIdx2] = *NeedsCommute;
    [[maybe_unused]] bool Commuted =
        TII->commuteInstruction(True, /*NewMI=*/false, OpIdx1, OpIdx2);
    assert(Commuted && "Failed to commute True?");
    Info = Capstone::lookupMaskedIntrinsicByUnmasked(True.getOpcode());
  }

  True.setDesc(TII->get(Info->MaskedPseudo));

  // Insert the mask operand.
  // TODO: Increment MaskOpIdx by number of explicit defs?
  True.insert(True.operands_begin() + Info->MaskOpIdx +
                  True.getNumExplicitDefs(),
              MachineOperand::CreateReg(MaskOp.getReg(), false));

  // Update the passthru, AVL and policy.
  True.getOperand(True.getNumExplicitDefs()).setReg(FalseReg);
  True.removeOperand(CapstoneII::getVLOpNum(True.getDesc()));
  True.insert(True.operands_begin() + CapstoneII::getVLOpNum(True.getDesc()),
              MinVL);
  True.getOperand(CapstoneII::getVecPolicyOpNum(True.getDesc())).setImm(Policy);

  MRI->replaceRegWith(True.getOperand(0).getReg(), MI.getOperand(0).getReg());
  // Now that True is masked, constrain its operands from vr -> vrnov0.
  for (MachineOperand &MO : True.explicit_operands()) {
    if (!MO.isReg() || !MO.getReg().isVirtual())
      continue;
    MRI->constrainRegClass(
        MO.getReg(), True.getRegClassConstraint(MO.getOperandNo(), TII, TRI));
  }
  MI.eraseFromParent();

  return true;
}

bool CapstoneVectorPeephole::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  // Skip if the vector extension is not enabled.
  ST = &MF.getSubtarget<CapstoneSubtarget>();
  if (!ST->hasVInstructions())
    return false;

  TII = ST->getInstrInfo();
  MRI = &MF.getRegInfo();
  TRI = MRI->getTargetRegisterInfo();

  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : make_early_inc_range(MBB))
      Changed |= foldVMergeToMask(MI);

    for (MachineInstr &MI : make_early_inc_range(MBB)) {
      Changed |= convertToVLMAX(MI);
      Changed |= tryToReduceVL(MI);
      Changed |= convertToUnmasked(MI);
      Changed |= convertToWholeRegister(MI);
      Changed |= convertAllOnesVMergeToVMv(MI);
      Changed |= convertSameMaskVMergeToVMv(MI);
      if (foldUndefPassthruVMV_V_V(MI)) {
        Changed |= true;
        continue; // MI is erased
      }
      Changed |= foldVMV_V_V(MI);
    }
  }

  return Changed;
}

FunctionPass *llvm::createCapstoneVectorPeepholePass() {
  return new CapstoneVectorPeephole();
}
