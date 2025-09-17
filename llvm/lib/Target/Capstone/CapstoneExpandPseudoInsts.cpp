//===-- CapstoneExpandPseudoInsts.cpp - Expand pseudo instructions -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a pass that expands pseudo instructions into target
// instructions. This pass should be run after register allocation but before
// the post-regalloc scheduling pass.
//
//===----------------------------------------------------------------------===//

#include "Capstone.h"
#include "CapstoneInstrInfo.h"
#include "CapstoneTargetMachine.h"

#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/MC/MCContext.h"

using namespace llvm;

#define Capstone_EXPAND_PSEUDO_NAME "Capstone pseudo instruction expansion pass"
#define Capstone_PRERA_EXPAND_PSEUDO_NAME "Capstone Pre-RA pseudo instruction expansion pass"

namespace {

class CapstoneExpandPseudo : public MachineFunctionPass {
public:
  const CapstoneSubtarget *STI;
  const CapstoneInstrInfo *TII;
  static char ID;

  CapstoneExpandPseudo() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return Capstone_EXPAND_PSEUDO_NAME; }

private:
  bool expandMBB(MachineBasicBlock &MBB);
  bool expandMI(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                MachineBasicBlock::iterator &NextMBBI);
  bool expandCCOp(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                  MachineBasicBlock::iterator &NextMBBI);
  bool expandVMSET_VMCLR(MachineBasicBlock &MBB,
                         MachineBasicBlock::iterator MBBI, unsigned Opcode);
  bool expandMV_FPR16INX(MachineBasicBlock &MBB,
                         MachineBasicBlock::iterator MBBI);
  bool expandMV_FPR32INX(MachineBasicBlock &MBB,
                         MachineBasicBlock::iterator MBBI);
  bool expandRV32ZdinxStore(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator MBBI);
  bool expandRV32ZdinxLoad(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MBBI);
  bool expandPseudoReadVLENBViaVSETVLIX0(MachineBasicBlock &MBB,
                                         MachineBasicBlock::iterator MBBI);
#ifndef NDEBUG
  unsigned getInstSizeInBytes(const MachineFunction &MF) const {
    unsigned Size = 0;
    for (auto &MBB : MF)
      for (auto &MI : MBB)
        Size += TII->getInstSizeInBytes(MI);
    return Size;
  }
#endif
};

char CapstoneExpandPseudo::ID = 0;

bool CapstoneExpandPseudo::runOnMachineFunction(MachineFunction &MF) {
  STI = &MF.getSubtarget<CapstoneSubtarget>();
  TII = STI->getInstrInfo();

#ifndef NDEBUG
  const unsigned OldSize = getInstSizeInBytes(MF);
#endif

  bool Modified = false;
  for (auto &MBB : MF)
    Modified |= expandMBB(MBB);

#ifndef NDEBUG
  const unsigned NewSize = getInstSizeInBytes(MF);
  assert(OldSize >= NewSize);
#endif
  return Modified;
}

bool CapstoneExpandPseudo::expandMBB(MachineBasicBlock &MBB) {
  bool Modified = false;

  MachineBasicBlock::iterator MBBI = MBB.begin(), E = MBB.end();
  while (MBBI != E) {
    MachineBasicBlock::iterator NMBBI = std::next(MBBI);
    Modified |= expandMI(MBB, MBBI, NMBBI);
    MBBI = NMBBI;
  }

  return Modified;
}

bool CapstoneExpandPseudo::expandMI(MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator MBBI,
                                 MachineBasicBlock::iterator &NextMBBI) {
  // CapstoneInstrInfo::getInstSizeInBytes expects that the total size of the
  // expanded instructions for each pseudo is correct in the Size field of the
  // tablegen definition for the pseudo.
  switch (MBBI->getOpcode()) {
  case Capstone::PseudoMV_FPR16INX:
    return expandMV_FPR16INX(MBB, MBBI);
  case Capstone::PseudoMV_FPR32INX:
    return expandMV_FPR32INX(MBB, MBBI);
  case Capstone::PseudoRV32ZdinxSD:
    return expandRV32ZdinxStore(MBB, MBBI);
  case Capstone::PseudoRV32ZdinxLD:
    return expandRV32ZdinxLoad(MBB, MBBI);
  case Capstone::PseudoCCMOVGPRNoX0:
  case Capstone::PseudoCCMOVGPR:
  case Capstone::PseudoCCADD:
  case Capstone::PseudoCCSUB:
  case Capstone::PseudoCCAND:
  case Capstone::PseudoCCOR:
  case Capstone::PseudoCCXOR:
  case Capstone::PseudoCCADDW:
  case Capstone::PseudoCCSUBW:
  case Capstone::PseudoCCSLL:
  case Capstone::PseudoCCSRL:
  case Capstone::PseudoCCSRA:
  case Capstone::PseudoCCADDI:
  case Capstone::PseudoCCSLLI:
  case Capstone::PseudoCCSRLI:
  case Capstone::PseudoCCSRAI:
  case Capstone::PseudoCCANDI:
  case Capstone::PseudoCCORI:
  case Capstone::PseudoCCXORI:
  case Capstone::PseudoCCSLLW:
  case Capstone::PseudoCCSRLW:
  case Capstone::PseudoCCSRAW:
  case Capstone::PseudoCCADDIW:
  case Capstone::PseudoCCSLLIW:
  case Capstone::PseudoCCSRLIW:
  case Capstone::PseudoCCSRAIW:
  case Capstone::PseudoCCANDN:
  case Capstone::PseudoCCORN:
  case Capstone::PseudoCCXNOR:
  case Capstone::PseudoCCNDS_BFOS:
  case Capstone::PseudoCCNDS_BFOZ:
    return expandCCOp(MBB, MBBI, NextMBBI);
  case Capstone::PseudoVMCLR_M_B1:
  case Capstone::PseudoVMCLR_M_B2:
  case Capstone::PseudoVMCLR_M_B4:
  case Capstone::PseudoVMCLR_M_B8:
  case Capstone::PseudoVMCLR_M_B16:
  case Capstone::PseudoVMCLR_M_B32:
  case Capstone::PseudoVMCLR_M_B64:
    // vmclr.m vd => vmxor.mm vd, vd, vd
    return expandVMSET_VMCLR(MBB, MBBI, Capstone::VMXOR_MM);
  case Capstone::PseudoVMSET_M_B1:
  case Capstone::PseudoVMSET_M_B2:
  case Capstone::PseudoVMSET_M_B4:
  case Capstone::PseudoVMSET_M_B8:
  case Capstone::PseudoVMSET_M_B16:
  case Capstone::PseudoVMSET_M_B32:
  case Capstone::PseudoVMSET_M_B64:
    // vmset.m vd => vmxnor.mm vd, vd, vd
    return expandVMSET_VMCLR(MBB, MBBI, Capstone::VMXNOR_MM);
  case Capstone::PseudoReadVLENBViaVSETVLIX0:
    return expandPseudoReadVLENBViaVSETVLIX0(MBB, MBBI);
  }

  return false;
}

bool CapstoneExpandPseudo::expandCCOp(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator MBBI,
                                   MachineBasicBlock::iterator &NextMBBI) {

  MachineFunction *MF = MBB.getParent();
  MachineInstr &MI = *MBBI;
  DebugLoc DL = MI.getDebugLoc();

  MachineBasicBlock *TrueBB = MF->CreateMachineBasicBlock(MBB.getBasicBlock());
  MachineBasicBlock *MergeBB = MF->CreateMachineBasicBlock(MBB.getBasicBlock());

  MF->insert(++MBB.getIterator(), TrueBB);
  MF->insert(++TrueBB->getIterator(), MergeBB);

  // We want to copy the "true" value when the condition is true which means
  // we need to invert the branch condition to jump over TrueBB when the
  // condition is false.
  auto CC = static_cast<CapstoneCC::CondCode>(MI.getOperand(3).getImm());
  CC = CapstoneCC::getOppositeBranchCondition(CC);

  // Insert branch instruction.
  BuildMI(MBB, MBBI, DL, TII->get(CapstoneCC::getBrCond(CC)))
      .addReg(MI.getOperand(1).getReg())
      .addReg(MI.getOperand(2).getReg())
      .addMBB(MergeBB);

  Register DestReg = MI.getOperand(0).getReg();
  assert(MI.getOperand(4).getReg() == DestReg);

  if (MI.getOpcode() == Capstone::PseudoCCMOVGPR ||
      MI.getOpcode() == Capstone::PseudoCCMOVGPRNoX0) {
    // Add MV.
    BuildMI(TrueBB, DL, TII->get(Capstone::ADDI), DestReg)
        .add(MI.getOperand(5))
        .addImm(0);
  } else {
    unsigned NewOpc;
    switch (MI.getOpcode()) {
    default:
      llvm_unreachable("Unexpected opcode!");
    case Capstone::PseudoCCADD:   NewOpc = Capstone::ADD;   break;
    case Capstone::PseudoCCSUB:   NewOpc = Capstone::SUB;   break;
    case Capstone::PseudoCCSLL:   NewOpc = Capstone::SLL;   break;
    case Capstone::PseudoCCSRL:   NewOpc = Capstone::SRL;   break;
    case Capstone::PseudoCCSRA:   NewOpc = Capstone::SRA;   break;
    case Capstone::PseudoCCAND:   NewOpc = Capstone::AND;   break;
    case Capstone::PseudoCCOR:    NewOpc = Capstone::OR;    break;
    case Capstone::PseudoCCXOR:   NewOpc = Capstone::XOR;   break;
    case Capstone::PseudoCCADDI:  NewOpc = Capstone::ADDI;  break;
    case Capstone::PseudoCCSLLI:  NewOpc = Capstone::SLLI;  break;
    case Capstone::PseudoCCSRLI:  NewOpc = Capstone::SRLI;  break;
    case Capstone::PseudoCCSRAI:  NewOpc = Capstone::SRAI;  break;
    case Capstone::PseudoCCANDI:  NewOpc = Capstone::ANDI;  break;
    case Capstone::PseudoCCORI:   NewOpc = Capstone::ORI;   break;
    case Capstone::PseudoCCXORI:  NewOpc = Capstone::XORI;  break;
    case Capstone::PseudoCCADDW:  NewOpc = Capstone::ADDW;  break;
    case Capstone::PseudoCCSUBW:  NewOpc = Capstone::SUBW;  break;
    case Capstone::PseudoCCSLLW:  NewOpc = Capstone::SLLW;  break;
    case Capstone::PseudoCCSRLW:  NewOpc = Capstone::SRLW;  break;
    case Capstone::PseudoCCSRAW:  NewOpc = Capstone::SRAW;  break;
    case Capstone::PseudoCCADDIW: NewOpc = Capstone::ADDIW; break;
    case Capstone::PseudoCCSLLIW: NewOpc = Capstone::SLLIW; break;
    case Capstone::PseudoCCSRLIW: NewOpc = Capstone::SRLIW; break;
    case Capstone::PseudoCCSRAIW: NewOpc = Capstone::SRAIW; break;
    case Capstone::PseudoCCANDN:  NewOpc = Capstone::ANDN;  break;
    case Capstone::PseudoCCORN:   NewOpc = Capstone::ORN;   break;
    case Capstone::PseudoCCXNOR:  NewOpc = Capstone::XNOR;  break;
    case Capstone::PseudoCCNDS_BFOS: NewOpc = Capstone::NDS_BFOS; break;
    case Capstone::PseudoCCNDS_BFOZ: NewOpc = Capstone::NDS_BFOZ; break;
    }

    if (NewOpc == Capstone::NDS_BFOZ || NewOpc == Capstone::NDS_BFOS) {
      BuildMI(TrueBB, DL, TII->get(NewOpc), DestReg)
          .add(MI.getOperand(5))
          .add(MI.getOperand(6))
          .add(MI.getOperand(7));
    } else {
      BuildMI(TrueBB, DL, TII->get(NewOpc), DestReg)
          .add(MI.getOperand(5))
          .add(MI.getOperand(6));
    }
  }

  TrueBB->addSuccessor(MergeBB);

  MergeBB->splice(MergeBB->end(), &MBB, MI, MBB.end());
  MergeBB->transferSuccessors(&MBB);

  MBB.addSuccessor(TrueBB);
  MBB.addSuccessor(MergeBB);

  NextMBBI = MBB.end();
  MI.eraseFromParent();

  // Make sure live-ins are correctly attached to this new basic block.
  LivePhysRegs LiveRegs;
  computeAndAddLiveIns(LiveRegs, *TrueBB);
  computeAndAddLiveIns(LiveRegs, *MergeBB);

  return true;
}

bool CapstoneExpandPseudo::expandVMSET_VMCLR(MachineBasicBlock &MBB,
                                          MachineBasicBlock::iterator MBBI,
                                          unsigned Opcode) {
  DebugLoc DL = MBBI->getDebugLoc();
  Register DstReg = MBBI->getOperand(0).getReg();
  const MCInstrDesc &Desc = TII->get(Opcode);
  BuildMI(MBB, MBBI, DL, Desc, DstReg)
      .addReg(DstReg, RegState::Undef)
      .addReg(DstReg, RegState::Undef);
  MBBI->eraseFromParent(); // The pseudo instruction is gone now.
  return true;
}

bool CapstoneExpandPseudo::expandMV_FPR16INX(MachineBasicBlock &MBB,
                                          MachineBasicBlock::iterator MBBI) {
  DebugLoc DL = MBBI->getDebugLoc();
  const TargetRegisterInfo *TRI = STI->getRegisterInfo();
  Register DstReg = TRI->getMatchingSuperReg(
      MBBI->getOperand(0).getReg(), Capstone::sub_16, &Capstone::GPRRegClass);
  Register SrcReg = TRI->getMatchingSuperReg(
      MBBI->getOperand(1).getReg(), Capstone::sub_16, &Capstone::GPRRegClass);

  BuildMI(MBB, MBBI, DL, TII->get(Capstone::ADDI), DstReg)
      .addReg(SrcReg, getKillRegState(MBBI->getOperand(1).isKill()))
      .addImm(0);

  MBBI->eraseFromParent(); // The pseudo instruction is gone now.
  return true;
}

bool CapstoneExpandPseudo::expandMV_FPR32INX(MachineBasicBlock &MBB,
                                          MachineBasicBlock::iterator MBBI) {
  DebugLoc DL = MBBI->getDebugLoc();
  const TargetRegisterInfo *TRI = STI->getRegisterInfo();
  Register DstReg = TRI->getMatchingSuperReg(
      MBBI->getOperand(0).getReg(), Capstone::sub_32, &Capstone::GPRRegClass);
  Register SrcReg = TRI->getMatchingSuperReg(
      MBBI->getOperand(1).getReg(), Capstone::sub_32, &Capstone::GPRRegClass);

  BuildMI(MBB, MBBI, DL, TII->get(Capstone::ADDI), DstReg)
      .addReg(SrcReg, getKillRegState(MBBI->getOperand(1).isKill()))
      .addImm(0);

  MBBI->eraseFromParent(); // The pseudo instruction is gone now.
  return true;
}

// This function expands the PseudoRV32ZdinxSD for storing a double-precision
// floating-point value into memory by generating an equivalent instruction
// sequence for RV32.
bool CapstoneExpandPseudo::expandRV32ZdinxStore(MachineBasicBlock &MBB,
                                             MachineBasicBlock::iterator MBBI) {
  DebugLoc DL = MBBI->getDebugLoc();
  const TargetRegisterInfo *TRI = STI->getRegisterInfo();
  Register Lo =
      TRI->getSubReg(MBBI->getOperand(0).getReg(), Capstone::sub_gpr_even);
  Register Hi =
      TRI->getSubReg(MBBI->getOperand(0).getReg(), Capstone::sub_gpr_odd);
  if (Hi == Capstone::DUMMY_REG_PAIR_WITH_X0)
    Hi = Capstone::X0;

  auto MIBLo = BuildMI(MBB, MBBI, DL, TII->get(Capstone::SW))
                   .addReg(Lo, getKillRegState(MBBI->getOperand(0).isKill()))
                   .addReg(MBBI->getOperand(1).getReg())
                   .add(MBBI->getOperand(2));

  MachineInstrBuilder MIBHi;
  if (MBBI->getOperand(2).isGlobal() || MBBI->getOperand(2).isCPI()) {
    assert(MBBI->getOperand(2).getOffset() % 8 == 0);
    MBBI->getOperand(2).setOffset(MBBI->getOperand(2).getOffset() + 4);
    MIBHi = BuildMI(MBB, MBBI, DL, TII->get(Capstone::SW))
                .addReg(Hi, getKillRegState(MBBI->getOperand(0).isKill()))
                .add(MBBI->getOperand(1))
                .add(MBBI->getOperand(2));
  } else {
    assert(isInt<12>(MBBI->getOperand(2).getImm() + 4));
    MIBHi = BuildMI(MBB, MBBI, DL, TII->get(Capstone::SW))
                .addReg(Hi, getKillRegState(MBBI->getOperand(0).isKill()))
                .add(MBBI->getOperand(1))
                .addImm(MBBI->getOperand(2).getImm() + 4);
  }

  MachineFunction *MF = MBB.getParent();
  SmallVector<MachineMemOperand *> NewLoMMOs;
  SmallVector<MachineMemOperand *> NewHiMMOs;
  for (const MachineMemOperand *MMO : MBBI->memoperands()) {
    NewLoMMOs.push_back(MF->getMachineMemOperand(MMO, 0, 4));
    NewHiMMOs.push_back(MF->getMachineMemOperand(MMO, 4, 4));
  }
  MIBLo.setMemRefs(NewLoMMOs);
  MIBHi.setMemRefs(NewHiMMOs);

  MBBI->eraseFromParent();
  return true;
}

// This function expands PseudoRV32ZdinxLoad for loading a double-precision
// floating-point value from memory into an equivalent instruction sequence for
// RV32.
bool CapstoneExpandPseudo::expandRV32ZdinxLoad(MachineBasicBlock &MBB,
                                            MachineBasicBlock::iterator MBBI) {
  DebugLoc DL = MBBI->getDebugLoc();
  const TargetRegisterInfo *TRI = STI->getRegisterInfo();
  Register Lo =
      TRI->getSubReg(MBBI->getOperand(0).getReg(), Capstone::sub_gpr_even);
  Register Hi =
      TRI->getSubReg(MBBI->getOperand(0).getReg(), Capstone::sub_gpr_odd);
  assert(Hi != Capstone::DUMMY_REG_PAIR_WITH_X0 && "Cannot write to X0_Pair");

  MachineInstrBuilder MIBLo, MIBHi;

  // If the register of operand 1 is equal to the Lo register, then swap the
  // order of loading the Lo and Hi statements.
  bool IsOp1EqualToLo = Lo == MBBI->getOperand(1).getReg();
  // Order: Lo, Hi
  if (!IsOp1EqualToLo) {
    MIBLo = BuildMI(MBB, MBBI, DL, TII->get(Capstone::LW), Lo)
                .addReg(MBBI->getOperand(1).getReg())
                .add(MBBI->getOperand(2));
  }

  if (MBBI->getOperand(2).isGlobal() || MBBI->getOperand(2).isCPI()) {
    auto Offset = MBBI->getOperand(2).getOffset();
    assert(Offset % 8 == 0);
    MBBI->getOperand(2).setOffset(Offset + 4);
    MIBHi = BuildMI(MBB, MBBI, DL, TII->get(Capstone::LW), Hi)
                .addReg(MBBI->getOperand(1).getReg())
                .add(MBBI->getOperand(2));
    MBBI->getOperand(2).setOffset(Offset);
  } else {
    assert(isInt<12>(MBBI->getOperand(2).getImm() + 4));
    MIBHi = BuildMI(MBB, MBBI, DL, TII->get(Capstone::LW), Hi)
                .addReg(MBBI->getOperand(1).getReg())
                .addImm(MBBI->getOperand(2).getImm() + 4);
  }

  // Order: Hi, Lo
  if (IsOp1EqualToLo) {
    MIBLo = BuildMI(MBB, MBBI, DL, TII->get(Capstone::LW), Lo)
                .addReg(MBBI->getOperand(1).getReg())
                .add(MBBI->getOperand(2));
  }

  MachineFunction *MF = MBB.getParent();
  SmallVector<MachineMemOperand *> NewLoMMOs;
  SmallVector<MachineMemOperand *> NewHiMMOs;
  for (const MachineMemOperand *MMO : MBBI->memoperands()) {
    NewLoMMOs.push_back(MF->getMachineMemOperand(MMO, 0, 4));
    NewHiMMOs.push_back(MF->getMachineMemOperand(MMO, 4, 4));
  }
  MIBLo.setMemRefs(NewLoMMOs);
  MIBHi.setMemRefs(NewHiMMOs);

  MBBI->eraseFromParent();
  return true;
}

bool CapstoneExpandPseudo::expandPseudoReadVLENBViaVSETVLIX0(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI) {
  DebugLoc DL = MBBI->getDebugLoc();
  Register Dst = MBBI->getOperand(0).getReg();
  unsigned Mul = MBBI->getOperand(1).getImm();
  CapstoneVType::VLMUL VLMUL = CapstoneVType::encodeLMUL(Mul, /*Fractional=*/false);
  unsigned VTypeImm = CapstoneVType::encodeVTYPE(
      VLMUL, /*SEW=*/8, /*TailAgnostic=*/true, /*MaskAgnostic=*/true);

  BuildMI(MBB, MBBI, DL, TII->get(Capstone::PseudoVSETVLIX0))
      .addReg(Dst, RegState::Define)
      .addReg(Capstone::X0, RegState::Kill)
      .addImm(VTypeImm);

  MBBI->eraseFromParent();
  return true;
}

class CapstonePreRAExpandPseudo : public MachineFunctionPass {
public:
  const CapstoneSubtarget *STI;
  const CapstoneInstrInfo *TII;
  static char ID;

  CapstonePreRAExpandPseudo() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
  StringRef getPassName() const override {
    return Capstone_PRERA_EXPAND_PSEUDO_NAME;
  }

private:
  bool expandMBB(MachineBasicBlock &MBB);
  bool expandMI(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                MachineBasicBlock::iterator &NextMBBI);
  bool expandAuipcInstPair(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MBBI,
                           MachineBasicBlock::iterator &NextMBBI,
                           unsigned FlagsHi, unsigned SecondOpcode);
  bool expandLoadLocalAddress(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator MBBI,
                              MachineBasicBlock::iterator &NextMBBI);
  bool expandLoadGlobalAddress(MachineBasicBlock &MBB,
                               MachineBasicBlock::iterator MBBI,
                               MachineBasicBlock::iterator &NextMBBI);
  bool expandLoadTLSIEAddress(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator MBBI,
                              MachineBasicBlock::iterator &NextMBBI);
  bool expandLoadTLSGDAddress(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator MBBI,
                              MachineBasicBlock::iterator &NextMBBI);
  bool expandLoadTLSDescAddress(MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator MBBI,
                                MachineBasicBlock::iterator &NextMBBI);

#ifndef NDEBUG
  unsigned getInstSizeInBytes(const MachineFunction &MF) const {
    unsigned Size = 0;
    for (auto &MBB : MF)
      for (auto &MI : MBB)
        Size += TII->getInstSizeInBytes(MI);
    return Size;
  }
#endif
};

char CapstonePreRAExpandPseudo::ID = 0;

bool CapstonePreRAExpandPseudo::runOnMachineFunction(MachineFunction &MF) {
  STI = &MF.getSubtarget<CapstoneSubtarget>();
  TII = STI->getInstrInfo();

#ifndef NDEBUG
  const unsigned OldSize = getInstSizeInBytes(MF);
#endif

  bool Modified = false;
  for (auto &MBB : MF)
    Modified |= expandMBB(MBB);

#ifndef NDEBUG
  const unsigned NewSize = getInstSizeInBytes(MF);
  assert(OldSize >= NewSize);
#endif
  return Modified;
}

bool CapstonePreRAExpandPseudo::expandMBB(MachineBasicBlock &MBB) {
  bool Modified = false;

  MachineBasicBlock::iterator MBBI = MBB.begin(), E = MBB.end();
  while (MBBI != E) {
    MachineBasicBlock::iterator NMBBI = std::next(MBBI);
    Modified |= expandMI(MBB, MBBI, NMBBI);
    MBBI = NMBBI;
  }

  return Modified;
}

bool CapstonePreRAExpandPseudo::expandMI(MachineBasicBlock &MBB,
                                      MachineBasicBlock::iterator MBBI,
                                      MachineBasicBlock::iterator &NextMBBI) {

  switch (MBBI->getOpcode()) {
  case Capstone::PseudoLLA:
    return expandLoadLocalAddress(MBB, MBBI, NextMBBI);
  case Capstone::PseudoLGA:
    return expandLoadGlobalAddress(MBB, MBBI, NextMBBI);
  case Capstone::PseudoLA_TLS_IE:
    return expandLoadTLSIEAddress(MBB, MBBI, NextMBBI);
  case Capstone::PseudoLA_TLS_GD:
    return expandLoadTLSGDAddress(MBB, MBBI, NextMBBI);
  case Capstone::PseudoLA_TLSDESC:
    return expandLoadTLSDescAddress(MBB, MBBI, NextMBBI);
  }
  return false;
}

bool CapstonePreRAExpandPseudo::expandAuipcInstPair(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    MachineBasicBlock::iterator &NextMBBI, unsigned FlagsHi,
    unsigned SecondOpcode) {
  MachineFunction *MF = MBB.getParent();
  MachineInstr &MI = *MBBI;
  DebugLoc DL = MI.getDebugLoc();

  Register DestReg = MI.getOperand(0).getReg();
  Register ScratchReg =
      MF->getRegInfo().createVirtualRegister(&Capstone::GPRRegClass);

  MachineOperand &Symbol = MI.getOperand(1);
  Symbol.setTargetFlags(FlagsHi);
  MCSymbol *AUIPCSymbol = MF->getContext().createNamedTempSymbol("pcrel_hi");

  MachineInstr *MIAUIPC =
      BuildMI(MBB, MBBI, DL, TII->get(Capstone::AUIPC), ScratchReg).add(Symbol);
  MIAUIPC->setPreInstrSymbol(*MF, AUIPCSymbol);

  MachineInstr *SecondMI =
      BuildMI(MBB, MBBI, DL, TII->get(SecondOpcode), DestReg)
          .addReg(ScratchReg)
          .addSym(AUIPCSymbol, CapstoneII::MO_PCREL_LO);

  if (MI.hasOneMemOperand())
    SecondMI->addMemOperand(*MF, *MI.memoperands_begin());

  MI.eraseFromParent();
  return true;
}

bool CapstonePreRAExpandPseudo::expandLoadLocalAddress(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    MachineBasicBlock::iterator &NextMBBI) {
  return expandAuipcInstPair(MBB, MBBI, NextMBBI, CapstoneII::MO_PCREL_HI,
                             Capstone::ADDI);
}

bool CapstonePreRAExpandPseudo::expandLoadGlobalAddress(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    MachineBasicBlock::iterator &NextMBBI) {
  unsigned SecondOpcode = STI->is64Bit() ? Capstone::LD : Capstone::LW;
  return expandAuipcInstPair(MBB, MBBI, NextMBBI, CapstoneII::MO_GOT_HI,
                             SecondOpcode);
}

bool CapstonePreRAExpandPseudo::expandLoadTLSIEAddress(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    MachineBasicBlock::iterator &NextMBBI) {
  unsigned SecondOpcode = STI->is64Bit() ? Capstone::LD : Capstone::LW;
  return expandAuipcInstPair(MBB, MBBI, NextMBBI, CapstoneII::MO_TLS_GOT_HI,
                             SecondOpcode);
}

bool CapstonePreRAExpandPseudo::expandLoadTLSGDAddress(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    MachineBasicBlock::iterator &NextMBBI) {
  return expandAuipcInstPair(MBB, MBBI, NextMBBI, CapstoneII::MO_TLS_GD_HI,
                             Capstone::ADDI);
}

bool CapstonePreRAExpandPseudo::expandLoadTLSDescAddress(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    MachineBasicBlock::iterator &NextMBBI) {
  MachineFunction *MF = MBB.getParent();
  MachineInstr &MI = *MBBI;
  DebugLoc DL = MI.getDebugLoc();

  const auto &STI = MF->getSubtarget<CapstoneSubtarget>();
  unsigned SecondOpcode = STI.is64Bit() ? Capstone::LD : Capstone::LW;

  Register FinalReg = MI.getOperand(0).getReg();
  Register DestReg =
      MF->getRegInfo().createVirtualRegister(&Capstone::GPRRegClass);
  Register ScratchReg =
      MF->getRegInfo().createVirtualRegister(&Capstone::GPRRegClass);

  MachineOperand &Symbol = MI.getOperand(1);
  Symbol.setTargetFlags(CapstoneII::MO_TLSDESC_HI);
  MCSymbol *AUIPCSymbol = MF->getContext().createNamedTempSymbol("tlsdesc_hi");

  MachineInstr *MIAUIPC =
      BuildMI(MBB, MBBI, DL, TII->get(Capstone::AUIPC), ScratchReg).add(Symbol);
  MIAUIPC->setPreInstrSymbol(*MF, AUIPCSymbol);

  BuildMI(MBB, MBBI, DL, TII->get(SecondOpcode), DestReg)
      .addReg(ScratchReg)
      .addSym(AUIPCSymbol, CapstoneII::MO_TLSDESC_LOAD_LO);

  BuildMI(MBB, MBBI, DL, TII->get(Capstone::ADDI), Capstone::X10)
      .addReg(ScratchReg)
      .addSym(AUIPCSymbol, CapstoneII::MO_TLSDESC_ADD_LO);

  BuildMI(MBB, MBBI, DL, TII->get(Capstone::PseudoTLSDESCCall), Capstone::X5)
      .addReg(DestReg)
      .addImm(0)
      .addSym(AUIPCSymbol, CapstoneII::MO_TLSDESC_CALL);

  BuildMI(MBB, MBBI, DL, TII->get(Capstone::ADD), FinalReg)
      .addReg(Capstone::X10)
      .addReg(Capstone::X4);

  MI.eraseFromParent();
  return true;
}

} // end of anonymous namespace

INITIALIZE_PASS(CapstoneExpandPseudo, "capstone-expand-pseudo",
                Capstone_EXPAND_PSEUDO_NAME, false, false)

INITIALIZE_PASS(CapstonePreRAExpandPseudo, "capstone-prera-expand-pseudo",
                Capstone_PRERA_EXPAND_PSEUDO_NAME, false, false)

namespace llvm {

FunctionPass *createCapstoneExpandPseudoPass() { return new CapstoneExpandPseudo(); }
FunctionPass *createCapstonePreRAExpandPseudoPass() { return new CapstonePreRAExpandPseudo(); }

} // end of namespace llvm
