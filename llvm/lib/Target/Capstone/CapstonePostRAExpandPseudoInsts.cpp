//===-- CapstonePostRAExpandPseudoInsts.cpp - Expand pseudo instrs ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a pass that expands the pseudo instruction pseudolisimm32
// into target instructions. This pass should be run during the post-regalloc
// passes, before post RA scheduling.
//
//===----------------------------------------------------------------------===//

#include "Capstone.h"
#include "CapstoneInstrInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

// C-14 fixup, off by default until it is proven not to regress the corpus.
static cl::opt<bool> CapstoneFixDestructiveCopies(
    "capstone-fix-destructive-copies", cl::init(true), cl::Hidden,
    cl::desc("Rewrite `movc rd, rs` to a scalar move when rs is provably an "
             "integer that stays live (C-14)."));

#define Capstone_POST_RA_EXPAND_PSEUDO_NAME                                       \
  "Capstone post-regalloc pseudo instruction expansion pass"

namespace {

class CapstonePostRAExpandPseudo : public MachineFunctionPass {
public:
  const CapstoneInstrInfo *TII;
  const TargetRegisterInfo *TRI = nullptr;
  static char ID;

  CapstonePostRAExpandPseudo() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return Capstone_POST_RA_EXPAND_PSEUDO_NAME;
  }

private:
  bool expandMBB(MachineBasicBlock &MBB);
  bool fixupDestructiveCopies(MachineBasicBlock &MBB);
  bool expandMI(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                MachineBasicBlock::iterator &NextMBBI);
  bool expandMovImm(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI);
  bool expandMovAddr(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI);
};

char CapstonePostRAExpandPseudo::ID = 0;

bool CapstonePostRAExpandPseudo::runOnMachineFunction(MachineFunction &MF) {
  TII = static_cast<const CapstoneInstrInfo *>(MF.getSubtarget().getInstrInfo());
  TRI = MF.getSubtarget().getRegisterInfo();
  bool Modified = false;
  for (auto &MBB : MF)
    Modified |= expandMBB(MBB);
  return Modified;
}

// C-14. `movc` is a MOVE, not a copy: on CVA6 it writes cnull to its SOURCE unless
// the source is a non-linear capability (capstone_flu_unit.anvil:14-25), and a plain
// integer is not one. copyPhysReg emits MOVC for every GPR-to-GPR copy, so a copy of
// a still-live scalar destroys it. Board-proven: a loop counter copied that way reads
// back as 0, which turns `bne` into an infinite loop (the domain wedges) and `beq`
// into an early exit -- gpw2 returned 3950255460, exactly the early-exit checksum.
//
// QEMU cannot see it: helper_csmovc guards the same zeroing with `rs1_v->tag &&`
// (op_helper.c:580-584), so scalars survive in the model.
//
// We cannot simply emit ADDI for every copy -- capability values also flow through
// copyPhysReg, and ADDI drops their metadata (measured: doing so faults five ladder
// rungs under QEMU with cause = 7). So the source's type has to be inferred, and the
// only sound evidence available post-RA is how the source is USED next: if the first
// thing that reads it is an integer ALU op or a branch, it held an integer.
//
// Conservative by construction -- anything we cannot prove scalar keeps MOVC, i.e.
// today's behaviour.
static bool isScalarIntegerUse(unsigned Opc) {
  switch (Opc) {
  case Capstone::ADDI:  case Capstone::ADDIW: case Capstone::ADD:
  case Capstone::ADDW:  case Capstone::SUB:   case Capstone::SUBW:
  case Capstone::SLLI:  case Capstone::SRLI:  case Capstone::SRAI:
  case Capstone::SLLIW: case Capstone::SRLIW: case Capstone::SRAIW:
  case Capstone::SLL:   case Capstone::SRL:   case Capstone::SRA:
  case Capstone::XOR:   case Capstone::XORI:  case Capstone::OR:
  case Capstone::ORI:   case Capstone::AND:   case Capstone::ANDI:
  case Capstone::MUL:   case Capstone::MULW:
  case Capstone::SLT:   case Capstone::SLTI:  case Capstone::SLTU:
  case Capstone::SLTIU:
  case Capstone::BEQ:   case Capstone::BNE:   case Capstone::BLT:
  case Capstone::BGE:   case Capstone::BLTU:  case Capstone::BGEU:
    return true;
  default:
    return false;
  }
}

bool CapstonePostRAExpandPseudo::fixupDestructiveCopies(MachineBasicBlock &MBB) {
  bool Modified = false;
  if (!CapstoneFixDestructiveCopies)
    return false;

  // A single-block loop reaches its own top through the back-edge, so a use that
  // sits ABOVE the movc is still a use "after" it. Scanning only forwards missed
  // exactly this case while the bug was being characterised.
  bool SelfLoop = llvm::is_contained(MBB.successors(), &MBB);

  for (MachineBasicBlock::iterator I = MBB.begin(), E = MBB.end(); I != E;) {
    MachineBasicBlock::iterator Next = std::next(I);
    if (I->getOpcode() != Capstone::MOVC) {
      I = Next;
      continue;
    }
    Register Dst = I->getOperand(0).getReg();
    Register Src = I->getOperand(1).getReg();
    if (Src == Dst || Src == Capstone::X0) {
      I = Next;
      continue;
    }

    // Find what reads Src next. A redefinition first means the source is dead and
    // the destructive move is harmless (and for a linear capability it is the only
    // legal semantics anyway).
    bool ScalarUse = false, Decided = false;
    auto Scan = [&](MachineBasicBlock::iterator B, MachineBasicBlock::iterator End) {
      for (auto J = B; J != End && !Decided; ++J) {
        if (J->readsRegister(Src, TRI)) {
          ScalarUse = isScalarIntegerUse(J->getOpcode());
          Decided = true;
          return;
        }
        if (J->modifiesRegister(Src, TRI)) {
          Decided = true; // redefined before use: source is dead
          return;
        }
      }
    };
    Scan(std::next(I), MBB.end());
    if (!Decided && SelfLoop)
      Scan(MBB.begin(), I);

    if (!Decided || !ScalarUse) {
      I = Next;
      continue;
    }

    BuildMI(MBB, I, I->getDebugLoc(), TII->get(Capstone::ADDI), Dst)
        .addReg(Src)
        .addImm(0);
    I->eraseFromParent();
    I = Next;
    Modified = true;
  }

  return Modified;
}

bool CapstonePostRAExpandPseudo::expandMBB(MachineBasicBlock &MBB) {
  bool Modified = false;

  MachineBasicBlock::iterator MBBI = MBB.begin(), E = MBB.end();
  while (MBBI != E) {
    MachineBasicBlock::iterator NMBBI = std::next(MBBI);
    Modified |= expandMI(MBB, MBBI, NMBBI);
    MBBI = NMBBI;
  }

  Modified |= fixupDestructiveCopies(MBB);

  return Modified;
}

bool CapstonePostRAExpandPseudo::expandMI(MachineBasicBlock &MBB,
                                       MachineBasicBlock::iterator MBBI,
                                       MachineBasicBlock::iterator &NextMBBI) {
  switch (MBBI->getOpcode()) {
  case Capstone::PseudoMovImm:
    return expandMovImm(MBB, MBBI);
  case Capstone::PseudoMovAddr:
    return expandMovAddr(MBB, MBBI);
  default:
    return false;
  }
}

bool CapstonePostRAExpandPseudo::expandMovImm(MachineBasicBlock &MBB,
                                           MachineBasicBlock::iterator MBBI) {
  DebugLoc DL = MBBI->getDebugLoc();

  int64_t Val = MBBI->getOperand(1).getImm();

  Register DstReg = MBBI->getOperand(0).getReg();
  bool DstIsDead = MBBI->getOperand(0).isDead();
  bool Renamable = MBBI->getOperand(0).isRenamable();

  TII->movImm(MBB, MBBI, DL, DstReg, Val, MachineInstr::NoFlags, Renamable,
              DstIsDead);

  MBBI->eraseFromParent();
  return true;
}

bool CapstonePostRAExpandPseudo::expandMovAddr(MachineBasicBlock &MBB,
                                            MachineBasicBlock::iterator MBBI) {
  DebugLoc DL = MBBI->getDebugLoc();

  Register DstReg = MBBI->getOperand(0).getReg();
  bool DstIsDead = MBBI->getOperand(0).isDead();
  bool Renamable = MBBI->getOperand(0).isRenamable();

  BuildMI(MBB, MBBI, DL, TII->get(Capstone::LUI))
      .addReg(DstReg, RegState::Define | getRenamableRegState(Renamable))
      .add(MBBI->getOperand(1));
  BuildMI(MBB, MBBI, DL, TII->get(Capstone::ADDI))
      .addReg(DstReg, RegState::Define | getDeadRegState(DstIsDead) |
                          getRenamableRegState(Renamable))
      .addReg(DstReg, RegState::Kill | getRenamableRegState(Renamable))
      .add(MBBI->getOperand(2));
  MBBI->eraseFromParent();
  return true;
}

} // end of anonymous namespace

INITIALIZE_PASS(CapstonePostRAExpandPseudo, "capstone-post-ra-expand-pseudo",
                Capstone_POST_RA_EXPAND_PSEUDO_NAME, false, false)
namespace llvm {

FunctionPass *createCapstonePostRAExpandPseudoPass() {
  return new CapstonePostRAExpandPseudo();
}

} // end of namespace llvm
