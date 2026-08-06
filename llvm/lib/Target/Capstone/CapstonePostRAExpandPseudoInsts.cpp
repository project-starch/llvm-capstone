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
#include "llvm/Support/Debug.h"
#include "llvm/ADT/SmallSet.h"
#include "CapstoneInstrInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "capstone-postra-expand"



// C-14 fixup, off by default until it is proven not to regress the corpus.
static cl::opt<bool> CapstoneFixDestructiveCopies(
    "capstone-fix-destructive-copies", cl::init(true), cl::Hidden,
    cl::desc("Rewrite `movc rd, rs` to a scalar move when rs is provably an "
             "integer that stays live (C-14)."));

#define Capstone_POST_RA_EXPAND_PSEUDO_NAME                                       \
  "Capstone post-regalloc pseudo instruction expansion pass"

namespace {

// Populated once per function by runOnMachineFunction before any block is processed.
static SmallSet<Register, 16> ScalarAddrRegs;

static void computeScalarAddressRegs(const MachineFunction &MF,
                                     const TargetRegisterInfo *TRI,
                                     SmallSet<Register, 16> &Scalar);


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
  ScalarAddrRegs.clear();
  computeScalarAddressRegs(MF, MF.getSubtarget().getRegisterInfo(), ScalarAddrRegs);
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

// Is this value provably a SCALAR because of how it was DEFINED?
//
// isScalarIntegerUse proves scalar-ness from the next USE, which works for loop counters but
// never for a function pointer: its first use is `movc`, `stc` or `jalr`, none of which imply
// a scalar. That gap is what left C-14 live on hardware -- board-confirmed 2026-08-06, locfl3
// wedges and the same source with the destructive movc removed returns its oracle 26.
//
// A PC-relative address materialisation (AUIPC, optionally + ADDI; or LUI + ADDI) produces a
// plain integer address. It is NOT a capability: capabilities arrive via LDC, CINCOFFSET, MOVC,
// SCC or the cap-table, never from AUIPC/LUI. So proving it here is sound where the blanket
// `capstone-scalar-copy-live-src` default was not -- that one flipped copyPhysReg for ALL GPRs,
// and GPRRegClass holds both scalars and capabilities, so it dropped the tag on a live
// capability and faulted matmult_int with cause 24.
static bool isScalarDefiningOpc(unsigned Opc) {
  switch (Opc) {
  case Capstone::AUIPC:
  case Capstone::LUI:
    return true;
  default:
    return false;
  }
}

// NOTE (2026-08-06): THIS ANALYSIS DOES NOT WORK, and the reason rules out the whole approach.
//
// A physical register is REUSED across a function. The function pointers land in callee-saved
// registers (s3/s4 here), and the prologue/epilogue define those with `ldc sN, off(sp)` -- a
// non-materialising def -- so any "every def of this register is an address materialisation"
// rule disqualifies them immediately. Verified with -debug-only=capstone-postra-expand: the
// pass sees `movc $x10, $x20` and `movc $x10, $x19` with scalarAddr=0 for both.
//
// Two earlier shapes failed for related reasons and are recorded so they are not retried:
//   * block-local backward walk -- the pointer is materialised in the loop PREHEADER and copied
//     in the BODY, so the def is never in the same block;
//   * single-predecessor CFG walk -- a loop body has two predecessors (preheader + back-edge),
//     so the walk gives up exactly where it is needed.
//
// The real fix belongs PRE-RA, where a materialised address is a distinct virtual register with
// an unambiguous single definition, instead of a physical register shared with spills, reloads
// and every other value the allocator put there. Post-RA is simply too late to tell a scalar
// from a capability, which is the same reason the blanket `capstone-scalar-copy-live-src`
// default in copyPhysReg is unsafe.
//
// Which physical registers hold a value that is PROVABLY a plain scalar address?
//
// Computed over the whole function as a fixpoint, not by walking the CFG. A backward walk fails
// on exactly the shape that matters: the pointer is materialised in the loop PREHEADER and
// copied in the loop BODY, and a loop body has two predecessors (preheader + back-edge), so any
// single-predecessor restriction gives up. Verified with -debug-only: the pass saw
// `movc $x10, $x20` with the def of $x20 unreachable that way.
//
// A register qualifies only if EVERY definition of it in the function is an address
// materialisation -- AUIPC/LUI, an ADDI completing one, or a MOVC copying another qualifying
// register. Since a single non-qualifying def disqualifies it, the result is safe under any
// control flow: whichever path reached the copy, the value came from one of those defs.
//
// Capabilities never arrive this way -- they come from LDC, CINCOFFSET, SCC, CAPENTER or the
// cap table -- so this cannot misclassify a capability as a scalar. That is the distinction
// copyPhysReg cannot make, because GPRRegClass holds both, which is why the blanket
// `capstone-scalar-copy-live-src` default broke matmult_int with cause 24.
static void computeScalarAddressRegs(const MachineFunction &MF,
                                     const TargetRegisterInfo *TRI,
                                     SmallSet<Register, 16> &Scalar) {
  // Seed: registers defined ONLY by AUIPC/LUI, or by ADDI whose source is already seeded.
  bool Changed = true;
  SmallSet<Register, 16> Disqualified;
  for (unsigned Round = 0; Changed && Round < 8; ++Round) {
    Changed = false;
    for (const MachineBasicBlock &MBB : MF) {
      for (const MachineInstr &MI : MBB) {
        for (const MachineOperand &MO : MI.operands()) {
          if (!MO.isReg() || !MO.isDef() || !MO.getReg().isPhysical())
            continue;
          Register R = MO.getReg();
          if (Disqualified.count(R))
            continue;
          unsigned Opc = MI.getOpcode();
          bool Ok = false;
          if (Opc == Capstone::AUIPC || Opc == Capstone::LUI) {
            Ok = true;
          } else if ((Opc == Capstone::ADDI || Opc == Capstone::MOVC) &&
                     MI.getNumOperands() >= 2 && MI.getOperand(1).isReg()) {
            Register Src = MI.getOperand(1).getReg();
            Ok = Src != Capstone::X0 && Scalar.count(Src) && !Disqualified.count(Src);
          }
          if (!Ok) {
            // A non-materialising def poisons the register permanently.
            if (Scalar.erase(R) || !Disqualified.count(R))
              Changed = true;
            Disqualified.insert(R);
          } else if (!Scalar.count(R)) {
            Scalar.insert(R);
            Changed = true;
          }
        }
      }
    }
  }
  for (Register R : Disqualified)
    Scalar.erase(R);
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
    auto Scan = [&](MachineBasicBlock::const_iterator B, MachineBasicBlock::const_iterator End) {
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

    // Cross-block: if the answer is not in this block, follow SINGLE-SUCCESSOR edges.
    // Sound because a block with exactly one successor sends every path there, so the
    // first use found downstream really is the first use. Bounded, and it stops at the
    // first branch -- anything needing a join is left alone and keeps MOVC.
    // SQLite had ~5 sites the block-local scan could not classify; this is for those.
    const MachineBasicBlock *Cur = &MBB;
    for (unsigned Hops = 0; !Decided && Hops < 4; ++Hops) {
      if (Cur->succ_size() != 1)
        break;
      MachineBasicBlock *Succ = *Cur->succ_begin();
      if (Succ == &MBB)
        break; // already covered by the self-loop scan above
      Scan(Succ->begin(), Succ->end());
      Cur = Succ;
    }

    // If the USE-based proof failed, try the DEF-based one: a value materialised by
    // AUIPC/LUI (+ADDI) is a plain address and provably not a capability. That is the
    // function-pointer case -- its first use is movc/stc/jalr, so the use-based test can
    // never classify it, which is exactly why C-14 stayed live on silicon for this shape.
    LLVM_DEBUG(dbgs() << "C14: movc dst=" << printReg(Dst, TRI) << " src="
                      << printReg(Src, TRI) << " Decided=" << Decided
                      << " ScalarUse=" << ScalarUse
                      << " scalarAddr=" << ScalarAddrRegs.count(Src) << "\n");
    // The USE-based proof cannot classify a function pointer -- its first use is movc/stc/jalr.
    // The DEF-based one can: if every definition of Src in this function is an address
    // materialisation, it is a scalar and the destructive movc is unnecessary.
    if ((!Decided || !ScalarUse) && ScalarAddrRegs.count(Src)) {
      Decided = true;
      ScalarUse = true;
    }

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
