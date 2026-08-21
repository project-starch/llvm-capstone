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
#include "llvm/CodeGen/ReachingDefAnalysis.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "capstone-postra-expand"



// C-14 fixup. ON by default since 8f332e5d9d4d (the comment here said "off by default"
// until 2026-08-06, three weeks after that stopped being true). Setting it to false is the
// correct A/B control: it restores the exact codegen that wedged locfl3 on silicon.
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

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<ReachingDefAnalysis>();
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
  // MOVCs whose SOURCE the reaching-def proof showed to be a plain scalar. Computed for
  // the whole function BEFORE any rewriting, because ReachingDefAnalysis indexes
  // instructions and this pass erases them.
  SmallPtrSet<const MachineInstr *, 16> RDAScalarSrcMovc;
  void computeRDAScalarMovcs(MachineFunction &MF, ReachingDefAnalysis &RDA);

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
  RDAScalarSrcMovc.clear();
  if (CapstoneFixDestructiveCopies)
    computeRDAScalarMovcs(MF, getAnalysis<ReachingDefAnalysis>());
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
// HISTORY (2026-08-06). This whole-function rule was once labelled "THIS ANALYSIS DOES NOT
// WORK ... the fix belongs PRE-RA". That was half right, and the half that was wrong cost a
// board slot, so both halves are recorded.
//
// RIGHT: as a rule on its own it is too coarse, and for the reason given -- a physical
// register is REUSED across a function, so one non-materialising def anywhere (a reload
// `ldc sN, off(sp)`, a call result) disqualifies every site. `loc1` is the proof: this rule
// rewrites NOTHING there, and `loc1_kernel.h` calls itself "smallest possible form of the
// construct that wedges". `locfl3` being fixed by it is luck of register allocation.
//
// WRONG: "post-RA is simply too late to tell a scalar from a capability". It is not. The
// missing ingredient was not SSA, it was asking per SITE instead of per FUNCTION --
// isScalarByReachingDef below does that with ReachingDefAnalysis and handles every shape
// this rule misses. Two earlier attempts had failed and were over-generalised from:
//   * block-local backward walk -- the pointer is materialised in the loop PREHEADER and
//     copied in the BODY, so the def is never in the same block;
//   * single-predecessor CFG walk -- a loop body has two predecessors (preheader +
//     back-edge), so the walk gives up exactly where it is needed.
// Reaching definitions answer both; neither justified abandoning post-RA.
//
// The rule is KEPT because it is cheap and still catches sites where every def in the
// function is a materialisation but the reaching-def query cannot see them.
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

  // A LIVE-IN arrives from OUTSIDE the function, so the def scan below never sees the value
  // the register actually holds on entry. "Every def is an address materialisation" is then
  // VACUOUSLY true for an incoming pointer argument whose register is later reused as scratch
  // for an indirect call target -- and the copy that saves the incoming capability gets
  // rewritten to ADDI, dropping the tag.
  //
  //     void domain_main(unsigned *res, unsigned f) {
  //       unsigned g = gate; v(); w(); *res = f + g;   // v/w return void, so no implicit
  //     }                                              // def of $x10 from the calls
  //
  // $x10's only defs are then the AUIPC/ADDI pairs computing @v and @w, all of them
  // materialisations, and `movc s2, a0` becomes `mv s2, a0`. Measured exposure across the
  // corpus is currently zero, so this is latent rather than shipping -- but it is the same
  // shape as the cause-24 fault the blanket `capstone-scalar-copy-live-src` default produced
  // in matmult_int, which is exactly the failure this pass exists to avoid.
  //
  // DO NOT extend this to regmask clobbers. a2/a3/a4 are call-clobbered in locfl3, and
  // treating a regmask as a def would disqualify precisely the function-pointer registers
  // this pass must rewrite, re-creating the silicon wedge.
  //
  // Only ever REMOVES registers from Scalar, so it is monotonically conservative: strictly
  // fewer rewrites, never more.
  for (const auto &LI : MF.getRegInfo().liveins())
    Disqualified.insert(LI.first);

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
          } else if ((Opc == Capstone::ADDI || Opc == Capstone::ADDIW ||
                      Opc == Capstone::MOVC) &&
                     MI.getNumOperands() >= 2 && MI.getOperand(1).isReg()) {
            Register Src = MI.getOperand(1).getReg();
            // X0 counts as a scalar SOURCE. `li rd, imm` is `ADDI rd, x0, imm`, the most
            // provably-scalar thing in the ISA, and `movc rd, zero` reads the hardwired
            // zero register, which is not a capability either. Excluding it (the previous
            // `Src != X0`) is what left matmult_int's loop counters unprovable: `a4` is
            // `li a4, 1` plus `addi a4, a4, 1`, and rejecting the `li` disqualified the
            // whole chain, so `movc t1, a4` stood and zeroed the counter on the first
            // iteration. ADDIW is here for the same reason -- `addiw s2, s2, 1` is the
            // other half of that chain.
            Ok = Src == Capstone::X0 ||
                 (Scalar.count(Src) && !Disqualified.count(Src));
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

// PER-SITE scalar proof, via reaching definitions.
//
// This supersedes the whole-function per-register fixpoint above as the primary rule,
// because that fixpoint asks the wrong question. It asks "is EVERY def of this register
// anywhere in the function a materialisation", so one unrelated reuse of the register
// poisons every site. `loc1` is the counterexample that forced this, and it matters
// because loc1_kernel.h calls itself "smallest possible form of the construct that
// wedges":
//
//     1103d0: auipc a0, 0x0
//     1103d4: auipc a2, 0x0
//     1103d8: addi  s4, a0, 0x8c     <- &loc1_f0, loop-invariant
//     1103dc: addi  s5, a2, 0xb0     <- &loc1_f1
//     ...
//     110408: movc  a0, s5           <- DESTROYS s5, live across all 8 iterations
//     110410: movc  a0, s4           <- DESTROYS s4
//
// The fixpoint rewrites NOTHING here (verified: the pass on and off produce byte-identical
// images). `a0` is reused as the indirect-call target and return value, so it is
// disqualified; s4/s5 are ADDIs off `a0`, so they inherit the disqualification. locfl3 is
// only fixed because its pointers happened to land in a2/a3/a4, which have no other def --
// i.e. by luck of register allocation, not by the analysis. That is exactly what the NOTE
// above predicted, and it is why locfl3 alone is not an acceptable regression test.
//
// The right question is per-SITE: which definition actually reaches THIS movc? At 0x110408
// the unique reaching def of s5 is `addi s5, a2, 0xb0`, and the unique reaching def of a2
// there is `auipc a2` -- a plain PC-relative address, provably not a capability. Whatever
// else `a0` or `a2` hold elsewhere in the function is irrelevant to this site.
//
// getUniqueReachingMIDef returns null when several defs reach (a join, a loop-carried
// value), so an ambiguous site keeps MOVC. Conservative by construction.
//
// ADDI from X0 counts: `li rd, imm` is the most provably-scalar thing in the ISA. The
// fixpoint above explicitly excludes it (`Src != Capstone::X0`), which is why matmult_int
// keeps `movc t1, a4` / `movc t2, a3` over `li a4, 1` / `li a3, 2`.
// A single def is proof only if it is a materialisation or forwards one.
static bool isScalarDefChain(MachineInstr *Def, ReachingDefAnalysis &RDA,
                             unsigned Depth,
                             SmallPtrSetImpl<MachineInstr *> &Visited);

static bool isScalarByReachingDef(MachineInstr *MI, Register Reg,
                                  ReachingDefAnalysis &RDA, unsigned Depth,
                                  SmallPtrSetImpl<MachineInstr *> &Visited) {
  if (Depth > 6 || !Reg.isPhysical())
    return false;
  if (Reg == Capstone::X0)
    return true;

  // ALL reaching defs, not just a unique one. A loop counter has two -- the initialiser
  // and its own increment on the back-edge -- so getUniqueReachingMIDef returns null for
  // exactly the shape that matters. matmult_int:
  //     110430: movc s2, zero          <- initialiser
  //     110454: movc a0, s2            <- the destructive copy, zeroing the OUTER index
  //     110478: addiw s2, s2, 0x1      <- increment, reaches 110454 via the back-edge
  // Both defs are scalar, so the value at 110454 is scalar on every path. The
  // whole-function fixpoint cannot say so either, because `s2` is genuinely a CAPABILITY
  // elsewhere in the same function (`cincoffset s2, gp, zero` at 110044, `ldc s2, 0x50(sp)`
  // at 1104f8) -- which is the precise reason the question has to be asked per site.
  //
  // Visited breaks the self-reference: `addiw s2, s2, 1` reaches itself round the loop.
  // Treating an already-visited def as satisfied is sound here because the recursion only
  // ever concludes "scalar" when EVERY def on every path is a materialisation -- a cycle
  // adds no new way for a capability to enter.
  // A REGISTER THAT IS LIVE-IN TO THE FUNCTION CAN NEVER BE PROVEN SCALAR HERE,
  // because the value it arrives with has no defining MachineInstr and therefore
  // cannot appear in Defs at all. Without this the proof runs over an incomplete
  // set, and the `Visited` cycle rule then closes it over nothing:
  //
  //     529:  $x9  = MOVC $x10        <- the copy being judged
  //     580:  $x10 = MOVC $x9         <- reaches 529 via the back-edge
  //
  // Those are the ONLY defs of $x10 in mrb_method_search_vm. Judging 529 asks about
  // $x10, gets 580, which forwards to $x9, whose def is 529 -- already in Visited,
  // so "already on the path: contributes no new source" returns true. Every path is
  // the cycle, no path reaches a materialisation, and the rule concludes scalar for
  // the incoming `mrb`, which is a live capability. The rewritten `mv` drops its tag
  // and mruby faults with cause 24 at the next store through it (C-14, measured
  // 2026-08-21 under both ABIs).
  //
  // The cycle rule is sound only when some path leaves the cycle and lands on a real
  // materialisation. For a live-in the only such path is the function entry, which is
  // exactly what RDA cannot show us -- so refuse.
  //
  // computeScalarAddressRegs already disqualifies live-ins for the same reason
  // (`for (const auto &LI : MF.getRegInfo().liveins()) Disqualified.insert(...)`).
  // This rule was simply missing the guard its sibling had.
  if (MI->getMF()->getRegInfo().isLiveIn(Reg))
    return false;

  SmallPtrSet<MachineInstr *, 8> Defs;
  RDA.getGlobalReachingDefs(MI, Reg, Defs);
  if (Defs.empty())
    return false; // not visible: prove nothing
  for (MachineInstr *Def : Defs)
    if (!isScalarDefChain(Def, RDA, Depth, Visited))
      return false;
  return true;
}

static bool isScalarDefChain(MachineInstr *Def, ReachingDefAnalysis &RDA,
                             unsigned Depth,
                             SmallPtrSetImpl<MachineInstr *> &Visited) {
  if (!Visited.insert(Def).second)
    return true; // already on the path: contributes no new source of a capability
  switch (Def->getOpcode()) {
  case Capstone::AUIPC:
  case Capstone::LUI:
    return true;
  case Capstone::ADDI:
  case Capstone::ADDIW:
  case Capstone::MOVC:
    // Completes a materialisation, or forwards one. Recurse on the value it reads.
    if (Def->getNumOperands() >= 2 && Def->getOperand(1).isReg())
      return isScalarByReachingDef(Def, Def->getOperand(1).getReg(), RDA, Depth + 1,
                                   Visited);
    return false;
  default:
    // LDC / CINCOFFSET / SCC / CAPENTER / a call result / a reload -- may be a capability.
    return false;
  }
}

void CapstonePostRAExpandPseudo::computeRDAScalarMovcs(MachineFunction &MF,
                                                       ReachingDefAnalysis &RDA) {
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : MBB) {
      if (MI.getOpcode() != Capstone::MOVC || MI.getNumOperands() < 2)
        continue;
      if (!MI.getOperand(1).isReg())
        continue;
      Register Src = MI.getOperand(1).getReg();
      if (Src == Capstone::X0 || Src == MI.getOperand(0).getReg())
        continue;
      SmallPtrSet<MachineInstr *, 8> Visited;
      if (isScalarByReachingDef(&MI, Src, RDA, 0, Visited))
        RDAScalarSrcMovc.insert(&MI);
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
    bool RDAScalar = RDAScalarSrcMovc.count(&*I) != 0;
    LLVM_DEBUG(dbgs() << "C14: movc dst=" << printReg(Dst, TRI) << " src="
                      << printReg(Src, TRI) << " Decided=" << Decided
                      << " ScalarUse=" << ScalarUse
                      << " scalarAddr=" << ScalarAddrRegs.count(Src)
                      << " rdaScalar=" << RDAScalar << "\n");
    // The USE-based proof cannot classify a function pointer -- its first use is movc/stc/jalr.
    // Two DEF-based proofs cover it. The reaching-def one is per-site and strictly stronger;
    // the whole-function fixpoint is kept because it still catches sites where the unique
    // reaching def is not visible (e.g. a value materialised in another block that RDA
    // reports as several defs) while every def in the function is a materialisation anyway.
    if ((!Decided || !ScalarUse) && (RDAScalar || ScalarAddrRegs.count(Src))) {
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
