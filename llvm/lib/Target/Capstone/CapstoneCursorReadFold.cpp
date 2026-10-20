//===-- CapstoneCursorReadFold.cpp - Read a cursor where the ALU can ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A capability's address reaches integer code through PseudoTRUNC_CAP, an
// integer write (`addi rd, rs, 0`) of the capability register's low half. The
// write exists for C-31: a bare sub-register read coalesces into the capability
// register itself, whose metadata shadow stays tagged on the RTL, and a tagged
// register as the integer operand of cincoffset, scc or shrink raises
// UNEXPECTED_OPERAND there (capstone_flu_unit.anvil:30). The write clears the
// shadow (alu-write-clears-shadow.S) and so protects every later use.
//
// Most reads of an address need no such protection. A compare, a branch, a
// subtraction or a mask reads the register's low half on both implementations,
// and its own result is an integer write. QEMU reads gpr[i].val, the cursor.
// On the RTL the capability metadata travels with every operand
// (issue_read_operands.sv packs it unconditionally), but only the capability
// units look at it: core/alu.sv, mult.sv and multiplier.sv never reference
// cap_metadata, the branch unit reads it only for CJALR and CBNZ
// (branch_unit.sv:95-101), and UNEXPECTED_OPERAND is the FLU's check on a
// capability instruction (capstone_flu_unit.anvil:35-36); all read at
// capstone-ariane 19ac4c22ccfe (virtual-profile-2-c). An earlier citation of
// issue_read_operands.sv:1663 named S-12's WAW guard, not an opcode gate. When
// EVERY reader of the extracted address is such an instruction, the address can
// be read straight from the capability register and the pseudo goes:
//
//   mv   a4, a0               ->   bne  a0, a2, .L
//   bne  a4, a2, .L
//
// Readers that keep the pseudo: a capability instruction's integer operand
// (C-31), a store, a copy (PHI, COPY, a call argument, a return), a select's
// data operands, any reader that could later become a copy of the address
// (see readsAsIntegerOnly), and anything not listed -- a value that leaves
// through one of those could later be read as a register that still looks
// tagged. The fold therefore never creates a new integer value that lives on;
// it only moves a read closer to the register it came from.
//
// Measured on the SPEC test inputs (instruction attempts, Capstone over the
// plain RISC-V build): the surplus of `mv` and `movc` was 22% of gcc's compiled
// code gap, 49% of imagick's and 13% of nab's; the loop exits `mv; bne` and the
// pointer differences `mv; mv; sub` are this shape.
//
//===----------------------------------------------------------------------===//

#include "Capstone.h"
#include "CapstoneInstrInfo.h"
#include "CapstoneSubtarget.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

using namespace llvm;

#define DEBUG_TYPE "capstone-cursor-read-fold"
#define CAPSTONE_CURSOR_READ_FOLD_NAME "Capstone cursor read folding"

STATISTIC(NumFolded, "Number of cursor reads folded into their readers");

namespace {
class CapstoneCursorReadFold : public MachineFunctionPass {
public:
  static char ID;

  CapstoneCursorReadFold() : MachineFunctionPass(ID) {}
  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
  StringRef getPassName() const override {
    return CAPSTONE_CURSOR_READ_FOLD_NAME;
  }
};
} // end anonymous namespace

char CapstoneCursorReadFold::ID = 0;
INITIALIZE_PASS(CapstoneCursorReadFold, DEBUG_TYPE,
                CAPSTONE_CURSOR_READ_FOLD_NAME, false, false)

FunctionPass *llvm::createCapstoneCursorReadFoldPass() {
  return new CapstoneCursorReadFold();
}

// A reader the fold may hand the capability register to: it reads the operand
// as a plain integer, and in that operand position it can never turn into a
// copy of the operand later. The copy forms this backend recognises
// (isCopyInstr, simplifyInstruction, and MachineCopyPropagation after
// allocation) copy the register that is NOT x0 in `add/or/xor rd, rs, x0`,
// `sub rd, rs, x0`, `addi rd, rs, 0` and a shift by x0 or by 0; any register
// operand may still become x0 after this pass (a COPY chain the coalescer
// folds, a computed zero a peephole proves), so a reader is admitted only
// where no value of its other operands makes it a copy of ours:
//   - a branch writes no register; a comparison writes 0 or 1;
//   - an immediate form with an immediate that is not the identity;
//   - a subtraction with the address as the subtrahend (`sub rd, x, addr`);
//   - `and` and `mul`, which have no copy form.
// `add`, `or`, `xor`, `sub rd, addr, x` and the register shifts are refused.
// The pass runs last among the machine SSA optimisations (after the
// combiner's reassociation, the peepholes and the W-instruction rewrites);
// what follows before allocation (pseudo expansion, base-offset merging)
// rewrites none of the admitted forms into a copy of their operand.
static bool readsAsIntegerOnly(const MachineInstr &MI, unsigned OpNo) {
  auto ImmIsNot = [&](unsigned ImmOp, std::initializer_list<int64_t> Bad) {
    const MachineOperand &MO = MI.getOperand(ImmOp);
    if (!MO.isImm())
      return false;
    for (int64_t B : Bad)
      if (MO.getImm() == B)
        return false;
    return true;
  };
  switch (MI.getOpcode()) {
  case Capstone::BEQ:
  case Capstone::BNE:
  case Capstone::BLT:
  case Capstone::BGE:
  case Capstone::BLTU:
  case Capstone::BGEU:
  case Capstone::SLT:
  case Capstone::SLTU:
  case Capstone::AND:
  case Capstone::MUL:
  case Capstone::MULW:
    return true;
  case Capstone::SLTI:
  case Capstone::SLTIU:
    return OpNo == 1;
  case Capstone::ADDI:
  case Capstone::ADDIW:
  case Capstone::ORI:
  case Capstone::XORI:
  case Capstone::SLLI:
  case Capstone::SRLI:
  case Capstone::SRAI:
    return OpNo == 1 && ImmIsNot(2, {0});
  case Capstone::ANDI:
    return OpNo == 1 && ImmIsNot(2, {0, -1});
  case Capstone::SUB:
  case Capstone::SUBW:
    return OpNo == 2;
  case Capstone::Select_GPR_Using_CC_GPR:
  case Capstone::Select_GPRCAP_Using_CC_GPR:
    // (outs $dst), (ins $lhs, $rhs, $cc, $truev, $falsev): the compared
    // operands only; the data operands become the result through a PHI.
    return OpNo == 1 || OpNo == 2;
  default:
    return false;
  }
}

bool CapstoneCursorReadFold::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  MachineRegisterInfo &MRI = MF.getRegInfo();
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : llvm::make_early_inc_range(MBB)) {
      if (MI.getOpcode() != Capstone::PseudoTRUNC_CAP)
        continue;
      Register Dst = MI.getOperand(0).getReg();
      const MachineOperand &SrcMO = MI.getOperand(1);
      if (!Dst.isVirtual() || !SrcMO.isReg() || !SrcMO.getReg().isVirtual())
        continue;
      // The source is the address half of the capability register: selection
      // emits `%addr:gpr = COPY %cap.sub_cap_addr` and the pseudo reads %addr.
      // The readers take that operand; the coalescer then folds the copy into
      // the capability register itself, the bare sub-register read that is
      // exactly right for an integer reader and wrong for a capability one.
      Register Src = SrcMO.getReg();
      unsigned SubIdx = SrcMO.getSubReg();

      bool AllIntegerReaders = true;
      for (const MachineOperand &Use : MRI.use_operands(Dst)) {
        const MachineInstr &User = *Use.getParent();
        // A reader that is itself a copy (`addi rd, rs, 0` and the other forms
        // isCopyInstr recognises) writes nothing of its own: copy propagation
        // would carry the capability register through it to wherever its
        // result goes, which is the C-31 escape this pass must not open.
        if (Use.isTied() || !Use.isUse() || TII->isCopyInstr(User) ||
            !readsAsIntegerOnly(User, User.getOperandNo(&Use))) {
          AllIntegerReaders = false;
          break;
        }
      }
      if (!AllIntegerReaders)
        continue;

      // The readers now use Src, which may live on past a use that killed it
      // before: no use of Src keeps a kill flag (the rewritten ones and the
      // untouched ones alike).
      for (MachineOperand &Use :
           llvm::make_early_inc_range(MRI.use_operands(Dst))) {
        Use.setReg(Src);
        Use.setSubReg(SubIdx);
      }
      MRI.clearKillFlags(Src);
      MI.eraseFromParent();
      ++NumFolded;
      Changed = true;
    }
  }
  return Changed;
}
