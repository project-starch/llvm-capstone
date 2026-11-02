//===- CapstoneFoldMemOffset.cpp - Fold ADDI into memory offsets ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// Look for ADDIs that can be removed by folding their immediate into later
// load/store addresses. There may be other arithmetic instructions between the
// addi and load/store that we need to reassociate through. If the final result
// of the arithmetic is only used by load/store addresses, we can fold the
// offset into the all the load/store as long as it doesn't create an offset
// that is too large.
//
//===---------------------------------------------------------------------===//

#include "Capstone.h"
#include "CapstoneSubtarget.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include <queue>

using namespace llvm;

#define DEBUG_TYPE "capstone-fold-mem-offset"
#define Capstone_FOLD_MEM_OFFSET_NAME "Capstone Fold Memory Offset"

namespace {

class CapstoneFoldMemOffset : public MachineFunctionPass {
public:
  static char ID;

  CapstoneFoldMemOffset() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  bool foldOffset(Register OrigReg, int64_t InitialOffset,
                  const MachineRegisterInfo &MRI,
                  DenseMap<MachineInstr *, int64_t> &FoldableInstrs);

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override { return Capstone_FOLD_MEM_OFFSET_NAME; }
};

// Wrapper class around a std::optional to allow accumulation.
class FoldableOffset {
  std::optional<int64_t> Offset;

public:
  bool hasValue() const { return Offset.has_value(); }
  int64_t getValue() const { return *Offset; }

  FoldableOffset &operator=(int64_t RHS) {
    Offset = RHS;
    return *this;
  }

  FoldableOffset &operator+=(int64_t RHS) {
    if (!Offset)
      Offset = 0;
    Offset = (uint64_t)*Offset + (uint64_t)RHS;
    return *this;
  }

  int64_t operator*() { return *Offset; }
};

} // end anonymous namespace

char CapstoneFoldMemOffset::ID = 0;
INITIALIZE_PASS(CapstoneFoldMemOffset, DEBUG_TYPE, Capstone_FOLD_MEM_OFFSET_NAME,
                false, false)

FunctionPass *llvm::createCapstoneFoldMemOffsetPass() {
  return new CapstoneFoldMemOffset();
}

// Walk forward from the ADDI looking for arithmetic instructions we can
// analyze or memory instructions that use it as part of their address
// calculation. For each arithmetic instruction we lookup how the offset
// contributes to the value in that register use that information to
// calculate the contribution to the output of this instruction.
// Only addition and left shift are supported.
// FIXME: Add multiplication by constant. The constant will be in a register.
bool CapstoneFoldMemOffset::foldOffset(
    Register OrigReg, int64_t InitialOffset, const MachineRegisterInfo &MRI,
    DenseMap<MachineInstr *, int64_t> &FoldableInstrs) {
  // Map to hold how much the offset contributes to the value of this register.
  DenseMap<Register, int64_t> RegToOffsetMap;

  // Capability cursors the fold leaves in place (CIncOffset / CIncOffsetImm
  // results downstream of the root), and the registers that end up as the
  // base of a folded memory op. Erasing the root moves every surviving cursor
  // by the root's constant, and a cursor that drifts more than ~2 KiB outside
  // its object loses its tag (R-60). A surviving cursor that is the base of a
  // folded access sits within that access's simm12 displacement of an
  // in-bounds address, inside the window; one that only feeds further cursor
  // arithmetic has nothing bounding it: with bounds [0x10000,0x10010) and
  // cursor 0x10000, `CIncOffsetImm 2047; CIncOffset -3072; CIncOffset 1025;
  // LBU 0` visits 0x107ff, 0xfbff, 0x10000 (all representable), while the
  // folded form visits 0xf400 first, untagged, and the in-bounds load faults.
  // So every surviving capability intermediate must be a folded base, or the
  // whole fold is vetoed (checked after the walk).
  DenseSet<Register> CapIntermediates, FoldedBases;

  // Insert root offset into the map.
  RegToOffsetMap[OrigReg] = InitialOffset;

  std::queue<Register> Worklist;
  Worklist.push(OrigReg);

  while (!Worklist.empty()) {
    Register Reg = Worklist.front();
    Worklist.pop();

    if (!Reg.isVirtual())
      return false;

    for (auto &User : MRI.use_nodbg_instructions(Reg)) {
      FoldableOffset Offset;

      switch (User.getOpcode()) {
      default:
        return false;
      case Capstone::ADD:
      // A cursor add moves a capability by a register amount; the folded
      // constant rides on whichever operand carries it (the capability base
      // for `cincoffset cap, idx`, where the dynamic idx is not in the map).
      // Same accumulation as ADD.
      case Capstone::CIncOffset:
        if (auto I = RegToOffsetMap.find(User.getOperand(1).getReg());
            I != RegToOffsetMap.end())
          Offset = I->second;
        if (auto I = RegToOffsetMap.find(User.getOperand(2).getReg());
            I != RegToOffsetMap.end())
          Offset += I->second;
        if (User.getOpcode() == Capstone::CIncOffset)
          CapIntermediates.insert(User.getOperand(0).getReg());
        break;
      // A constant cursor move downstream of the root stays in place with its
      // own immediate (only the root is erased), so just the folded delta
      // flows through it, as through ADD. Adding its immediate here counted
      // it twice: 538.imagick_r CloneImageInfo then stored at s7+0x3c+0x238
      // for s7+0x40 and faulted on bounds (cause 28).
      case Capstone::CIncOffsetImm:
        if (auto I = RegToOffsetMap.find(User.getOperand(1).getReg());
            I != RegToOffsetMap.end())
          Offset = I->second;
        CapIntermediates.insert(User.getOperand(0).getReg());
        break;
      case Capstone::SH1ADD:
        if (auto I = RegToOffsetMap.find(User.getOperand(1).getReg());
            I != RegToOffsetMap.end())
          Offset = (uint64_t)I->second << 1;
        if (auto I = RegToOffsetMap.find(User.getOperand(2).getReg());
            I != RegToOffsetMap.end())
          Offset += I->second;
        break;
      case Capstone::SH2ADD:
        if (auto I = RegToOffsetMap.find(User.getOperand(1).getReg());
            I != RegToOffsetMap.end())
          Offset = (uint64_t)I->second << 2;
        if (auto I = RegToOffsetMap.find(User.getOperand(2).getReg());
            I != RegToOffsetMap.end())
          Offset += I->second;
        break;
      case Capstone::SH3ADD:
        if (auto I = RegToOffsetMap.find(User.getOperand(1).getReg());
            I != RegToOffsetMap.end())
          Offset = (uint64_t)I->second << 3;
        if (auto I = RegToOffsetMap.find(User.getOperand(2).getReg());
            I != RegToOffsetMap.end())
          Offset += I->second;
        break;
      case Capstone::ADD_UW:
      case Capstone::SH1ADD_UW:
      case Capstone::SH2ADD_UW:
      case Capstone::SH3ADD_UW:
        // Don't fold through the zero extended input.
        if (User.getOperand(1).getReg() == Reg)
          return false;
        if (auto I = RegToOffsetMap.find(User.getOperand(2).getReg());
            I != RegToOffsetMap.end())
          Offset = I->second;
        break;
      case Capstone::SLLI: {
        unsigned ShAmt = User.getOperand(2).getImm();
        if (auto I = RegToOffsetMap.find(User.getOperand(1).getReg());
            I != RegToOffsetMap.end())
          Offset = (uint64_t)I->second << ShAmt;
        break;
      }
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
      case Capstone::FLD:
      case Capstone::SD:
      case Capstone::FSD:
      // Capability load/store (128-bit). Same operand order: dest/value, base,
      // simm12. Every memory op on this target already takes a GPCR base, so
      // these opcodes are the capability-addressed forms.
      case Capstone::LDC:
      case Capstone::STC: {
        // Can't fold into store value.
        if (User.getOperand(0).getReg() == Reg)
          return false;

        // Existing offset must be immediate.
        if (!User.getOperand(2).isImm())
          return false;

        // Require at least one operation between the ADDI and the load/store.
        // We have other optimizations that should handle the simple case.
        if (User.getOperand(1).getReg() == OrigReg)
          return false;

        auto I = RegToOffsetMap.find(User.getOperand(1).getReg());
        if (I == RegToOffsetMap.end())
          return false;

        int64_t LocalOffset = User.getOperand(2).getImm();
        assert(isInt<12>(LocalOffset));
        int64_t CombinedOffset = (uint64_t)LocalOffset + (uint64_t)I->second;
        if (!isInt<12>(CombinedOffset))
          return false;

        FoldableInstrs[&User] = CombinedOffset;
        FoldedBases.insert(User.getOperand(1).getReg());
        continue;
      }
      }

      // If we reach here we should have an accumulated offset.
      assert(Offset.hasValue() && "Expected an offset");

      // If the offset is new or changed, add the destination register to the
      // work list.
      int64_t OffsetVal = Offset.getValue();
      auto P =
          RegToOffsetMap.try_emplace(User.getOperand(0).getReg(), OffsetVal);
      if (P.second) {
        Worklist.push(User.getOperand(0).getReg());
      } else if (P.first->second != OffsetVal) {
        P.first->second = OffsetVal;
        Worklist.push(User.getOperand(0).getReg());
      }
    }
  }

  // Every surviving capability cursor must be the base of a folded access
  // (see the comment at CapIntermediates); otherwise the fold could move it
  // out of the representable window.
  for (Register R : CapIntermediates)
    if (!FoldedBases.count(R))
      return false;

  return true;
}

bool CapstoneFoldMemOffset::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  // This optimization may increase size by preventing compression.
  if (MF.getFunction().hasOptSize())
    return false;

  MachineRegisterInfo &MRI = MF.getRegInfo();

  bool MadeChange = false;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : llvm::make_early_inc_range(MBB)) {
      // FIXME: We can support ADDIW from an LUI+ADDIW pair if the result is
      // equivalent to LUI+ADDI.
      // CIncOffsetImm is the capability cursor-move analogue of ADDI: same
      // operand order (dest, base reg, simm12) and a pure constant move, so it
      // roots a fold the same way. It lets a constant cursor offset sink past a
      // dynamic `cincoffset` into the load/store immediate (libm exp).
      unsigned RootOpc = MI.getOpcode();
      if (RootOpc != Capstone::ADDI && RootOpc != Capstone::CIncOffsetImm)
        continue;

      // We only want to optimize register roots.
      if (!MI.getOperand(1).isReg() || !MI.getOperand(2).isImm())
        continue;

      // Ignore 'li' (ADDI from x0). A cincoffsetimm base is always a capability
      // register, never x0.
      if (RootOpc == Capstone::ADDI && MI.getOperand(1).getReg() == Capstone::X0)
        continue;

      int64_t Offset = MI.getOperand(2).getImm();
      assert(isInt<12>(Offset));

      DenseMap<MachineInstr *, int64_t> FoldableInstrs;

      if (!foldOffset(MI.getOperand(0).getReg(), Offset, MRI, FoldableInstrs))
        continue;

      if (FoldableInstrs.empty())
        continue;

      // We can fold this ADDI.
      // Rewrite all the instructions.
      for (auto [MemMI, NewOffset] : FoldableInstrs)
        MemMI->getOperand(2).setImm(NewOffset);

      MRI.replaceRegWith(MI.getOperand(0).getReg(), MI.getOperand(1).getReg());
      MRI.clearKillFlags(MI.getOperand(1).getReg());
      MI.eraseFromParent();
    }
  }

  return MadeChange;
}
