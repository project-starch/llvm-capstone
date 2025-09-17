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
        if (auto I = RegToOffsetMap.find(User.getOperand(1).getReg());
            I != RegToOffsetMap.end())
          Offset = I->second;
        if (auto I = RegToOffsetMap.find(User.getOperand(2).getReg());
            I != RegToOffsetMap.end())
          Offset += I->second;
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
      case Capstone::FSD: {
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
      if (MI.getOpcode() != Capstone::ADDI)
        continue;

      // We only want to optimize register ADDIs.
      if (!MI.getOperand(1).isReg() || !MI.getOperand(2).isImm())
        continue;

      // Ignore 'li'.
      if (MI.getOperand(1).getReg() == Capstone::X0)
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
