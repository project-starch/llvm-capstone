//===-- CapstoneRedundantShrink.cpp - Drop narrowings nothing can exceed -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// selectLGA narrows a data global's capability to the object, [&g, &g +
// sizeof(g)), with SHRINKTO (or li + add + SHRINK), so that an access leaving
// the object faults. When EVERY reader of the narrowed capability is a load or
// store that uses it as the base at a constant offset, and each access lies
// inside the object (0 <= offset and offset + width <= size), no access through
// it can leave the object, so none of those accesses can fault on the narrowed
// bounds. The readers can take the un-narrowed base instead and the narrowing
// goes. The base is the same address with the bounds of the image's data
// authority (the gp-derived PseudoCapGlobalBase, NONLIN under every ABI), so
// every access reads and writes the same bytes.
//
// What goes with it is the narrowing instruction's OWN check: a SHRINKTO or
// SHRINK faults when the requested end lies past the base's bounds. That fires
// when the object does not fit the authority the unit believes it has (an
// adjust-1 SHRINKTO ending exactly at the end of gp's grant, or a declaration
// whose real definition is smaller than this unit's type). So by default only
// exact, non-interposable DEFINITIONS are handled, whose size the unit knows.
// With -capstone-drop-redundant-shrink-declarations a pure declaration is
// handled too, and the furthest byte its remaining accesses reach is recorded
// in the non-allocated section .capstone.size_assume (8-byte LE size, then the
// NUL-terminated name). capstone/tests/capstone-size-assume-check.py fails a
// linked image whose definition is smaller, missing or duplicated.
//
//   scc      a0, gp, a0          ->   scc  a0, gp, a0
//   shrinkto a0, a0, 17               ldc  a1, 0(a0)
//   ldc      a1, 0(a0)
//
// A capability that leaves through anything else keeps its narrowing: stored as
// a value, passed to a call or returned, copied into a PHI, used by arithmetic
// (cincoffset, scc, shrink), compared, or used by a memory access whose width is
// unknown or whose offset is not an immediate. A plain COPY is followed to its
// readers. Only global bases are handled (the SHRINKTO / SHRINK source is a
// PseudoCapGlobalBase); stack objects keep theirs.
//
// The size of a SHRINKTO is its immediate minus -capstone-shrinkto-end-adjust
// (the object's true size); a SHRINK's is the constant in `add end, cursor, k`
// with `cursor` the base's own address operand.
//
// Measured on the SPEC test inputs (QEMU per-PC profile, perlbench test on
// the unified stack with -capstone-direct-calls -fomit-frame-pointer): a lower
// bound of 9.0 SHRINKTO and 0.7 SHRINK per 1,000 Plain instructions are of
// this shape, most in S_regmatch's loop (`ld a0, off(sp); scc; shrinkto; ldc
// a0, 0(..)` of a 16-byte global).
//
//===----------------------------------------------------------------------===//

#include "Capstone.h"
#include "CapstoneInstrInfo.h"
#include "CapstoneMachineFunctionInfo.h"
#include "CapstoneSubtarget.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/GlobalVariable.h"

using namespace llvm;

#define DEBUG_TYPE "capstone-redundant-shrink"
#define CAPSTONE_REDUNDANT_SHRINK_NAME "Capstone redundant global narrowing"

STATISTIC(NumDropped, "Number of global narrowings no access can exceed");

// CapstoneISelDAGToDAG.cpp: the SHRINKTO length is sizeof(g) plus this.
extern cl::opt<unsigned> CapstoneShrinkToEndAdjust;

static cl::opt<bool> CapstoneRedundantShrinkDeclarations(
    "capstone-drop-redundant-shrink-declarations", cl::Hidden,
    cl::desc("Also drop the narrowing of a global this unit only declares "
             "(not extern_weak): its size is this unit's belief, recorded in "
             ".capstone.size_assume for capstone-size-assume-check.py"),
    cl::init(false));

namespace {
class CapstoneRedundantShrink : public MachineFunctionPass {
public:
  static char ID;

  CapstoneRedundantShrink() : MachineFunctionPass(ID) {}
  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
  StringRef getPassName() const override {
    return CAPSTONE_REDUNDANT_SHRINK_NAME;
  }
};
} // end anonymous namespace

char CapstoneRedundantShrink::ID = 0;
INITIALIZE_PASS(CapstoneRedundantShrink, DEBUG_TYPE,
                CAPSTONE_REDUNDANT_SHRINK_NAME, false, false)

FunctionPass *llvm::createCapstoneRedundantShrinkPass() {
  return new CapstoneRedundantShrink();
}

// The value of a virtual register built from constants by li (addi x0),
// lui and addi/addiw, or nothing.
static std::optional<int64_t> constantOf(const MachineRegisterInfo &MRI,
                                         Register R, unsigned Depth = 0) {
  if (!R.isVirtual() || Depth > 4)
    return std::nullopt;
  const MachineInstr *D = MRI.getUniqueVRegDef(R);
  if (!D)
    return std::nullopt;
  switch (D->getOpcode()) {
  case Capstone::LUI:
    if (!D->getOperand(1).isImm())
      return std::nullopt;
    return SignExtend64<32>(D->getOperand(1).getImm() << 12);
  case Capstone::ADDI:
  case Capstone::ADDIW: {
    if (!D->getOperand(1).isReg() || !D->getOperand(2).isImm())
      return std::nullopt;
    int64_t Imm = D->getOperand(2).getImm();
    std::optional<int64_t> Base =
        D->getOperand(1).getReg() == Capstone::X0
            ? std::optional<int64_t>(0)
            : constantOf(MRI, D->getOperand(1).getReg(), Depth + 1);
    if (!Base)
      return std::nullopt;
    int64_t V = *Base + Imm;
    return D->getOpcode() == Capstone::ADDIW ? SignExtend64<32>(V) : V;
  }
  case TargetOpcode::COPY:
    if (D->getOperand(1).isReg() && D->getOperand(1).getReg() == Capstone::X0)
      return 0;
    return constantOf(MRI, D->getOperand(1).getReg(), Depth + 1);
  default:
    return std::nullopt;
  }
}

// Every reader of Reg (through plain virtual-register COPYs) uses it only as
// the base of a load or store whose access lies inside [0, Size).
static bool onlyInBoundsBase(const MachineRegisterInfo &MRI,
                             const CapstoneInstrInfo &TII,
                             const TargetRegisterInfo *TRI, Register Reg,
                             int64_t Size, int64_t &MaxEnd,
                             unsigned Depth = 0) {
  if (Depth > 4)
    return false;
  for (const MachineOperand &Use : MRI.use_nodbg_operands(Reg)) {
    const MachineInstr &User = *Use.getParent();
    if (User.isCopy()) {
      Register D = User.getOperand(0).getReg();
      if (!D.isVirtual() || User.getOperand(0).getSubReg() ||
          Use.getSubReg() ||
          !onlyInBoundsBase(MRI, TII, TRI, D, Size, MaxEnd, Depth + 1))
        return false;
      continue;
    }
    if (Use.getSubReg() || Use.isTied() || !User.mayLoadOrStore() ||
        User.hasUnmodeledSideEffects() || User.hasOrderedMemoryRef() ||
        User.isCall())
      return false;
    const MachineOperand *BaseOp = nullptr;
    int64_t Offset = 0;
    LocationSize Width = LocationSize::precise(0);
    if (!TII.getMemOperandWithOffsetWidth(User, BaseOp, Offset, Width, TRI) ||
        BaseOp != &Use)
      return false;
    // The base must be the only operand naming the register: a store whose
    // value is the capability itself carries it out of the access.
    for (const MachineOperand &MO : User.operands())
      if (&MO != &Use && MO.isReg() && MO.getReg() == Reg)
        return false;
    if (!Width.hasValue() || Width.isScalable())
      return false;
    int64_t W = Width.getValue().getFixedValue();
    if (Offset < 0 || W <= 0 || Offset + W > Size)
      return false;
    MaxEnd = std::max(MaxEnd, Offset + W);
  }
  return true;
}

bool CapstoneRedundantShrink::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  MachineRegisterInfo &MRI = MF.getRegInfo();
  const auto &STI = MF.getSubtarget<CapstoneSubtarget>();
  const CapstoneInstrInfo &TII = *STI.getInstrInfo();
  const TargetRegisterInfo *TRI = STI.getRegisterInfo();
  bool Changed = false;

  // The size comes from this translation unit's type for the global. For a
  // declaration, or a definition the link may replace (weak, common, linkonce),
  // that type can be larger than the object the link resolves to (an ODR size
  // mismatch). The narrowing then still covers only the believed extent, but
  // its own trap (SHRINKTO / SHRINK past the authority's end) can catch such an
  // object placed at the end of the image, and dropping it would lose that. So
  // by default only a global this unit defines exactly is eligible.
  //
  // -capstone-drop-redundant-shrink-declarations also admits a declaration
  // (never an extern-weak one, nor a weak, common or linkonce definition),
  // and records how many bytes from its start the dropped narrowing's accesses
  // reach. The AsmPrinter emits the records into .capstone.size_assume and
  // capstone/tests/capstone-size-assume-check.py fails a linked image whose
  // definition is smaller, so a size mismatch is a link error, not a silent
  // overread.
  // Returns the global, or null when the base is not an eligible global.
  auto EligibleGlobal = [&](Register R) -> const GlobalVariable * {
    const MachineInstr *D = R.isVirtual() ? MRI.getUniqueVRegDef(R) : nullptr;
    if (!D || D->getOpcode() != Capstone::PseudoCapGlobalBase)
      return nullptr;
    const MachineOperand &Addr = D->getOperand(1);
    const MachineInstr *LLA = Addr.isReg() && Addr.getReg().isVirtual()
                                  ? MRI.getUniqueVRegDef(Addr.getReg())
                                  : nullptr;
    if (!LLA || LLA->getOpcode() != Capstone::PseudoLLA ||
        !LLA->getOperand(1).isGlobal())
      return nullptr;
    const auto *GV = dyn_cast<GlobalVariable>(LLA->getOperand(1).getGlobal());
    if (!GV)
      return nullptr;
    if (!GV->isDeclaration() && GV->hasExactDefinition() &&
        !GV->isInterposable())
      return GV;
    if (CapstoneRedundantShrinkDeclarations && GV->isDeclaration() &&
        !GV->hasExternalWeakLinkage())
      return GV;
    return nullptr;
  };

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : llvm::make_early_inc_range(MBB)) {
      Register Dst, Src;
      int64_t Size;
      if (MI.getOpcode() == Capstone::SHRINKTO) {
        // (outs $rd), (ins $rs1, $imm12)
        Dst = MI.getOperand(0).getReg();
        Src = MI.getOperand(1).getReg();
        if (!MI.getOperand(2).isImm())
          continue;
        Size = MI.getOperand(2).getImm() - (int64_t)CapstoneShrinkToEndAdjust;
      } else if (MI.getOpcode() == Capstone::SHRINK) {
        // (outs $rd), (ins $cap_in, $rs1, $rs2), $rd tied to $cap_in:
        // [rs1, rs2). Only end = rs1 + k with rs1 the base's own address.
        Dst = MI.getOperand(0).getReg();
        Src = MI.getOperand(1).getReg();
        Register Lo = MI.getOperand(2).getReg(), Hi = MI.getOperand(3).getReg();
        const MachineInstr *Base = Src.isVirtual() ? MRI.getUniqueVRegDef(Src)
                                                   : nullptr;
        if (!Base || Base->getOpcode() != Capstone::PseudoCapGlobalBase ||
            !Base->getOperand(1).isReg() || Base->getOperand(1).getReg() != Lo)
          continue;
        const MachineInstr *End =
            Hi.isVirtual() ? MRI.getUniqueVRegDef(Hi) : nullptr;
        if (!End || End->getOpcode() != Capstone::ADD)
          continue;
        Register A = End->getOperand(1).getReg(), B = End->getOperand(2).getReg();
        std::optional<int64_t> K =
            A == Lo ? constantOf(MRI, B) : B == Lo ? constantOf(MRI, A)
                                                   : std::nullopt;
        if (!K)
          continue;
        Size = *K;
      } else {
        continue;
      }
      const GlobalVariable *GV = Dst.isVirtual() ? EligibleGlobal(Src) : nullptr;
      if (!GV || Size <= 0 || MRI.getRegClass(Dst) != MRI.getRegClass(Src))
        continue;
      int64_t MaxEnd = 0;
      if (!onlyInBoundsBase(MRI, TII, TRI, Dst, Size, MaxEnd))
        continue;
      if (GV->isDeclaration())
        MF.getInfo<CapstoneMachineFunctionInfo>()->addSizeAssumption(
            GV, std::max<int64_t>(MaxEnd, 1));
      MRI.replaceRegWith(Dst, Src);
      MRI.clearKillFlags(Src);
      MI.eraseFromParent();
      ++NumDropped;
      Changed = true;
    }
  }
  return Changed;
}
