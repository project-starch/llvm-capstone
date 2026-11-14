//===----- CapstoneMergeBaseOffset.cpp - Optimise address calculations  ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Merge the offset of address calculation into the offset field
// of instructions in a global address lowering sequence.
//
//===----------------------------------------------------------------------===//

#include "Capstone.h"
#include "CapstoneTargetMachine.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetOptions.h"
#include <optional>
using namespace llvm;

#define DEBUG_TYPE "capstone-merge-base-offset"
#define Capstone_MERGE_BASE_OFFSET_NAME "Capstone Merge Base Offset"
bool capstoneImageGpAbiActive(); // CapstoneISelDAGToDAG.cpp

STATISTIC(NumPoolBasesMerged, "Pool addresses whose several bases were merged");
STATISTIC(NumGlobalLoFolded,
          "gp-derived global bases whose %pcrel_lo was folded into their accesses");

// The global-address fold (foldIntoCapGlobalMemoryOps below). It forms a
// cursor the program never formed -- the auipc's partial address, up to 2 KiB
// from the accessed address -- which the cursor-preservation rule
// (capstone/docs/design/capability-loop-addressing.md) otherwise refuses. The
// project lead relaxed that rule for this fold alone on 2026-11-13, on two
// conditions that this pass and the image link script provide between them:
// the derived capability is only ever the base of loads and stores (no
// surviving cursor: nothing copies it, stores it, compares it or bounds it),
// and every such partial cursor lies inside gp's bounds, which the virtual
// image's link script guarantees with a margin of at least 2 KiB on both
// sides of every symbol reachable through gp (link.ld: the text segment in
// front, an explicit pad after .bss). Default off.
static cl::opt<bool> CapstoneFoldGlobalLo(
    "capstone-fold-global-lo", cl::Hidden, cl::init(false),
    cl::desc("Under the image-gp ABI, fold a global's %pcrel_lo into the "
             "loads and stores through its gp-derived capability (auipc; scc "
             "gp; ld %pcrel_lo(cap)); needs the image link script's 2 KiB "
             "gp margins"));

// Linker gp relaxation of global accesses. On a fold above whose auipc is a
// global's (not a constant pool's) and whose result feeds nothing but the scc,
// the scc becomes PseudoCapGlobalBaseGpRel: the same instruction plus an
// R_Capstone_GP_SCC relocation against the auipc's label. That relocation is
// the compiler's statement that the auipc's value reaches only this scc, and
// the scc's only through accesses that carry the label's %pcrel_lo, which is
// what the fold already requires. lld (--relax-gp) may then delete the auipc and
// the scc and rebase each access on gp, provided the global lies within 2 KiB of
// __global_pointer$ and every one of those relocations is marked relaxable.
// That form reads gp's own cursor, which must therefore be __global_pointer$ in
// every context (the runtime's convention under this option). Default off.
static cl::opt<bool> CapstoneGpRelaxGlobals(
    "capstone-gp-relax-globals", cl::Hidden, cl::init(false),
    cl::desc("Mark folded global derivations (auipc; scc gp; access "
             "%pcrel_lo) for lld's gp relaxation (R_Capstone_GP_SCC); needs "
             "-capstone-fold-global-lo, +relax, and a runtime that keeps gp's "
             "cursor at __global_pointer$"));

STATISTIC(NumGpSccMarked, "Number of global bases marked for gp relaxation");

namespace {

class CapstoneMergeBaseOffsetOpt : public MachineFunctionPass {
  const CapstoneSubtarget *ST = nullptr;
  MachineRegisterInfo *MRI;

public:
  static char ID;
  bool runOnMachineFunction(MachineFunction &Fn) override;
  bool detectFoldable(MachineInstr &Hi, MachineInstr *&Lo);

  bool detectAndFoldOffset(MachineInstr &Hi, MachineInstr &Lo);
  bool foldOffset(MachineInstr &Hi, MachineInstr &Lo, MachineInstr &Tail,
                  int64_t Offset);
  bool foldLargeOffset(MachineInstr &Hi, MachineInstr &Lo,
                       MachineInstr &TailAdd, Register GSReg);
  bool foldShiftedOffset(MachineInstr &Hi, MachineInstr &Lo,
                         MachineInstr &TailShXAdd, Register GSReg);

  bool foldIntoMemoryOps(MachineInstr &Hi, MachineInstr &Lo);
  bool foldIntoCapGlobalMemoryOps(MachineInstr &Hi, MachineInstr &Lo);
  bool mergePoolBases(MachineInstr &Hi, MachineInstr &Lo);
  bool mergeGlobalBaseUses(MachineInstr &Lo, Register LoReg);

  // Whether any instruction of the function writes gp (c3 or a part of it):
  // then two PseudoCapGlobalBase of one address may derive from different gp
  // values and are not merged.
  bool GpModified = false;

  CapstoneMergeBaseOffsetOpt() : MachineFunctionPass(ID) {}

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().setIsSSA();
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override {
    return Capstone_MERGE_BASE_OFFSET_NAME;
  }
};
} // end anonymous namespace

char CapstoneMergeBaseOffsetOpt::ID = 0;
INITIALIZE_PASS(CapstoneMergeBaseOffsetOpt, DEBUG_TYPE,
                Capstone_MERGE_BASE_OFFSET_NAME, false, false)

// Detect either of the patterns:
//
// 1. (medlow pattern):
//   lui   vreg1, %hi(s)
//   addi  vreg2, vreg1, %lo(s)
//
// 2. (medany pattern):
// .Lpcrel_hi1:
//   auipc vreg1, %pcrel_hi(s)
//   addi  vreg2, vreg1, %pcrel_lo(.Lpcrel_hi1)
//
// The pattern is only accepted if:
//    1) The first instruction has only one use, which is the ADDI.
//    2) The address operands have the appropriate type, reflecting the
//       lowering of a global address or constant pool using medlow or medany.
//    3) The offset value in the Global Address or Constant Pool is 0.
bool CapstoneMergeBaseOffsetOpt::detectFoldable(MachineInstr &Hi,
                                             MachineInstr *&Lo) {
  if (Hi.getOpcode() != Capstone::LUI && Hi.getOpcode() != Capstone::AUIPC &&
      Hi.getOpcode() != Capstone::PseudoMovAddr)
    return false;

  const MachineOperand &HiOp1 = Hi.getOperand(1);
  unsigned ExpectedFlags =
      Hi.getOpcode() == Capstone::AUIPC ? CapstoneII::MO_PCREL_HI : CapstoneII::MO_HI;
  if (HiOp1.getTargetFlags() != ExpectedFlags)
    return false;

  if (!(HiOp1.isGlobal() || HiOp1.isCPI() || HiOp1.isBlockAddress()) ||
      HiOp1.getOffset() != 0)
    return false;

  if (Hi.getOpcode() == Capstone::PseudoMovAddr) {
    // Most of the code should handle it correctly without modification by
    // setting Lo and Hi both point to PseudoMovAddr
    Lo = &Hi;
  } else {
    Register HiDestReg = Hi.getOperand(0).getReg();
    if (!MRI->hasOneUse(HiDestReg))
      return false;

    Lo = &*MRI->use_instr_begin(HiDestReg);
    if (Lo->getOpcode() != Capstone::ADDI)
      return false;
  }

  const MachineOperand &LoOp2 = Lo->getOperand(2);
  if (Hi.getOpcode() == Capstone::LUI || Hi.getOpcode() == Capstone::PseudoMovAddr) {
    if (LoOp2.getTargetFlags() != CapstoneII::MO_LO ||
        !(LoOp2.isGlobal() || LoOp2.isCPI() || LoOp2.isBlockAddress()) ||
        LoOp2.getOffset() != 0)
      return false;
  } else {
    assert(Hi.getOpcode() == Capstone::AUIPC);
    if (LoOp2.getTargetFlags() != CapstoneII::MO_PCREL_LO ||
        LoOp2.getType() != MachineOperand::MO_MCSymbol)
      return false;
  }

  if (HiOp1.isGlobal()) {
    LLVM_DEBUG(dbgs() << "  Found lowered global address: "
                      << *HiOp1.getGlobal() << "\n");
  } else if (HiOp1.isBlockAddress()) {
    LLVM_DEBUG(dbgs() << "  Found lowered basic address: "
                      << *HiOp1.getBlockAddress() << "\n");
  } else if (HiOp1.isCPI()) {
    LLVM_DEBUG(dbgs() << "  Found lowered constant pool: " << HiOp1.getIndex()
                      << "\n");
  }

  return true;
}

// Update the offset in Hi and Lo instructions.
// Delete the tail instruction and update all the uses to use the
// output from Lo.
bool CapstoneMergeBaseOffsetOpt::foldOffset(MachineInstr &Hi, MachineInstr &Lo,
                                         MachineInstr &Tail, int64_t Offset) {
  assert(isInt<32>(Offset) && "Unexpected offset");

  // If Hi is an AUIPC, don't fold the offset if it is outside the bounds of
  // the global object. The object may be within 2GB of the PC, but addresses
  // outside of the object might not be.
  if (Hi.getOpcode() == Capstone::AUIPC && Hi.getOperand(1).isGlobal()) {
    const GlobalValue *GV = Hi.getOperand(1).getGlobal();
    Type *Ty = GV->getValueType();
    if (!Ty->isSized() || Offset < 0 ||
        (uint64_t)Offset > GV->getDataLayout().getTypeAllocSize(Ty))
      return false;
  }

  // Put the offset back in Hi and the Lo
  Hi.getOperand(1).setOffset(Offset);
  if (Hi.getOpcode() != Capstone::AUIPC)
    Lo.getOperand(2).setOffset(Offset);
  // Delete the tail instruction.
  MRI->constrainRegClass(Lo.getOperand(0).getReg(),
                         MRI->getRegClass(Tail.getOperand(0).getReg()));
  MRI->replaceRegWith(Tail.getOperand(0).getReg(), Lo.getOperand(0).getReg());
  Tail.eraseFromParent();
  LLVM_DEBUG(dbgs() << "  Merged offset " << Offset << " into base.\n"
                    << "     " << Hi << "     " << Lo;);
  return true;
}

// Detect patterns for large offsets that are passed into an ADD instruction.
// If the pattern is found, updates the offset in Hi and Lo instructions
// and deletes TailAdd and the instructions that produced the offset.
//
//                     Base address lowering is of the form:
//                       Hi:  lui   vreg1, %hi(s)
//                       Lo:  addi  vreg2, vreg1, %lo(s)
//                       /                                  \
//                      /                                    \
//                     /                                      \
//                    /  The large offset can be of two forms: \
//  1) Offset that has non zero bits in lower      2) Offset that has non zero
//     12 bits and upper 20 bits                      bits in upper 20 bits only
//   OffseLUI: lui   vreg3, 4
// OffsetTail: addi  voff, vreg3, 188                OffsetTail: lui  voff, 128
//                    \                                        /
//                     \                                      /
//                      \                                    /
//                       \                                  /
//                         TailAdd: add  vreg4, vreg2, voff
bool CapstoneMergeBaseOffsetOpt::foldLargeOffset(MachineInstr &Hi,
                                              MachineInstr &Lo,
                                              MachineInstr &TailAdd,
                                              Register GAReg) {
  assert((TailAdd.getOpcode() == Capstone::ADD) && "Expected ADD instruction!");
  Register Rs = TailAdd.getOperand(1).getReg();
  Register Rt = TailAdd.getOperand(2).getReg();
  Register Reg = Rs == GAReg ? Rt : Rs;

  // Can't fold if the register has more than one use.
  if (!Reg.isVirtual() || !MRI->hasOneUse(Reg))
    return false;
  // This can point to an ADDI(W) or a LUI:
  MachineInstr &OffsetTail = *MRI->getVRegDef(Reg);
  if (OffsetTail.getOpcode() == Capstone::ADDI ||
      OffsetTail.getOpcode() == Capstone::ADDIW) {
    // The offset value has non zero bits in both %hi and %lo parts.
    // Detect an ADDI that feeds from a LUI instruction.
    MachineOperand &AddiImmOp = OffsetTail.getOperand(2);
    if (AddiImmOp.getTargetFlags() != CapstoneII::MO_None)
      return false;
    Register AddiReg = OffsetTail.getOperand(1).getReg();
    int64_t OffLo = AddiImmOp.getImm();

    // Handle rs1 of ADDI is X0.
    if (AddiReg == Capstone::X0) {
      LLVM_DEBUG(dbgs() << "  Offset Instrs: " << OffsetTail);
      if (!foldOffset(Hi, Lo, TailAdd, OffLo))
        return false;
      OffsetTail.eraseFromParent();
      return true;
    }

    MachineInstr &OffsetLui = *MRI->getVRegDef(AddiReg);
    MachineOperand &LuiImmOp = OffsetLui.getOperand(1);
    if (OffsetLui.getOpcode() != Capstone::LUI ||
        LuiImmOp.getTargetFlags() != CapstoneII::MO_None ||
        !MRI->hasOneUse(OffsetLui.getOperand(0).getReg()))
      return false;
    int64_t Offset = SignExtend64<32>(LuiImmOp.getImm() << 12);
    Offset += OffLo;
    // RV32 ignores the upper 32 bits. ADDIW sign extends the result.
    if (!ST->is64Bit() || OffsetTail.getOpcode() == Capstone::ADDIW)
      Offset = SignExtend64<32>(Offset);
    // We can only fold simm32 offsets.
    if (!isInt<32>(Offset))
      return false;
    LLVM_DEBUG(dbgs() << "  Offset Instrs: " << OffsetTail
                      << "                 " << OffsetLui);
    if (!foldOffset(Hi, Lo, TailAdd, Offset))
      return false;
    OffsetTail.eraseFromParent();
    OffsetLui.eraseFromParent();
    return true;
  } else if (OffsetTail.getOpcode() == Capstone::LUI) {
    // The offset value has all zero bits in the lower 12 bits. Only LUI
    // exists.
    LLVM_DEBUG(dbgs() << "  Offset Instr: " << OffsetTail);
    int64_t Offset = SignExtend64<32>(OffsetTail.getOperand(1).getImm() << 12);
    if (!foldOffset(Hi, Lo, TailAdd, Offset))
      return false;
    OffsetTail.eraseFromParent();
    return true;
  }
  return false;
}

// Detect patterns for offsets that are passed into a SHXADD instruction.
// The offset has 1, 2, or 3 trailing zeros and fits in simm13, simm14, simm15.
// The constant is created with addi voff, x0, C, and shXadd is used to
// fill insert the trailing zeros and do the addition.
// If the pattern is found, updates the offset in Hi and Lo instructions
// and deletes TailShXAdd and the instructions that produced the offset.
//
// Hi:         lui     vreg1, %hi(s)
// Lo:         addi    vreg2, vreg1, %lo(s)
// OffsetTail: addi    voff, x0, C
// TailAdd:    shXadd  vreg4, voff, vreg2
bool CapstoneMergeBaseOffsetOpt::foldShiftedOffset(MachineInstr &Hi,
                                                MachineInstr &Lo,
                                                MachineInstr &TailShXAdd,
                                                Register GAReg) {
  assert((TailShXAdd.getOpcode() == Capstone::SH1ADD ||
          TailShXAdd.getOpcode() == Capstone::SH2ADD ||
          TailShXAdd.getOpcode() == Capstone::SH3ADD) &&
         "Expected SHXADD instruction!");

  if (GAReg != TailShXAdd.getOperand(2).getReg())
    return false;

  // The first source is the shifted operand.
  Register Rs1 = TailShXAdd.getOperand(1).getReg();

  // Can't fold if the register has more than one use.
  if (!Rs1.isVirtual() || !MRI->hasOneUse(Rs1))
    return false;
  // This can point to an ADDI X0, C.
  MachineInstr &OffsetTail = *MRI->getVRegDef(Rs1);
  if (OffsetTail.getOpcode() != Capstone::ADDI)
    return false;
  if (!OffsetTail.getOperand(1).isReg() ||
      OffsetTail.getOperand(1).getReg() != Capstone::X0 ||
      !OffsetTail.getOperand(2).isImm())
    return false;

  int64_t Offset = OffsetTail.getOperand(2).getImm();
  assert(isInt<12>(Offset) && "Unexpected offset");

  unsigned ShAmt;
  switch (TailShXAdd.getOpcode()) {
  default: llvm_unreachable("Unexpected opcode");
  case Capstone::SH1ADD: ShAmt = 1; break;
  case Capstone::SH2ADD: ShAmt = 2; break;
  case Capstone::SH3ADD: ShAmt = 3; break;
  }

  Offset = (uint64_t)Offset << ShAmt;

  LLVM_DEBUG(dbgs() << "  Offset Instr: " << OffsetTail);
  if (!foldOffset(Hi, Lo, TailShXAdd, Offset))
    return false;
  OffsetTail.eraseFromParent();
  return true;
}

bool CapstoneMergeBaseOffsetOpt::detectAndFoldOffset(MachineInstr &Hi,
                                                  MachineInstr &Lo) {
  Register DestReg = Lo.getOperand(0).getReg();

  // Look for arithmetic instructions we can get an offset from.
  // We might be able to remove the arithmetic instructions by folding the
  // offset into the LUI+ADDI.
  if (!MRI->hasOneUse(DestReg))
    return false;

  // Lo has only one use.
  MachineInstr &Tail = *MRI->use_instr_begin(DestReg);
  switch (Tail.getOpcode()) {
  default:
    LLVM_DEBUG(dbgs() << "Don't know how to get offset from this instr:"
                      << Tail);
    break;
  case Capstone::ADDI: {
    // Offset is simply an immediate operand.
    int64_t Offset = Tail.getOperand(2).getImm();

    // We might have two ADDIs in a row.
    Register TailDestReg = Tail.getOperand(0).getReg();
    if (MRI->hasOneUse(TailDestReg)) {
      MachineInstr &TailTail = *MRI->use_instr_begin(TailDestReg);
      if (TailTail.getOpcode() == Capstone::ADDI) {
        Offset += TailTail.getOperand(2).getImm();
        LLVM_DEBUG(dbgs() << "  Offset Instrs: " << Tail << TailTail);
        if (!foldOffset(Hi, Lo, TailTail, Offset))
          return false;
        Tail.eraseFromParent();
        return true;
      }
    }

    LLVM_DEBUG(dbgs() << "  Offset Instr: " << Tail);
    return foldOffset(Hi, Lo, Tail, Offset);
  }
  case Capstone::ADD:
    // The offset is too large to fit in the immediate field of ADDI.
    // This can be in two forms:
    // 1) LUI hi_Offset followed by:
    //    ADDI lo_offset
    //    This happens in case the offset has non zero bits in
    //    both hi 20 and lo 12 bits.
    // 2) LUI (offset20)
    //    This happens in case the lower 12 bits of the offset are zeros.
    return foldLargeOffset(Hi, Lo, Tail, DestReg);
  case Capstone::SH1ADD:
  case Capstone::SH2ADD:
  case Capstone::SH3ADD:
    // The offset is too large to fit in the immediate field of ADDI.
    // It may be encoded as (SH2ADD (ADDI X0, C), DestReg) or
    // (SH3ADD (ADDI X0, C), DestReg).
    return foldShiftedOffset(Hi, Lo, Tail, DestReg);
  }

  return false;
}

bool CapstoneMergeBaseOffsetOpt::foldIntoMemoryOps(MachineInstr &Hi,
                                                MachineInstr &Lo) {
  Register DestReg = Lo.getOperand(0).getReg();

  // If all the uses are memory ops with the same offset, we can transform:
  //
  // 1. (medlow pattern):
  // Hi:   lui vreg1, %hi(foo)          --->  lui vreg1, %hi(foo+8)
  // Lo:   addi vreg2, vreg1, %lo(foo)  --->  lw vreg3, lo(foo+8)(vreg1)
  // Tail: lw vreg3, 8(vreg2)
  //
  // 2. (medany pattern):
  // Hi: 1:auipc vreg1, %pcrel_hi(s)         ---> auipc vreg1, %pcrel_hi(foo+8)
  // Lo:   addi  vreg2, vreg1, %pcrel_lo(1b) ---> lw vreg3, %pcrel_lo(1b)(vreg1)
  // Tail: lw vreg3, 8(vreg2)

  std::optional<int64_t> CommonOffset;
  DenseMap<const MachineInstr *, SmallVector<unsigned>>
      InlineAsmMemoryOpIndexesMap;
  for (const MachineInstr &UseMI : MRI->use_instructions(DestReg)) {
    switch (UseMI.getOpcode()) {
    default:
      LLVM_DEBUG(dbgs() << "Not a load or store instruction: " << UseMI);
      return false;
    case Capstone::LB:
    case Capstone::LH:
    case Capstone::LH_INX:
    case Capstone::LW:
    case Capstone::LW_INX:
    case Capstone::LBU:
    case Capstone::LHU:
    case Capstone::LWU:
    case Capstone::LD:
    case Capstone::LD_RV32:
    case Capstone::FLH:
    case Capstone::FLW:
    case Capstone::FLD:
    case Capstone::SB:
    case Capstone::SH:
    case Capstone::SH_INX:
    case Capstone::SW:
    case Capstone::SW_INX:
    case Capstone::SD:
    case Capstone::SD_RV32:
    case Capstone::FSH:
    case Capstone::FSW:
    case Capstone::FSD: {
      if (UseMI.getOperand(1).isFI())
        return false;
      // Register defined by Lo should not be the value register.
      if (DestReg == UseMI.getOperand(0).getReg())
        return false;
      assert(DestReg == UseMI.getOperand(1).getReg() &&
             "Expected base address use");
      // All load/store instructions must use the same offset.
      int64_t Offset = UseMI.getOperand(2).getImm();
      if (CommonOffset && Offset != CommonOffset)
        return false;
      CommonOffset = Offset;
      break;
    }
    case Capstone::INLINEASM:
    case Capstone::INLINEASM_BR: {
      SmallVector<unsigned> InlineAsmMemoryOpIndexes;
      unsigned NumOps = 0;
      for (unsigned I = InlineAsm::MIOp_FirstOperand;
           I < UseMI.getNumOperands(); I += 1 + NumOps) {
        const MachineOperand &FlagsMO = UseMI.getOperand(I);
        // Should be an imm.
        if (!FlagsMO.isImm())
          continue;

        const InlineAsm::Flag Flags(FlagsMO.getImm());
        NumOps = Flags.getNumOperandRegisters();

        // Memory constraints have two operands.
        if (NumOps != 2 || !Flags.isMemKind()) {
          // If the register is used by something other than a memory
          // constraint, we should not fold.
          for (unsigned J = 0; J < NumOps; ++J) {
            const MachineOperand &MO = UseMI.getOperand(I + 1 + J);
            if (MO.isReg() && MO.getReg() == DestReg)
              return false;
          }
          continue;
        }

        // We can't do this for constraint A because AMO instructions don't have
        // an immediate offset field.
        if (Flags.getMemoryConstraintID() == InlineAsm::ConstraintCode::A)
          return false;

        const MachineOperand &AddrMO = UseMI.getOperand(I + 1);
        if (!AddrMO.isReg() || AddrMO.getReg() != DestReg)
          continue;

        const MachineOperand &OffsetMO = UseMI.getOperand(I + 2);
        if (!OffsetMO.isImm())
          continue;

        // All inline asm memory operands must use the same offset.
        int64_t Offset = OffsetMO.getImm();
        if (CommonOffset && Offset != CommonOffset)
          return false;
        CommonOffset = Offset;
        InlineAsmMemoryOpIndexes.push_back(I + 1);
      }
      InlineAsmMemoryOpIndexesMap.insert(
          std::make_pair(&UseMI, InlineAsmMemoryOpIndexes));
      break;
    }
    }
  }

  // We found a common offset.
  // Update the offsets in global address lowering.
  // We may have already folded some arithmetic so we need to add to any
  // existing offset.
  int64_t NewOffset = Hi.getOperand(1).getOffset() + *CommonOffset;
  // RV32 ignores the upper 32 bits.
  if (!ST->is64Bit())
    NewOffset = SignExtend64<32>(NewOffset);
  // We can only fold simm32 offsets.
  if (!isInt<32>(NewOffset))
    return false;

  Hi.getOperand(1).setOffset(NewOffset);
  MachineOperand &ImmOp = Lo.getOperand(2);
  // Expand PseudoMovAddr into LUI
  if (Hi.getOpcode() == Capstone::PseudoMovAddr) {
    auto *TII = ST->getInstrInfo();
    Hi.setDesc(TII->get(Capstone::LUI));
    Hi.removeOperand(2);
  }

  if (Hi.getOpcode() != Capstone::AUIPC)
    ImmOp.setOffset(NewOffset);

  // Update the immediate in the load/store instructions to add the offset.
  for (MachineInstr &UseMI :
       llvm::make_early_inc_range(MRI->use_instructions(DestReg))) {
    if (UseMI.getOpcode() == Capstone::INLINEASM ||
        UseMI.getOpcode() == Capstone::INLINEASM_BR) {
      auto &InlineAsmMemoryOpIndexes = InlineAsmMemoryOpIndexesMap[&UseMI];
      for (unsigned I : InlineAsmMemoryOpIndexes) {
        MachineOperand &MO = UseMI.getOperand(I + 1);
        switch (ImmOp.getType()) {
        case MachineOperand::MO_GlobalAddress:
          MO.ChangeToGA(ImmOp.getGlobal(), ImmOp.getOffset(),
                        ImmOp.getTargetFlags());
          break;
        case MachineOperand::MO_MCSymbol:
          MO.ChangeToMCSymbol(ImmOp.getMCSymbol(), ImmOp.getTargetFlags());
          MO.setOffset(ImmOp.getOffset());
          break;
        case MachineOperand::MO_BlockAddress:
          MO.ChangeToBA(ImmOp.getBlockAddress(), ImmOp.getOffset(),
                        ImmOp.getTargetFlags());
          break;
        default:
          report_fatal_error("unsupported machine operand type");
          break;
        }
      }
    } else {
      UseMI.removeOperand(2);
      UseMI.addOperand(ImmOp);
    }
  }

  // Prevent Lo (originally PseudoMovAddr, which is also pointed by Hi) from
  // being erased
  if (&Lo == &Hi)
    return true;

  MRI->replaceRegWith(Lo.getOperand(0).getReg(), Hi.getOperand(0).getReg());
  Lo.eraseFromParent();
  return true;
}

// A constant-pool entry reached through a gp-derived capability keeps its
// complete address:
//
// Hi:   1:auipc vreg1, %pcrel_hi(.LCPI)
// Lo:   addi  vreg2, vreg1, %pcrel_lo(1b)
// Base: vcap = PseudoCapGlobalBase vreg2
// Tail: fld f, 0(vcap)
//
// The %pcrel_lo is NOT folded into the loads (deleting Lo, deriving the
// capability from the auipc's partial address), although only the access is
// checked against gp's bounds: the cursor in between may not lie anywhere:
// a cursor the address policy refuses is written untagged (isa.md, "address
// policy"). The partial address is the entry's address minus its signed low
// 12 bits, up to 2048 bytes above the entry: with H = 2^38, gp bounds
// [H-4096, H), the auipc at H-4096 and the entry at H-16 the relocation splits
// into hi +4096 and lo -16, scc of the high part alone forms H, refused under
// sv39, and the `fld -16` off it faults at the unchanged effective address.
// The low part is a link-time value, so no compile-time test can tell the
// safe shapes apart, and the rule (no rewrite changes the value of a
// surviving cursor) leaves no fold: the capability is derived from the entry's
// address, which the program forms.
//
// Exception, behind -capstone-fold-global-lo (2026-11-13): the fold above IS
// made for a global or pool entry whose base has no use but as the base of
// loads and stores at one common offset, when the image link script keeps a
// 2 KiB margin inside gp's bounds around every gp-reachable symbol. The
// partial cursor is then within gp's bounds, so the address policy cannot
// refuse it, and it never survives the accesses (foldIntoCapGlobalMemoryOps).
//
// Several PseudoCapGlobalBase of one constant-pool address. MachineCSE's PRE
// hoists the PseudoLLA (auipc + addi, no register operands) to a common
// dominator of its users but not the PseudoCapGlobalBase (a virtual-register
// operand), so after it a hoisted Lo can feed one base per using block:
//
//   bb.0:  %lo = ADDI %hi, %pcrel_lo(...)
//   bb.1:  %c1 = PseudoCapGlobalBase %lo, implicit $x3
//   bb.2:  %c2 = PseudoCapGlobalBase %lo, implicit $x3
//
// 544.nab_r's mme34 then spilled the 64-bit address and re-derived the
// capability (ld + scc) in every loop that read a pool entry. This keeps one
// base and replaces the others with it. The primary is chosen by INSTRUCTION
// order, never by use-list order (whose order is the order the operands were
// (re)added in, which earlier passes change): the first base after Lo in Lo's
// block if there is one, which every other base is dominated by (same block:
// later; other block: Lo's block dominates every use of Lo in SSA); otherwise
// the first base in function order, moved to just after Lo, where it dominates
// all of them. No cursor value changes: every base forms the same cursor from
// the same address operand and the same gp, and PseudoCapGlobalBase expands to
// a plain scc of gp under the image-gp ABI (no delin, no shrink), which cannot
// fault, so moving it above a branch hoists no derivation above a guard. The
// derivation must keep the same authority and lifetime, so the merge is
// refused when anything in the function writes gp (GpModified): then the two
// bases could derive from different gp values. Calls do not count: gp is a
// reserved register the ABI preserves across calls, which every existing use
// of gp after a call already relies on.
bool CapstoneMergeBaseOffsetOpt::mergeGlobalBaseUses(MachineInstr &Lo,
                                                     Register LoReg) {
  if (GpModified)
    return false;
  SmallVector<MachineInstr *, 4> Bases;
  for (MachineOperand &MO : MRI->use_operands(LoReg)) {
    MachineInstr *UseMI = MO.getParent();
    // A debug use is not a base; conservatively refuse (as the single-use
    // path does for the capability).
    if (UseMI->getOpcode() != Capstone::PseudoCapGlobalBase ||
        UseMI->getOperandNo(&MO) != 1)
      return false;
    Bases.push_back(UseMI);
  }
  if (Bases.size() < 2)
    return false;

  MachineBasicBlock *LoMBB = Lo.getParent();
  MachineInstr *Primary = nullptr;
  for (MachineBasicBlock::iterator It = std::next(Lo.getIterator()),
                                   E = LoMBB->end();
       It != E; ++It) {
    if (It->getOpcode() == Capstone::PseudoCapGlobalBase &&
        It->getOperand(1).getReg() == LoReg) {
      Primary = &*It;
      break;
    }
  }
  if (!Primary) {
    for (MachineBasicBlock &MBB : *LoMBB->getParent()) {
      for (MachineInstr &MI : MBB) {
        if (llvm::is_contained(Bases, &MI)) {
          Primary = &MI;
          break;
        }
      }
      if (Primary)
        break;
    }
    assert(Primary && "a use of LoReg is in no block");
    LoMBB->splice(std::next(Lo.getIterator()), Primary->getParent(), Primary);
  }

  Register PrimaryCap = Primary->getOperand(0).getReg();
  for (MachineInstr *B : Bases) {
    if (B == Primary)
      continue;
    MRI->replaceRegWith(B->getOperand(0).getReg(), PrimaryCap);
    B->eraseFromParent();
  }
  // The primary may now be read after what was the last use of either value.
  MRI->clearKillFlags(LoReg);
  MRI->clearKillFlags(PrimaryCap);
  LLVM_DEBUG(dbgs() << "merged " << Bases.size() << " pool bases of " << Lo
                    << "  into " << *Primary);
  return true;
}

// Hi:   1:auipc vreg1, %pcrel_hi(s)          1:auipc vreg1, %pcrel_hi(s+8)
// Lo:   addi  vreg2, vreg1, %pcrel_lo(1b)  -->  (deleted)
// Base: vcap = PseudoCapGlobalBase vreg2       vcap = PseudoCapGlobalBase vreg1
// Tail: ld vreg3, 8(vcap)                      ld vreg3, %pcrel_lo(1b)(vcap)
//
// Only under -capstone-fold-global-lo and the image-gp ABI (a plain scc of gp:
// no delin, no shrink), only when nothing in the function writes gp, and only
// when vcap's every use is the base (never the value) of a load or store with
// one common immediate offset. A SHRINK/SHRINKTO, copy, call argument, store of
// vcap, cincoffset or debug use leaves the address whole. The cursor of vcap is
// the partial address: see CapstoneFoldGlobalLo for the rule this relies on.
bool CapstoneMergeBaseOffsetOpt::foldIntoCapGlobalMemoryOps(MachineInstr &Hi,
                                                         MachineInstr &Lo) {
  if (!CapstoneFoldGlobalLo || !capstoneImageGpAbiActive() || GpModified)
    return false;
  if (Hi.getOpcode() != Capstone::AUIPC || &Lo == &Hi ||
      Lo.getOpcode() != Capstone::ADDI)
    return false;
  const MachineOperand &HiOp = Hi.getOperand(1);
  if (!HiOp.isGlobal() && !HiOp.isCPI())
    return false;
  Register LoReg = Lo.getOperand(0).getReg();
  if (!MRI->hasOneUse(LoReg))
    return false;
  MachineInstr &Base = *MRI->use_instr_begin(LoReg);
  if (Base.getOpcode() != Capstone::PseudoCapGlobalBase ||
      Base.getOperand(1).getReg() != LoReg)
    return false;
  Register CapReg = Base.getOperand(0).getReg();
  if (!CapReg.isVirtual() || MRI->use_empty(CapReg))
    return false;

  std::optional<int64_t> CommonOffset;
  for (const MachineOperand &MO : MRI->use_operands(CapReg)) {
    const MachineInstr &UseMI = *MO.getParent();
    switch (UseMI.getOpcode()) {
    default:
      return false;
    case Capstone::LB:
    case Capstone::LH:
    case Capstone::LW:
    case Capstone::LBU:
    case Capstone::LHU:
    case Capstone::LWU:
    case Capstone::LD:
    case Capstone::LDC:
    case Capstone::FLH:
    case Capstone::FLW:
    case Capstone::FLD:
    case Capstone::SB:
    case Capstone::SH:
    case Capstone::SW:
    case Capstone::SD:
    case Capstone::STC:
    case Capstone::FSH:
    case Capstone::FSW:
    case Capstone::FSD:
      break;
    }
    // vcap must be the base (operand 1), never the loaded or stored value.
    if (UseMI.getOperandNo(&MO) != 1 || !UseMI.getOperand(2).isImm() ||
        (UseMI.getOperand(0).isReg() && UseMI.getOperand(0).getReg() == CapReg))
      return false;
    int64_t Offset = UseMI.getOperand(2).getImm();
    if (CommonOffset && Offset != *CommonOffset)
      return false;
    CommonOffset = Offset;
  }

  int64_t NewOffset = HiOp.getOffset() + *CommonOffset;
  if (!isInt<32>(NewOffset))
    return false;
  Hi.getOperand(1).setOffset(NewOffset);
  const MachineOperand &ImmOp = Lo.getOperand(2); // %pcrel_lo of Hi's label
  for (MachineInstr &UseMI :
       llvm::make_early_inc_range(MRI->use_instructions(CapReg))) {
    UseMI.removeOperand(2);
    UseMI.addOperand(ImmOp);
  }
  Register HiReg = Hi.getOperand(0).getReg();
  Base.getOperand(1).setReg(HiReg);
  MRI->clearKillFlags(HiReg);
  LLVM_DEBUG(dbgs() << "folded %pcrel_lo of " << Hi << "  into the accesses of "
                    << Base);
  // ImmOp is Lo's operand: take the label before Lo goes.
  MCSymbol *HiLabel = ImmOp.isMCSymbol() ? ImmOp.getMCSymbol() : nullptr;
  Lo.eraseFromParent();
  ++NumGlobalLoFolded;

  // The gp-relaxation mark (see CapstoneGpRelaxGlobals): a global's auipc whose
  // only use is now this scc. Every use of CapReg carries HiLabel's %pcrel_lo.
  if (CapstoneGpRelaxGlobals && HiOp.isGlobal() && HiLabel &&
      MRI->hasOneNonDBGUse(HiReg)) {
    MachineInstr *NewBase =
        BuildMI(*Base.getParent(), Base, Base.getDebugLoc(),
                ST->getInstrInfo()->get(Capstone::PseudoCapGlobalBaseGpRel),
                CapReg)
            .addReg(HiReg)
            .addSym(HiLabel, CapstoneII::MO_GP_SCC);
    LLVM_DEBUG(dbgs() << "marked for gp relaxation: " << *NewBase);
    (void)NewBase;
    Base.eraseFromParent();
    ++NumGpSccMarked;
  }
  return true;
}

bool CapstoneMergeBaseOffsetOpt::mergePoolBases(MachineInstr &Hi,
                                                MachineInstr &Lo) {
  // Only under the image-gp ABI, where PseudoCapGlobalBase is a plain scc of
  // gp (see mergeGlobalBaseUses).
  if (!capstoneImageGpAbiActive())
    return false;
  if (Hi.getOpcode() != Capstone::AUIPC || !Hi.getOperand(1).isCPI() ||
      &Lo == &Hi)
    return false;
  Register LoReg = Lo.getOperand(0).getReg();
  // Several bases of one pool address: merge them into one, each keeping
  // Lo's complete address.
  if (MRI->hasOneUse(LoReg) || !mergeGlobalBaseUses(Lo, LoReg))
    return false;
  ++NumPoolBasesMerged;
  return true;
}

bool CapstoneMergeBaseOffsetOpt::runOnMachineFunction(MachineFunction &Fn) {
  if (skipFunction(Fn.getFunction()))
    return false;

  ST = &Fn.getSubtarget<CapstoneSubtarget>();

  bool MadeChange = false;
  MRI = &Fn.getRegInfo();

  // Any register write of gp (c3 or a part of it, e.g. an inline-asm clobber)
  // disables mergeGlobalBaseUses. Call regmasks are not consulted: gp is
  // reserved and preserved across calls by the ABI.
  const TargetRegisterInfo *TRI = ST->getRegisterInfo();
  GpModified = false;
  for (MachineBasicBlock &MBB : Fn) {
    for (MachineInstr &MI : MBB) {
      for (const MachineOperand &MO : MI.operands()) {
        if (MO.isReg() && MO.isDef() && MO.getReg().isPhysical() &&
            TRI->regsOverlap(MO.getReg(), Capstone::C3)) {
          GpModified = true;
          break;
        }
      }
      if (GpModified)
        break;
    }
    if (GpModified)
      break;
  }

  for (MachineBasicBlock &MBB : Fn) {
    LLVM_DEBUG(dbgs() << "MBB: " << MBB.getName() << "\n");
    for (MachineInstr &Hi : MBB) {
      MachineInstr *Lo = nullptr;
      if (!detectFoldable(Hi, Lo))
        continue;
      MadeChange |= detectAndFoldOffset(Hi, *Lo);
      if (foldIntoMemoryOps(Hi, *Lo))
        MadeChange = true;
      else if (foldIntoCapGlobalMemoryOps(Hi, *Lo))
        MadeChange = true;
      else
        MadeChange |= mergePoolBases(Hi, *Lo);
    }
  }

  return MadeChange;
}

/// Returns an instance of the Merge Base Offset Optimization pass.
FunctionPass *llvm::createCapstoneMergeBaseOffsetOptPass() {
  return new CapstoneMergeBaseOffsetOpt();
}
