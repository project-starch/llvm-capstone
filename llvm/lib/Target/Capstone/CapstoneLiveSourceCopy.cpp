//===-- CapstoneLiveSourceCopy.cpp - copies that must not destroy a live source -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// MOVC is the only instruction that copies a whole capability register, and on
// today's silicon it writes cnull into its SOURCE unless the source is a tagged
// non-linear capability (capstone_flu_unit.anvil:6-27). A register that holds an
// integer -- an untagged value -- is destroyed by the copy, silently, and a
// later read of it sees 0 (C-32: SQLite's lookaside switched off on the board;
// musl's iconv_open handed a zero descriptor). QEMU keeps the source unless
// CAPSTONE_MOVC_NULL_SCALAR=1 asks it to behave as the RTL does.
//
// Whether a register holds an integer is known only at run time, so no choice
// of instruction by type can avoid it. Liveness can: a MOVC whose source is
// never read again destroys nothing anyone looks at. This pass keeps those and
// rewrites every MOVC whose source IS read again into a copy through a 16-byte
// stack slot,
//
//     stc src, off(fr)      ; STC leaves an integer or NONLIN source alone,
//     ldc dst, off(fr)      ; and nulls a LINEAR one, as MOVC does; LDC then
//                           ; clears the slot of a LINEAR value, so a move is
//                           ; still a move (CapstoneISASemantics.md, ldc/stc).
//
// It runs late -- after every pass that could turn a dead-source copy into a
// live-source one (both MachineCopyPropagations, BranchFolder, tail
// duplication, the post-RA scheduler) and before BranchRelaxation, since each
// rewrite adds four bytes -- so every earlier pass may keep treating COPY and
// MOVC as a pure copy. The slot is reserved before the frame is laid out
// (processFunctionBeforeFrameFinalized, from capstoneNeedsLiveSourceCopySlot,
// which is conservative because it runs before those passes).
//
// The rule is on by default. A bitstream whose MOVC leaves an untagged source
// alone (Q-04 option b) passes +movc-keeps-integer-source and turns it off.
// A second, check-only instance after MakeCompressible refuses to emit a
// live-source MOVC that BranchRelaxation or MakeCompressible might have
// produced. It is not the last pass: the machine outliner and addPreEmitPass2
// (move-merge, push/pop, pseudo expansion) follow. None of them builds a MOVC
// (the only builders are copyPhysReg and eliminateFrameIndex, the latter inside
// PEI with an sp/fp source), and none adds a read of a register it was not
// already given as an operand, so a MOVC's dead source stays dead through them.
//
//===----------------------------------------------------------------------===//

#include "Capstone.h"
#include "CapstoneFrameLowering.h"
#include "CapstoneInstrInfo.h"
#include "CapstoneMachineFunctionInfo.h"
#include "CapstoneSubtarget.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LiveRegUnits.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

#define DEBUG_TYPE "capstone-live-source-copy"
#define PASS_NAME "Capstone live-source capability copy"

STATISTIC(NumConverted, "MOVCs with a live source rewritten as STC+LDC");
STATISTIC(NumKeptDead, "MOVCs kept: source not read again");
STATISTIC(NumKeptExempt, "MOVCs kept: source always tagged or null");
STATISTIC(NumReusedSlot, "Rewritten copies that reused the slot without a new STC");

// Registers whose content is always a tagged NONLIN capability or null, so a
// MOVC from them never destroys an integer: c0 (hardwired null), sp, gp, tp,
// and the frame and base pointers when the function has them. tp is on the list
// on the same assumption musl's hand-written `movc` of tp already makes
// (arch/capstone64/pthread_arch.h). Deliberately NOT MRI.isReserved(): a
// register reserved with -ffixed-xN can hold an integer.
static bool isExemptSource(Register R, const MachineFunction &MF) {
  if (R == Capstone::C0 || R == Capstone::C2 || R == Capstone::C3 ||
      R == Capstone::C4)
    return true;
  const auto &STI = MF.getSubtarget<CapstoneSubtarget>();
  const CapstoneFrameLowering *TFL = STI.getFrameLowering();
  if (TFL->hasFP(MF) && R == Capstone::C8)
    return true;
  if (TFL->hasBP(MF) && R == Capstone::C9)
    return true;
  return false;
}

bool llvm::capstoneIsLiveSourceCopyCandidate(const MachineInstr &MI,
                                             const MachineFunction &MF) {
  if (!MI.isCopy() && MI.getOpcode() != Capstone::MOVC)
    return false;
  Register Dst = MI.getOperand(0).getReg();
  Register Src = MI.getOperand(1).getReg();
  if (!Dst.isPhysical() || !Src.isPhysical() || Dst == Src)
    return false;
  if (!Capstone::GPCRRegClass.contains(Dst) ||
      !Capstone::GPCRRegClass.contains(Src))
    return false;
  return !isExemptSource(Src, MF);
}

// Whether a candidate copy can still have a LIVE source when the rewrite runs.
// This is asked before the frame is laid out, and several passes run between
// then and the rewrite, so it must over-approximate. A dead source stays dead
// unless one of these holds, and each is checked:
//  (a) the source is live out of the block, so a later pass can place a read
//      after the copy.
//  (b) another instruction in the block reads the same value of the source,
//      before or after the copy: the post-RA scheduler can move such a read
//      across the copy. A redefinition of the source ends the value only if it
//      can not be deleted later. MachineLateInstrsCleanup deletes a definition
//      that repeats one already reaching it, and it only considers instructions
//      that read nothing but constants and the frame register (its isCandidate);
//      such a redefinition is looked through (mayBeDeletedLate).
//  (c) a later instruction in the block reads the destination through an
//      EXPLICIT operand. MachineCopyPropagation can forward that read to the
//      source. It never forwards into implicit operands (forwardUses skips
//      them), so a copy into a return or call register that nothing else
//      reads cannot become live-source that way.
//  (d) a predecessor's terminator reads the source, or the source is live into
//      another successor of that predecessor. BranchFolder hoists code common to
//      all successors into the predecessor, in front of its terminator.
// An error here is loud, not silent: the rewrite reports a fatal error when it
// needs the slot this said it would not. Functions with calls reserve the slot
// whenever they have a candidate at all (capstoneNeedsLiveSourceCopySlot), so
// this precision only matters to leaf functions, where it saves a frame.
// A definition MachineLateInstrsCleanup might delete as redundant: the shape its
// isCandidate accepts, over-approximated. No side effects, no memory access, and
// no register read except the zero register and the stack and frame pointers.
static bool mayBeDeletedLate(const MachineInstr &MI) {
  if (MI.isInlineAsm() || MI.isCall() || MI.hasUnmodeledSideEffects() ||
      MI.mayLoadOrStore())
    return false;
  for (const MachineOperand &MO : MI.uses())
    if (MO.isReg() && MO.getReg() &&
        !is_contained({Capstone::X0, Capstone::C0, Capstone::X2, Capstone::C2,
                       Capstone::X8, Capstone::C8},
                      MO.getReg().asMCReg()))
      return false;
  return true;
}

static bool copyMayNeedSlot(const MachineInstr &Copy,
                            const TargetRegisterInfo &TRI) {
  const MachineBasicBlock &MBB = *Copy.getParent();
  Register Dst = Copy.getOperand(0).getReg();
  Register Src = Copy.getOperand(1).getReg();
  LiveRegUnits Out(TRI);
  Out.addLiveOuts(MBB);
  if (!Out.available(Src))
    return true; // (a)
  bool ReadSinceDef = false, After = false, SrcRedefined = false;
  for (const MachineInstr &MI : MBB) {
    if (&MI == &Copy) {
      if (ReadSinceDef)
        return true; // (b), a read before the copy
      After = true;
      continue;
    }
    if (MI.isDebugInstr())
      continue;
    bool Reads = MI.readsRegister(Src, &TRI);
    bool EndsValue = MI.modifiesRegister(Src, &TRI) && !mayBeDeletedLate(MI);
    if (!After) {
      if (EndsValue)
        ReadSinceDef = false;
      else if (Reads)
        ReadSinceDef = true;
      continue;
    }
    if (!SrcRedefined && Reads)
      return true; // (b), a read after the copy
    for (const MachineOperand &MO : MI.explicit_uses())
      if (MO.isReg() && MO.getReg() && TRI.regsOverlap(MO.getReg(), Dst))
        return true; // (c)
    if (EndsValue)
      SrcRedefined = true;
  }
  for (const MachineBasicBlock *Pred : MBB.predecessors()) {
    for (const MachineInstr &T : Pred->terminators())
      if (T.readsRegister(Src, &TRI))
        return true; // (d)
    for (const MachineBasicBlock *Succ : Pred->successors())
      if (Succ != &MBB)
        for (const auto &LI : Succ->liveins())
          if (TRI.regsOverlap(LI.PhysReg, Src))
            return true; // (d)
  }
  return false;
}

bool llvm::capstoneNeedsLiveSourceCopySlot(const MachineFunction &MF) {
  if (MF.getSubtarget<CapstoneSubtarget>().movcKeepsIntegerSource())
    return false;
  const TargetRegisterInfo &TRI = *MF.getSubtarget().getRegisterInfo();
  const bool TracksLiveness = MF.getProperties().hasTracksLiveness();
  // A function with calls has a frame anyway; there the slot costs 16 bytes of
  // stack and no instruction, so it is reserved for any candidate.
  const bool HasCalls = MF.getFrameInfo().hasCalls();
  for (const MachineBasicBlock &MBB : MF)
    for (const MachineInstr &MI : MBB)
      if (capstoneIsLiveSourceCopyCandidate(MI, MF) &&
          (!TracksLiveness || HasCalls || copyMayNeedSlot(MI, TRI)))
        return true;
  return false;
}

namespace {
class CapstoneLiveSourceCopy : public MachineFunctionPass {
  bool CheckOnly;

public:
  static char ID;
  explicit CapstoneLiveSourceCopy(bool CheckOnly = false)
      : MachineFunctionPass(ID), CheckOnly(CheckOnly) {}
  StringRef getPassName() const override { return PASS_NAME; }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
  // No skipFunction(): this is correctness, and the -O0 and optnone images
  // (the SQLite silicon domain among them) need it as much as any other.
  bool runOnMachineFunction(MachineFunction &MF) override;
};
} // end anonymous namespace

char CapstoneLiveSourceCopy::ID = 0;
INITIALIZE_PASS(CapstoneLiveSourceCopy, DEBUG_TYPE, PASS_NAME, false, false)

FunctionPass *llvm::createCapstoneLiveSourceCopyPass(bool CheckOnly) {
  return new CapstoneLiveSourceCopy(CheckOnly);
}

bool CapstoneLiveSourceCopy::runOnMachineFunction(MachineFunction &MF) {
  const auto &STI = MF.getSubtarget<CapstoneSubtarget>();
  if (STI.movcKeepsIntegerSource())
    return false;
  const CapstoneInstrInfo *TII = STI.getInstrInfo();
  const TargetRegisterInfo *TRI = STI.getRegisterInfo();
  const TargetFrameLowering *TFL = STI.getFrameLowering();
  const bool TracksLiveness = MF.getProperties().hasTracksLiveness();

  // Collect first, rewrite after: the backward walk must see the block as it is.
  SmallVector<MachineInstr *, 16> LiveSource;
  for (MachineBasicBlock &MBB : MF) {
    LiveRegUnits Live(*TRI);
    Live.addLiveOuts(MBB);
    for (MachineInstr &MI : llvm::reverse(MBB)) {
      if (MI.isDebugInstr())
        continue;
      // Before stepping past MI, Live is the liveness just AFTER it: exactly the
      // question "is the source read again".
      if (MI.getOpcode() == Capstone::MOVC) {
        Register Src = MI.getOperand(1).getReg();
        if (MI.getOperand(0).getReg() == Src)
          ; // a self-copy destroys nothing
        else if (isExemptSource(Src, MF))
          ++NumKeptExempt;
        else if (!TracksLiveness || !Live.available(Src))
          LiveSource.push_back(&MI);
        else
          ++NumKeptDead;
      }
      Live.stepBackward(MI);
    }
  }
  if (LiveSource.empty())
    return false;

  if (CheckOnly)
    reportFatalInternalError(
        Twine("capstone-live-source-copy: a live-source MOVC was created after "
              "the rewrite ran, in ") + MF.getName());

  auto *RVFI = MF.getInfo<CapstoneMachineFunctionInfo>();
  int FI = RVFI->getLiveSourceCopyFrameIndex();
  if (FI < 0)
    reportFatalInternalError(
        Twine("capstone-live-source-copy: no copy slot was reserved in ") +
        MF.getName() + " (a copy the frame lowering did not see)");

  // The slot is registered with the scavenger only for its placement (see
  // processFunctionBeforeFrameFinalized). A spill the scavenger or anything else
  // put there could be live across one of our copies, and our STC would
  // overwrite it, so any other access is a hard error rather than a hazard.
  for (const MachineBasicBlock &MBB : MF)
    for (const MachineInstr &MI : MBB)
      for (const MachineMemOperand *MMO : MI.memoperands())
        if (const auto *PSV = dyn_cast_or_null<FixedStackPseudoSourceValue>(
                MMO->getPseudoValue()))
          if (PSV->getFrameIndex() == FI)
            reportFatalInternalError(
                Twine("capstone-live-source-copy: the copy slot is already "
                      "used by another pass (the register scavenger ran out of "
                      "its own slots?) in ") + MF.getName());

  Register FrameReg;
  StackOffset Off = TFL->getFrameIndexReference(MF, FI, FrameReg);
  if (Off.getScalable() != 0 || !isInt<12>(Off.getFixed()) ||
      !Capstone::GPCRRegClass.contains(FrameReg))
    reportFatalInternalError(
        Twine("capstone-live-source-copy: copy slot not addressable by a "
              "12-bit offset from a capability frame register in ") +
        MF.getName());
  assert((FrameReg != Capstone::C2 || TFL->hasReservedCallFrame(MF)) &&
         "an sp-relative slot needs a reserved call frame");

  MachineFrameInfo &MFI = MF.getFrameInfo();
  SmallPtrSet<MachineInstr *, 16> ToRewrite(LiveSource.begin(), LiveSource.end());
  // Forward, block by block, so one STC can serve several copies of the same
  // source. That is faithful to a MOVC chain for every type: an integer or a
  // NONLIN value reloads unchanged each time; a LINEAR one is taken by the first
  // LDC, which clears the slot, so later LDCs read null, as later MOVCs from a
  // source the first MOVC nulled would. The slot keeps its value until the
  // source is redefined, or a call, another store or an instruction with
  // unmodeled side effects intervenes.
  for (MachineBasicBlock &MBB : MF) {
    Register SlotHolds;
    for (MachineInstr &MI : llvm::make_early_inc_range(MBB)) {
      if (!ToRewrite.count(&MI)) {
        if (SlotHolds && (MI.modifiesRegister(SlotHolds, TRI) || MI.isCall() ||
                          MI.mayStore() || MI.hasUnmodeledSideEffects()))
          SlotHolds = Register();
        continue;
      }
      const DebugLoc &DL = MI.getDebugLoc();
      Register Dst = MI.getOperand(0).getReg();
      Register Src = MI.getOperand(1).getReg();
      if (SlotHolds != Src) {
        MachineMemOperand *StoreMMO = MF.getMachineMemOperand(
            MachinePointerInfo::getFixedStack(MF, FI), MachineMemOperand::MOStore,
            MFI.getObjectSize(FI), MFI.getObjectAlign(FI));
        BuildMI(MBB, MI, DL, TII->get(Capstone::STC))
            .addReg(Src)
            .addReg(FrameReg)
            .addImm(Off.getFixed())
            .addMemOperand(StoreMMO)
            .setMIFlags(MI.getFlags());
        SlotHolds = Src;
      } else {
        ++NumReusedSlot;
      }
      MachineMemOperand *LoadMMO = MF.getMachineMemOperand(
          MachinePointerInfo::getFixedStack(MF, FI), MachineMemOperand::MOLoad,
          MFI.getObjectSize(FI), MFI.getObjectAlign(FI));
      BuildMI(MBB, MI, DL, TII->get(Capstone::LDC), Dst)
          .addReg(FrameReg)
          .addImm(Off.getFixed())
          .addMemOperand(LoadMMO)
          .setMIFlags(MI.getFlags());
      if (TRI->regsOverlap(Dst, SlotHolds))
        SlotHolds = Register();
      MI.eraseFromParent();
      ++NumConverted;
    }
  }
  return true;
}
