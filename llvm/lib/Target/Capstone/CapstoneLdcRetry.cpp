//===------------ CapstoneLdcRetry.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// S-07 INSTRUMENT: retry an `ldc` whose result comes back untagged.
//
// WHAT S-07 IS. On the Capstone FPGA a capability read back from memory
// sporadically arrives untagged, so the next capability consumer raises
// mcause 25 (UNEXPECTED_OPERAND) and the domain wedges. Four wedges, in four
// unrelated functions in four different builds, share one instruction shape
// byte for byte: two ADJACENT `ldc`s where the second's rs1 is the first's rd.
//
// WHY A COMPILER PASS. Every software probe tried so far has been defeated by
// the same structural problem: instrumenting one call site simply moves the
// wedge to the next uninstrumented one, and a wedge destroys both reporting
// channels (the return value, and everything the domain buffered for the host,
// which is only read when the domain RETURNS). Covering every `ldc` at once is
// the only way a probe can be the thing that fires.
//
// WHAT IT DISCRIMINATES. Because a capability op cannot issue while its rs1 is
// the destination of an in-flight instruction (stall_waw_rs1 in
// issue_read_operands.sv, whose rd_clobber_gpr is driven by still_issued and so
// holds until COMMIT), the second `ldc` of a dependent pair cannot issue until
// the first has committed. So when it sees NOT_CAP, the FIRST `ldc` retired
// NOT_CAP, and there are exactly two possibilities:
//
//   (a) the load syncer bypassed the response to the scalar LOAD_WB port, where
//       the scoreboard zeroes cap_result at writeback -- the value was still
//       INTACT IN MEMORY, so re-issuing the identical load returns it TAGGED;
//   (b) the load genuinely returned tag=0 -- memory lost it, and a retry is
//       also untagged.
//
// So the retry is not a blind workaround: it is the discriminator. If the
// instrumented build stops wedging, memory was fine and the defect is in
// register delivery.
//
// THE SHAPE EMITTED, per `ldc`:
//
//        ldc  rd, <addr>
//        lcc  rtmp, rd, 1        ; field 1 = TOTAL type query, 7 == NOT_CAP.
//        addi rtmp2, rtmp, -7    ; TOTAL means it REPORTS rather than raising,
//        bnez rtmp2, skip        ; which is what lets generated code branch on it.
//        ldc  rd2, <addr>        ; retry the IDENTICAL address
//   skip:
//        rd3 = PHI [rd, head], [rd2, retry]
//
// A KNOWN CONFOUND, STATED UP FRONT. The `lcc` sits between the producing `ldc`
// and its dependent consumer, so it SERIALIZES exactly the shape under test.
// An earlier workaround in this backend hit this and its commit message records
// it: putting the type query between the two loads "consumes the result and
// serialises exactly the overlap the rung exists to create". Therefore a clean
// run of an instrumented build does NOT by itself prove the retry repaired
// anything -- it may equally mean the extra instruction masked the defect. A
// WEDGE is unambiguous; a clean run needs the cold-path hit counter (a later
// phase) before it carries a verdict.
//
//===----------------------------------------------------------------------===//

#include "Capstone.h"
#include "CapstoneInstrInfo.h"
#include "CapstoneSubtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "capstone-ldc-retry"
#define PASS_NAME "Capstone S-07 untagged-ldc retry"

// LCC field 1 is the TOTAL type query: it answers 7 for a non-capability and
// does NOT raise, which is the whole reason generated code can test it. On
// silicon predating the S-06 enabler the query RAISES on plain data, so this is
// off by default and must stay that way.
static constexpr int64_t LccFieldType = 1;
static constexpr int64_t NotCapTypeValue = 7;

static cl::opt<bool> RetryUntaggedLdc(
    "capstone-retry-untagged-ldc", cl::Hidden,
    cl::desc("S-07 instrument: after every ldc, query the loaded value's type "
             "and re-issue the identical load if it came back NOT_CAP"),
    cl::init(false));

static cl::opt<bool> DoubleLdc(
    "capstone-double-ldc", cl::Hidden,
    cl::desc("S-07 mitigation: emit every ldc TWICE and use the second result. "
             "Same question as -capstone-retry-untagged-ldc (does re-reading "
             "return a good value?) at a tenth of the code size, because it "
             "needs no type query, no branch, no scratch register and no PHI"),
    cl::init(false));

namespace {

class CapstoneLdcRetry : public MachineFunctionPass {
public:
  static char ID;

  CapstoneLdcRetry() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  /// Unconditional-reload form; see -capstone-double-ldc.
  bool instrumentDouble(MachineInstr &MI);

public:

  StringRef getPassName() const override { return PASS_NAME; }

  // Deliberately NOT setPreservesCFG(): this pass introduces a branch per
  // instrumented load.
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
  bool instrument(MachineInstr &MI);
};

} // end anonymous namespace

/// Rewrite one `ldc` into the guarded/retried form described in the file
/// header. Returns true if anything changed.
bool CapstoneLdcRetry::instrument(MachineInstr &MI) {
  MachineBasicBlock *HeadMBB = MI.getParent();
  MachineFunction &MF = *HeadMBB->getParent();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const CapstoneInstrInfo &TII = *MF.getSubtarget<CapstoneSubtarget>().getInstrInfo();
  const DebugLoc &DL = MI.getDebugLoc();

  Register OrigDef = MI.getOperand(0).getReg();

  // Physical registers would need a scavenger and cannot carry a PHI; this pass
  // runs pre-RA precisely so that it does not have to deal with them.
  if (OrigDef.isPhysical())
    return false;

  const TargetRegisterClass *RC = MRI.getRegClass(OrigDef);

  // Redirect every existing use of the loaded value to a new register, which
  // will be defined by the PHI that merges the original load with the retry.
  // replaceRegWith also rewrites this instruction's own def, so put it back.
  Register MergedReg = MRI.createVirtualRegister(RC);
  MRI.replaceRegWith(OrigDef, MergedReg);
  MI.getOperand(0).setReg(OrigDef);

  // Split the block after the load: head | retry | skip.
  MachineBasicBlock *RetryMBB = MF.CreateMachineBasicBlock(HeadMBB->getBasicBlock());
  MachineBasicBlock *SkipMBB = MF.CreateMachineBasicBlock(HeadMBB->getBasicBlock());
  MF.insert(std::next(HeadMBB->getIterator()), RetryMBB);
  MF.insert(std::next(RetryMBB->getIterator()), SkipMBB);

  SkipMBB->splice(SkipMBB->begin(), HeadMBB, std::next(MI.getIterator()),
                  HeadMBB->end());
  SkipMBB->transferSuccessorsAndUpdatePHIs(HeadMBB);

  HeadMBB->addSuccessor(RetryMBB);
  HeadMBB->addSuccessor(SkipMBB);
  RetryMBB->addSuccessor(SkipMBB);

  // head: ask the type, and skip the retry unless it is NOT_CAP.
  // LCC reads a capability FIELD into an integer register, and the compare that
  // follows is integer arithmetic. Only the reloaded value is a capability.
  const TargetRegisterClass *IntRC = &Capstone::GPRRegClass;
  Register TypeReg = MRI.createVirtualRegister(IntRC);
  Register DiffReg = MRI.createVirtualRegister(IntRC);
  BuildMI(HeadMBB, DL, TII.get(Capstone::LCC), TypeReg)
      .addReg(OrigDef)
      .addImm(LccFieldType);
  BuildMI(HeadMBB, DL, TII.get(Capstone::ADDI), DiffReg)
      .addReg(TypeReg)
      .addImm(-NotCapTypeValue);
  BuildMI(HeadMBB, DL, TII.get(Capstone::BNE))
      .addReg(DiffReg)
      .addReg(Capstone::X0)
      .addMBB(SkipMBB);

  // retry: re-issue the IDENTICAL load. The address operands are cloned rather
  // than reconstructed, so this works whether the original addresses a register
  // plus an immediate or a frame index.
  Register RetryReg = MRI.createVirtualRegister(RC);
  MachineInstrBuilder Retry =
      BuildMI(RetryMBB, DL, TII.get(Capstone::LDC), RetryReg);
  for (unsigned I = 1, E = MI.getNumOperands(); I != E; ++I) {
    MachineOperand MO = MI.getOperand(I);
    // The address is now read in the retry block too, so neither copy may claim
    // to kill it. Without this the verifier rejects the function with "virtual
    // register killed in block, but needed live out" -- the original load
    // killed the address register, and the retry block is its successor.
    // Guarded: setIsKill asserts on anything that is not a register use, and
    // an LDC's address may be an immediate or a frame index.
    if (MO.isReg() && MO.isUse())
      MO.setIsKill(false);
    Retry.add(MO);
  }
  Retry.cloneMemRefs(MI);

  // clearKillFlags covers every use of the register in the function, which is
  // what is wanted here: the original load's operand must lose its kill too.
  for (const MachineOperand &MO : MI.uses())
    if (MO.isReg() && MO.getReg().isVirtual())
      MRI.clearKillFlags(MO.getReg());

  // skip: whichever load produced a capability is the value everything else
  // already refers to.
  BuildMI(*SkipMBB, SkipMBB->begin(), DL, TII.get(TargetOpcode::PHI), MergedReg)
      .addReg(OrigDef)
      .addMBB(HeadMBB)
      .addReg(RetryReg)
      .addMBB(RetryMBB);

  return true;
}

/// Emit the load a second time and let every consumer read the SECOND result.
///
/// WHY THIS MODE EXISTS. The guarded form above costs ~43 bytes per site at -O0 --
/// a type query, a compare, a branch, a duplicated load and a PHI whose merge gets
/// spilled. Over SQLite's 54,844 `ldc` sites that is +2.25 MiB, which pushes the
/// image past `code_len + max(code_len, DATA)` into an order-11 allocation that
/// `__get_free_pages` cannot satisfy, so the domain fails to be created and the
/// experiment never runs. Restricting instrumentation to loads whose result feeds
/// another capability op does not save enough either: measured on the real
/// artifact, that is 32% of sites and still lands in order-11.
///
/// This form answers the SAME question -- does re-reading the identical address
/// return a good value? -- in one instruction, with no control flow, no scratch
/// register and no PHI. If the defect is a transient bad load, the second read
/// repairs it; if memory itself holds an untagged granule, both reads return it
/// and the domain still wedges. That is the same discrimination the guarded form
/// gives, so nothing is lost by taking the cheap one.
///
/// The second load deliberately carries NO memoperands. LLVM treats a load with
/// no memoperand as touching unknown memory, which stops anything downstream from
/// folding the pair back into a single access -- the failure mode that once turned
/// a repeat-the-load-N-times ladder on this project into ONE load regardless of N,
/// and reported with total confidence. The artifact is still disassembled and the
/// loads counted before any board time; this only removes the obvious way to lose.
bool CapstoneLdcRetry::instrumentDouble(MachineInstr &MI) {
  MachineBasicBlock &MBB = *MI.getParent();
  MachineFunction &MF = *MBB.getParent();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const CapstoneInstrInfo &TII =
      *MF.getSubtarget<CapstoneSubtarget>().getInstrInfo();
  const DebugLoc &DL = MI.getDebugLoc();

  Register OrigDef = MI.getOperand(0).getReg();
  if (OrigDef.isPhysical())
    return false;
  MachineOperand &Base = MI.getOperand(1);
  if (!Base.isReg() || Base.getReg().isPhysical())
    return false;

  const TargetRegisterClass *RC = MRI.getRegClass(OrigDef);
  Register NewDef = MRI.createVirtualRegister(RC);

  // Point every existing consumer at the second load's result, then give the
  // first load its own register back -- replaceRegWith rewrites the def too.
  MRI.replaceRegWith(OrigDef, NewDef);
  MI.getOperand(0).setReg(OrigDef);

  // The base has to stay live across the first load for the second to use it.
  Base.setIsKill(false);

  BuildMI(MBB, std::next(MI.getIterator()), DL, TII.get(Capstone::LDC), NewDef)
      .add(MI.getOperand(1))
      .add(MI.getOperand(2));
  return true;
}

bool CapstoneLdcRetry::runOnMachineFunction(MachineFunction &MF) {
  // Deliberately NOT skipFunction(MF). That honours `optnone`, which clang puts
  // on EVERY function at -O0 -- and the silicon SQLite domain is built at -O0,
  // so skipping there would disable this pass in exactly the configuration
  // where S-07 was measured. The same mistake was made once already by the
  // S-06 granule-copy workaround and had to be corrected.
  if (!RetryUntaggedLdc && !DoubleLdc)
    return false;

  // Collect first: instrumenting splits blocks, which invalidates iteration.
  // Collecting also means the retry loads this pass inserts are never
  // themselves instrumented, which would otherwise not terminate.
  SmallVector<MachineInstr *, 16> Loads;
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : MBB)
      if (MI.getOpcode() == Capstone::LDC)
        Loads.push_back(&MI);

  bool Changed = false;
  for (MachineInstr *MI : Loads)
    Changed |= DoubleLdc ? instrumentDouble(*MI) : instrument(*MI);

  return Changed;
}

INITIALIZE_PASS(CapstoneLdcRetry, DEBUG_TYPE, PASS_NAME, false, false)

char CapstoneLdcRetry::ID = 0;

FunctionPass *llvm::createCapstoneLdcRetryPass() { return new CapstoneLdcRetry(); }
