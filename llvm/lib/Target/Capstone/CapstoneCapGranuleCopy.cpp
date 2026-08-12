//===- CapstoneCapGranuleCopy.cpp - guard capability-grained copies -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// S-06 WORKAROUND, CODEGEN SIDE. Expands a 16-byte-aligned constant-size
// llvm.memcpy into per-granule IR in which the capability store is CONDITIONAL
// on the granule actually holding a capability.
//
// THE DEFECT. On this silicon an `ldc`/`stc` pair used to copy PLAIN data can
// destroy the granule's high 8 bytes. `ldc` of an untagged granule loads
// metadata 0; the pipeline then re-encodes that zero against the CURSOR, and
// compress_bounds switches to its "cursorless" scheme when bounds.start ==
// cursor -- which decompress_bounds(0, cursor) makes true exactly when the
// cursor's low 14 bits are zero. That manufactures a NONZERO metadata out of an
// all-zero input, so the `stc` asserts st_wr_cap, writes both banks, and the
// high half is lost. The trigger is `low half % 0x4000 == 0`, NOT `== 0`;
// measured in RTL simulation (s06-lowhalf-zero.S / -swap.S, TESTNUM 5).
//
// WHY A PASS AND NOT MORE OF THE DAG WORKAROUND. The only correct repair is to
// ASK -- LCC field 1 is total on enabler silicon, answering 7 for a
// non-capability -- and then store the capability only when there is one. That
// is a BRANCH, and SelectionDAG's memcpy expansion cannot introduce control
// flow, which is why the existing EmitTargetCodeForMemcpy workaround
// (-capstone-memcpy-high-half-fixup) ends in an UNCONDITIONAL `stc` and is
// therefore still wrong for exactly the granules that trigger the defect. At
// the IR level the branch is free.
//
// THE SEQUENCE, per 16-byte granule:
//
//     lo  = load i64  src+off
//     hi  = load i64  src+off+8
//     cap = load ptr200 src+off            <-- ALL loads before ANY store
//     store volatile i64 lo -> dst+off
//     store volatile i64 hi -> dst+off+8
//     ty  = llvm.capstone.cap.get.type(cap)
//     if (ty != 7) store ptr200 cap -> dst+off
//
// READ EVERYTHING FIRST is correctness, not scheduling taste. A plain store
// CLEARS the target granule's tag (wt_dcache_mem.sv:419-422, an unconditional
// overwrite of cap_tag_q, not an OR). For an exact self-copy -- which `*d = *s`
// with d == s is, and which clang lowers to a memcpy -- storing first would
// clear the tag of the very line the `ldc` is about to read, so a live
// capability would be destroyed. The same fact is why the tempting branchless
// alternative (store the capability first, then repair both halves with plain
// stores) is UNSOUND: the trailing plain stores would strip the tag from every
// real capability, regardless of whether the bytes they write are identical.
//
// THE PLAIN STORES MUST STAY VOLATILE. By the compiler's own model the `stc`
// writes all 16 bytes, so without volatile they are dead and DSE regenerates
// the unfixed sequence. That has happened before: for a copy into a stack slot
// the pre-writes vanished entirely and the output was byte-identical to the
// unfixed build.
//
//===----------------------------------------------------------------------===//

#include "Capstone.h"
#include "CapstoneTargetMachine.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsCapstone.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

#define DEBUG_TYPE "capstone-cap-granule-copy"
#define PASS_NAME "Capstone guarded capability-granule copy"

STATISTIC(NumExpanded, "Number of memcpys expanded into guarded granule copies");
STATISTIC(NumGranules, "Number of 16-byte granules given a guarded store");

// Default OFF, matching every other S-06 workaround in this target. It changes
// emitted geometry -- a compare and a branch per granule -- and the published
// BEEBS numbers were measured without it. It also emits LCC field 1, which is
// only total on enabler silicon; on older silicon the query RAISES on plain
// data, so a build with this flag on requires the matching bitstream.
static cl::opt<bool> GuardCapGranuleCopies(
    "capstone-guard-cap-granule-copies", cl::Hidden,
    cl::desc("S-06 workaround: expand 16-byte-aligned memcpy into per-granule "
             "copies whose capability store is guarded by an LCC type query"),
    cl::init(false));

// Same ceiling as the DAG workaround it supersedes. Above this,
// PreISelIntrinsicLowering has already turned the memcpy into a libcall, which
// lands on the C memcpy -- itself already carrying the guarded copy -- so large
// copies are covered by the library rather than here.
static cl::opt<unsigned> MaxGuardedCopyBytes(
    "capstone-guard-cap-granule-copy-max-bytes", cl::Hidden,
    cl::desc("Only expand copies of at most N bytes"), cl::init(512));

namespace {

class CapstoneCapGranuleCopy : public FunctionPass {
public:
  static char ID;
  CapstoneCapGranuleCopy() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override;
  StringRef getPassName() const override { return PASS_NAME; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    // Deliberately NOT setPreservesCFG(): this pass introduces a branch per
    // granule. Claiming otherwise would leave stale dominator/loop info behind.
    AU.addRequired<TargetPassConfig>();
  }

private:
  bool expand(MemCpyInst *MC);
};

} // end anonymous namespace

/// Is this a copy we can and should rewrite?
static bool isGuardableCopy(MemCpyInst *MC, uint64_t &Bytes) {
  // A volatile copy must not have its stores multiplied or reordered.
  if (MC->isVolatile())
    return false;

  auto *Len = dyn_cast<ConstantInt>(MC->getLength());
  if (!Len)
    return false;
  Bytes = Len->getZExtValue();

  // Only the capability-grained shape is at risk: it is the only one lowered to
  // ldc/stc. Anything else already copies with scalar units and cannot lose a
  // high half.
  if (Bytes == 0 || (Bytes % 16) != 0)
    return false;
  if (Bytes > MaxGuardedCopyBytes)
    return false;

  // Both ends must really be 16-byte aligned, or the backend would not have
  // used capability-grained units either. getDestAlign()/getSourceAlign() are
  // MaybeAlign; an absent alignment is not an assertion of alignment.
  MaybeAlign DstA = MC->getDestAlign();
  MaybeAlign SrcA = MC->getSourceAlign();
  if (!DstA || !SrcA || *DstA < Align(16) || *SrcA < Align(16))
    return false;

  return true;
}

bool CapstoneCapGranuleCopy::expand(MemCpyInst *MC) {
  uint64_t Bytes;
  if (!isGuardableCopy(MC, Bytes))
    return false;

  LLVMContext &Ctx = MC->getContext();
  Module *M = MC->getModule();
  Type *I8 = Type::getInt8Ty(Ctx);
  Type *I64 = Type::getInt64Ty(Ctx);

  Value *Dst = MC->getRawDest();
  Value *Src = MC->getRawSource();

  // A capability is `ptr addrspace(200)` and AS200 is NON-INTEGRAL: never
  // ptrtoint it, and do pointer arithmetic with i8 GEPs, which lower to
  // cincoffset/cincoffsetimm. A plain ISD::ADD here would select to `addi`,
  // which STRIPS the tag -- a shipped bug in the DAG version of this
  // workaround, seen as `addi a5, s0, -0xd8; sd a4, 0x0(a5)`.
  Type *CapTy = Src->getType();
  if (!CapTy->isPointerTy() || CapTy != Dst->getType())
    return false;

  Function *GetType = Intrinsic::getOrInsertDeclaration(
      M, Intrinsic::capstone_cap_get_type, {CapTy});

  IRBuilder<> B(MC);
  const uint64_t GranuleCount = Bytes / 16;

  for (uint64_t G = 0; G != GranuleCount; ++G) {
    const uint64_t Off = G * 16;

    Value *SrcLo = B.CreateConstInBoundsGEP1_64(I8, Src, Off);
    Value *SrcHi = B.CreateConstInBoundsGEP1_64(I8, Src, Off + 8);
    Value *DstLo = B.CreateConstInBoundsGEP1_64(I8, Dst, Off);
    Value *DstHi = B.CreateConstInBoundsGEP1_64(I8, Dst, Off + 8);

    // --- every read of the source happens before any write to the destination
    Value *Lo = B.CreateAlignedLoad(I64, SrcLo, Align(8), "cgc.lo");
    Value *Hi = B.CreateAlignedLoad(I64, SrcHi, Align(8), "cgc.hi");
    Value *Cap = B.CreateAlignedLoad(CapTy, SrcLo, Align(16), "cgc.cap");

    // --- then the plain writes, which lay down BOTH halves correctly
    B.CreateAlignedStore(Lo, DstLo, Align(8), /*isVolatile=*/true);
    B.CreateAlignedStore(Hi, DstHi, Align(8), /*isVolatile=*/true);

    // --- ask, and store the capability only if there is one
    Value *Ty = B.CreateCall(GetType, {Cap}, "cgc.ty");
    Value *IsCap = B.CreateICmpNE(Ty, ConstantInt::get(I64, 7), "cgc.iscap");

    Instruction *ThenTerm =
        SplitBlockAndInsertIfThen(IsCap, &*B.GetInsertPoint(),
                                  /*Unreachable=*/false);
    IRBuilder<> TB(ThenTerm);
    TB.CreateAlignedStore(Cap, DstLo, Align(16));

    // Continue emitting the remaining granules in the join block.
    B.SetInsertPoint(ThenTerm->getSuccessor(0)->getFirstNonPHIIt());
    ++NumGranules;
  }

  MC->eraseFromParent();
  ++NumExpanded;
  return true;
}

bool CapstoneCapGranuleCopy::runOnFunction(Function &F) {
  // Deliberately NOT skipFunction(F). That honours `optnone`, which clang puts on
  // EVERY function at -O0 -- and the silicon SQLite domain is built at -O0, so
  // skipping there would disable this pass in exactly the configuration where the
  // defect was measured. Verified: with skipFunction in place the emitted code was
  // byte-identical with the flag on and off, i.e. the fix was silently inert. This
  // is a correctness workaround for a hardware defect, not an optimisation, so it
  // runs regardless of optimisation attributes.
  if (!GuardCapGranuleCopies)
    return false;

  // Collect first: expand() splits blocks, which invalidates iteration.
  SmallVector<MemCpyInst *, 8> Worklist;
  for (BasicBlock &BB : F)
    for (Instruction &I : BB)
      if (auto *MC = dyn_cast<MemCpyInst>(&I))
        Worklist.push_back(MC);

  bool Changed = false;
  for (MemCpyInst *MC : Worklist)
    Changed |= expand(MC);
  return Changed;
}

INITIALIZE_PASS_BEGIN(CapstoneCapGranuleCopy, DEBUG_TYPE, PASS_NAME, false,
                      false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(CapstoneCapGranuleCopy, DEBUG_TYPE, PASS_NAME, false, false)

char CapstoneCapGranuleCopy::ID = 0;

FunctionPass *llvm::createCapstoneCapGranuleCopyPass() {
  return new CapstoneCapGranuleCopy();
}
