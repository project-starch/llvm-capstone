//===-- CapstoneMemcpyAlignDispatch.cpp - inline small copies of unknown alignment -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// C-81 sends a constant-size copy of a granule or more whose 16-byte alignment
// the compiler cannot prove to a `memcpy` call (findOptimalMemOpLowering): an
// `align 8` copy may still hold a capability at a 16-aligned address, and a
// scalar copy would drop its tag. Its comment names what was missing, a
// run-time alignment test in the expansion. This pass is that test, done in IR
// so that both arms are ordinary code for SelectionDAG:
//
//   if (((uintptr_t)d | (uintptr_t)s) & 15) == 0
//     llvm.memcpy.inline(d, s, N), align 16 on both: C-78's capability-grained
//                                  copy, one ldc/stc per granule, tags kept;
//   else
//     two aligned i64 loads, then two i64 stores: no whole source granule
//                                  lands on a whole destination granule, so
//                                  no tag can survive, and none does.
//
// That is the rule the runtime's memcpy applies to this size
// (ports/musl-capstone/runtime/string_bounds_safe.c, copy_forward: granules
// move as capabilities only when the two cursors agree mod 16, and then only
// the whole granules inside the range). For N = 16 with both cursors
// 8-aligned, a copy that is not 16-aligned on both ends covers no whole
// destination granule that a whole source granule lands on.
//
// N = 16 ONLY. At N = 32 with both cursors 8 mod 16 the middle 16 bytes ARE a
// whole granule on both sides, which memcpy copies as a capability and a word
// copy would strip; a larger N needs a third arm (head word, granules, tail
// word), not this one.
//
// The aligned arm is the same ldc/stc the runtime's granule loop issues. No
// lcc guard: R-29 is fixed on silicon (2026-10-05), and the memcpy the
// virtual SDK's images link has none (CAPSTONE_MEMCPY_PLAIN_GUARD is off).
//
// Every load precedes every store in both arms, so an exact self-copy
// (d == s, which an aggregate self-assignment lowers to) keeps its bytes and
// tags; llvm.memcpy permits no other overlap.
//
// Scope: llvm.memcpy (not .inline, not volatile) of the constant 16, both
// operands in the capability address space with a known alignment of at
// least 8 but not both 16. A source in constant memory is left alone
// (SelectionDAG copies it as bytes it can read). Default off.
//
//===----------------------------------------------------------------------===//

#include "Capstone.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

#define DEBUG_TYPE "capstone-memcpy-dispatch"  // the pass name; it cannot be the flag's (opt registers both)
#define PASS_NAME "Capstone run-time alignment dispatch for small copies"

STATISTIC(NumDispatched, "Copies given a run-time alignment test");

static cl::opt<bool> EnableMemcpyAlignDispatch(
    "capstone-memcpy-align-dispatch", cl::Hidden, cl::init(false),
    cl::desc("Inline a constant-size copy whose 16-byte alignment is unknown "
             "(C-81) as a run-time alignment test: capability copy when both "
             "cursors are 16-aligned, aligned words otherwise"));

namespace {

class CapstoneMemcpyAlignDispatch : public FunctionPass {
public:
  static char ID;
  CapstoneMemcpyAlignDispatch() : FunctionPass(ID) {}
  bool runOnFunction(Function &F) override;
  StringRef getPassName() const override { return PASS_NAME; }
};

} // end anonymous namespace

static bool isCandidate(MemCpyInst *MC) {
  if (MC->isForceInlined() || MC->isVolatile())
    return false;
  auto *Len = dyn_cast<ConstantInt>(MC->getLength());
  if (!Len)
    return false;
  if (Len->getZExtValue() != 16) // see the header: larger sizes need a third arm
    return false;
  unsigned AS = 200;
  if (MC->getDestAddressSpace() != AS || MC->getSourceAddressSpace() != AS)
    return false;
  Align DA = MC->getDestAlign().valueOrOne(), SA = MC->getSourceAlign().valueOrOne();
  if (DA < Align(8) || SA < Align(8))
    return false;
  if (DA >= Align(16) && SA >= Align(16))
    return false; // C-78 already copies it as capabilities, inline
  // A source in constant memory is plain bytes SelectionDAG reads itself.
  if (auto *GV = dyn_cast<GlobalVariable>(getUnderlyingObject(MC->getSource())))
    if (GV->isConstant())
      return false;
  return true;
}

static void dispatch(MemCpyInst *MC) {
  uint64_t N = cast<ConstantInt>(MC->getLength())->getZExtValue();
  Value *D = MC->getDest(), *S = MC->getSource();
  LLVMContext &Ctx = MC->getContext();
  Type *I64 = Type::getInt64Ty(Ctx), *I128 = Type::getInt128Ty(Ctx);

  IRBuilder<> B(MC);
  // The cursor, as clang lowers (uintptr_t)p: ptrtoint to the pointer's
  // width, then the low 64 bits.
  Value *DI = B.CreateTrunc(B.CreatePtrToInt(D, I128), I64);
  Value *SI = B.CreateTrunc(B.CreatePtrToInt(S, I128), I64);
  Value *Low = B.CreateAnd(B.CreateOr(DI, SI), ConstantInt::get(I64, 15));
  Value *Aligned = B.CreateICmpEQ(Low, ConstantInt::get(I64, 0));

  Instruction *ThenT, *ElseT;
  SplitBlockAndInsertIfThenElse(Aligned, MC, &ThenT, &ElseT);

  // Both cursors 16-aligned: the capability-grained inline copy.
  B.SetInsertPoint(ThenT);
  CallInst *Cap = B.CreateMemCpyInline(D, Align(16), S, Align(16),
                                       ConstantInt::get(I64, N));
  Cap->copyMetadata(*MC);

  // Otherwise aligned words: every load before every store.
  B.SetInsertPoint(ElseT);
  SmallVector<Value *, 8> Words;
  for (uint64_t Off = 0; Off < N; Off += 8) {
    Value *P = B.CreateConstInBoundsGEP1_64(B.getInt8Ty(), S, Off);
    Words.push_back(B.CreateAlignedLoad(I64, P, Align(8)));
  }
  for (uint64_t Off = 0, K = 0; Off < N; Off += 8, ++K) {
    Value *P = B.CreateConstInBoundsGEP1_64(B.getInt8Ty(), D, Off);
    B.CreateAlignedStore(Words[K], P, Align(8));
  }

  MC->eraseFromParent();
  ++NumDispatched;
}

bool CapstoneMemcpyAlignDispatch::runOnFunction(Function &F) {
  if (!EnableMemcpyAlignDispatch || skipFunction(F))
    return false;
  SmallVector<MemCpyInst *, 16> Work;
  for (BasicBlock &BB : F)
    for (Instruction &I : BB)
      if (auto *MC = dyn_cast<MemCpyInst>(&I))
        if (isCandidate(MC))
          Work.push_back(MC);
  for (MemCpyInst *MC : Work)
    dispatch(MC);
  return !Work.empty();
}

char CapstoneMemcpyAlignDispatch::ID = 0;

INITIALIZE_PASS(CapstoneMemcpyAlignDispatch, DEBUG_TYPE, PASS_NAME, false, false)

FunctionPass *llvm::createCapstoneMemcpyAlignDispatchPass() {
  return new CapstoneMemcpyAlignDispatch();
}
