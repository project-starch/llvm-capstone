//===-- CapstoneRecoverProvenance.cpp - capability back from an address ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// C that computes a pointer through an integer -- align it down, step it on,
// set or clear a flag in its low bits, and cast back:
//
//     (T *)(((uintptr_t)p + 63) & ~(uintptr_t)63)
//
// is an integer round trip, and on this target an integer holds only the
// address: the pointer cast back is untagged and traps on first use. Where the
// integer demonstrably came from ONE pointer in the same function, this pass
// rebuilds the result as that pointer moved to the computed address:
//
//     inttoptr(X)  ->  getelementptr i8, p, (X - ptrtoint(p))
//
// The address is exactly the one the source computed; the capability -- tag,
// bounds, permissions -- is p's. Nothing is widened: if X lies outside p's
// bounds, an access through the result traps as any out-of-bounds access does.
// This is the rule a capability-carrying uintptr_t gives CHERI C (arithmetic on
// the address, provenance from the one capability operand), applied where the
// provenance is still visible in the IR, before instruction selection loses it.
//
// "From one pointer" is decided on the IR, not guessed:
//   - a leaf is a ptrtoint of a capability (that pointer is a source), or
//     anything else -- a constant, an argument, a load, a call -- which
//     contributes an integer and no source; constant expressions count as
//     the operations they spell, so `(uintptr_t)&global` is a source too;
//   - a load from a local slot every store to which is visible here -- at -O0
//     every local variable is one -- has the sources of what is stored there;
//   - add, or, xor, and, and the extensions and truncations between the
//     address width and the i128 carrier propagate the sources of both sides;
//     sub propagates its left side's, and a sub whose RIGHT side has a source
//     is a difference of addresses: no pointer, and the whole value is left
//     alone; select and phi take the union of their inputs;
//   - any other operation (multiply, shift, divide, compare, ...) is fine on
//     plain integers -- `(uintptr_t)p + i * 8` is the common shape -- but if a
//     pointer's address flows INTO one, its result is no longer that pointer's
//     address moved, and the whole value is left alone.
// The rewrite happens only when exactly one source remains and it dominates
// the cast. Two sources (hashing two pointers, a difference plus a pointer)
// keep the IR's answer, an untagged value, as before.
//
// A pointer that must carry NO authority -- an address kept on purpose, say for
// a zero-byte allocation -- is written with the address taken explicitly,
// `(T *)__builtin_capstone_cap_get_cursor(p)`: an integer produced by a call is
// not a source, so the result stays untagged. `(T *)(uintptr_t)p` used to give
// that too and now gives p back.
//
// Not covered, and cannot be here: an integer that went through any other
// MEMORY (a struct field of type uintptr_t, a list link, an arena address kept
// as a number, a local whose address escapes). Its tag is gone when it is
// stored, and nothing in the function knows where it came from.
// -Wcapstone-pointer-roundtrip is the diagnostic for that case.
//
// Why a source chosen this way is never worse than no rewrite: the result has
// the source's authority and the program's own address, nothing more. Where the
// address lies outside the source's bounds the access traps, as the untagged
// pointer would have; the pass cannot grant what the function does not hold.
//
//===----------------------------------------------------------------------===//

#include "Capstone.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/InstSimplifyFolder.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/ReplaceConstant.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "capstone-provenance"
#define PASS_NAME "Capstone: recover a capability from an address computed from one pointer"

// Off switch, for measuring what the pass changes and for the QEMU test's
// control arm (the same program with the round trips left untagged).
static cl::opt<bool> EnableRecoverProvenance(
    "capstone-recover-provenance", cl::Hidden, cl::init(true),
    cl::desc("Rebuild an integer round trip computed from one capability as "
             "that capability moved (CapstoneRecoverProvenance)"));

namespace {

class CapstoneRecoverProvenance : public FunctionPass {
public:
  static char ID;
  CapstoneRecoverProvenance() : FunctionPass(ID) {}
  bool runOnFunction(Function &F) override;
  StringRef getPassName() const override { return PASS_NAME; }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.setPreservesCFG();
  }
};

// The capability address space: pointers there are capabilities.
constexpr unsigned CapAS = 200;

struct SourceFinder {
  // Distinct source pointers found so far. More than one ends the search.
  SmallVector<Value *, 2> Sources;
  SmallPtrSet<const Value *, 16> Visited;
  bool Poisoned = false; // an address difference, or an opaque operation
  // Nesting of the independent searches carriesAddress starts. Each has its
  // own Visited set, so a cycle through a phi and a multiply would otherwise
  // recurse without end; past the limit the answer is the conservative one
  // ("it may carry an address"), which only ever means "do not rewrite".
  unsigned Depth = 0;
  static constexpr unsigned MaxDepth = 4;

  void addSource(Value *P) {
    if (!is_contained(Sources, P))
      Sources.push_back(P);
  }

  // True if V has a source or is itself poisoned, searched on its own.
  bool carriesAddress(Value *V) const {
    if (Depth >= MaxDepth)
      return true;
    SourceFinder Sub;
    Sub.Depth = Depth + 1;
    Sub.walk(V);
    return Sub.Poisoned || !Sub.Sources.empty();
  }

  // A load from a local slot: at -O0 `uintptr_t t = (uintptr_t)p | 1;` is a
  // store to one and every later read of t a load, so the round trip reaches
  // the cast through memory. The slot's value is one of the values stored into
  // it, so its sources are theirs -- provided every store is visible, which
  // holds when the slot is only ever loaded from and stored to by address.
  // Anything else (its address passed on, stored, or offset; a volatile or
  // atomic access; a store of another type) means it may hold anything: no
  // source.
  void walkSlot(LoadInst *LI) {
    auto *A = dyn_cast<AllocaInst>(LI->getPointerOperand());
    if (!A || !LI->isSimple())
      return;
    SmallVector<Value *, 4> Stored;
    for (User *U : A->users()) {
      if (auto *L = dyn_cast<LoadInst>(U)) {
        if (!L->isSimple() || L->getType() != LI->getType())
          return;
      } else if (auto *S = dyn_cast<StoreInst>(U)) {
        if (S->getPointerOperand() != A || !S->isSimple() ||
            S->getValueOperand()->getType() != LI->getType())
          return;
        Stored.push_back(S->getValueOperand());
      } else if (auto *II = dyn_cast<IntrinsicInst>(U);
                 !II || !II->isLifetimeStartOrEnd()) {
        return;
      }
    }
    for (Value *V : Stored)
      walk(V);
  }

  // Walk V, collecting sources; sets Poisoned when the value cannot be one
  // pointer moved.
  void walk(Value *V) {
    if (Poisoned || Sources.size() > 1)
      return;
    if (!Visited.insert(V).second)
      return;
    // An instruction or a constant expression: `(uintptr_t)&g + 8` is folded
    // into one, at -O0 as well, and means the same as the instructions.
    auto *I = dyn_cast<Operator>(V);
    if (!I)
      return; // a constant integer or an argument: an integer, no source
    switch (I->getOpcode()) {
    case Instruction::PtrToInt: {
      Value *P = I->getOperand(0);
      if (P->getType()->getPointerAddressSpace() == CapAS)
        addSource(P);
      return;
    }
    case Instruction::Add:
    case Instruction::Or:
    case Instruction::Xor:
    case Instruction::And:
      walk(I->getOperand(0));
      walk(I->getOperand(1));
      return;
    case Instruction::Sub:
      if (carriesAddress(I->getOperand(1))) {
        Poisoned = true; // p - q: a distance, not a pointer
        return;
      }
      walk(I->getOperand(0));
      return;
    case Instruction::ZExt:
    case Instruction::SExt:
    case Instruction::Trunc:
    case Instruction::Freeze:
      walk(I->getOperand(0));
      return;
    case Instruction::Select:
      walk(I->getOperand(1));
      walk(I->getOperand(2));
      return;
    case Instruction::PHI:
      for (Value *In : cast<PHINode>(I)->incoming_values())
        walk(In);
      return;
    default:
      if (auto *LI = dyn_cast<LoadInst>(I))
        return walkSlot(LI);
      // A call: an integer from elsewhere, no source (its tag, if it ever had
      // one, is already gone).
      if (isa<CallBase>(I))
        return;
      // Anything else is arithmetic that does not move an address: harmless
      // on plain integers, disqualifying when an address flows into it.
      for (Value *Op : I->operands())
        if (carriesAddress(Op)) {
          Poisoned = true;
          return;
        }
      return;
    }
  }
};

// The offset of V from P's address, when V is P's address plus or minus other
// integers: `p + e` gives e, `p - e` gives -e, through the extensions the i128
// carrier adds. Built directly rather than as V - addr(p), because this pass runs
// after the optimizer and nothing would fold `(p - n) - p` back to `-n`. Returns
// nullptr for anything else (a mask, a select, a phi); the caller then uses the
// general V - addr(p), which is right for every shape.
static Value *linearOffset(Value *V, Value *P, Type *IdxTy, IRBuilderBase &B,
                           unsigned Depth = 0) {
  if (Depth > 8)
    return nullptr;
  auto *I = dyn_cast<Operator>(V);
  if (!I)
    return nullptr;
  auto hasSource = [&](Value *X) {
    SourceFinder SF;
    SF.walk(X);
    return is_contained(SF.Sources, P);
  };
  auto asIdx = [&](Value *X) { return B.CreateSExtOrTrunc(X, IdxTy); };
  switch (I->getOpcode()) {
  case Instruction::PtrToInt:
    return I->getOperand(0) == P ? ConstantInt::get(IdxTy, 0) : nullptr;
  case Instruction::ZExt:
  case Instruction::SExt:
  case Instruction::Trunc:
    return linearOffset(I->getOperand(0), P, IdxTy, B, Depth + 1);
  case Instruction::Or:
    if (!isa<PossiblyDisjointInst>(I) ||
        !cast<PossiblyDisjointInst>(I)->isDisjoint())
      return nullptr;
    [[fallthrough]];
  case Instruction::Add: {
    Value *L = I->getOperand(0), *R = I->getOperand(1);
    if (!hasSource(L))
      std::swap(L, R);
    Value *D = linearOffset(L, P, IdxTy, B, Depth + 1);
    return D ? B.CreateAdd(D, asIdx(R)) : nullptr;
  }
  case Instruction::Sub: {
    Value *D = linearOffset(I->getOperand(0), P, IdxTy, B, Depth + 1);
    return D ? B.CreateSub(D, asIdx(I->getOperand(1))) : nullptr;
  }
  default:
    return nullptr;
  }
}

// True if C, a constant, contains the address of a capability: a ptrtoint of
// an address-space-200 pointer somewhere inside it.
static bool hasCapabilityAddress(const Constant *C,
                                 SmallPtrSetImpl<const Constant *> &Seen) {
  if (!Seen.insert(C).second)
    return false;
  auto *CE = dyn_cast<ConstantExpr>(C);
  if (!CE)
    return false;
  if (CE->getOpcode() == Instruction::PtrToInt &&
      CE->getOperand(0)->getType()->getPointerAddressSpace() == CapAS)
    return true;
  for (const Use &Op : CE->operands())
    if (hasCapabilityAddress(cast<Constant>(Op), Seen))
      return true;
  return false;
}

// Collect every constant `inttoptr` to a capability, used by an instruction of
// F (directly or inside another constant expression), whose operand contains a
// capability's address, and turn it into an instruction.
static bool expandConstantRoundTrips(Function &F) {
  SetVector<Constant *> Found;
  SmallPtrSet<const Constant *, 16> Visited;
  SmallVector<Constant *, 8> Stack;
  for (Instruction &I : instructions(F))
    for (Use &Op : I.operands())
      if (auto *CE = dyn_cast<ConstantExpr>(Op.get()))
        Stack.push_back(CE);
  while (!Stack.empty()) {
    auto *CE = cast<ConstantExpr>(Stack.pop_back_val());
    if (!Visited.insert(CE).second)
      continue;
    if (CE->getOpcode() == Instruction::IntToPtr &&
        CE->getType()->getPointerAddressSpace() == CapAS) {
      SmallPtrSet<const Constant *, 16> Seen;
      if (hasCapabilityAddress(CE->getOperand(0), Seen))
        Found.insert(CE);
    }
    for (Use &Op : CE->operands())
      if (auto *Inner = dyn_cast<ConstantExpr>(Op.get()))
        Stack.push_back(Inner);
  }
  if (Found.empty())
    return false;
  return convertUsersOfConstantsToInstructions(Found.getArrayRef(), &F,
                                               /*RemoveDeadConstants=*/false,
                                               /*IncludeSelf=*/true);
}

} // namespace

bool CapstoneRecoverProvenance::runOnFunction(Function &F) {
  // Not skipFunction(): that honours optnone, which clang puts on EVERY function
  // at -O0, and this pass is not an optimization -- without it the -O0 build of
  // a round trip traps. Measured: the -O0 QEMU arm halted on its first
  // round-tripped pointer while hand-written IR (no optnone) was rewritten.
  if (!EnableRecoverProvenance)
    return false;
  // A round trip made of constants only -- `(T *)((uintptr_t)&g + 8)` -- is a
  // constant expression, even at -O0, and never an instruction this pass would
  // see. Where one carries a capability's address, expand it into
  // instructions here so the loop below treats it like any other.
  bool Changed = expandConstantRoundTrips(F);
  DominatorTree &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();

  SmallVector<IntToPtrInst *, 8> Casts;
  for (Instruction &I : instructions(F))
    if (auto *ITP = dyn_cast<IntToPtrInst>(&I))
      if (ITP->getType()->getPointerAddressSpace() == CapAS &&
          !isa<ConstantData>(ITP->getOperand(0)))
        Casts.push_back(ITP);

  for (IntToPtrInst *ITP : Casts) {
    SourceFinder SF;
    SF.walk(ITP->getOperand(0));
    if (SF.Poisoned || SF.Sources.size() != 1)
      continue;
    Value *P = SF.Sources.front();
    if (auto *PI = dyn_cast<Instruction>(P); PI && !DT.dominates(PI, ITP))
      continue;

    // InstSimplifyFolder: `0 + e` and the like fold as they are built, since
    // no optimization pass runs after this one.
    IRBuilder<InstSimplifyFolder> B(ITP->getContext(),
                                    InstSimplifyFolder(F.getDataLayout()));
    B.SetInsertPoint(ITP);
    Value *X = ITP->getOperand(0);
    Type *IntTy = X->getType();
    Type *IdxTy =
        F.getDataLayout().getIndexType(ITP->getType()); // i64 on capstone64
    Value *Delta = linearOffset(X, P, IdxTy, B);
    if (!Delta) {
      Value *Base = B.CreatePtrToInt(P, IntTy, P->getName() + ".addr");
      Delta = B.CreateSExtOrTrunc(B.CreateSub(X, Base, "prov.delta"), IdxTy);
    }
    Value *NewP = B.CreateGEP(B.getInt8Ty(), P, Delta, ITP->getName() + ".prov");
    if (NewP->getType() != ITP->getType())
      NewP = B.CreateAddrSpaceCast(NewP, ITP->getType());
    ITP->replaceAllUsesWith(NewP);
    ITP->eraseFromParent();
    Changed = true;
  }
  return Changed;
}

char CapstoneRecoverProvenance::ID = 0;

INITIALIZE_PASS_BEGIN(CapstoneRecoverProvenance, DEBUG_TYPE, PASS_NAME, false,
                      false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(CapstoneRecoverProvenance, DEBUG_TYPE, PASS_NAME, false,
                    false)

FunctionPass *llvm::createCapstoneRecoverProvenancePass() {
  return new CapstoneRecoverProvenance();
}
