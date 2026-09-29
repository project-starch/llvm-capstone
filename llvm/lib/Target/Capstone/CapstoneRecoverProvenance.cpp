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
//     alone; a select, a phi, and a slot with more than one store need EVERY
//     input to be the same one pointer moved -- an input that is a plain
//     integer would make the result an address that pointer never held;
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
// that too and now gives p back when p is proven not to need a consuming move.
//
// Not covered, and cannot be here: an integer that went through any other
// MEMORY (a struct field of type uintptr_t, a list link, an arena address kept
// as a number, a local whose address escapes). Its tag is gone when it is
// stored, and nothing in the function knows where it came from.
// -Wcapstone-pointer-roundtrip is the diagnostic for that case.
//
// What the rewrite is worth, stated exactly: the result has the source's
// authority and the program's own address, nothing more. Where the address lies
// outside the source's bounds the access traps, as the untagged pointer would
// have, and no rewrite can grant authority the function does not already hold,
// because each one is a GEP of a capability that dominates the cast.
//
// It does add a trap of its own, though, and that decides WHERE it may run.
// `cincoffset` raises UNEXPECTED_OPERAND when its base register holds no
// capability (capstone_flu_unit.anvil, CINCOFFSET; cause 24 in QEMU), and a null
// pointer holds none -- so rewriting `(T *)(((uintptr_t)p + 15) & ~15)` on a p
// that may be null turns a function that returned an address nobody used into
// one that traps. CHERI's cincoffset has no such rule, which is why its
// uintptr_t model needs no condition here; whether this target adopts it is an
// open decision (docs/plans/2026-09-24-scc-cincoffset-untagged.md). Until it is
// taken, a round trip is rebuilt only where the source CERTAINLY HOLDS a
// capability -- an alloca, a global, a pointer past a null test, or one kept in
// a stack slot that holds only those, and never an integer cast to a pointer,
// however non-zero -- or where the rewrite adds no offset at all,
// `(T *)(uintptr_t)p` being p itself. This is the question C-19 asks before
// speculating a GEP on a capability, answered for this target rather than by
// isKnownNonZero alone; see holdsCapability().
// This validity check is separate from linearity: neither a non-null address
// nor a successful dereference proves that an additional use is non-consuming.
//
// The rewrite also adds a USE of the source, and a use is not free for every
// capability. `cincoffset` with rd != rs1 nulls a LINEAR rs1, so a round trip
// rebuilt on a LINEAR value that the program goes on using would leave that use
// reading null. Arguments, loads from unknown memory and ordinary call results
// can hold LINEAR capabilities too: the proposed linearity contract is not
// enforced, and assembly entry points pass live LINEAR arguments today. Only
// sources proven not to need a consuming move may be recovered. See
// mayBeLinear(), which also covers the no-offset rewrite.
//
//===----------------------------------------------------------------------===//

#include "Capstone.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/InstSimplifyFolder.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsCapstone.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/ReplaceConstant.h"
#include "llvm/Analysis/ValueTracking.h"
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
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.setPreservesCFG();
  }
};

// The capability address space: pointers there are capabilities.
constexpr unsigned CapAS = 200;

// The stores of the local slot LI reads, when every access to it is visible:
// the slot is only ever loaded from and stored to by address, with simple
// accesses of LI's type. At -O0 every local variable is such a slot. False when
// the slot may be written some other way, or LI does not read a slot at all.
static bool slotStores(LoadInst *LI, SmallVectorImpl<StoreInst *> &Stores) {
  auto *A = dyn_cast<AllocaInst>(LI->getPointerOperand());
  if (!A || !LI->isSimple())
    return false;
  for (User *U : A->users()) {
    if (auto *L = dyn_cast<LoadInst>(U)) {
      if (!L->isSimple() || L->getType() != LI->getType())
        return false;
    } else if (auto *S = dyn_cast<StoreInst>(U)) {
      if (S->getPointerOperand() != A || !S->isSimple() ||
          S->getValueOperand()->getType() != LI->getType())
        return false;
      Stores.push_back(S);
    } else if (auto *II = dyn_cast<IntrinsicInst>(U);
               !II || !II->isLifetimeStartOrEnd()) {
      return false;
    }
  }
  return true;
}

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
  // The value whose own definition this search is resolving, and whether the
  // search reached it again. A cursor's back edge does: `c = (uintptr_t)p` in
  // the preheader, `c += 8` on the latch, so one input of the phi is the phi
  // itself plus a step. That input carries no source of its own and must not be
  // read as an integer from somewhere else. For a stack slot the node is the
  // alloca, reached through a load.
  const Value *CycleRoot = nullptr;
  bool HitCycle = false;

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
    Sub.CycleRoot = CycleRoot;
    Sub.walk(V);
    // Reaching the node being resolved counts as carrying an address: it holds
    // one by the time the back edge runs.
    return Sub.Poisoned || !Sub.Sources.empty() || Sub.HitCycle;
  }

  // A value that is one of several -- a select, a phi, or a load from a slot
  // written in more than one place -- is one pointer moved only if EVERY input
  // is that same pointer moved. Taking the UNION instead (which this did until
  // 2026-09-28) attached the source's tag, bounds and permissions to an address
  // the source never held: in `c ? (uintptr_t)p : x` the false arm is the
  // caller's own integer, and the rewrite handed p's authority back at it.
  // Root is the node being resolved, for the back-edge case CycleRoot names.
  void joinInputs(ArrayRef<Value *> Inputs, const Value *Root) {
    if (Depth >= MaxDepth) {
      Poisoned = true; // too deep to decide: never rewrite
      return;
    }
    Value *Common = nullptr;
    bool Foreign = false;
    for (Value *In : Inputs) {
      SourceFinder Sub;
      Sub.Depth = Depth + 1;
      Sub.CycleRoot = Root;
      Sub.walk(In);
      if (Sub.Poisoned || Sub.Sources.size() > 1) {
        Poisoned = true;
        return;
      }
      if (Sub.Sources.empty()) {
        // Only the back edge of a cycle through this node may carry no source:
        // its address is this node's own, moved. Anything else is an integer
        // from elsewhere, and this node may hold it instead of the address.
        Foreign |= !Sub.HitCycle;
        continue;
      }
      if (Common && Common != Sub.Sources.front()) {
        Poisoned = true; // two pointers meet here
        return;
      }
      Common = Sub.Sources.front();
    }
    if (!Common)
      return; // plain integers throughout: no source, and nothing disqualified
    if (Foreign) {
      Poisoned = true;
      return;
    }
    addSource(Common);
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
    if (A == CycleRoot) { // the back edge of a cursor kept in a slot
      HitCycle = true;
      return;
    }
    SmallVector<StoreInst *, 4> Stores;
    if (!slotStores(LI, Stores))
      return;
    SmallVector<Value *, 4> Stored;
    for (StoreInst *S : Stores)
      Stored.push_back(S->getValueOperand());
    // Every store, not just one: a slot written with p on one path and a
    // foreign integer on the other is the -O0 spelling of the select above.
    joinInputs(Stored, A);
  }

  // Walk V, collecting sources; sets Poisoned when the value cannot be one
  // pointer moved.
  void walk(Value *V) {
    if (Poisoned || Sources.size() > 1)
      return;
    if (V == CycleRoot) {
      HitCycle = true;
      return;
    }
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
    case Instruction::Select: {
      Value *Arms[] = {I->getOperand(1), I->getOperand(2)};
      joinInputs(Arms, I);
      return;
    }
    case Instruction::PHI: {
      SmallVector<Value *, 4> In(cast<PHINode>(I)->incoming_values());
      joinInputs(In, I);
      return;
    }
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
  case Instruction::Trunc: {
    // Only through a cast that keeps the whole address. The i128 carrier's
    // widening and narrowing do, `(uint32_t)(uintptr_t)p` does not: looking
    // through that one would report the offset of the UNtruncated address, an
    // address the program never computed. Returning nullptr is not a refusal --
    // the caller then builds the general `V - addr(p)`, which is right here too.
    unsigned IdxBits = IdxTy->getIntegerBitWidth();
    if (I->getType()->getIntegerBitWidth() < IdxBits ||
        I->getOperand(0)->getType()->getIntegerBitWidth() < IdxBits)
      return nullptr;
    return linearOffset(I->getOperand(0), P, IdxTy, B, Depth + 1);
  }
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

// True if X is exactly P's address: `ptrtoint p`, through the operations that
// keep the whole of it. The rewrite is then P ITSELF -- no offset, so no
// `cincoffset` -- which is safe even on a source that may be null or untagged,
// since it adds no operation the untagged answer did not already survive.
//
// This is musl's `call()`, `((void (*)(void))(uintptr_t)p)()`, the shape #86 had
// to override, and it is why the mask below is not an optional nicety: clang
// takes the address out of the i128 carrier with an AND, not a truncation
// (`%0 = ptrtoint ptr addrspace(200) %p to i128; %conv = and i128 %0, 2^64-1`).
// Read that as ordinary arithmetic and musl's own atexit() goes back to calling
// through `mv`, which drops the tag: cause 24 at the cjalr, measured 2026-09-28.
static bool isPlainAddressOf(const Value *X, const Value *P, unsigned IdxBits) {
  while (const auto *I = dyn_cast<Operator>(X)) {
    switch (I->getOpcode()) {
    case Instruction::PtrToInt:
      return I->getOperand(0) == P;
    case Instruction::ZExt:
    case Instruction::SExt:
    case Instruction::Trunc:
      // Only a cast that keeps every bit of the address.
      if (I->getType()->getIntegerBitWidth() < IdxBits ||
          I->getOperand(0)->getType()->getIntegerBitWidth() < IdxBits)
        return false;
      X = I->getOperand(0);
      continue;
    case Instruction::And: {
      // A mask that is all ones across the address keeps it; one that clears a
      // bit of it is an alignment, and moves the pointer.
      Value *L = I->getOperand(0), *R = I->getOperand(1);
      if (!isa<ConstantInt>(R))
        std::swap(L, R);
      const auto *C = dyn_cast<ConstantInt>(R);
      if (!C || C->getValue().countr_one() < IdxBits)
        return false;
      X = L;
      continue;
    }
    default:
      return false;
    }
  }
  return false;
}

// True if a load or store through P dominates the query point. A load or store
// through a register holding no capability traps, so if that one ran, P holds
// one. This is generic LLVM's own deduction, which it restricts to address
// space 0 because NullPointerIsDefined is true for every other -- so on this
// target it has to be made here, and without it no HEAP pointer would ever
// qualify (malloc may return null, and nothing else says otherwise).
static bool accessDominates(Value *P, const SimplifyQuery &Q,
                            const DominatorTree &DT) {
  if (!Q.CxtI)
    return false;
  const Function *F = Q.CxtI->getFunction();
  for (User *U : P->users()) {
    auto *UI = dyn_cast<Instruction>(U);
    if (!UI || UI->getFunction() != F)
      continue;
    if (getLoadStorePointerOperand(UI) == P && DT.dominates(UI, Q.CxtI))
      return true;
    // `p->field` is a GEP of P and then the access; the GEP is a cincoffset
    // on P, which would have trapped on its own if P held no capability.
    auto *GEP = dyn_cast<GetElementPtrInst>(UI);
    if (!GEP || GEP->getPointerOperand() != P)
      continue;
    for (User *GU : GEP->users())
      if (auto *GI = dyn_cast<Instruction>(GU))
        if (getLoadStorePointerOperand(GI) == GEP && DT.dominates(GI, Q.CxtI))
          return true;
  }
  return false;
}

// True if P certainly holds a capability here -- certainly not the null pointer,
// which is `{cursor 0, cap_type 0}` and makes `cincoffset` raise
// UNEXPECTED_OPERAND, and certainly not an integer spelled as a pointer, which
// holds no capability either.
//
// isKnownNonZero cannot answer this on this target, in either direction.
// Generic LLVM treats any non-zero address space as one where null may be a
// valid address, so it declines to call a global there non-null ("Other address
// spaces may have null as a valid address for a global") and restricts its
// alloca and its dereference rules to address space 0 -- while every capability
// pointer here lives in address space 200. Asked alone it would decline a
// global, an alloca and a pointer the function has already dereferenced, which
// is nearly every round trip there is. And it looks through `inttoptr`, so it
// calls `(T *)(w | 1)` non-null, and every constant integer cast to a pointer --
// `(char *)-1`, MAP_FAILED, SIG_IGN -- as well. None of those is a capability.
//
// So every value whose origin is visible here is decided by what it is made
// of: an alloca or a global (not extern-weak, not absolute) holds a capability
// by construction, the stack allocation and the linker-materialized address
// being real capabilities, tag and all; a GEP holds one if its base does; a
// select, a phi or a local slot holds one if every input does; a constant or an
// inttoptr never does. isKnownNonZero is asked only where the origin is out of
// sight -- an argument, a load from memory, a call -- for what it does decide
// there: a `nonnull` argument, a pointer past a null test, an `llvm.assume`.
// One more proof holds for any value: an access through it that dominates the
// cast (accessDominates), which is how a heap pointer qualifies at all.
//
// The local slot is the -O0 shape, the level this pass exists for: every local
// pointer lives in one and each use is a fresh load, which no query about an
// SSA value can answer. When every store to it is visible and stores a
// capability, and one of them dominates the load, what the load returns is one
// of them. The same rule for the slot and for the select is what makes -O0 and
// -O2 agree on the same source.
static bool holdsCapability(Value *P, const SimplifyQuery &Q,
                            const DominatorTree &DT,
                            SmallPtrSetImpl<const PHINode *> &OnPath,
                            unsigned Depth = 0) {
  if (Depth > 6)
    return false;
  if (isa<AllocaInst>(P))
    return true;
  if (auto *GV = dyn_cast<GlobalValue>(P))
    return !GV->isAbsoluteSymbolRef() && !GV->hasExternalWeakLinkage();
  if (auto *GEP = dyn_cast<GEPOperator>(P))
    return holdsCapability(GEP->getPointerOperand(), Q, DT, OnPath, Depth + 1);
  // Every other constant is an integer spelled as a pointer: null, `(T *)-1`,
  // an inttoptr of anything.
  if (isa<Constant>(P))
    return false;
  if (accessDominates(P, Q, DT))
    return true;
  auto *I = dyn_cast<Instruction>(P);
  if (!I)
    return isKnownNonZero(P, Q); // an argument
  auto Holds = [&](Value *V, const SimplifyQuery &VQ) {
    return holdsCapability(V, VQ, DT, OnPath, Depth + 1);
  };
  switch (I->getOpcode()) {
  case Instruction::IntToPtr:
    return false; // an integer, however non-zero
  case Instruction::Freeze:
    return Holds(I->getOperand(0), Q);
  case Instruction::Select:
    return Holds(I->getOperand(1), Q) && Holds(I->getOperand(2), Q);
  case Instruction::PHI: {
    auto *PN = cast<PHINode>(I);
    // Back at a phi already being decided: a cycle, such as a cursor stepped
    // around a loop, adds no origin of its own.
    if (!OnPath.insert(PN).second)
      return true;
    bool All = true;
    for (unsigned K = 0, E = PN->getNumIncomingValues(); All && K != E; ++K)
      All = Holds(PN->getIncomingValue(K),
                  Q.getWithInstruction(PN->getIncomingBlock(K)->getTerminator()));
    OnPath.erase(PN);
    return All;
  }
  case Instruction::Load: {
    auto *LI = cast<LoadInst>(I);
    SmallVector<StoreInst *, 4> Stores;
    if (!slotStores(LI, Stores))
      return isKnownNonZero(P, Q); // memory this function cannot see into
    bool Dominated = false;
    for (StoreInst *S : Stores) {
      if (!Holds(S->getValueOperand(), Q.getWithInstruction(S)))
        return false;
      Dominated |= DT.dominates(S, LI);
    }
    // No store at all means the load reads an uninitialized slot; one that does
    // not dominate means it may.
    return Dominated;
  }
  case Instruction::Call:
  case Instruction::Invoke: {
    auto *CB = cast<CallBase>(I);
    // A call that returns one of its arguments (`returned`, llvm.ptrmask, ...)
    // is decided by that argument, which isKnownNonZero would look through
    // unchecked.
    if (Value *Arg = getArgumentAliasingToReturnedPointer(CB, false))
      return Holds(Arg, Q);
    return isKnownNonZero(P, Q);
  }
  default:
    return false;
  }
}

// True if P may hold a capability that the rewrite's new use would consume.
//
// Reading an address with ptrtoint is not a consumer. The rewrite's GEP is
// one -- `cincoffset rd, rs1`
// with rd != rs1 nulls a LINEAR rs1 on the RTL and in QEMU -- and so is any use
// of the value itself, which is what the no-offset rewrite produces. In
//     log((char *)(((uintptr_t)cap + 15) & ~15)); store(slot, cap);
// the store would then save null. So a value is refused as a source when any
// origin of it is a capability builtin other than DELIN, which returns NONLIN,
// or inline assembly, which is how SPLIT is written. SHRINK, TIGHTEN and SCC
// preserve their operand's type; a call known to return an argument preserves
// that argument's type too. Local objects and global addresses are
// NONLIN by construction. Arguments, opaque loads and ordinary call results
// carry no linearity guarantee: start-fpga-nogp.S passes a LINEAR scratch
// argument, and revoke-on-free allocators store MREV results in global arrays.
// A null test or a dominating dereference proves neither case NONLIN.
static bool mayBeLinear(Value *P, SmallPtrSetImpl<const Value *> &Seen,
                        unsigned Depth = 0) {
  if (Depth > 8)
    return true; // too deep to decide: never rewrite
  if (!Seen.insert(P).second)
    return false; // already on the way to an answer
  if (isa<Constant>(P) || isa<AllocaInst>(P))
    return false;
  if (isa<Argument>(P))
    return true;
  auto MayBe = [&](Value *V) { return mayBeLinear(V, Seen, Depth + 1); };
  if (auto *GEP = dyn_cast<GEPOperator>(P))
    return MayBe(GEP->getPointerOperand());
  auto *I = dyn_cast<Instruction>(P);
  if (!I)
    return true;
  switch (I->getOpcode()) {
  case Instruction::IntToPtr:
    return false; // no capability at all
  case Instruction::Freeze:
  case Instruction::AddrSpaceCast:
    return MayBe(I->getOperand(0));
  case Instruction::Select:
    return MayBe(I->getOperand(1)) || MayBe(I->getOperand(2));
  case Instruction::PHI:
    return any_of(cast<PHINode>(I)->incoming_values(), MayBe);
  case Instruction::Load: {
    // At -O0 a builtin's result reaches its uses through a local slot.
    auto *LI = cast<LoadInst>(I);
    SmallVector<StoreInst *, 4> Stores;
    if (slotStores(LI, Stores))
      return any_of(Stores,
                    [&](StoreInst *S) { return MayBe(S->getValueOperand()); });
    // Without all stores visible, the loaded capability may be LINEAR,
    // regardless of whether the storage itself is local, global or indirect.
    return true;
  }
  case Instruction::Call:
  case Instruction::Invoke: {
    auto *CB = cast<CallBase>(I);
    if (CB->isInlineAsm())
      return true;
    if (auto *II = dyn_cast<IntrinsicInst>(CB)) {
      switch (II->getIntrinsicID()) {
      case Intrinsic::capstone_cap_delin:
        return false;
      case Intrinsic::capstone_cap_shrink:
      case Intrinsic::capstone_cap_tighten:
      case Intrinsic::capstone_cap_scc:
        return MayBe(II->getArgOperand(0));
      case Intrinsic::capstone_cap_init:
      case Intrinsic::capstone_cap_mrev:
      case Intrinsic::capstone_cap_seal:
      case Intrinsic::capstone_cap_drop:
      case Intrinsic::capstone_cap_revoke:
      case Intrinsic::capstone_cap_call:
      case Intrinsic::capstone_cap_enter:
      case Intrinsic::capstone_cap_ccsrrw:
        return true;
      default:
        break;
      }
    }
    if (Value *Arg = getArgumentAliasingToReturnedPointer(CB, false))
      return MayBe(Arg);
    // Neither an ordinary call nor an unmodelled intrinsic has an enforced
    // NONLIN return contract. Pointer operands alone do not prove its result.
    return true;
  }
  default:
    return true; // an origin not modelled here: do not rewrite
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
  AssumptionCache &AC = getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);

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

    const DataLayout &DL = F.getDataLayout();
    Value *X = ITP->getOperand(0);
    Type *IntTy = X->getType();
    Type *IdxTy = DL.getIndexType(ITP->getType()); // i64 on capstone64
    // Where the address is the source's own, the answer is the source: emit no
    // offset at all rather than one that folds away later, so nothing downstream
    // has to fold a cincoffset by zero off a base that may hold no capability.
    bool SameAddress = isPlainAddressOf(X, P, IdxTy->getIntegerBitWidth());
    // Moving a base that may be null is the one way this rewrite can trap where
    // the untagged answer did not (see the note on cincoffset at the top).
    SmallPtrSet<const PHINode *, 4> OnPath;
    if (!SameAddress &&
        !holdsCapability(P, SimplifyQuery(DL, &DT, &AC, ITP), DT, OnPath))
      continue;
    // Either rewrite adds a use of P, which a value that may be LINEAR cannot
    // take (see the note on linearity at the top).
    SmallPtrSet<const Value *, 16> Seen;
    if (mayBeLinear(P, Seen))
      continue;

    // InstSimplifyFolder: `0 + e` and the like fold as they are built, since
    // no optimization pass runs after this one.
    IRBuilder<InstSimplifyFolder> B(ITP->getContext(), InstSimplifyFolder(DL));
    B.SetInsertPoint(ITP);
    Value *NewP = P;
    if (!SameAddress) {
      Value *Delta = linearOffset(X, P, IdxTy, B);
      if (!Delta) {
        Value *Base = B.CreatePtrToInt(P, IntTy, P->getName() + ".addr");
        Delta = B.CreateSExtOrTrunc(B.CreateSub(X, Base, "prov.delta"), IdxTy);
      }
      NewP = B.CreateGEP(B.getInt8Ty(), P, Delta, ITP->getName() + ".prov");
    }
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
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(CapstoneRecoverProvenance, DEBUG_TYPE, PASS_NAME, false,
                    false)

FunctionPass *llvm::createCapstoneRecoverProvenancePass() {
  return new CapstoneRecoverProvenance();
}
