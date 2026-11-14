//===-- CapstoneMisalignedCapAccess.cpp - under-aligned capability accesses -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// C-69. A load or store of a capability whose 16-byte alignment the IR does
// not state reaches SelectionDAG as an under-aligned access, and legalization
// expands it through a stack slot in scalar pieces (two ld/sd at align 8,
// sixteen lbu/sb at align 1) followed by an ldc. A scalar access never carries
// a tag, so the capability loaded that way is untagged even when its address
// was 16-aligned at run time and the granule held a tagged capability.
//
// SROA produces exactly this from the C idiom that copies a pointer through
// memory of unknown alignment:
//
//   void av_freep(void *arg) { void *val; memcpy(&val, arg, sizeof(val)); ... }
//
// becomes `load ptr addrspace(200), ptr addrspace(200) %arg, align 1`, so the
// memcpy that C-81 would have sent to musl (which tests the run-time alignment
// and keeps the tag) is gone before code generation. FFmpeg's av_freep then
// frees an untagged pointer (cause 24 in __libc_free). Clang makes the same
// access directly from `*(T **)((char *)&obj + runtime_offset)`, libavfilter's
// FF_FIELD_AT (capstone/tests/compiler-repros/C69-underaligned-capability-bytecopy).
//
// A misaligned ldc/stc traps, so the access cannot simply be raised to
// align 16. This pass tests the address at run time instead:
//
//   if ((uintptr_t)p & 15) == 0   the same access at align 16: ldc/stc, and
//                                 capability members at their ABI offsets,
//                                 so the tag survives;
//   else                          the access unchanged: a misaligned granule
//                                 holds no tag, and the scalar expansion
//                                 gives the bytes, as memcpy would.
//
// That is musl memcpy's result for a 16-byte copy into or out of a 16-aligned
// object (string_bounds_safe.c: granules move as capabilities only when both
// cursors agree mod 16).
//
// An address whose alignment is provably 16 (an alloca, an aligned global, a
// masked pointer) has its access re-aligned in place, with no test.
//
// Scope: non-atomic loads and stores (volatile included: each arm performs the
// one access) whose type is or contains a pointer in a non-integral address
// space -- a scalar capability, or a struct, array or vector holding one --
// with an alignment below that address space's pointer ABI alignment.
// Atomic accesses are left alone: AtomicExpand, which runs earlier, makes every
// 16-byte atomic a library call. Functions marked optnone are not skipped.
//
// Not covered: a capability member of a PACKED aggregate. The aligned arm
// re-aligns the access's base only; a member at an offset that is not a
// multiple of 16 stays scalar in both arms, so when the base is misaligned by
// exactly the complement of that offset the member is aligned at run time and
// still loses its tag (memcpy would keep it). Natural layout puts every
// capability member at a multiple of 16, where the base test decides it. Off with
// -capstone-misaligned-cap-dispatch=false, which restores the old lowering;
// -capstone-misaligned-cap-log=<file> lists the functions it changes (or, off,
// would change).
//
//===----------------------------------------------------------------------===//

#include "Capstone.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;

#define DEBUG_TYPE "capstone-misaligned-cap"
#define PASS_NAME "Capstone run-time alignment test for under-aligned capability accesses"

STATISTIC(NumDispatched, "Under-aligned capability accesses given a run-time alignment test");
STATISTIC(NumRealigned, "Under-aligned capability accesses whose address is provably aligned");

static cl::opt<bool> EnableMisalignedCapDispatch(
    "capstone-misaligned-cap-dispatch", cl::Hidden, cl::init(true),
    cl::desc("Give a load or store of a capability whose alignment is below "
             "the capability's a run-time alignment test (C-69): a capability "
             "access when the address is aligned, the scalar expansion "
             "otherwise"));

static cl::opt<std::string> MisalignedCapLog(
    "capstone-misaligned-cap-log", cl::Hidden, cl::init(""),
    cl::desc("Append one line per function with under-aligned capability "
             "accesses (tested, re-aligned) to this file; '-' is stderr. "
             "Written with the dispatch off too, as a census"));

namespace {

class CapstoneMisalignedCapAccess : public FunctionPass {
public:
  static char ID;
  CapstoneMisalignedCapAccess() : FunctionPass(ID) {}
  bool runOnFunction(Function &F) override;
  StringRef getPassName() const override { return PASS_NAME; }
};

} // end anonymous namespace

// The strictest pointer ABI alignment of a non-integral pointer inside T, or
// none when T holds no such pointer.
static std::optional<Align> capabilityAlign(Type *T, const DataLayout &DL) {
  if (auto *PT = dyn_cast<PointerType>(T)) {
    if (!DL.isNonIntegralPointerType(PT))
      return std::nullopt;
    return DL.getPointerABIAlignment(PT->getAddressSpace());
  }
  if (auto *VT = dyn_cast<VectorType>(T))
    return capabilityAlign(VT->getElementType(), DL);
  if (auto *AT = dyn_cast<ArrayType>(T))
    return capabilityAlign(AT->getElementType(), DL);
  if (auto *ST = dyn_cast<StructType>(T)) {
    std::optional<Align> Max;
    for (Type *E : ST->elements())
      if (std::optional<Align> A = capabilityAlign(E, DL))
        Max = Max ? std::max(*Max, *A) : *A;
    return Max;
  }
  return std::nullopt;
}

// The alignment the access needs for its capabilities to be moved as
// capabilities, when it lacks it.
static std::optional<Align> neededAlign(Instruction &I, const DataLayout &DL) {
  Type *T;
  Align Have;
  if (auto *LI = dyn_cast<LoadInst>(&I)) {
    if (LI->isAtomic())
      return std::nullopt;
    T = LI->getType();
    Have = LI->getAlign();
  } else if (auto *SI = dyn_cast<StoreInst>(&I)) {
    if (SI->isAtomic())
      return std::nullopt;
    T = SI->getValueOperand()->getType();
    Have = SI->getAlign();
  } else {
    return std::nullopt;
  }
  std::optional<Align> Need = capabilityAlign(T, DL);
  if (!Need || Have >= *Need)
    return std::nullopt;
  return Need;
}

static void setAlign(Instruction &I, Align A) {
  if (auto *LI = dyn_cast<LoadInst>(&I))
    LI->setAlignment(A);
  else
    cast<StoreInst>(&I)->setAlignment(A);
}

static void dispatch(Instruction &I, Align Need, const DataLayout &DL) {
  Value *Ptr = getLoadStorePointerOperand(&I);
  LLVMContext &Ctx = I.getContext();
  Type *IntPtr = DL.getIntPtrType(Ctx, Ptr->getType()->getPointerAddressSpace());
  Type *I64 = Type::getInt64Ty(Ctx);

  IRBuilder<> B(&I);
  // The cursor, as clang lowers (uintptr_t)p: ptrtoint to the pointer's
  // width, then the low 64 bits.
  Value *Cursor = B.CreatePtrToInt(Ptr, IntPtr);
  if (IntPtr != I64)
    Cursor = B.CreateTrunc(Cursor, I64);
  Value *Low = B.CreateAnd(Cursor, ConstantInt::get(I64, Need.value() - 1));
  Value *Aligned = B.CreateICmpEQ(Low, ConstantInt::get(I64, 0));

  Instruction *ThenT, *ElseT;
  SplitBlockAndInsertIfThenElse(Aligned, &I, &ThenT, &ElseT);
  BasicBlock *Tail = I.getParent();

  // Aligned: the same access as a capability access.
  Instruction *Cap = I.clone();
  setAlign(*Cap, Need);
  Cap->insertBefore(ThenT->getIterator());
  // Otherwise: the access as it was.
  I.moveBefore(ElseT->getIterator());

  if (!I.getType()->isVoidTy()) {
    Cap->takeName(&I);
    PHINode *PN = PHINode::Create(I.getType(), 2, Cap->getName());
    PN->insertBefore(Tail->begin());
    I.replaceAllUsesWith(PN);
    PN->addIncoming(Cap, ThenT->getParent());
    PN->addIncoming(&I, ElseT->getParent());
  }
  ++NumDispatched;
}

bool CapstoneMisalignedCapAccess::runOnFunction(Function &F) {
  // No skipFunction: this is a correctness lowering, like AtomicExpand, and
  // clang marks every function optnone at -O0, which skipFunction skips.
  if (!EnableMisalignedCapDispatch && MisalignedCapLog.empty())
    return false;
  const DataLayout &DL = F.getDataLayout();
  SmallVector<std::pair<Instruction *, Align>, 8> Work;
  for (BasicBlock &BB : F)
    for (Instruction &I : BB)
      if (std::optional<Align> Need = neededAlign(I, DL))
        Work.push_back({&I, *Need});
  unsigned Tested = 0, Realigned = 0;
  for (auto [I, Need] : Work) {
    // Known alignment is read before any change, so the census is the same
    // with the dispatch on or off.
    if (getKnownAlignment(getLoadStorePointerOperand(I), DL, I) >= Need) {
      ++Realigned;
      if (EnableMisalignedCapDispatch) {
        setAlign(*I, Need);
        ++NumRealigned;
      }
      continue;
    }
    ++Tested;
    if (EnableMisalignedCapDispatch)
      dispatch(*I, Need, DL);
  }
  if (!Work.empty() && !MisalignedCapLog.empty()) {
    std::string Line;
    raw_string_ostream(Line)
        << "capstone-misaligned-cap: " << F.getParent()->getSourceFileName()
        << " " << F.getName() << " tested " << Tested << " realigned "
        << Realigned << (EnableMisalignedCapDispatch ? " on" : " off") << "\n";
    if (MisalignedCapLog == "-") {
      errs() << Line;
    } else {
      std::error_code EC;
      raw_fd_ostream OS(MisalignedCapLog, EC,
                        sys::fs::OF_Append | sys::fs::OF_Text);
      if (!EC)
        OS << Line;
      else
        report_fatal_error(Twine("capstone-misaligned-cap-log: ") + EC.message());
    }
  }
  return EnableMisalignedCapDispatch && !Work.empty();
}

char CapstoneMisalignedCapAccess::ID = 0;

INITIALIZE_PASS(CapstoneMisalignedCapAccess, DEBUG_TYPE, PASS_NAME, false, false)

FunctionPass *llvm::createCapstoneMisalignedCapAccessPass() {
  return new CapstoneMisalignedCapAccess();
}
