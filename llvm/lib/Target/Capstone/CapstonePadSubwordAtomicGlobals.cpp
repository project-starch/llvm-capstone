//===-- CapstonePadSubwordAtomicGlobals.cpp - Let a sub-word atomic's word fit ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// C-74. Without byte or halfword atomics (Zabha), AtomicExpand turns an i8 or
// i16 cmpxchg or atomicrmw into a loop on the aligned 32-bit word that holds
// it. A global's capability is bounded to the global, so when the global's size
// is not a multiple of 4, or it is not 4-aligned, the word can reach past the
// bound and the lr.w faults (cause 28 on QEMU): a lone `static uint8_t`
// locked with __atomic_compare_exchange_n, or CPython's standalone static
// PyMutex.
//
// This pass gives every defined global that is the target of a visible sub-word
// cmpxchg or atomicrmw (the pointer operand, through constant offsets and casts)
// a size that is a multiple of 4 and an alignment of at least 4, so the aligned
// word of any of its bytes lies inside its bounds. The padding is at the end, in
// a wrapper struct { original type, [pad x i8] }; every use is redirected to the
// original type's start, so nothing reads the padding. Globals reached only
// through a pointer the module cannot see are not covered (an escaped global
// defined elsewhere); full LTO makes the test program-wide. A heap object is
// bounded by its allocation and is outside this pass: see ISSUES C-74.
//
// Only globals with such an atomic change, so other programs compile exactly as
// before. Option C in plans/2026-10-16-c74-subword-atomic-bounds.md, native
// byte and halfword atomics, would make this pass unnecessary.
//
//===----------------------------------------------------------------------===//

#include "Capstone.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"

using namespace llvm;

#define DEBUG_TYPE "capstone-pad-subword-atomic-globals"
#define PASS_NAME "Capstone pad globals targeted by sub-word atomics"

STATISTIC(NumPadded, "Globals padded or aligned for a sub-word atomic");

namespace {
class CapstonePadSubwordAtomicGlobals : public ModulePass {
public:
  static char ID;
  CapstonePadSubwordAtomicGlobals() : ModulePass(ID) {}
  bool runOnModule(Module &M) override;
  StringRef getPassName() const override { return PASS_NAME; }
};
} // end anonymous namespace

char CapstonePadSubwordAtomicGlobals::ID = 0;
INITIALIZE_PASS(CapstonePadSubwordAtomicGlobals, DEBUG_TYPE, PASS_NAME, false,
                false)

ModulePass *llvm::createCapstonePadSubwordAtomicGlobalsPass() {
  return new CapstonePadSubwordAtomicGlobals();
}

// The defined global a pointer is based on, through constant offsets and casts.
static GlobalVariable *baseGlobal(Value *Ptr) {
  Value *V = Ptr->stripInBoundsConstantOffsets();
  V = V->stripPointerCastsAndAliases();
  auto *GV = dyn_cast<GlobalVariable>(V);
  if (!GV || GV->isDeclaration() || GV->hasSection() ||
      GV->isThreadLocal())
    return nullptr;
  return GV;
}

bool CapstonePadSubwordAtomicGlobals::runOnModule(Module &M) {
  const DataLayout &DL = M.getDataLayout();
  SmallPtrSet<GlobalVariable *, 8> Targets;
  for (Function &F : M)
    for (Instruction &I : instructions(F)) {
      Value *Ptr = nullptr;
      Type *Ty = nullptr;
      if (auto *CX = dyn_cast<AtomicCmpXchgInst>(&I)) {
        Ptr = CX->getPointerOperand();
        Ty = CX->getCompareOperand()->getType();
      } else if (auto *RMW = dyn_cast<AtomicRMWInst>(&I)) {
        Ptr = RMW->getPointerOperand();
        Ty = RMW->getValOperand()->getType();
      }
      if (!Ptr || !Ty->isSized() || DL.getTypeStoreSize(Ty) >= 4)
        continue;
      if (GlobalVariable *GV = baseGlobal(Ptr))
        Targets.insert(GV);
    }

  bool Changed = false;
  for (GlobalVariable *GV : Targets) {
    Type *Ty = GV->getValueType();
    uint64_t Size = DL.getTypeAllocSize(Ty);
    Align A = GV->getAlign().value_or(DL.getPreferredAlign(GV));
    uint64_t Padded = alignTo(Size, 4);
    if (Padded == Size && A >= Align(4))
      continue;
    if (Padded == Size) {
      GV->setAlignment(std::max(A, Align(4)));
      ++NumPadded;
      Changed = true;
      continue;
    }
    LLVMContext &Ctx = M.getContext();
    Type *PadTy = ArrayType::get(Type::getInt8Ty(Ctx), Padded - Size);
    StructType *WrapTy = StructType::get(Ctx, {Ty, PadTy});
    Constant *Init = GV->hasInitializer()
                         ? ConstantStruct::get(WrapTy, {GV->getInitializer(),
                                                        Constant::getNullValue(PadTy)})
                         : nullptr;
    auto *NewGV = new GlobalVariable(
        M, WrapTy, GV->isConstant(), GV->getLinkage(), Init, "", GV,
        GV->getThreadLocalMode(), GV->getAddressSpace(),
        GV->isExternallyInitialized());
    NewGV->copyAttributesFrom(GV);
    NewGV->copyMetadata(GV, 0);
    NewGV->setAlignment(std::max(A, Align(4)));
    NewGV->takeName(GV);
    // The original object starts at offset 0 of the wrapper.
    Constant *Start = ConstantExpr::getPointerCast(NewGV, GV->getType());
    GV->replaceAllUsesWith(Start);
    GV->eraseFromParent();
    ++NumPadded;
    Changed = true;
  }
  return Changed;
}
