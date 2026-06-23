//===----- CapstoneCapGlobalInit.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Capstone capability-global initialization (constructor-codegen).
//
// On Capstone a capability tag is out-of-band metadata that cannot be encoded in
// a static ELF image, so a file-scope global that holds a capability (a pointer
// global, a string table like `char *nums[]`, a function-pointer table) loads
// with the address bits but *no tag*; the first dereference faults. There is no
// capability relocation / runtime fixup in the toolchain.
//
// This pass resolves that for the common cases by synthesizing a per-module
// initializer function `__capstone_cap_init` that stores each capability-global
// element at runtime with an ordinary store. Normal instruction selection lowers
// each store to a properly tagged capability store (`stc`), materializing the
// target as a bounded capability derived from the global data root -- exactly the
// pattern validated in tests/runtime-qemu/static-cap-typed-load-repro/. The
// domain runtime (`start.S`) calls `__capstone_cap_init` before `domain_main`.
//
// The static-image initializer is left intact (its untagged bytes are overwritten
// before first use); the stores are marked volatile so they are never elided as
// redundant-with-initializer. Two holder shapes are handled, matching the GCT
// metadata analysis in CapstoneAsmPrinter.cpp: a one-field struct wrapping a
// single addrspace(200) pointer, and an array of addrspace(200) pointers. Only
// elements whose target is a GlobalVariable or Function are materialized; null
// elements need no tag.
//
// Design note + rationale (constructor-codegen vs a GCT runtime consumer):
// capstone/agent-handoff/design/capability-globals-init-decision.md.
//
//===----------------------------------------------------------------------===//

#include "Capstone.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"

using namespace llvm;

#define DEBUG_TYPE "capstone-cap-global-init"
#define PASS_NAME "Capstone capability-global initialization"

static constexpr unsigned CapAS = 200;

namespace {

class CapstoneCapGlobalInit : public ModulePass {
public:
  static char ID;

  CapstoneCapGlobalInit() : ModulePass(ID) {}

  bool runOnModule(Module &M) override;

  StringRef getPassName() const override { return PASS_NAME; }
};

} // end anonymous namespace

// Returns true if Ty is a capability pointer (addrspace 200).
static bool isCapPtr(Type *Ty) {
  return Ty->isPointerTy() && Ty->getPointerAddressSpace() == CapAS;
}

// A capability-global element that must be materialized at runtime: a target
// (GlobalVariable/Function) that needs a tag, stored back into its holder slot.
static bool needsMaterialization(Constant *FieldInit) {
  if (!FieldInit || isa<ConstantPointerNull>(FieldInit))
    return false;
  const Value *Stripped = FieldInit->stripPointerCasts();
  return isa<GlobalVariable>(Stripped) || isa<Function>(Stripped);
}

bool CapstoneCapGlobalInit::runOnModule(Module &M) {
  LLVMContext &Ctx = M.getContext();
  Type *I64 = Type::getInt64Ty(Ctx);

  // Collect (holder, value-type, GEP-index, element-constant) work items first,
  // so we only create the init function when there is something to do.
  struct StoreItem {
    GlobalVariable *Holder;
    Type *AggTy;     // the holder's value type (array/struct) for the GEP
    uint64_t Index;  // element/field index within the holder
    Constant *Value; // the capability to store (a Global/Function reference)
  };
  SmallVector<StoreItem, 16> Items;

  for (GlobalVariable &GV : M.globals()) {
    if (!GV.hasInitializer())
      continue;
    // Never touch LLVM intrinsic/metadata globals (llvm.used, llvm.compiler.used,
    // llvm.global_ctors, ...): they have appending linkage / the llvm.metadata
    // section and are not runtime data to materialize.
    if (GV.hasAppendingLinkage() || GV.isThreadLocal() ||
        GV.getName().starts_with("llvm.") ||
        GV.getSection() == "llvm.metadata")
      continue;
    Type *Ty = GV.getValueType();
    Constant *Init = GV.getInitializer();

    if (auto *ST = dyn_cast<StructType>(Ty)) {
      // One-field struct wrapping a single capability pointer.
      if (ST->getNumElements() != 1 || !isCapPtr(ST->getElementType(0)))
        continue;
      auto *CS = dyn_cast<ConstantStruct>(Init);
      if (!CS || CS->getNumOperands() != 1)
        continue;
      Constant *F = CS->getOperand(0);
      if (needsMaterialization(F))
        Items.push_back({&GV, Ty, 0, F});
    } else if (auto *AT = dyn_cast<ArrayType>(Ty)) {
      // Array of capability pointers.
      if (!isCapPtr(AT->getElementType()))
        continue;
      auto *CA = dyn_cast<ConstantArray>(Init);
      if (!CA)
        continue;
      for (unsigned I = 0, E = CA->getNumOperands(); I != E; ++I) {
        Constant *F = CA->getOperand(I);
        if (needsMaterialization(F))
          Items.push_back({&GV, Ty, I, F});
      }
    }
  }

  if (Items.empty())
    return false;

  // Synthesize (or reuse) `void __capstone_cap_init(void)` in the program
  // address space, matching ordinary Capstone functions, and emit the stores.
  unsigned ProgAS = M.getDataLayout().getProgramAddressSpace();
  FunctionType *FTy = FunctionType::get(Type::getVoidTy(Ctx), /*isVarArg=*/false);
  Function *InitFn = Function::Create(FTy, GlobalValue::ExternalLinkage, ProgAS,
                                      "__capstone_cap_init", &M);
  InitFn->setVisibility(GlobalValue::DefaultVisibility);

  BasicBlock *BB = BasicBlock::Create(Ctx, "entry", InitFn);
  IRBuilder<> B(BB);

  for (const StoreItem &It : Items) {
    Value *Idx[] = {ConstantInt::get(I64, 0), ConstantInt::get(I64, It.Index)};
    Value *Slot = B.CreateInBoundsGEP(It.AggTy, It.Holder, Idx);
    // Volatile so the store is never elided as redundant with the (untagged)
    // static initializer; it must run to set the tag in place.
    B.CreateStore(It.Value, Slot, /*isVolatile=*/true);
  }
  B.CreateRetVoid();

  return true;
}

INITIALIZE_PASS(CapstoneCapGlobalInit, DEBUG_TYPE, PASS_NAME, false, false)

char CapstoneCapGlobalInit::ID = 0;

ModulePass *llvm::createCapstoneCapGlobalInitPass() {
  return new CapstoneCapGlobalInit();
}
