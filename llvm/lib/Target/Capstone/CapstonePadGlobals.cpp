//===-- CapstonePadGlobals.cpp - Representable bounds for globals ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// -capstone-pad-globals (default off): every data global this module DEFINES is
// aligned and tail-padded so that its bounds [&G, &G + length(sizeof G)) are
// exactly representable in the compact bounds encoding
// (MCTargetDesc/CapstoneRepresentable.h). Instruction selection narrows a
// global to the same length under the same flag (CapstoneISelDAGToDAG.cpp, the
// SHRINK in selectLGA), so a module that only declares G forms the same bounds
// as the one that defines it.
//
// Why: the global's capability is SHRINK(scc(gp, &G), sizeof G), and on vp2 an
// inexact SHRINK result is rounded OUTWARD (base down, end up) through the
// register file's compressed form, covering bytes of the neighbouring objects:
// an authority gain, and a difference from QEMU, which stores the requested
// bounds (isa.md, "SHRINK's result on the compact profile", rule (2)).
//
// Only objects of 4096 bytes or more are touched: below that every length is
// exact at any base. Above it the tail pad is less than one granule,
// 2^(msb(size) - 9) bytes, so under 0.4% of the object.
//
// A MIXED build is UNSAFE, not merely inexact: a module built with the flag
// narrows a global it only declares to the padded length, and if the defining
// module was built without it, that capability covers up to a granule of the
// next object. The flag must be uniform across every module of an image, the
// runtime, libc and builtins included (runtime/cmake/VirtualCodegen.cmake's
// CAPSTONE_VIRTUAL_PAD_GLOBALS records it in sdk.json).
//
// Skipped, deliberately: declarations, thread-locals, unsized types, llvm.*
// globals, appending linkage, and globals in an explicit section (a linker-set
// array is read as a contiguous run of its members, so padding one would break
// the walk). Instruction selection skips a global with a section too; a
// module that declares one WITHOUT its section narrows it to the padded length,
// at most one granule past the object (within vp2's own outward rounding of
// it). Large mergeable constants ARE padded: the pad turns them into a struct,
// so they leave the mergeable-string section, which costs only merging.
//
// STACK OBJECTS (the frame rule, under the same flag): a static alloca of 4096
// bytes or more becomes [length(size) x i8] at the representable alignment (lifetime
// markers name only the pointer here). The frame object's SHRINK
// (CapstoneISelDAGToDAG.cpp, narrowToFrameObjectBounds) reads the frame object's
// size, so it then narrows to exactly representable bounds; an alignment above
// the stack's 16 bytes realigns the frame (andi + scc sp), as any over-aligned
// alloca does. Not covered: dynamic allocas (a run-time size has no static pad);
// spill slots, which hold no object a capability names. The rule is local to the
// function, so unlike globals a mixed build cannot widen a stack object.
//
//===----------------------------------------------------------------------===//

#include "Capstone.h"
#include "MCTargetDesc/CapstoneRepresentable.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

// Not "capstone-pad-globals": opt registers every legacy pass as an option
// named by its argument, which would collide with the flag below.
#define DEBUG_TYPE "capstone-pad-globals-pass"
#define PASS_NAME "Capstone representable global padding"

cl::opt<bool> llvm::CapstonePadGlobals(
    "capstone-pad-globals", cl::init(false),
    cl::desc("Capstone: align and tail-pad every defined global of 4096 bytes "
             "or more to exactly representable bounds, and narrow globals to "
             "that length. Must be uniform across an image (see "
             "CapstonePadGlobals.cpp)."));

namespace {

class CapstonePadGlobalsPass : public ModulePass {
public:
  static char ID;

  CapstonePadGlobalsPass() : ModulePass(ID) {}

  bool runOnModule(Module &M) override;

  StringRef getPassName() const override { return PASS_NAME; }
};

} // end anonymous namespace

static bool eligible(const GlobalVariable &GV) {
  return !GV.isDeclaration() && !GV.isThreadLocal() && !GV.hasSection() &&
         !GV.hasAppendingLinkage() && !GV.getName().starts_with("llvm.") &&
         GV.getValueType()->isSized();
}

bool CapstonePadGlobalsPass::runOnModule(Module &M) {
  if (!CapstonePadGlobals)
    return false;
  const DataLayout &DL = M.getDataLayout();
  SmallVector<GlobalVariable *, 16> Work;
  for (GlobalVariable &GV : M.globals())
    if (eligible(GV) && DL.getTypeAllocSize(GV.getValueType()) >= 4096)
      Work.push_back(&GV);

  bool Changed = false;
  for (GlobalVariable *GV : Work) {
    uint64_t Size = DL.getTypeAllocSize(GV->getValueType());
    uint64_t Length = CapstoneRepresentable::length(Size);
    Align Need(CapstoneRepresentable::alignment(Size));
    Align Have = GV->getAlign().value_or(DL.getPreferredAlign(GV));
    if (Length == Size) {
      if (Have < Need) {
        GV->setAlignment(Need);
        Changed = true;
      }
      continue;
    }
    LLVMContext &Ctx = M.getContext();
    ArrayType *Pad = ArrayType::get(Type::getInt8Ty(Ctx), Length - Size);
    StructType *Padded = StructType::get(Ctx, {GV->getValueType(), Pad});
    assert(DL.getTypeAllocSize(Padded) == Length &&
           "pad did not reach the representable length");
    Constant *Init = nullptr;
    if (GV->hasInitializer()) {
      Constant *Old = GV->getInitializer();
      Init = Old->isNullValue()
                 ? Constant::getNullValue(Padded)
                 : ConstantStruct::get(Padded,
                                       {Old, Constant::getNullValue(Pad)});
    }
    auto *New = new GlobalVariable(
        M, Padded, GV->isConstant(), GV->getLinkage(), Init, "", GV,
        GV->getThreadLocalMode(), GV->getAddressSpace(),
        GV->isExternallyInitialized());
    New->copyAttributesFrom(GV);
    New->setAlignment(std::max(Have, Need));
    New->setComdat(GV->getComdat());
    New->copyMetadata(GV, 0);
    New->takeName(GV);
    GV->replaceAllUsesWith(New);
    GV->eraseFromParent();
    LLVM_DEBUG(dbgs() << "capstone-pad-globals: " << New->getName() << " "
                      << Size << " -> " << Length << " align "
                      << std::max(Have, Need).value() << "\n");
    Changed = true;
  }

  // The frame rule: static allocas of 4096 bytes or more.
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    SmallVector<AllocaInst *, 8> Allocas;
    for (Instruction &I : F.getEntryBlock())
      if (auto *AI = dyn_cast<AllocaInst>(&I))
        if (AI->isStaticAlloca())
          if (std::optional<TypeSize> TS = AI->getAllocationSize(DL))
            if (!TS->isScalable() && TS->getFixedValue() >= 4096)
              Allocas.push_back(AI);
    for (AllocaInst *AI : Allocas) {
      uint64_t Size = AI->getAllocationSize(DL)->getFixedValue();
      uint64_t Length = CapstoneRepresentable::length(Size);
      Align Need(CapstoneRepresentable::alignment(Size));
      Align Have = AI->getAlign();
      if (Length == Size) {
        if (Have < Need) {
          AI->setAlignment(Need);
          Changed = true;
        }
        continue;
      }
      auto *New = new AllocaInst(
          ArrayType::get(Type::getInt8Ty(M.getContext()), Length),
          AI->getAddressSpace(), nullptr, std::max(Have, Need), "", AI->getIterator());
      New->takeName(AI);
      New->setDebugLoc(AI->getDebugLoc());
      AI->replaceAllUsesWith(New);
      AI->eraseFromParent();
      LLVM_DEBUG(dbgs() << "capstone-pad-globals: alloca " << New->getName() << " in "
                        << F.getName() << " " << Size << " -> " << Length << " align "
                        << std::max(Have, Need).value() << "\n");
      Changed = true;
    }
  }
  return Changed;
}

INITIALIZE_PASS(CapstonePadGlobalsPass, DEBUG_TYPE, PASS_NAME, false, false)

char CapstonePadGlobalsPass::ID = 0;

ModulePass *llvm::createCapstonePadGlobalsPass() {
  return new CapstonePadGlobalsPass();
}
