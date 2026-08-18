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
// redundant-with-initializer. The initializer constant is walked *recursively*,
// so a capability-pointer slot at any depth is materialized: a bare pointer
// global, a one-field struct, a flat pointer array, and — the case SQLite's
// builtin-function table needs — an array of structs / arbitrarily nested
// aggregates. Only leaves whose target is a GlobalVariable or Function are
// materialized; null elements need no tag.
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
#include "llvm/Support/CommandLine.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"

using namespace llvm;

#define DEBUG_TYPE "capstone-cap-global-init"

// DIAGNOSTIC ONLY -- emit at most the first N capability stores.
//
// This function is straight-line (one store per capability leaf, no loop) and is called
// exactly once, through the .capstone_cap_init table, so it cannot be bisected from the
// glue side the way the carve loop can with INTERP_BUILD_LIMIT. Building with a limit and
// halving it is what converts "cap-init faults somewhere" into "leaf K faults".
//
// Measured 2026-07-31: skipping cap-init entirely clears an OOB fault at the domain's
// first entry (a 16-byte capability accessed at +32), so the offending store is one of
// these leaves. Never ship a build with this set: the leaves past the limit keep their
// untagged static initialisers, and using one is a capability fault on QEMU and a silent
// pipeline stall on this silicon (issue R-5).
static cl::opt<unsigned> CapstoneCapInitLimit(
    "capstone-cap-init-limit", cl::init(0), cl::Hidden,
    cl::desc("Capstone: emit only the first N capability initializer stores "
             "(0 = all). Diagnostic bisect knob; never use in a shipping build."));

static cl::opt<bool> CapstoneCapInitPrint(
    "capstone-cap-init-print", cl::init(false), cl::Hidden,
    cl::desc("Capstone: print each capability initializer leaf (index, holder, path, "
             "target) as it is emitted. Pairs with -capstone-cap-init-limit."));
#define PASS_NAME "Capstone capability-global initialization"

static constexpr unsigned CapAS = 200;
// SelectionDAG is formed one basic block at a time. Keep the synthetic straight-line
// initializer in modest blocks so a module with hundreds of pointer leaves does not
// make all of their holder/target capabilities live in one enormous DAG. Capstone uses
// the ordinary GPR register class for both scalars and capabilities, so a register-
// allocator spill is an `sd` and drops the capability tag. The corresponding reload is
// selected as `ldc`, yielding an untagged base at the next initializer store. Splitting
// the IR block bounds that pressure without adding calls or changing store order.
static constexpr unsigned CapInitStoresPerBlock = 32;

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
  // Peel pointer casts and constant-offset GEPs to the underlying target. A
  // pointer global initialized to an *interior* address of another global --
  //   const unsigned char *p = &otherGlobal[N];
  // (SQLite's sqlite3aLTb/aEQb/aGTb = &sqlite3UpperToLower[256(+6/+12)-OP_Ne]) --
  // is a ConstantExpr GEP with non-zero indices. stripPointerCasts() only strips
  // zero-index GEPs, so such a global was never recognized as referencing a
  // GlobalVariable and its slot was left untagged, faulting on first use. We peel
  // the GEP/cast chain here regardless of the `inbounds` flag: SQLite deliberately
  // uses an index that is only in-bounds thanks to appended array elements, so
  // clang does not mark the GEP inbounds and stripInBoundsConstantOffsets() would
  // miss it. A global-initializer GEP always has constant indices, so peeling
  // operand 0 is safe. The full interior pointer (with offset) is still what we
  // store, so the tagged capability lands at the correct cursor.
  const Constant *C = FieldInit;
  while (auto *CE = dyn_cast<ConstantExpr>(C)) {
    unsigned Op = CE->getOpcode();
    if (Op != Instruction::GetElementPtr && Op != Instruction::BitCast &&
        Op != Instruction::AddrSpaceCast)
      break;
    C = CE->getOperand(0);
  }
  return isa<GlobalVariable>(C) || isa<Function>(C);
}

namespace {
// A capability-pointer leaf inside a (possibly nested) global initializer that
// must be materialized at runtime, identified by the GEP index path from the
// holder global down to the slot.
struct StoreItem {
  GlobalVariable *Holder;
  Type *AggTy;                   // holder's value type, for the GEP base
  SmallVector<uint64_t, 4> Path; // element/field indices from holder to slot
  Constant *Value;               // the capability to store (a Global/Function ref)
};
} // end anonymous namespace

// Recursively walk a constant initializer, recording every capability-pointer
// leaf that references a global/function. The original two flat shapes (a
// one-field struct and a flat pointer array) are just shallow cases; arrays of
// structs and other nested aggregates are now covered.
static void collectCapInits(GlobalVariable *Holder, Type *AggTy, Constant *C,
                            SmallVectorImpl<uint64_t> &Path,
                            SmallVectorImpl<StoreItem> &Items) {
  if (!C || isa<ConstantAggregateZero>(C) || isa<UndefValue>(C))
    return;
  if (auto *CS = dyn_cast<ConstantStruct>(C)) {
    for (unsigned I = 0, E = CS->getNumOperands(); I != E; ++I) {
      Path.push_back(I);
      collectCapInits(Holder, AggTy, CS->getOperand(I), Path, Items);
      Path.pop_back();
    }
    return;
  }
  if (auto *CA = dyn_cast<ConstantArray>(C)) {
    for (unsigned I = 0, E = CA->getNumOperands(); I != E; ++I) {
      Path.push_back(I);
      collectCapInits(Holder, AggTy, CA->getOperand(I), Path, Items);
      Path.pop_back();
    }
    return;
  }
  // Leaf: only a capability-pointer slot referencing a global/function gets a tag.
  if (isCapPtr(C->getType()) && needsMaterialization(C))
    Items.push_back(
        {Holder, AggTy, SmallVector<uint64_t, 4>(Path.begin(), Path.end()), C});
}

bool CapstoneCapGlobalInit::runOnModule(Module &M) {
  LLVMContext &Ctx = M.getContext();
  Type *I64 = Type::getInt64Ty(Ctx);

  // Collect work items first, so we only create the init function when there is
  // something to materialize.
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
    SmallVector<uint64_t, 4> Path;
    collectCapInits(&GV, GV.getValueType(), GV.getInitializer(), Path, Items);
  }

  if (Items.empty())
    return false;

  // Synthesize `void __capstone_cap_init(void)` in the program address space
  // (matching ordinary Capstone functions) and emit the stores. It is given
  // *internal* linkage rather than exported under a fixed name: a fixed external
  // name would collide at link time when more than one object in a multi-module
  // program has capability globals (e.g. CoreMark). The AsmPrinter registers
  // this function by emitting a PC-relative `.capstone_cap_init` table entry.
  unsigned ProgAS = M.getDataLayout().getProgramAddressSpace();
  FunctionType *FTy = FunctionType::get(Type::getVoidTy(Ctx), /*isVarArg=*/false);
  Function *InitFn = Function::Create(FTy, GlobalValue::InternalLinkage, ProgAS,
                                      "__capstone_cap_init", &M);

  BasicBlock *BB = BasicBlock::Create(Ctx, "entry", InitFn);
  IRBuilder<> B(BB);

  Type *I32 = Type::getInt32Ty(Ctx);
  unsigned Emitted = 0;
  for (const StoreItem &It : Items) {
    if (CapstoneCapInitLimit && Emitted >= CapstoneCapInitLimit)
      break;
    if (Emitted != 0 && Emitted % CapInitStoresPerBlock == 0) {
      BasicBlock *Next = BasicBlock::Create(
          Ctx, "cap.init." + Twine(Emitted / CapInitStoresPerBlock), InitFn);
      B.CreateBr(Next);
      B.SetInsertPoint(Next);
    }
    // Companion to the limit: once a bisect names an index, this says WHICH global that
    // index is, without having to re-derive it from a disassembly or an IR dump.
    if (CapstoneCapInitPrint) {
      errs() << "capstone-cap-init: leaf " << Emitted << " holder="
             << It.Holder->getName() << " path=";
      for (uint64_t P : It.Path)
        errs() << "[" << P << "]";
      errs() << " value=" << It.Value->getName()
             << " holder_size="
             << M.getDataLayout().getTypeAllocSize(It.AggTy) << "\n";
    }
    ++Emitted;
    // GEP index path: a leading 0 to step through the holder pointer, then the
    // recorded element/field indices down to the capability slot. Struct member
    // indices must be i32; array indices are i64 -- pick per level by walking
    // the holder type.
    SmallVector<Value *, 5> Idx;
    Idx.push_back(ConstantInt::get(I64, 0));
    Type *CurTy = It.AggTy;
    for (uint64_t P : It.Path) {
      if (auto *ST = dyn_cast<StructType>(CurTy)) {
        Idx.push_back(ConstantInt::get(I32, P));
        CurTy = ST->getElementType(P);
      } else if (auto *AT = dyn_cast<ArrayType>(CurTy)) {
        Idx.push_back(ConstantInt::get(I64, P));
        CurTy = AT->getElementType();
      } else {
        Idx.push_back(ConstantInt::get(I64, P));
      }
    }
    Value *Slot = B.CreateInBoundsGEP(It.AggTy, It.Holder, Idx);
    // Volatile so the store is never elided as redundant with the (untagged)
    // static initializer; it must run to set the tag in place.
    B.CreateStore(It.Value, Slot, /*isVolatile=*/true);
  }
  B.CreateRetVoid();

  // The init function is internal (no cross-module name collision). The
  // AsmPrinter emits a PC-relative entry for it into the `.capstone_cap_init`
  // table (one per module with capability globals), which start.S runs before
  // domain_main. A standard .init_array of absolute function addresses cannot be
  // used: the domain is loaded at a runtime base and processes no load-time
  // relocations, so absolute (link-time) addresses are stale; the table stores
  // `initfn - entry` offsets, which are position-independent.
  return true;
}

INITIALIZE_PASS(CapstoneCapGlobalInit, DEBUG_TYPE, PASS_NAME, false, false)

char CapstoneCapGlobalInit::ID = 0;

ModulePass *llvm::createCapstoneCapGlobalInitPass() {
  return new CapstoneCapGlobalInit();
}
