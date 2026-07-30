//===----- CapstoneMergeStrConstants.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Merge private read-only string constants into a single container global, so
// they consume ONE cap-table slot instead of hundreds.
//
// WHY THIS EXISTS -- it is a hardware limit, measured, not a size optimization.
// Under the gp-captable ABI every global gets its own capability, and the entry
// glue carves each one with a `split`. Every `split` allocates a revocation node.
// The prototype board's allocator head is 10 bits initialised to 3
// (capstone_rev_node.anvil:160,168), so only 1021 allocations exist. SQLite needs
// 1059 carves and overflowed the pool on silicon 2026-07-31: head read back 74
// with the overflow flag set, i.e. 1095 allocations, wrapped by 74. Reuse of live
// node ids can splice the revocation list into a cycle, and since every `stc`
// blocks on a rev-node query with no timeout, the core then stalls forever with
// no trap. That is invisible to software: the overflow flag reaches only a debug
// LED, and there is no CSR, interrupt or watchdog for it anywhere in the design.
//
// 885 of SQLite's 1059 globals are anonymous string literals -- 84% of the carves
// for objects that are read-only and whose addresses are interchangeable. Merging
// just those takes the domain from 1095 allocations to ~222.
//
// SCOPE IS DELIBERATELY NARROW, and this is an ABI decision, not a tuning knob.
// Only private/internal, constant, unnamed_addr byte-array globals are merged --
// i.e. string literals. Named globals keep their own capability, which preserves
// per-object bounds on exactly the objects an overread is interesting on
// (sqlite3UpperToLower, zKWText, sqlite3CtypeMap, ...). The cost accepted here is
// that the merged literals lose mutual bounds against each other. Note that cost
// is read AND write, not merely overread: the glue's carve is a bare `split` with
// no permission tightening, so a .rodata capability is read-write today.
// Rationale, alternatives considered and the decision:
// capstone/agent-handoff/design/carve-count-vs-revnode-pool-decision.md
//
// WHY NOT THE GENERIC GlobalMerge PASS. It merges the writable and .bss buckets
// unconditionally (only the constant bucket is gated), so it cannot express
// "constants only", let alone "string literals only" -- and merging writable
// globals is precisely what this ABI must not do silently. It also groups by use
// within a function, which fragments the set and defeats the purpose here.
//
// ORDERING. This must run BEFORE CapstoneCapGlobalInit. That pass walks global
// initializers for capability leaves; after merging, a pointer-to-literal
// initializer is a ConstantExpr GEP into the container, which its
// needsMaterialization() already peels to the underlying GlobalVariable. Running
// it the other way round would leave cap-init holding references to globals this
// pass then erases.
//
//===----------------------------------------------------------------------===//

#include "Capstone.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "capstone-merge-str-constants"
#define PASS_NAME "Capstone merge private string constants"

// OFF until the lowering gap below is closed. The pass itself works -- it takes SQLite
// from 1059 carves to 174, which is the whole point -- but references to the container
// currently lower through the INTEGER auipc/PseudoLLA fallback instead of the cap-table
// path, so __capstone_cap_init stores untagged addresses into pointer globals and the
// domain jumps to 0 on its first entry (cause = 1, pc = 0x0). lowerGlobalAddress DOES
// handle a global-plus-offset correctly (CapstoneISelLowering.cpp:9958-9981: ldc the slot,
// then CIncOffset by the interior offset), so the reference is not arriving there as a
// GlobalAddress with an offset -- most likely the container's address space, or the
// ConstantExpr GEP not folding into a GlobalAddress SDNode. Fix that, then flip this on.
static cl::opt<bool> CapstoneMergeStrings(
    "capstone-merge-string-constants", cl::init(false), cl::Hidden,
    cl::desc("Capstone: merge private read-only string constants into one "
             "cap-table slot (gp-captable ABI only)"));

// Cap on how many bytes land in one container. Not a correctness limit -- the
// access path is `ldc` of the slot then `cincoffset` by the interior offset, so
// any offset is reachable -- but a bound keeps the blast radius of the lost
// mutual bounds finite, and keeps each container's length comfortably inside the
// representable-bounds window the carve needs.
// 4096 is a REPRESENTABILITY limit, not a tuning choice. Capability bounds are compressed,
// and the encoder's granule for a length L is 1 << (max(0, floor(log2 L) - 12) + 3): below
// 4096 bytes that is 8, so the glue's 16-byte carve alignment always yields exact bounds.
// A single 21,211-byte container needs 32-byte granularity, which the carve does not
// provide -- the resulting capability is not exactly representable and the domain died
// before it ever reached domain_main. Several small containers save just as many carves as
// one big one (885 literals -> ~6 slots either way).
static cl::opt<unsigned> CapstoneMergeStrMaxBytes(
    "capstone-merge-string-max-bytes", cl::init(4096), cl::Hidden,
    cl::desc("Capstone: maximum bytes per merged string container"));

extern cl::opt<bool> CapstoneGpCaptable;

namespace {
class CapstoneMergeStrConstants : public ModulePass {
public:
  static char ID;
  CapstoneMergeStrConstants() : ModulePass(ID) {}
  bool runOnModule(Module &M) override;
  StringRef getPassName() const override { return PASS_NAME; }
};
} // end anonymous namespace

// A string literal for our purposes: private/internal linkage (so no other module
// can refer to it by name), constant, unnamed_addr (so its exact address is
// explicitly not semantically significant -- this is what makes merging legal),
// an array of bytes, with a plain initializer and no special placement.
static bool isMergeableStringConstant(const GlobalVariable &GV) {
  if (!GV.hasInitializer() || !GV.isConstant())
    return false;
  if (!GV.hasPrivateLinkage() && !GV.hasInternalLinkage())
    return false;
  // unnamed_addr is the licence to merge: it states that the address is not
  // significant, only the contents. Without it, code could legitimately compare
  // two literals' addresses and observe the merge.
  if (!GV.hasGlobalUnnamedAddr())
    return false;
  if (GV.hasSection() || GV.isThreadLocal() || GV.hasComdat())
    return false;
  auto *AT = dyn_cast<ArrayType>(GV.getValueType());
  if (!AT || !AT->getElementType()->isIntegerTy(8))
    return false;
  // Keep the container in one address space.
  return GV.getAddressSpace() == 0 || GV.getAddressSpace() == 200;
}

bool CapstoneMergeStrConstants::runOnModule(Module &M) {
  if (!CapstoneMergeStrings || !CapstoneGpCaptable)
    return false;

  const DataLayout &DL = M.getDataLayout();

  SmallVector<GlobalVariable *, 64> Cands;
  unsigned AS = 0;
  bool ASSet = false;
  for (GlobalVariable &GV : M.globals()) {
    if (!isMergeableStringConstant(GV))
      continue;
    // All members must share an address space, since the container has one.
    if (!ASSet) {
      AS = GV.getAddressSpace();
      ASSet = true;
    } else if (GV.getAddressSpace() != AS) {
      continue;
    }
    Cands.push_back(&GV);
  }
  // Two is the minimum that saves anything; one would just rename it.
  if (Cands.size() < 2)
    return false;

  bool Changed = false;
  size_t Idx = 0;
  unsigned ContainerNo = 0;
  while (Idx < Cands.size()) {
    // Lay out one container: concatenate members, honouring each member's
    // alignment by padding, and give the container the maximum alignment of its
    // members so every interior offset keeps the alignment it had standalone.
    SmallVector<Constant *, 64> Fields;
    SmallVector<GlobalVariable *, 64> Members;
    SmallVector<uint64_t, 64> Offsets;
    uint64_t Size = 0;
    Align MaxAlign(1);

    while (Idx < Cands.size()) {
      GlobalVariable *GV = Cands[Idx];
      uint64_t GVSize = DL.getTypeAllocSize(GV->getValueType());
      Align GVAlign = GV->getAlign().value_or(
          DL.getPrefTypeAlign(GV->getValueType()));
      uint64_t Pad = offsetToAlignment(Size, GVAlign);
      if (!Members.empty() && Size + Pad + GVSize > CapstoneMergeStrMaxBytes)
        break;
      if (Pad) {
        Fields.push_back(ConstantAggregateZero::get(
            ArrayType::get(Type::getInt8Ty(M.getContext()), Pad)));
        Size += Pad;
      }
      Offsets.push_back(Size);
      Fields.push_back(GV->getInitializer());
      Members.push_back(GV);
      Size += GVSize;
      MaxAlign = std::max(MaxAlign, GVAlign);
      ++Idx;
    }
    if (Members.size() < 2) {
      // A lone member (e.g. one literal larger than the cap) is left alone --
      // merging it with nothing would only rename it.
      continue;
    }

    // Packed: the padding is explicit above, so the struct layout must not add
    // any of its own or the recorded offsets would be wrong.
    StructType *STy = StructType::get(M.getContext(), {}, /*isPacked=*/true);
    SmallVector<Type *, 64> FieldTys;
    for (Constant *F : Fields)
      FieldTys.push_back(F->getType());
    STy = StructType::create(M.getContext(), FieldTys,
                             ("__capstone_merged_strs." + Twine(ContainerNo)).str(),
                             /*isPacked=*/true);
    Constant *Init = ConstantStruct::get(STy, Fields);

    auto *Container = new GlobalVariable(
        M, STy, /*isConstant=*/true, GlobalValue::PrivateLinkage, Init,
        ".L__capstone_merged_strs." + Twine(ContainerNo), nullptr,
        GlobalValue::NotThreadLocal, AS);
    Container->setAlignment(MaxAlign);
    Container->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);

    // Replace each member with a pointer to its slice of the container. The
    // members are byte arrays and the container is packed, so a byte-offset GEP
    // reproduces the original address exactly.
    Type *I8 = Type::getInt8Ty(M.getContext());
    for (size_t I = 0; I < Members.size(); ++I) {
      GlobalVariable *GV = Members[I];
      Constant *Off = ConstantInt::get(Type::getInt64Ty(M.getContext()),
                                       Offsets[I]);
      Constant *Ptr = ConstantExpr::getGetElementPtr(I8, Container, Off,
                                                     /*InBounds=*/true);
      GV->replaceAllUsesWith(Ptr);
      GV->eraseFromParent();
    }
    ++ContainerNo;
    Changed = true;
  }
  return Changed;
}

char CapstoneMergeStrConstants::ID = 0;

INITIALIZE_PASS(CapstoneMergeStrConstants, DEBUG_TYPE, PASS_NAME, false, false)

ModulePass *llvm::createCapstoneMergeStrConstantsPass() {
  return new CapstoneMergeStrConstants();
}
