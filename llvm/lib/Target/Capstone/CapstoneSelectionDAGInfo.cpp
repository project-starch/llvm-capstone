//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CapstoneSelectionDAGInfo.h"
#include "CapstoneISelLowering.h"
#include "CapstoneSubtarget.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/Support/CommandLine.h"

#define GET_SDNODE_DESC
#include "CapstoneGenSDNodeInfo.inc"

using namespace llvm;

// S-06 WORKAROUND, codegen side. On the CVA6 silicon an `ldc`/`stc` round trip of PLAIN,
// untagged data keeps only its LOW 64 bits (see ISSUES.md S-06 and
// capstone/tests/fpga-repros/S06-untagged-ldc-stc-high-half/). A 16-byte-aligned copy is
// expanded into capability-grained `ldc`/`stc` pairs for EVERY chunk -- including chunks that
// hold no pointer -- so `struct { void *p; unsigned long x, y; }` silently loses `y`. Measured
// on the board with a rung that calls no memcpy at all: 66 where 64 is correct.
//
// The library-level fix in memcpy cannot reach this, because the compiler emits these copies
// directly for aggregate assignment. Routing them to a libcall instead was tried and does not
// work on this ABI (see -capstone-lower-memops-via-libcall in CapstoneISelLowering.cpp), so the
// sequence is emitted INLINE here, which introduces no call.
//
// Per 16-byte chunk: plain-store BOTH 64-bit halves first, then lay the `ldc`/`stc` on top.
// Branchless, and correct for both kinds of chunk because of how the cache gates the store
// (`st_wr_cap = |wr_user_i`):
//   - the chunk IS a capability -> its metadata is non-zero, so the `stc` writes both banks and
//     restores the tag, overwriting the plain stores;
//   - the chunk is PLAIN data   -> the `ldc` yields zero metadata, so the `stc` degrades to a
//     single-bank store that never touches the high half, and the plain store survives.
// Validated on both kinds of chunk in RTL simulation (capstone-ariane
// verif/tests/custom/capstone/untagged-ldc-stc-fixup.S arm E) and on silicon at the primitive
// level.
//
// Default OFF: it triples the store traffic of every aligned aggregate copy and changes the
// emitted geometry that the published BEEBS numbers were measured with.
// Bisection knob: only apply the fixup to copies of at most this many bytes. Exists because the
// flag works on a small rung and faults inside SQLite's CREATE TABLE, and narrowing by size is
// the cheapest way to find which copies are responsible without mapping a runtime pc back to
// the image.
static cl::opt<unsigned> CapstoneMemcpyHighHalfFixupMaxBytes(
    "capstone-memcpy-high-half-fixup-max-bytes", cl::Hidden,
    cl::desc("S-06 workaround: only fix copies of at most N bytes (bisection aid)"),
    cl::init(512));

cl::opt<bool> CapstoneMemcpyHighHalfFixup(
    "capstone-memcpy-high-half-fixup", cl::Hidden,
    cl::desc("S-06 workaround: expand 16-byte-aligned memcpy as plain-store-both-halves then "
             "ldc/stc, so an untagged chunk keeps its high 64 bits"),
    cl::init(false));

// DIAGNOSTIC ONLY -- NEVER SHIP A BUILD WITH THIS ON.
//
// Drops the trailing capability store from the fixup sequence, leaving `ld, ld, ldc, sd, sd`.
// That does NOT preserve tags, so any real capability copied by this memcpy is destroyed and the
// domain will misbehave in ways unrelated to what is being measured. It exists for exactly one
// experiment.
//
// WHY. The fixup above is correct in RTL simulation and on an isolated silicon rung, yet wedges
// SQLite with mcause 25 INVALID_CAPABILITY, and that failure is UNEXPLAINED -- its only
// hypothesis was refuted on 2026-08-11. Nothing localises it to the reads or the writes. This
// build keeps the fixup's SOURCE traffic byte-identical and removes only the `stc`, so a matched
// pair against the unmodified fixup differs in exactly one thing:
//
//   wedges  -> the cause is source-side (`ld, ld, ldc`), and every proposed fix that reads the
//              source that way inherits it -- including the LCC-query design under consideration
//   returns -> the cause is destination-side, and that design is on the right side of the line
//
// It answers the question without knowing the mechanism, which is why it is worth a board slot.
static cl::opt<bool> CapstoneMemcpyFixupNoStc(
    "capstone-memcpy-fixup-no-stc", cl::Hidden,
    cl::desc("DIAGNOSTIC ONLY, breaks capability copies: omit the trailing stc from the S-06 "
             "fixup, to localise the SQLite wedge to the source or destination side"),
    cl::init(false));

// DIAGNOSTIC ONLY, but unlike -no-stc this one is INTERPRETABLE.
//
// Drops the two plain stores, leaving `ld, ld, ldc, stc`. Source traffic stays byte-identical to
// the full fixup, and -- crucially -- the capability store REMAINS, so capabilities are still
// copied correctly and the domain is not sabotaged in a second, independent way.
//
// WHY THIS EXISTS. The -no-stc variant above wedged on silicon with mcause 25, and that result
// is CONFOUNDED and must not be cited as localisation: removing the stc means a copied pointer
// never gets its tag, so a later dereference faults with exactly mcause 25 for reasons that have
// nothing to do with the source sequence. That arm differs from the fixup in TWO respects and
// therefore measures whichever one you did not intend.
//
// RESULT, AND ITS RETRACTION. This arm RETURNED on the board while the full fixup WEDGED, and
// that was recorded as "the plain stores are the trigger, the source reads are exonerated".
// THAT CONCLUSION IS WITHDRAWN -- this arm is confounded too, in the mirror image of the one
// above. Dropping the plain stores leaves the destination written ONLY by the `stc`, which under
// S-06 loses the high half, so this build produces the SAME CORRUPT DATA as the unfixed baseline.
// It is the baseline plus dead volatile loads. The variable that separates it from the full
// fixup is therefore not the store pattern but whether the copy is CORRECT -- and "repairing the
// data makes SQLite run deeper and meet a different silicon fault" is an explanation this
// project had already recorded, consistent with all three constructions that have wedged.
//
// Both diagnostic arms held SOURCE TRAFFIC constant and let DATA CORRECTNESS float. Any future
// arm must state which variables it holds fixed, and needs N >= 3 -- the board table was N = 1.
//
// Neither flag currently supports a conclusion about where the wedge comes from.
static cl::opt<bool> CapstoneMemcpyFixupNoPlainStores(
    "capstone-memcpy-fixup-no-plain-stores", cl::Hidden,
    cl::desc("DIAGNOSTIC ONLY: omit the two plain stores from the S-06 fixup, keeping the source "
             "reads and the stc, to localise the SQLite wedge without breaking capability copies"),
    cl::init(false));

CapstoneSelectionDAGInfo::CapstoneSelectionDAGInfo()
    : SelectionDAGGenTargetInfo(CapstoneGenSDNodeInfo) {}

CapstoneSelectionDAGInfo::~CapstoneSelectionDAGInfo() = default;

void CapstoneSelectionDAGInfo::verifyTargetNode(const SelectionDAG &DAG,
                                             const SDNode *N) const {
#ifndef NDEBUG
  switch (N->getOpcode()) {
  default:
    return SelectionDAGGenTargetInfo::verifyTargetNode(DAG, N);
  case CapstoneISD::TUPLE_EXTRACT:
    assert(N->getNumOperands() == 2 && "Expected three operands!");
    assert(N->getOperand(1).getOpcode() == ISD::TargetConstant &&
           N->getOperand(1).getValueType() == MVT::i32 &&
           "Expected index to be an i32 target constant!");
    break;
  case CapstoneISD::TUPLE_INSERT:
    assert(N->getNumOperands() == 3 && "Expected three operands!");
    assert(N->getOperand(2).getOpcode() == ISD::TargetConstant &&
           N->getOperand(2).getValueType() == MVT::i32 &&
           "Expected index to be an i32 target constant!");
    break;
  case CapstoneISD::VQDOT_VL:
  case CapstoneISD::VQDOTU_VL:
  case CapstoneISD::VQDOTSU_VL: {
    assert(N->getNumValues() == 1 && "Expected one result!");
    assert(N->getNumOperands() == 5 && "Expected five operands!");
    EVT VT = N->getValueType(0);
    assert(VT.isScalableVector() && VT.getVectorElementType() == MVT::i32 &&
           "Expected result to be an i32 scalable vector");
    assert(N->getOperand(0).getValueType() == VT &&
           N->getOperand(1).getValueType() == VT &&
           N->getOperand(2).getValueType() == VT &&
           "Expected result and first 3 operands to have the same type!");
    EVT MaskVT = N->getOperand(3).getValueType();
    assert(MaskVT.isScalableVector() &&
           MaskVT.getVectorElementType() == MVT::i1 &&
           MaskVT.getVectorElementCount() == VT.getVectorElementCount() &&
           "Expected mask VT to be an i1 scalable vector with same number of "
           "elements as the result");
    assert((N->getOperand(4).getValueType() == MVT::i32 ||
            N->getOperand(4).getValueType() == MVT::i64) &&
           "Expect VL operand to be i32 or i64");
    break;
  }
  }
#endif
}

SDValue CapstoneSelectionDAGInfo::EmitTargetCodeForMemset(
    SelectionDAG &DAG, const SDLoc &dl, SDValue Chain, SDValue Dst, SDValue Src,
    SDValue Size, Align Alignment, bool isVolatile, bool AlwaysInline,
    MachinePointerInfo DstPtrInfo) const {
  const auto &Subtarget = DAG.getSubtarget<CapstoneSubtarget>();
  // We currently do this only for Xqcilsm
  if (!Subtarget.hasVendorXqcilsm())
    return SDValue();

  // Do this only if we know the size at compile time.
  ConstantSDNode *ConstantSize = dyn_cast<ConstantSDNode>(Size);
  if (!ConstantSize)
    return SDValue();

  uint64_t NumberOfBytesToWrite = ConstantSize->getZExtValue();

  // Do this only if it is word aligned and we write a multiple of 4 bytes.
  if (!(Alignment >= 4) || !((NumberOfBytesToWrite & 3) == 0))
    return SDValue();

  SmallVector<SDValue, 8> OutChains;
  SDValue SrcValueReplicated = DAG.getNode(ISD::ZERO_EXTEND, dl, MVT::i32, Src);
  int NumberOfWords = NumberOfBytesToWrite / 4;
  MachineFunction &MF = DAG.getMachineFunction();
  auto Volatile =
      isVolatile ? MachineMemOperand::MOVolatile : MachineMemOperand::MONone;

  // Helper for constructing the QC_SETWMI instruction
  auto getSetwmiNode = [&](uint8_t SizeWords, uint8_t OffsetSetwmi) -> SDValue {
    SDValue Ops[] = {Chain, SrcValueReplicated, Dst,
                     DAG.getTargetConstant(SizeWords, dl, MVT::i32),
                     DAG.getTargetConstant(OffsetSetwmi, dl, MVT::i32)};
    MachineMemOperand *BaseMemOperand = MF.getMachineMemOperand(
        DstPtrInfo.getWithOffset(OffsetSetwmi),
        MachineMemOperand::MOStore | Volatile, SizeWords * 4, Align(4));
    return DAG.getMemIntrinsicNode(CapstoneISD::QC_SETWMI, dl,
                                   DAG.getVTList(MVT::Other), Ops, MVT::i32,
                                   BaseMemOperand);
  };

  // If i8 type and constant non-zero value.
  if ((Src.getValueType() == MVT::i8) && !isNullConstant(Src))
    // Replicate byte to word by multiplication with 0x01010101.
    SrcValueReplicated =
        DAG.getNode(ISD::MUL, dl, MVT::i32, SrcValueReplicated,
                    DAG.getConstant(0x01010101ul, dl, MVT::i32));

  // We limit a QC_SETWMI to 16 words or less to improve interruptibility.
  // So for 1-16 words we use a single QC_SETWMI:
  //
  // QC_SETWMI reg1, N, 0(reg2)
  //
  // For 17-32 words we use two QC_SETWMI's with the first as 16 words and the
  // second for the remainder:
  //
  // QC_SETWMI reg1, 16, 0(reg2)
  // QC_SETWMI reg1, N, 64(reg2)
  //
  // For 33-48 words, we would like to use (16, 16, n), but that means the last
  // QC_SETWMI needs an offset of 128 which the instruction doesn't support.
  // So in this case we use a length of 15 for the second instruction and we do
  // the rest with the third instruction.
  // This means the maximum inlined number of words is 47 (for now):
  //
  // QC_SETWMI R2, R0, 16, 0
  // QC_SETWMI R2, R0, 15, 64
  // QC_SETWMI R2, R0, N, 124
  //
  // For 48 words or more, call the target independent memset
  if (NumberOfWords >= 48)
    return SDValue();

  if (NumberOfWords <= 16) {
    // 1 - 16 words
    return getSetwmiNode(NumberOfWords, 0);
  }

  if (NumberOfWords <= 32) {
    // 17 - 32 words
    OutChains.push_back(getSetwmiNode(NumberOfWords - 16, 64));
    OutChains.push_back(getSetwmiNode(16, 0));
  } else {
    // 33 - 47 words
    OutChains.push_back(getSetwmiNode(NumberOfWords - 31, 124));
    OutChains.push_back(getSetwmiNode(15, 64));
    OutChains.push_back(getSetwmiNode(16, 0));
  }

  return DAG.getNode(ISD::TokenFactor, dl, MVT::Other, OutChains);
}

// THE SINGLE SOURCE OF TRUTH for "does the fixup apply to this copy?".
//
// This exists because the predicate lives in TWO places that must agree exactly: the decline in
// CapstoneTargetLowering::findOptimalMemOpLowering (which stops the generic inline expansion so
// this hook is consulted at all) and the hook itself. When they disagree, `getMemcpy` finds that
// nobody will expand the copy and falls through to a LIBCALL -- silently, with no diagnostic.
//
// They HAD diverged, in three ways, all measured with the tree's own `llc` rather than reasoned
// about:
//   * volatile      -- checked here, not at the decline. A volatile 16-aligned copy became a libcall.
//   * max-bytes     -- checked here, not at the decline. This one also invalidated a recorded
//                      bisection result: restricting the fixup to 16 bytes was read as "one chunk
//                      is enough to reproduce", when it had actually converted every LARGER
//                      16-aligned copy to a libcall.
//   * memmove       -- `MemOp` carries no memcpy/memmove discriminator (`isMemcpy()` is just
//                      `!IsMemset`), so the decline fires for memmove too, and there was no
//                      EmitTargetCodeForMemmove override to catch it. See that override below.
//
// Keep every condition that is expressible from BOTH call sites in here. `isVolatile` is
// available at the decline as `!Op.allowOverlap()` (TargetLowering.h:143 sets
// `AllowOverlap = !IsVolatile`), so it belongs here rather than in the caller.
// NB: takes the ALIGNMENT CHECK RESULT, not an Align. MemOp::getDstAlign() carries
// `assert(!DstAlignCanChange)` and fires whenever the destination is a fresh alloca whose
// alignment may still be raised -- a common shape -- so the decline site must use
// MemOp::isAligned(), which handles that case, and pass the answer in.
bool llvm::capstoneShouldExpandCapMemcpyInline(uint64_t Bytes, bool Is16ByteAligned,
                                               bool IsVolatile) {
  if (!CapstoneMemcpyHighHalfFixup)
    return false;
  // A volatile copy must not have its stores multiplied.
  if (IsVolatile)
    return false;
  // Only the capability-grained shape needs fixing: that is the ONLY one lowered to ldc/stc.
  // Anything else already copies with scalar units and cannot lose a high half.
  if (Bytes == 0 || (Bytes % 16) != 0 || !Is16ByteAligned)
    return false;
  // Match the chunk budget the inline capability path uses.
  if ((Bytes / 16) > 32)
    return false;
  if (Bytes > CapstoneMemcpyHighHalfFixupMaxBytes)
    return false;
  return true;
}

SDValue CapstoneSelectionDAGInfo::EmitTargetCodeForMemcpy(
    SelectionDAG &DAG, const SDLoc &dl, SDValue Chain, SDValue Dst, SDValue Src,
    SDValue Size, Align Alignment, bool isVolatile, bool AlwaysInline,
    MachinePointerInfo DstPtrInfo, MachinePointerInfo SrcPtrInfo) const {
  ConstantSDNode *ConstantSize = dyn_cast<ConstantSDNode>(Size);
  if (!ConstantSize)
    return SDValue();
  uint64_t Bytes = ConstantSize->getZExtValue();

  if (!capstoneShouldExpandCapMemcpyInline(Bytes, Alignment >= Align(16), isVolatile))
    return SDValue();

  uint64_t NumChunks = Bytes / 16;

  // Strictly serial chain. The `ldc`/`stc` for a chunk MUST be ordered after that chunk's two
  // plain stores -- the whole construction depends on the capability store landing last.
  //
  // A serial chain expresses that intent, but DO NOT read it as a guarantee: at -O2 the machine
  // scheduler demonstrably DOES reorder these, hoisting all the plain stores together and
  // sinking the `stc`s to the end. The required per-chunk order survived there only because the
  // moves happened to go in that direction. The board result was taken at -O0, where no MI
  // scheduling runs, so it says nothing about higher optimisation levels. If this is ever
  // enabled above -O0, the ordering must be enforced properly -- e.g. by making the capability
  // store volatile too, or by emitting a pseudo that cannot be split -- and re-verified on the
  // emitted code, not assumed.
  // Capability-preserving pointer arithmetic. The generic offset helper builds an ISD::ADD,
  // which selects to an INTEGER `addi` whenever it is materialised instead of folded into the
  // addressing mode -- and `addi` STRIPS the capability, so the access on that address then
  // faults with UNEXPECTED_OPERAND. Not theoretical: it is the bug this hook shipped with, seen
  // in sqlite3Parser as
  //     addi a5, s0, -0xd8   ;   sd a4, 0x0(a5)      <- base is no longer a capability
  // while the neighbouring store, whose offset happened to fold into the immediate, was fine --
  // which is why it only showed up at SQLite scale and never on a small rung.
  // CapstoneISD::CIncOffset is the node that means "advance a capability's cursor".
  auto capOffset = [&](SDValue Ptr, uint64_t Off) -> SDValue {
    if (Off == 0)
      return Ptr;
    return DAG.getNode(CapstoneISD::CIncOffset, dl, Ptr.getValueType(), Ptr,
                       DAG.getConstant(Off, dl, MVT::i64));
  };

  for (uint64_t Chunk = 0; Chunk != NumChunks; ++Chunk) {
    uint64_t Base = Chunk * 16;

    // ORDER: read EVERYTHING from the source before writing ANYTHING to the destination.
  //
  // This is not a scheduling preference, it is correctness. A plain store CLEARS the target
  // line's capability tag. If the destination and source are the same 16 bytes -- an exact
  // self-copy, which `*d = *s` with `d == s` is, and which clang lowers to a memcpy -- then
  // pre-storing the halves first clears the tag of the line the `ldc` is about to read. The
  // `ldc` then sees untagged data, so the `stc` writes it back untagged and a LIVE CAPABILITY
  // IS DESTROYED. Measured: with the stores first, a domain that returned an error instead
  // wedged on silicon with mcause 25 INVALID_CAPABILITY, matched pair in one boot.
  //
  // DO NOT READ THE ABOVE AS "AND THAT FIXED THE WEDGE". It did not. This ordering is
  // necessary for self-copy correctness, and it is not sufficient: with the reads first, the
  // workaround STILL wedges SQLite with the same mcause 25. That failure is UNEXPLAINED --
  // its only hypothesis (a tag/eviction divergence between the D-cache's two `ruser` gates)
  // was refuted on 2026-08-11, because the refill gate's eight bits are a shadow-tag byte and
  // not capability metadata, so the two gates are the same predicate over different
  // encodings. See agent-handoff/ref/ISSUES.md under S-06. The workaround is therefore NOT
  // enabled anywhere, and this flag defaults off.
  //
  // Reading first also makes the sequence safe for any overlap inside a chunk, which the
  // original single ldc/stc had for free and a split sequence does not.
  //
  // The two plain stores MUST still be marked volatile: without it they are dead by the
  // compiler's own model -- it believes the `stc` writes all 16 bytes -- so DSE deletes them
  // and silently regenerates the unfixed sequence. Measured: for a copy into a stack slot the
  // pre-writes vanished entirely and the output was byte-identical to the unfixed build.
    // --- read the whole chunk first ---
    // Under -capstone-memcpy-fixup-no-plain-stores these two loads lose their only users, so
    // DCE would delete them and the image would carry plain `ldc, stc` -- i.e. the BASELINE,
    // silently measuring nothing. Same trap as the capability load below under -no-stc.
    auto SrcFlags = CapstoneMemcpyFixupNoPlainStores ? MachineMemOperand::MOVolatile
                                                     : MachineMemOperand::MONone;
    SDValue Lo = DAG.getLoad(MVT::i64, dl, Chain, capOffset(Src, Base),
                             SrcPtrInfo.getWithOffset(Base), Align(8), SrcFlags);
    Chain = Lo.getValue(1);
    SDValue Hi = DAG.getLoad(MVT::i64, dl, Chain, capOffset(Src, Base + 8),
                             SrcPtrInfo.getWithOffset(Base + 8), Align(8), SrcFlags);
    Chain = Hi.getValue(1);
    // Under -capstone-memcpy-fixup-no-stc this load has NO USERS, so plain DCE deletes it and
    // the resulting image would carry `ld, ld, sd, sd` -- measuring a sequence with no `ldc` in
    // it at all, and answering a question nobody asked. Marking it volatile keeps it. This is
    // the same failure mode the two plain stores already guard against below, and the reason
    // every arm of this experiment is disassembled before it is baked.
    SDValue Cap = DAG.getLoad(MVT::i128, dl, Chain, capOffset(Src, Base),
                              SrcPtrInfo.getWithOffset(Base), Align(16),
                              CapstoneMemcpyFixupNoStc
                                  ? MachineMemOperand::MOVolatile
                                  : MachineMemOperand::MONone);
    Chain = Cap.getValue(1);

    // --- then write it ---
    if (!CapstoneMemcpyFixupNoPlainStores) {
      Chain = DAG.getStore(Chain, dl, Lo, capOffset(Dst, Base),
                           DstPtrInfo.getWithOffset(Base), Align(8),
                           MachineMemOperand::MOVolatile);
      Chain = DAG.getStore(Chain, dl, Hi, capOffset(Dst, Base + 8),
                           DstPtrInfo.getWithOffset(Base + 8), Align(8),
                           MachineMemOperand::MOVolatile);
    }
    // Last, so a real capability's tag is restored over the plain stores.
    //
    // Skipped only under -capstone-memcpy-fixup-no-stc, which is a localisation experiment and
    // not a build anyone should run for real: without this store a copied capability keeps no
    // tag. The `ldc` above is deliberately still emitted even though its value is now unused,
    // because the whole point is to hold the SOURCE traffic byte-identical to the real fixup.
    if (!CapstoneMemcpyFixupNoStc)
      Chain = DAG.getStore(Chain, dl, Cap, capOffset(Dst, Base),
                           DstPtrInfo.getWithOffset(Base), Align(16));
  }

  return Chain;
}
