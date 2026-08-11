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

SDValue CapstoneSelectionDAGInfo::EmitTargetCodeForMemcpy(
    SelectionDAG &DAG, const SDLoc &dl, SDValue Chain, SDValue Dst, SDValue Src,
    SDValue Size, Align Alignment, bool isVolatile, bool AlwaysInline,
    MachinePointerInfo DstPtrInfo, MachinePointerInfo SrcPtrInfo) const {
  if (!CapstoneMemcpyHighHalfFixup)
    return SDValue();

  // A volatile copy must not have its stores multiplied; leave it to the generic path.
  if (isVolatile)
    return SDValue();

  ConstantSDNode *ConstantSize = dyn_cast<ConstantSDNode>(Size);
  if (!ConstantSize)
    return SDValue();
  uint64_t Bytes = ConstantSize->getZExtValue();

  // Only the capability-grained shape needs fixing: that is the ONLY one lowered to ldc/stc
  // (CapstoneTargetLowering::findOptimalMemOpLowering). Anything else already copies with
  // scalar units and cannot lose a high half, so hand it back to the generic expansion rather
  // than growing code for no reason.
  if (Bytes == 0 || (Bytes % 16) != 0 || Alignment < Align(16))
    return SDValue();

  // Match the chunk budget the inline capability path uses; above it the generic path emits a
  // libcall, which lands on the library memcpy and its own copy of this fix.
  uint64_t NumChunks = Bytes / 16;
  if (NumChunks > 32)
    return SDValue();
  if (Bytes > CapstoneMemcpyHighHalfFixupMaxBytes)
    return SDValue();

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

    // The two plain stores MUST be marked volatile. Without it they are DEAD by the compiler's
    // own model -- it believes the `stc` below writes all 16 bytes of the same chunk, so DSE
    // deletes them and silently regenerates the very sequence this is working around. Measured:
    // for a copy into a stack slot the pre-writes vanished entirely and the output was
    // byte-identical to the unfixed build. On this silicon the `stc` does NOT write the high
    // half of an untagged chunk, which is exactly the fact the optimiser cannot know.
    for (unsigned Half = 0; Half != 2; ++Half) {
      uint64_t Off = Base + Half * 8;
      SDValue Ld = DAG.getLoad(
          MVT::i64, dl, Chain,
          capOffset(Src, Off),
          SrcPtrInfo.getWithOffset(Off), Align(8));
      Chain = DAG.getStore(
          Ld.getValue(1), dl, Ld,
          capOffset(Dst, Off),
          DstPtrInfo.getWithOffset(Off), Align(8), MachineMemOperand::MOVolatile);
    }

    SDValue LdCap = DAG.getLoad(
        MVT::i128, dl, Chain,
        capOffset(Src, Base),
        SrcPtrInfo.getWithOffset(Base), Align(16));
    Chain = DAG.getStore(
        LdCap.getValue(1), dl, LdCap,
        capOffset(Dst, Base),
        DstPtrInfo.getWithOffset(Base), Align(16));
  }

  return Chain;
}
