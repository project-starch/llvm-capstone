//===- IntrinsicsCapstone.td - Defines Capstone intrinsics -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the Capstone-specific intrinsics.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Atomics

// Atomic Intrinsics have multiple versions for different access widths, which
// all follow one of the following signatures (depending on how many arguments
// they require).
//
// In fact, as these intrinsics take `llvm_anyptr_ty`, the given names are the
// canonical names, and the intrinsics used in the code will have a name
// suffixed with the pointer type they are specialised for (denoted `<p>` in the
// names below), in order to avoid type conflicts.

let TargetPrefix = "capstone" in {

  // T @llvm.<name>.<i>.<p>(any*, T, T, T imm);
  class CapstoneMaskedAtomicRMWFourArg
      : Intrinsic<[llvm_anyint_ty], [llvm_anyptr_ty, LLVMMatchType<0>,
                                     LLVMMatchType<0>, LLVMMatchType<0>],
                  [IntrArgMemOnly, NoCapture<ArgIndex<0>>, ImmArg<ArgIndex<3>>]>;
  // T @llvm.<name>.<i>.<p>(any*, T, T, T, T imm);
  class CapstoneMaskedAtomicRMWFiveArg
      : Intrinsic<[llvm_anyint_ty], [llvm_anyptr_ty, LLVMMatchType<0>,
                                     LLVMMatchType<0>, LLVMMatchType<0>,
                                     LLVMMatchType<0>],
                  [IntrArgMemOnly, NoCapture<ArgIndex<0>>, ImmArg<ArgIndex<4>>]>;

  // These intrinsics are intended only for internal compiler use (i.e. as
  // part of AtomicExpandpass via the emitMaskedAtomic*Intrinsic hooks). Their
  // names and semantics could change in the future.

  // @llvm.capstone.masked.atomicrmw.*.<i>.<p>(
  //   ptr addr, ixlen oparg, ixlen mask, ixlenimm ordering)
  def int_capstone_masked_atomicrmw_xchg : CapstoneMaskedAtomicRMWFourArg;
  def int_capstone_masked_atomicrmw_add : CapstoneMaskedAtomicRMWFourArg;
  def int_capstone_masked_atomicrmw_sub : CapstoneMaskedAtomicRMWFourArg;
  def int_capstone_masked_atomicrmw_nand : CapstoneMaskedAtomicRMWFourArg;
  def int_capstone_masked_atomicrmw_umax : CapstoneMaskedAtomicRMWFourArg;
  def int_capstone_masked_atomicrmw_umin : CapstoneMaskedAtomicRMWFourArg;
  // Signed min and max need an extra operand to do sign extension with.
  // @llvm.capstone.masked.atomicrmw.{max,min}.<i>.<p>(
  //   ptr addr, ixlen oparg, ixlen mask, ixlen shamt, ixlenimm ordering)
  def int_capstone_masked_atomicrmw_max : CapstoneMaskedAtomicRMWFiveArg;
  def int_capstone_masked_atomicrmw_min : CapstoneMaskedAtomicRMWFiveArg;

  // @llvm.capstone.masked.cmpxchg.<i>.<p>(
  //   ptr addr, ixlen cmpval, ixlen newval, ixlen mask, ixlenimm ordering)
  def int_capstone_masked_cmpxchg : CapstoneMaskedAtomicRMWFiveArg;

} // TargetPrefix = "capstone"

//===----------------------------------------------------------------------===//
// Bitmanip (Bit Manipulation) Extension

let TargetPrefix = "capstone" in {

  class CapstoneBitManipGPRIntrinsics
      : DefaultAttrsIntrinsic<[llvm_anyint_ty],
                              [LLVMMatchType<0>],
                              [IntrNoMem, IntrSpeculatable]>;
  class CapstoneBitManipGPRGPRIntrinsics
      : DefaultAttrsIntrinsic<[llvm_anyint_ty],
                              [LLVMMatchType<0>, LLVMMatchType<0>],
                              [IntrNoMem, IntrSpeculatable]>;

  // Zbb
  def int_capstone_orc_b : CapstoneBitManipGPRIntrinsics;

  // Zbc or Zbkc
  def int_capstone_clmul  : CapstoneBitManipGPRGPRIntrinsics;
  def int_capstone_clmulh : CapstoneBitManipGPRGPRIntrinsics;

  // Zbc
  def int_capstone_clmulr : CapstoneBitManipGPRGPRIntrinsics;

  // Zbkb
  def int_capstone_brev8 : CapstoneBitManipGPRIntrinsics;
  def int_capstone_zip   : CapstoneBitManipGPRIntrinsics;
  def int_capstone_unzip : CapstoneBitManipGPRIntrinsics;

  // Zbkx
  def int_capstone_xperm4  : CapstoneBitManipGPRGPRIntrinsics;
  def int_capstone_xperm8  : CapstoneBitManipGPRGPRIntrinsics;
} // TargetPrefix = "capstone"

//===----------------------------------------------------------------------===//
// May-Be-Operations

let TargetPrefix = "capstone" in {

  // Zimop
  def int_capstone_mopr
      : DefaultAttrsIntrinsic<[llvm_anyint_ty],
                              [LLVMMatchType<0>, LLVMMatchType<0>],
                              [IntrNoMem, IntrSpeculatable, ImmArg<ArgIndex<1>>]>;
  def int_capstone_moprr
      : DefaultAttrsIntrinsic<[llvm_anyint_ty],
                              [LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>],
                              [IntrNoMem, IntrSpeculatable, ImmArg<ArgIndex<2>>]>;
} // TargetPrefix = "capstone"

//===----------------------------------------------------------------------===//
// Vectors

// The intrinsic does not have any operand that must be extended.
defvar CapstoneNoScalarOperand = 0xF;

// The intrinsic does not have a VL operand.
// (e.g., capstone_vmv_x_s and capstone_vfmv_f_s)
defvar CapstoneNoVLOperand = 0x1F;

class CapstoneVIntrinsic {
  // These intrinsics may accept illegal integer values in their llvm_anyint_ty
  // operand, so they have to be extended.
  Intrinsic IntrinsicID = !cast<Intrinsic>(NAME);
  bits<4> ScalarOperand = CapstoneNoScalarOperand;
  bits<5> VLOperand = CapstoneNoVLOperand;
}

let TargetPrefix = "capstone" in {
  // We use anyint here but we only support XLen.
  def int_capstone_vsetvli   : DefaultAttrsIntrinsic<[llvm_anyint_ty],
                           /* AVL */  [LLVMMatchType<0>,
                           /* VSEW */  LLVMMatchType<0>,
                           /* VLMUL */ LLVMMatchType<0>],
                                      [IntrNoMem,
                                       ImmArg<ArgIndex<1>>,
                                       ImmArg<ArgIndex<2>>]>;
  def int_capstone_vsetvlimax : DefaultAttrsIntrinsic<[llvm_anyint_ty],
                            /* VSEW */ [LLVMMatchType<0>,
                            /* VLMUL */ LLVMMatchType<0>],
                                      [IntrNoMem,
                                       ImmArg<ArgIndex<0>>,
                                       ImmArg<ArgIndex<1>>]>;

  // For unit stride mask load
  // Input: (pointer, vl)
  class CapstoneUSMLoad
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [llvm_anyptr_ty, llvm_anyint_ty],
                    [NoCapture<ArgIndex<0>>, IntrReadMem, IntrArgMemOnly]>,
          CapstoneVIntrinsic {
    let VLOperand = 1;
  }
  // For unit stride load
  // Input: (passthru, pointer, vl)
  class CapstoneUSLoad
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_anyptr_ty, llvm_anyint_ty],
                    [NoCapture<ArgIndex<1>>, IntrReadMem, IntrArgMemOnly]>,
          CapstoneVIntrinsic {
    let VLOperand = 2;
  }
  // For unit stride fault-only-first load
  // Input: (passthru, pointer, vl)
  // Output: (data, vl)
  // NOTE: We model this with default memory properties since we model writing
  // VL as a side effect. IntrReadMem, IntrHasSideEffects does not work.
  class CapstoneUSLoadFF
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty, llvm_anyint_ty],
                    [LLVMMatchType<0>, llvm_anyptr_ty, LLVMMatchType<1>],
                    [NoCapture<ArgIndex<1>>]>,
                    CapstoneVIntrinsic {
    let VLOperand = 2;
  }
  // For unit stride load with mask
  // Input: (maskedoff, pointer, mask, vl, policy)
  class CapstoneUSLoadMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty ],
                    [LLVMMatchType<0>, llvm_anyptr_ty,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                     llvm_anyint_ty, LLVMMatchType<2>],
                    [NoCapture<ArgIndex<1>>, ImmArg<ArgIndex<4>>, IntrReadMem,
                     IntrArgMemOnly]>,
                    CapstoneVIntrinsic {
    let VLOperand = 3;
  }
  // For unit stride fault-only-first load with mask
  // Input: (maskedoff, pointer, mask, vl, policy)
  // Output: (data, vl)
  // NOTE: We model this with default memory properties since we model writing
  // VL as a side effect. IntrReadMem, IntrHasSideEffects does not work.
  class CapstoneUSLoadFFMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty, llvm_anyint_ty],
                    [LLVMMatchType<0>, llvm_anyptr_ty,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                     LLVMMatchType<1>, LLVMMatchType<1>],
                    [NoCapture<ArgIndex<1>>, ImmArg<ArgIndex<4>>]>, CapstoneVIntrinsic {
    let VLOperand = 3;
  }
  // For strided load with passthru operand
  // Input: (passthru, pointer, stride, vl)
  class CapstoneSLoad
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_anyptr_ty,
                     llvm_anyint_ty, LLVMMatchType<2>],
                    [NoCapture<ArgIndex<1>>, IntrReadMem]>, CapstoneVIntrinsic {
    let VLOperand = 3;
  }
  // For strided load with mask
  // Input: (maskedoff, pointer, stride, mask, vl, policy)
  class CapstoneSLoadMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty ],
                    [LLVMMatchType<0>, llvm_anyptr_ty, llvm_anyint_ty,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, LLVMMatchType<2>,
                     LLVMMatchType<2>],
                    [NoCapture<ArgIndex<1>>, ImmArg<ArgIndex<5>>, IntrReadMem]>,
                    CapstoneVIntrinsic {
    let VLOperand = 4;
  }
  // For indexed load with passthru operand
  // Input: (passthru, pointer, index, vl)
  class CapstoneILoad
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_anyptr_ty,
                     llvm_anyvector_ty, llvm_anyint_ty],
                    [NoCapture<ArgIndex<1>>, IntrReadMem]>, CapstoneVIntrinsic {
    let VLOperand = 3;
  }
  // For indexed load with mask
  // Input: (maskedoff, pointer, index, mask, vl, policy)
  class CapstoneILoadMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty ],
                    [LLVMMatchType<0>, llvm_anyptr_ty, llvm_anyvector_ty,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyint_ty,
                     LLVMMatchType<3>],
                    [NoCapture<ArgIndex<1>>, ImmArg<ArgIndex<5>>, IntrReadMem]>,
                    CapstoneVIntrinsic {
    let VLOperand = 4;
  }
  // For unit stride store
  // Input: (vector_in, pointer, vl)
  class CapstoneUSStore
        : DefaultAttrsIntrinsic<[],
                    [llvm_anyvector_ty, llvm_anyptr_ty, llvm_anyint_ty],
                    [NoCapture<ArgIndex<1>>, IntrWriteMem, IntrArgMemOnly]>,
          CapstoneVIntrinsic {
    let VLOperand = 2;
  }
  // For unit stride store with mask
  // Input: (vector_in, pointer, mask, vl)
  class CapstoneUSStoreMasked
        : DefaultAttrsIntrinsic<[],
                    [llvm_anyvector_ty, llvm_anyptr_ty,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                     llvm_anyint_ty],
                    [NoCapture<ArgIndex<1>>, IntrWriteMem, IntrArgMemOnly]>,
          CapstoneVIntrinsic {
    let VLOperand = 3;
  }
  // For strided store
  // Input: (vector_in, pointer, stride, vl)
  class CapstoneSStore
        : DefaultAttrsIntrinsic<[],
                    [llvm_anyvector_ty, llvm_anyptr_ty,
                     llvm_anyint_ty, LLVMMatchType<2>],
                    [NoCapture<ArgIndex<1>>, IntrWriteMem]>, CapstoneVIntrinsic {
    let VLOperand = 3;
  }
  // For stride store with mask
  // Input: (vector_in, pointer, stirde, mask, vl)
  class CapstoneSStoreMasked
        : DefaultAttrsIntrinsic<[],
                    [llvm_anyvector_ty, llvm_anyptr_ty, llvm_anyint_ty,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, LLVMMatchType<2>],
                    [NoCapture<ArgIndex<1>>, IntrWriteMem]>, CapstoneVIntrinsic {
    let VLOperand = 4;
  }
  // For indexed store
  // Input: (vector_in, pointer, index, vl)
  class CapstoneIStore
        : DefaultAttrsIntrinsic<[],
                    [llvm_anyvector_ty, llvm_anyptr_ty,
                     llvm_anyint_ty, llvm_anyint_ty],
                    [NoCapture<ArgIndex<1>>, IntrWriteMem]>, CapstoneVIntrinsic {
    let VLOperand = 3;
  }
  // For indexed store with mask
  // Input: (vector_in, pointer, index, mask, vl)
  class CapstoneIStoreMasked
        : DefaultAttrsIntrinsic<[],
                    [llvm_anyvector_ty, llvm_anyptr_ty, llvm_anyvector_ty,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyint_ty],
                    [NoCapture<ArgIndex<1>>, IntrWriteMem]>, CapstoneVIntrinsic {
    let VLOperand = 4;
  }
  // For destination vector type is the same as source vector.
  // Input: (passthru, vector_in, vl)
  class CapstoneUnaryAAUnMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, LLVMMatchType<0>, llvm_anyint_ty],
                    [IntrNoMem]>, CapstoneVIntrinsic {
    let VLOperand = 2;
  }
  // For destination vector type is the same as the source vector type
  // Input: (passthru, vector_in, vl, policy)
  class CapstoneUnaryAAUnMaskedZvk<bit IsVS>
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, !if(IsVS, llvm_anyvector_ty, LLVMMatchType<0>),
                     llvm_anyint_ty, !if(IsVS, LLVMMatchType<2>, LLVMMatchType<1>)],
                    [ImmArg<ArgIndex<3>>, IntrNoMem]>, CapstoneVIntrinsic {
    let VLOperand = 2;
  }

  multiclass CapstoneUnaryAAUnMaskedZvk<bit HasVV = 1, bit HasVS = 1> {
    if HasVV then
      def "int_capstone_" # NAME # "_vv" : CapstoneUnaryAAUnMaskedZvk<IsVS=0>;

    if HasVS then
      def "int_capstone_" # NAME # "_vs" : CapstoneUnaryAAUnMaskedZvk<IsVS=1>;
  }
  // For destination vector type is the same as first source vector (with mask).
  // Input: (vector_in, vector_in, mask, vl, policy)
  class CapstoneUnaryAAMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, LLVMMatchType<0>,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyint_ty,
                     LLVMMatchType<1>],
                    [ImmArg<ArgIndex<4>>, IntrNoMem]>, CapstoneVIntrinsic {
    let VLOperand = 3;
  }
  // For destination vector type is the same as source vector.
  // Input: (passthru, vector_in, frm, vl)
  class CapstoneUnaryAAUnMaskedRoundingMode
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, LLVMMatchType<0>, llvm_anyint_ty, LLVMMatchType<1>],
                    [ImmArg<ArgIndex<2>>, IntrNoMem]>, CapstoneVIntrinsic {
    let VLOperand = 3;
  }
  // For destination vector type is the same as first source vector (with mask).
  // Input: (vector_in, vector_in, mask, frm, vl, policy)
  class CapstoneUnaryAAMaskedRoundingMode
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, LLVMMatchType<0>,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyint_ty,
                     LLVMMatchType<1>, LLVMMatchType<1>],
                    [ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<5>>, IntrNoMem]>, CapstoneVIntrinsic {
    let VLOperand = 4;
  }
  // Input: (passthru, vector_in, mask, vl)
  class CapstoneCompress
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, LLVMMatchType<0>,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyint_ty],
                    [IntrNoMem]>, CapstoneVIntrinsic {
    let VLOperand = 3;
  }
  // For destination vector type is the same as first and second source vector.
  // Input: (vector_in, vector_in, vl)
  class CapstoneBinaryAAAUnMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, LLVMMatchType<0>, llvm_anyint_ty],
                    [IntrNoMem]>, CapstoneVIntrinsic {
    let VLOperand = 2;
  }
  // For destination vector type is the same as first and second source vector.
  // Input: (passthru, vector_in, int_vector_in, vl)
  class CapstoneRGatherVVUnMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, LLVMMatchType<0>,
                     LLVMVectorOfBitcastsToInt<0>, llvm_anyint_ty],
                    [IntrNoMem]>, CapstoneVIntrinsic {
    let VLOperand = 3;
  }
  // For destination vector type is the same as first and second source vector.
  // Input: (vector_in, vector_in, int_vector_in, vl, policy)
  class CapstoneRGatherVVMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, LLVMMatchType<0>, LLVMVectorOfBitcastsToInt<0>,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyint_ty,
                     LLVMMatchType<1>],
                    [ImmArg<ArgIndex<5>>, IntrNoMem]>, CapstoneVIntrinsic {
    let VLOperand = 4;
  }
  // Input: (passthru, vector_in, int16_vector_in, vl)
  class CapstoneRGatherEI16VVUnMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, LLVMMatchType<0>,
                     LLVMScalarOrSameVectorWidth<0, llvm_i16_ty>,
                     llvm_anyint_ty],
                    [IntrNoMem]>, CapstoneVIntrinsic {
    let VLOperand = 3;
  }
  // For destination vector type is the same as first and second source vector.
  // Input: (vector_in, vector_in, int16_vector_in, vl, policy)
  class CapstoneRGatherEI16VVMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, LLVMMatchType<0>,
                     LLVMScalarOrSameVectorWidth<0, llvm_i16_ty>,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyint_ty,
                     LLVMMatchType<1>],
                    [ImmArg<ArgIndex<5>>, IntrNoMem]>, CapstoneVIntrinsic {
    let VLOperand = 4;
  }
  // For destination vector type is the same as first source vector, and the
  // second operand is XLen.
  // Input: (passthru, vector_in, xlen_in, vl)
  class CapstoneGatherVXUnMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, LLVMMatchType<0>, llvm_anyint_ty,
                     LLVMMatchType<1>],
                    [IntrNoMem]>, CapstoneVIntrinsic {
    let VLOperand = 3;
  }
  // For destination vector type is the same as first source vector (with mask).
  // Second operand is XLen.
  // Input: (maskedoff, vector_in, xlen_in, mask, vl, policy)
  class CapstoneGatherVXMasked
       : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                   [LLVMMatchType<0>, LLVMMatchType<0>, llvm_anyint_ty,
                    LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, LLVMMatchType<1>,
                    LLVMMatchType<1>],
                   [ImmArg<ArgIndex<5>>, IntrNoMem]>, CapstoneVIntrinsic {
    let VLOperand = 4;
  }
  // For destination vector type is the same as first source vector.
  // Input: (passthru, vector_in, vector_in/scalar_in, vl)
  class CapstoneBinaryAAXUnMasked<bit IsVI = 0>
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, LLVMMatchType<0>, llvm_any_ty,
                     llvm_anyint_ty],
                    !listconcat([IntrNoMem],
                                !if(IsVI, [ImmArg<ArgIndex<2>>], []))>,
                    CapstoneVIntrinsic {
    let ScalarOperand = 2;
    let VLOperand = 3;
  }
  // For destination vector type is the same as the source vector type.
  // Input: (passthru, vector_in, vector_in/scalar_in, vl, policy)
  class CapstoneBinaryAAXUnMaskedZvk<bit IsVI = 0>
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                                [LLVMMatchType<0>, LLVMMatchType<0>,
                                 llvm_any_ty, llvm_anyint_ty, LLVMMatchType<2>],
                                !listconcat([ImmArg<ArgIndex<4>>, IntrNoMem],
                                            !if(IsVI, [ImmArg<ArgIndex<2>>], []))>,
                                CapstoneVIntrinsic {
    let ScalarOperand = 2;
    let VLOperand = 3;
  }
  // For destination vector type is the same as first source vector (with mask).
  // Input: (maskedoff, vector_in, vector_in/scalar_in, mask, vl, policy)
  class CapstoneBinaryAAXMasked
       : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                   [LLVMMatchType<0>, LLVMMatchType<0>, llvm_any_ty,
                    LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyint_ty,
                    LLVMMatchType<2>],
                   [ImmArg<ArgIndex<5>>, IntrNoMem]>, CapstoneVIntrinsic {
    let ScalarOperand = 2;
    let VLOperand = 4;
  }
  // For destination vector type is the same as first source vector.
  // Input: (passthru, vector_in, vector_in/scalar_in, frm, vl)
  class CapstoneBinaryAAXUnMaskedRoundingMode
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, LLVMMatchType<0>, llvm_any_ty,
                     llvm_anyint_ty, LLVMMatchType<2>],
                    [ImmArg<ArgIndex<3>>, IntrNoMem]>, CapstoneVIntrinsic {
    let ScalarOperand = 2;
    let VLOperand = 4;
  }
  // For destination vector type is the same as first source vector (with mask).
  // Input: (maskedoff, vector_in, vector_in/scalar_in, mask, frm, vl, policy)
  class CapstoneBinaryAAXMaskedRoundingMode
       : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                   [LLVMMatchType<0>, LLVMMatchType<0>, llvm_any_ty,
                    LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyint_ty,
                    LLVMMatchType<2>, LLVMMatchType<2>],
                   [ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<6>>, IntrNoMem]>, CapstoneVIntrinsic {
    let ScalarOperand = 2;
    let VLOperand = 5;
  }
  // For destination vector type is the same as first source vector. The
  // second source operand must match the destination type or be an XLen scalar.
  // Input: (passthru, vector_in, vector_in/scalar_in, vl)
  class CapstoneBinaryAAShiftUnMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, LLVMMatchType<0>, llvm_any_ty,
                     llvm_anyint_ty],
                    [IntrNoMem]>, CapstoneVIntrinsic {
    let VLOperand = 3;
  }
  // For destination vector type is the same as first source vector (with mask).
  // The second source operand must match the destination type or be an XLen scalar.
  // Input: (maskedoff, vector_in, vector_in/scalar_in, mask, vl, policy)
  class CapstoneBinaryAAShiftMasked
       : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                   [LLVMMatchType<0>, LLVMMatchType<0>, llvm_any_ty,
                    LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyint_ty,
                    LLVMMatchType<2>],
                   [ImmArg<ArgIndex<5>>, IntrNoMem]>, CapstoneVIntrinsic {
    let VLOperand = 4;
  }
  // For destination vector type is NOT the same as first source vector.
  // Input: (passthru, vector_in, vector_in/scalar_in, vl)
  class CapstoneBinaryABXUnMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_anyvector_ty, llvm_any_ty,
                     llvm_anyint_ty],
                    [IntrNoMem]>, CapstoneVIntrinsic {
    let ScalarOperand = 2;
    let VLOperand = 3;
  }
  // For destination vector type is NOT the same as first source vector (with mask).
  // Input: (maskedoff, vector_in, vector_in/scalar_in, mask, vl, policy)
  class CapstoneBinaryABXMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_anyvector_ty, llvm_any_ty,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyint_ty,
                     LLVMMatchType<3>],
                    [ImmArg<ArgIndex<5>>, IntrNoMem]>, CapstoneVIntrinsic {
    let ScalarOperand = 2;
    let VLOperand = 4;
  }
  // For destination vector type is NOT the same as first source vector.
  // Input: (passthru, vector_in, vector_in/scalar_in, frm, vl)
  class CapstoneBinaryABXUnMaskedRoundingMode
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_anyvector_ty, llvm_any_ty,
                     llvm_anyint_ty, LLVMMatchType<3>],
                    [ImmArg<ArgIndex<3>>, IntrNoMem]>, CapstoneVIntrinsic {
    let ScalarOperand = 2;
    let VLOperand = 4;
  }
  // For destination vector type is NOT the same as first source vector (with mask).
  // Input: (maskedoff, vector_in, vector_in/scalar_in, mask, frm, vl, policy)
  class CapstoneBinaryABXMaskedRoundingMode
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_anyvector_ty, llvm_any_ty,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyint_ty,
                     LLVMMatchType<3>, LLVMMatchType<3>],
                    [ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<6>>, IntrNoMem]>, CapstoneVIntrinsic {
    let ScalarOperand = 2;
    let VLOperand = 5;
  }
  // For destination vector type is NOT the same as first source vector. The
  // second source operand must match the destination type or be an XLen scalar.
  // Input: (passthru, vector_in, vector_in/scalar_in, vl)
  class CapstoneBinaryABShiftUnMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_anyvector_ty, llvm_any_ty,
                     llvm_anyint_ty],
                    [IntrNoMem]>, CapstoneVIntrinsic {
    let VLOperand = 3;
  }
  // For destination vector type is NOT the same as first source vector (with mask).
  // The second source operand must match the destination type or be an XLen scalar.
  // Input: (maskedoff, vector_in, vector_in/scalar_in, mask, vl, policy)
  class CapstoneBinaryABShiftMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_anyvector_ty, llvm_any_ty,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyint_ty,
                     LLVMMatchType<3>],
                    [ImmArg<ArgIndex<5>>, IntrNoMem]>, CapstoneVIntrinsic {
    let VLOperand = 4;
  }
  // For binary operations with V0 as input.
  // Input: (passthru, vector_in, vector_in/scalar_in, V0, vl)
  class CapstoneBinaryWithV0
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, LLVMMatchType<0>, llvm_any_ty,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                     llvm_anyint_ty],
                    [IntrNoMem]>, CapstoneVIntrinsic {
    let ScalarOperand = 2;
    let VLOperand = 4;
  }
  // For binary operations with mask type output and V0 as input.
  // Output: (mask type output)
  // Input: (vector_in, vector_in/scalar_in, V0, vl)
  class CapstoneBinaryMOutWithV0
        :DefaultAttrsIntrinsic<[LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>],
                   [llvm_anyvector_ty, llvm_any_ty,
                    LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                    llvm_anyint_ty],
                   [IntrNoMem]>, CapstoneVIntrinsic {
    let ScalarOperand = 1;
    let VLOperand = 3;
  }
  // For binary operations with mask type output.
  // Output: (mask type output)
  // Input: (vector_in, vector_in/scalar_in, vl)
  class CapstoneBinaryMOut
        : DefaultAttrsIntrinsic<[LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>],
                    [llvm_anyvector_ty, llvm_any_ty, llvm_anyint_ty],
                    [IntrNoMem]>, CapstoneVIntrinsic {
    let ScalarOperand = 1;
    let VLOperand = 2;
  }
  // For binary operations with mask type output without mask.
  // Output: (mask type output)
  // Input: (vector_in, vector_in/scalar_in, vl)
  class CapstoneCompareUnMasked
        : DefaultAttrsIntrinsic<[LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>],
                    [llvm_anyvector_ty, llvm_any_ty, llvm_anyint_ty],
                    [IntrNoMem]>, CapstoneVIntrinsic {
    let ScalarOperand = 1;
    let VLOperand = 2;
  }
  // For binary operations with mask type output with mask.
  // Output: (mask type output)
  // Input: (maskedoff, vector_in, vector_in/scalar_in, mask, vl)
  class CapstoneCompareMasked
        : DefaultAttrsIntrinsic<[LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>],
                    [LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                     llvm_anyvector_ty, llvm_any_ty,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyint_ty],
                    [IntrNoMem]>, CapstoneVIntrinsic {
    let ScalarOperand = 2;
    let VLOperand = 4;
  }
  // For FP classify operations.
  // Output: (bit mask type output)
  // Input: (passthru, vector_in, vl)
  class CapstoneClassifyUnMasked
        : DefaultAttrsIntrinsic<[LLVMVectorOfBitcastsToInt<0>],
                    [LLVMVectorOfBitcastsToInt<0>, llvm_anyvector_ty,
                      llvm_anyint_ty],
                    [IntrNoMem]>, CapstoneVIntrinsic {
    let VLOperand = 1;
  }
  // For FP classify operations with mask.
  // Output: (bit mask type output)
  // Input: (maskedoff, vector_in, mask, vl, policy)
  class CapstoneClassifyMasked
        : DefaultAttrsIntrinsic<[LLVMVectorOfBitcastsToInt<0>],
                    [LLVMVectorOfBitcastsToInt<0>, llvm_anyvector_ty,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, 
                     llvm_anyint_ty, LLVMMatchType<1>],
                    [IntrNoMem, ImmArg<ArgIndex<4>>]>, CapstoneVIntrinsic {
    let VLOperand = 3;
  }
  // For Saturating binary operations.
  // The destination vector type is the same as first source vector.
  // Input: (passthru, vector_in, vector_in/scalar_in, vl)
  class CapstoneSaturatingBinaryAAXUnMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, LLVMMatchType<0>, llvm_any_ty,
                     llvm_anyint_ty],
                    [IntrNoMem]>, CapstoneVIntrinsic {
    let ScalarOperand = 2;
    let VLOperand = 3;
  }
  // For Saturating binary operations with rounding-mode operand
  // The destination vector type is the same as first source vector.
  // Input: (passthru, vector_in, vector_in/scalar_in, vxrm, vl)
  class CapstoneSaturatingBinaryAAXUnMaskedRoundingMode
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, LLVMMatchType<0>, llvm_any_ty,
                     llvm_anyint_ty, LLVMMatchType<2>],
                    [ImmArg<ArgIndex<3>>, IntrNoMem]>, CapstoneVIntrinsic {
    let ScalarOperand = 2;
    let VLOperand = 4;
  }
  // For Saturating binary operations with mask.
  // The destination vector type is the same as first source vector.
  // Input: (maskedoff, vector_in, vector_in/scalar_in, mask, vl, policy)
  class CapstoneSaturatingBinaryAAXMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, LLVMMatchType<0>, llvm_any_ty,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyint_ty,
                     LLVMMatchType<2>],
                    [ImmArg<ArgIndex<5>>, IntrNoMem]>, CapstoneVIntrinsic {
    let ScalarOperand = 2;
    let VLOperand = 4;
  }
  // For Saturating binary operations with mask and rounding-mode operand
  // The destination vector type is the same as first source vector.
  // Input: (maskedoff, vector_in, vector_in/scalar_in, mask, vxrm, vl, policy)
  class CapstoneSaturatingBinaryAAXMaskedRoundingMode
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, LLVMMatchType<0>, llvm_any_ty,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyint_ty,
                     LLVMMatchType<2>, LLVMMatchType<2>],
                    [ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<6>>, IntrNoMem]>, CapstoneVIntrinsic {
    let ScalarOperand = 2;
    let VLOperand = 5;
  }
  // For Saturating binary operations.
  // The destination vector type is the same as first source vector.
  // The second source operand matches the destination type or is an XLen scalar.
  // Input: (passthru, vector_in, vector_in/scalar_in, vxrm, vl)
  class CapstoneSaturatingBinaryAAShiftUnMaskedRoundingMode
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, LLVMMatchType<0>, llvm_any_ty,
                     llvm_anyint_ty, LLVMMatchType<2>],
                    [ImmArg<ArgIndex<3>>, IntrNoMem]>,
                    CapstoneVIntrinsic {
    let VLOperand = 4;
  }
  // For Saturating binary operations with mask.
  // The destination vector type is the same as first source vector.
  // The second source operand matches the destination type or is an XLen scalar.
  // Input: (maskedoff, vector_in, vector_in/scalar_in, mask, vxrm, vl, policy)
  class CapstoneSaturatingBinaryAAShiftMaskedRoundingMode
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, LLVMMatchType<0>, llvm_any_ty,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyint_ty,
                     LLVMMatchType<2>, LLVMMatchType<2>],
                    [ImmArg<ArgIndex<4>>,ImmArg<ArgIndex<6>>, IntrNoMem]>,
                    CapstoneVIntrinsic {
    let VLOperand = 5;
  }
  // For Saturating binary operations.
  // The destination vector type is NOT the same as first source vector.
  // The second source operand matches the destination type or is an XLen scalar.
  // Input: (passthru, vector_in, vector_in/scalar_in, vxrm, vl)
  class CapstoneSaturatingBinaryABShiftUnMaskedRoundingMode
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_anyvector_ty, llvm_any_ty,
                     llvm_anyint_ty, LLVMMatchType<3>],
                    [ImmArg<ArgIndex<3>>, IntrNoMem]>,
                    CapstoneVIntrinsic {
    let VLOperand = 4;
  }
  // For Saturating binary operations with mask.
  // The destination vector type is NOT the same as first source vector (with mask).
  // The second source operand matches the destination type or is an XLen scalar.
  // Input: (maskedoff, vector_in, vector_in/scalar_in, mask, vxrm, vl, policy)
  class CapstoneSaturatingBinaryABShiftMaskedRoundingMode
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_anyvector_ty, llvm_any_ty,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyint_ty,
                     LLVMMatchType<3>, LLVMMatchType<3>],
                    [ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<6>>, IntrNoMem]>, CapstoneVIntrinsic {
    let VLOperand = 5;
  }
  // Input: (vector_in, vector_in, scalar_in, vl, policy)
  class CapstoneRVVSlideUnMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, LLVMMatchType<0>, llvm_anyint_ty,
                     LLVMMatchType<1>, LLVMMatchType<1>],
                    [ImmArg<ArgIndex<4>>, IntrNoMem]>, CapstoneVIntrinsic {
    let VLOperand = 3;
  }
  // Input: (vector_in, vector_in, vector_in/scalar_in, mask, vl, policy)
  class CapstoneRVVSlideMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, LLVMMatchType<0>, llvm_anyint_ty,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                     LLVMMatchType<1>, LLVMMatchType<1>],
                    [ImmArg<ArgIndex<5>>, IntrNoMem]>, CapstoneVIntrinsic {
    let VLOperand = 4;
  }
  // UnMasked Vector Multiply-Add operations, its first operand can not be undef.
  // Input: (vector_in, vector_in/scalar, vector_in, vl, policy)
  class CapstoneTernaryAAXAUnMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_any_ty, LLVMMatchType<0>,
                     llvm_anyint_ty, LLVMMatchType<2>],
                    [ImmArg<ArgIndex<4>>, IntrNoMem]>, CapstoneVIntrinsic {
    let ScalarOperand = 1;
    let VLOperand = 3;
  }
  // Masked Vector Multiply-Add operations, its first operand can not be undef.
  // Input: (vector_in, vector_in/scalar, vector_in, mask, vl, policy
  class CapstoneTernaryAAXAMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_any_ty, LLVMMatchType<0>,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                     llvm_anyint_ty, LLVMMatchType<2>],
                    [ImmArg<ArgIndex<5>>, IntrNoMem]>, CapstoneVIntrinsic {
    let ScalarOperand = 1;
    let VLOperand = 4;
  }
  // UnMasked Vector Multiply-Add operations, its first operand can not be undef.
  // Input: (vector_in, vector_in/scalar, vector_in, frm, vl, policy)
  class CapstoneTernaryAAXAUnMaskedRoundingMode
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_any_ty, LLVMMatchType<0>,
                     llvm_anyint_ty, LLVMMatchType<2>, LLVMMatchType<2>],
                    [ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<5>>, IntrNoMem]>,
                    CapstoneVIntrinsic {
    let ScalarOperand = 1;
    let VLOperand = 4;
  }
  // Masked Vector Multiply-Add operations, its first operand can not be undef.
  // Input: (vector_in, vector_in/scalar, vector_in, mask, frm, vl, policy
  class CapstoneTernaryAAXAMaskedRoundingMode
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_any_ty, LLVMMatchType<0>,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                     llvm_anyint_ty, LLVMMatchType<2>, LLVMMatchType<2>],
                    [ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<6>>, IntrNoMem]>,
                    CapstoneVIntrinsic {
    let ScalarOperand = 1;
    let VLOperand = 5;
  }
  // UnMasked Widening Vector Multiply-Add operations, its first operand can not be undef.
  // Input: (vector_in, vector_in/scalar, vector_in, vl, policy)
  class CapstoneTernaryWideUnMasked
        : DefaultAttrsIntrinsic< [llvm_anyvector_ty],
                     [LLVMMatchType<0>, llvm_any_ty, llvm_anyvector_ty,
                      llvm_anyint_ty, LLVMMatchType<3>],
                     [ImmArg<ArgIndex<4>>, IntrNoMem] >, CapstoneVIntrinsic {
    let ScalarOperand = 1;
    let VLOperand = 3;
  }
  // Masked Widening Vector Multiply-Add operations, its first operand can not be undef.
  // Input: (vector_in, vector_in/scalar, vector_in, mask, vl, policy
  class CapstoneTernaryWideMasked
        : DefaultAttrsIntrinsic< [llvm_anyvector_ty],
                     [LLVMMatchType<0>, llvm_any_ty, llvm_anyvector_ty,
                      LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                      llvm_anyint_ty, LLVMMatchType<3>],
                     [ImmArg<ArgIndex<5>>, IntrNoMem]>, CapstoneVIntrinsic {
    let ScalarOperand = 1;
    let VLOperand = 4;
  }
  // UnMasked Widening Vector Multiply-Add operations, its first operand can not be undef.
  // Input: (vector_in, vector_in/scalar, vector_in, frm, vl, policy)
  class CapstoneTernaryWideUnMaskedRoundingMode
        : DefaultAttrsIntrinsic< [llvm_anyvector_ty],
                     [LLVMMatchType<0>, llvm_any_ty, llvm_anyvector_ty,
                      llvm_anyint_ty, LLVMMatchType<3>, LLVMMatchType<3>],
                     [ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<5>>, IntrNoMem] >,
                     CapstoneVIntrinsic {
    let ScalarOperand = 1;
    let VLOperand = 4;
  }
  // Masked Widening Vector Multiply-Add operations, its first operand can not be undef.
  // Input: (vector_in, vector_in/scalar, vector_in, mask, frm, vl, policy
  class CapstoneTernaryWideMaskedRoundingMode
        : DefaultAttrsIntrinsic< [llvm_anyvector_ty],
                     [LLVMMatchType<0>, llvm_any_ty, llvm_anyvector_ty,
                      LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                      llvm_anyint_ty, LLVMMatchType<3>, LLVMMatchType<3>],
                     [ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<6>>, IntrNoMem]>,
                     CapstoneVIntrinsic {
    let ScalarOperand = 1;
    let VLOperand = 5;
  }
  // For Reduction ternary operations.
  // For destination vector type is the same as first and third source vector.
  // Input: (vector_in, vector_in, vector_in, vl)
  class CapstoneReductionUnMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_anyvector_ty, LLVMMatchType<0>,
                     llvm_anyint_ty],
                    [IntrNoMem]>, CapstoneVIntrinsic {
    let VLOperand = 3;
  }
  // For Reduction ternary operations with mask.
  // For destination vector type is the same as first and third source vector.
  // The mask type come from second source vector.
  // Input: (maskedoff, vector_in, vector_in, vector_in, mask, vl)
  class CapstoneReductionMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_anyvector_ty, LLVMMatchType<0>,
                     LLVMScalarOrSameVectorWidth<1, llvm_i1_ty>, llvm_anyint_ty],
                    [IntrNoMem]>, CapstoneVIntrinsic {
    let VLOperand = 4;
  }
  // For Reduction ternary operations.
  // For destination vector type is the same as first and third source vector.
  // Input: (vector_in, vector_in, vector_in, frm, vl)
  class CapstoneReductionUnMaskedRoundingMode
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_anyvector_ty, LLVMMatchType<0>,
                     llvm_anyint_ty, LLVMMatchType<2>],
                    [ImmArg<ArgIndex<3>>, IntrNoMem]>, CapstoneVIntrinsic {
    let VLOperand = 4;
  }
  // For Reduction ternary operations with mask.
  // For destination vector type is the same as first and third source vector.
  // The mask type come from second source vector.
  // Input: (vector_in, vector_in, vector_in, mask, frm, vl)
  class CapstoneReductionMaskedRoundingMode
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_anyvector_ty, LLVMMatchType<0>,
                     LLVMScalarOrSameVectorWidth<1, llvm_i1_ty>, llvm_anyint_ty,
                     LLVMMatchType<2>],
                    [ImmArg<ArgIndex<4>>, IntrNoMem]>, CapstoneVIntrinsic {
    let VLOperand = 5;
  }
  // For unary operations with scalar type output without mask
  // Output: (scalar type)
  // Input: (vector_in, vl)
  class CapstoneMaskedUnarySOutUnMasked
        : DefaultAttrsIntrinsic<[LLVMMatchType<1>],
                    [llvm_anyvector_ty, llvm_anyint_ty],
                    [IntrNoMem]>, CapstoneVIntrinsic {
    let VLOperand = 1;
  }
  // For unary operations with scalar type output with mask
  // Output: (scalar type)
  // Input: (vector_in, mask, vl)
  class CapstoneMaskedUnarySOutMasked
        : DefaultAttrsIntrinsic<[LLVMMatchType<1>],
                    [llvm_anyvector_ty, LLVMMatchType<0>, llvm_anyint_ty],
                    [IntrNoMem]>, CapstoneVIntrinsic {
    let VLOperand = 2;
  }
  // For destination vector type is NOT the same as source vector.
  // Input: (passthru, vector_in, vl)
  class CapstoneUnaryABUnMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_anyvector_ty, llvm_anyint_ty],
                    [IntrNoMem]>, CapstoneVIntrinsic {
    let VLOperand = 2;
  }
  // For destination vector type is NOT the same as source vector (with mask).
  // Input: (maskedoff, vector_in, mask, vl, policy)
  class CapstoneUnaryABMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_anyvector_ty,
                     LLVMScalarOrSameVectorWidth<1, llvm_i1_ty>,
                     llvm_anyint_ty, LLVMMatchType<2>],
                    [ImmArg<ArgIndex<4>>, IntrNoMem]>, CapstoneVIntrinsic {
    let VLOperand = 3;
  }
  // For unary operations with the same vector type in/out without mask
  // Output: (vector)
  // Input: (vector_in, vl)
  class CapstoneUnaryUnMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_anyint_ty],
                    [IntrNoMem]>, CapstoneVIntrinsic {
    let VLOperand = 1;
  }
  // For mask unary operations with mask type in/out with mask
  // Output: (mask type output)
  // Input: (mask type maskedoff, mask type vector_in, mask, vl)
  class CapstoneMaskedUnaryMOutMasked
        : DefaultAttrsIntrinsic<[llvm_anyint_ty],
                    [LLVMMatchType<0>, LLVMMatchType<0>,
                     LLVMMatchType<0>, llvm_anyint_ty],
                    [IntrNoMem]>, CapstoneVIntrinsic {
    let VLOperand = 3;
  }
  // Output: (vector)
  // Input: (vl)
  class CapstoneNullaryIntrinsic
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [llvm_anyint_ty], [IntrNoMem]>, CapstoneVIntrinsic {
    let VLOperand = 1;
  }
  // Output: (vector)
  // Input: (passthru, vl)
  class CapstoneID
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_anyint_ty],
                    [IntrNoMem]>, CapstoneVIntrinsic {
    let VLOperand = 1;
  }
  // For Conversion unary operations.
  // Input: (passthru, vector_in, vl)
  class CapstoneConversionUnMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_anyvector_ty, llvm_anyint_ty],
                    [IntrNoMem]>, CapstoneVIntrinsic {
    let VLOperand = 2;
  }
  // For Conversion unary operations with mask.
  // Input: (maskedoff, vector_in, mask, vl, policy)
  class CapstoneConversionMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_anyvector_ty,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyint_ty,
                     LLVMMatchType<2>],
                    [ImmArg<ArgIndex<4>>, IntrNoMem]>, CapstoneVIntrinsic {
    let VLOperand = 3;
  }
  // For Conversion unary operations.
  // Input: (passthru, vector_in, frm, vl)
  class CapstoneConversionUnMaskedRoundingMode
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_anyvector_ty, llvm_anyint_ty,
                     LLVMMatchType<2>],
                    [ImmArg<ArgIndex<2>>, IntrNoMem]>, CapstoneVIntrinsic {
    let VLOperand = 3;
  }
  // For Conversion unary operations with mask.
  // Input: (maskedoff, vector_in, mask, frm, vl, policy)
  class CapstoneConversionMaskedRoundingMode
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_anyvector_ty,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyint_ty,
                     LLVMMatchType<2>, LLVMMatchType<2>],
                    [ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<5>>, IntrNoMem]>, CapstoneVIntrinsic {
    let VLOperand = 4;
  }

  // For unit stride segment load
  // Input: (passthru, pointer, vl, sew)
  class CapstoneUSSegLoad
        : DefaultAttrsIntrinsic<[llvm_any_ty],
                                [LLVMMatchType<0>, llvm_anyptr_ty, llvm_anyint_ty,
                                 LLVMMatchType<2>],
                    [NoCapture<ArgIndex<1>>, ImmArg<ArgIndex<3>>, IntrReadMem,
                     IntrArgMemOnly]>,
          CapstoneVIntrinsic {
    let VLOperand = 2;
  }
  // For unit stride segment load with mask
  // Input: (maskedoff, pointer, mask, vl, policy, sew)
  class CapstoneUSSegLoadMasked
        : DefaultAttrsIntrinsic<[llvm_any_ty],
                                [LLVMMatchType<0>, llvm_anyptr_ty, llvm_anyvector_ty,
                                 llvm_anyint_ty, LLVMMatchType<3>, LLVMMatchType<3>],
                    [ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>,
                     NoCapture<ArgIndex<1>>, IntrReadMem, IntrArgMemOnly]>,
          CapstoneVIntrinsic {
    let VLOperand = 3;
  }

  // For unit stride fault-only-first segment load
  // Input: (passthru, pointer, vl, sew)
  // Output: (data, vl)
  // NOTE: We model this with default memory properties since we model writing
  // VL as a side effect. IntrReadMem, IntrHasSideEffects does not work.
  class CapstoneUSSegLoadFF
        : DefaultAttrsIntrinsic<[llvm_any_ty, llvm_anyint_ty],
                    [LLVMMatchType<0>, llvm_anyptr_ty, LLVMMatchType<1>, LLVMMatchType<1>],
                    [ImmArg<ArgIndex<3>>, NoCapture<ArgIndex<1>>]>, CapstoneVIntrinsic {
    let VLOperand = 2;
  }
  // For unit stride fault-only-first segment load with mask
  // Input: (maskedoff, pointer, mask, vl, policy, sew)
  // Output: (data, vl)
  // NOTE: We model this with default memory properties since we model writing
  // VL as a side effect. IntrReadMem, IntrHasSideEffects does not work.
  class CapstoneUSSegLoadFFMasked
        : DefaultAttrsIntrinsic<[llvm_any_ty, llvm_anyint_ty],
                     [LLVMMatchType<0>, llvm_anyptr_ty, llvm_anyvector_ty,
                      LLVMMatchType<1>, LLVMMatchType<1>, LLVMMatchType<1>],
                    [ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>, NoCapture<ArgIndex<1>>]>,
                    CapstoneVIntrinsic {
    let VLOperand = 3;
  }

  // For stride segment load
  // Input: (passthru, pointer, offset, vl, sew)
  class CapstoneSSegLoad
        : DefaultAttrsIntrinsic<[llvm_any_ty],
                    [LLVMMatchType<0>, llvm_anyptr_ty, llvm_anyint_ty,
                     LLVMMatchType<2>, LLVMMatchType<2>],
                    [ImmArg<ArgIndex<4>>, NoCapture<ArgIndex<1>>, IntrReadMem]>,
          CapstoneVIntrinsic {
    let VLOperand = 3;
  }
  // For stride segment load with mask
  // Input: (maskedoff, pointer, offset, mask, vl, policy, sew)
  class CapstoneSSegLoadMasked
        : DefaultAttrsIntrinsic<[llvm_any_ty],
                                [LLVMMatchType<0>, llvm_anyptr_ty,
                                 llvm_anyint_ty, llvm_anyvector_ty,
                                 LLVMMatchType<2>, LLVMMatchType<2>,
                                 LLVMMatchType<2>],
                    [ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>,
                     NoCapture<ArgIndex<1>>, IntrReadMem]>,
                    CapstoneVIntrinsic {
    let VLOperand = 4;
  }

  // For indexed segment load
  // Input: (passthru, pointer, index, vl, sew)
  class CapstoneISegLoad
        : DefaultAttrsIntrinsic<[llvm_any_ty],
                    [LLVMMatchType<0>, llvm_anyptr_ty, llvm_anyvector_ty,
                     llvm_anyint_ty, LLVMMatchType<3>],
                    [ImmArg<ArgIndex<4>>, NoCapture<ArgIndex<1>>, IntrReadMem]>,
          CapstoneVIntrinsic {
    let VLOperand = 3;
  }
  // For indexed segment load with mask
  // Input: (maskedoff, pointer, index, mask, vl, policy, sew)
  class CapstoneISegLoadMasked
        : DefaultAttrsIntrinsic<[llvm_any_ty],
                                [LLVMMatchType<0>, llvm_anyptr_ty,
                                 llvm_anyvector_ty, llvm_anyvector_ty,
                                 llvm_anyint_ty, LLVMMatchType<4>, LLVMMatchType<4>],
                    [ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>,
                     NoCapture<ArgIndex<1>>, IntrReadMem]>, CapstoneVIntrinsic {
    let VLOperand = 4;
  }

  // For unit stride segment store
  // Input: (value, pointer, vl, sew)
  class CapstoneUSSegStore
        : DefaultAttrsIntrinsic<[],
                                [llvm_any_ty, llvm_anyptr_ty, llvm_anyint_ty,
                                 LLVMMatchType<2>],
                    [ImmArg<ArgIndex<3>>, NoCapture<ArgIndex<1>>, IntrWriteMem,
                     IntrArgMemOnly]>,
          CapstoneVIntrinsic {
    let VLOperand = 2;
  }
  // For unit stride segment store with mask
  // Input: (value, pointer, mask, vl, sew)
  class CapstoneUSSegStoreMasked
        : DefaultAttrsIntrinsic<[],
                                [llvm_any_ty, llvm_anyptr_ty,
                                 llvm_anyvector_ty, llvm_anyint_ty, LLVMMatchType<3>],
                    [ImmArg<ArgIndex<4>>, NoCapture<ArgIndex<1>>, IntrWriteMem,
                     IntrArgMemOnly]>,
          CapstoneVIntrinsic {
    let VLOperand = 3;
  }

  // For stride segment store
  // Input: (value, pointer, offset, vl, sew)
  class CapstoneSSegStore
        : DefaultAttrsIntrinsic<[],
                                [llvm_any_ty, llvm_anyptr_ty, llvm_anyint_ty,
                                 LLVMMatchType<2>, LLVMMatchType<2>],
                    [ImmArg<ArgIndex<4>>, NoCapture<ArgIndex<1>>, IntrWriteMem]>,
          CapstoneVIntrinsic {
    let VLOperand = 3;
  }
  // For stride segment store with mask
  // Input: (value, pointer, offset, mask, vl, sew)
  class CapstoneSSegStoreMasked
        : DefaultAttrsIntrinsic<[],
                                [llvm_any_ty, llvm_anyptr_ty, llvm_anyint_ty,
                                 llvm_anyvector_ty, LLVMMatchType<2>,
                                 LLVMMatchType<2>],
                    [ImmArg<ArgIndex<5>>, NoCapture<ArgIndex<1>>, IntrWriteMem]>,
          CapstoneVIntrinsic {
    let VLOperand = 4;
  }

  // For indexed segment store
  // Input: (value, pointer, offset, vl, sew)
  class CapstoneISegStore
        : DefaultAttrsIntrinsic<[],
                                [llvm_any_ty, llvm_anyptr_ty, llvm_anyvector_ty,
                                 llvm_anyint_ty, LLVMMatchType<3>],
                    [ImmArg<ArgIndex<4>>, NoCapture<ArgIndex<1>>, IntrWriteMem]>,
          CapstoneVIntrinsic {
    let VLOperand = 3;
  }
  // For indexed segment store with mask
  // Input: (value, pointer, offset, mask, vl, sew)
  class CapstoneISegStoreMasked
        : DefaultAttrsIntrinsic<[],
                                [llvm_any_ty, llvm_anyptr_ty, llvm_anyvector_ty,
                                 llvm_anyvector_ty, llvm_anyint_ty,
                                 LLVMMatchType<4>],
                    [ImmArg<ArgIndex<5>>, NoCapture<ArgIndex<1>>, IntrWriteMem]>,
          CapstoneVIntrinsic {
    let VLOperand = 4;
  }

  multiclass CapstoneUSLoad {
    def "int_capstone_" # NAME : CapstoneUSLoad;
    def "int_capstone_" # NAME # "_mask" : CapstoneUSLoadMasked;
  }
  multiclass CapstoneUSLoadFF {
    def "int_capstone_" # NAME : CapstoneUSLoadFF;
    def "int_capstone_" # NAME # "_mask" : CapstoneUSLoadFFMasked;
  }
  multiclass CapstoneSLoad {
    def "int_capstone_" # NAME : CapstoneSLoad;
    def "int_capstone_" # NAME # "_mask" : CapstoneSLoadMasked;
  }
  multiclass CapstoneILoad {
    def "int_capstone_" # NAME : CapstoneILoad;
    def "int_capstone_" # NAME # "_mask" : CapstoneILoadMasked;
  }
  multiclass CapstoneUSStore {
    def "int_capstone_" # NAME : CapstoneUSStore;
    def "int_capstone_" # NAME # "_mask" : CapstoneUSStoreMasked;
  }
  multiclass CapstoneSStore {
    def "int_capstone_" # NAME : CapstoneSStore;
    def "int_capstone_" # NAME # "_mask" : CapstoneSStoreMasked;
  }

  multiclass CapstoneIStore {
    def "int_capstone_" # NAME : CapstoneIStore;
    def "int_capstone_" # NAME # "_mask" : CapstoneIStoreMasked;
  }
  multiclass CapstoneUnaryAA {
    def "int_capstone_" # NAME : CapstoneUnaryAAUnMasked;
    def "int_capstone_" # NAME # "_mask" : CapstoneUnaryAAMasked;
  }
  multiclass CapstoneUnaryAARoundingMode {
    def "int_capstone_" # NAME : CapstoneUnaryAAUnMaskedRoundingMode;
    def "int_capstone_" # NAME # "_mask" : CapstoneUnaryAAMaskedRoundingMode;
  }
  multiclass CapstoneUnaryAB {
    def "int_capstone_" # NAME : CapstoneUnaryABUnMasked;
    def "int_capstone_" # NAME # "_mask" : CapstoneUnaryABMasked;
  }
  // AAX means the destination type(A) is the same as the first source
  // type(A). X means any type for the second source operand.
  multiclass CapstoneBinaryAAX {
    def "int_capstone_" # NAME : CapstoneBinaryAAXUnMasked;
    def "int_capstone_" # NAME # "_mask" : CapstoneBinaryAAXMasked;
  }
  multiclass CapstoneBinaryAAXRoundingMode {
    def "int_capstone_" # NAME : CapstoneBinaryAAXUnMaskedRoundingMode;
    def "int_capstone_" # NAME # "_mask" : CapstoneBinaryAAXMaskedRoundingMode;
  }
  // Like CapstoneBinaryAAX, but the second operand is used a shift amount so it
  // must be a vector or an XLen scalar.
  multiclass CapstoneBinaryAAShift {
    def "int_capstone_" # NAME : CapstoneBinaryAAShiftUnMasked;
    def "int_capstone_" # NAME # "_mask" : CapstoneBinaryAAShiftMasked;
  }
  multiclass CapstoneRGatherVV {
    def "int_capstone_" # NAME : CapstoneRGatherVVUnMasked;
    def "int_capstone_" # NAME # "_mask" : CapstoneRGatherVVMasked;
  }
  multiclass CapstoneRGatherVX {
    def "int_capstone_" # NAME : CapstoneGatherVXUnMasked;
    def "int_capstone_" # NAME # "_mask" : CapstoneGatherVXMasked;
  }
  multiclass CapstoneRGatherEI16VV {
    def "int_capstone_" # NAME : CapstoneRGatherEI16VVUnMasked;
    def "int_capstone_" # NAME # "_mask" : CapstoneRGatherEI16VVMasked;
  }
  // ABX means the destination type(A) is different from the first source
  // type(B). X means any type for the second source operand.
  multiclass CapstoneBinaryABX {
    def "int_capstone_" # NAME : CapstoneBinaryABXUnMasked;
    def "int_capstone_" # NAME # "_mask" : CapstoneBinaryABXMasked;
  }
  multiclass CapstoneBinaryABXRoundingMode {
    def "int_capstone_" # NAME : CapstoneBinaryABXUnMaskedRoundingMode;
    def "int_capstone_" # NAME # "_mask" : CapstoneBinaryABXMaskedRoundingMode;
  }
  // Like CapstoneBinaryABX, but the second operand is used a shift amount so it
  // must be a vector or an XLen scalar.
  multiclass CapstoneBinaryABShift {
    def "int_capstone_" # NAME : CapstoneBinaryABShiftUnMasked;
    def "int_capstone_" # NAME # "_mask" : CapstoneBinaryABShiftMasked;
  }
  multiclass CapstoneBinaryWithV0 {
    def "int_capstone_" # NAME : CapstoneBinaryWithV0;
  }
  multiclass CapstoneBinaryMaskOutWithV0 {
    def "int_capstone_" # NAME : CapstoneBinaryMOutWithV0;
  }
  multiclass CapstoneBinaryMaskOut {
    def "int_capstone_" # NAME : CapstoneBinaryMOut;
  }
  multiclass CapstoneSaturatingBinaryAAX {
    def "int_capstone_" # NAME : CapstoneSaturatingBinaryAAXUnMasked;
    def "int_capstone_" # NAME # "_mask" : CapstoneSaturatingBinaryAAXMasked;
  }
  multiclass CapstoneSaturatingBinaryAAXRoundingMode {
    def "int_capstone_" # NAME : CapstoneSaturatingBinaryAAXUnMaskedRoundingMode;
    def "int_capstone_" # NAME # "_mask" : CapstoneSaturatingBinaryAAXMaskedRoundingMode;
  }
  multiclass CapstoneSaturatingBinaryAAShiftRoundingMode {
    def "int_capstone_" # NAME : CapstoneSaturatingBinaryAAShiftUnMaskedRoundingMode;
    def "int_capstone_" # NAME # "_mask" : CapstoneSaturatingBinaryAAShiftMaskedRoundingMode;
  }
  multiclass CapstoneSaturatingBinaryABShiftRoundingMode {
    def "int_capstone_" # NAME : CapstoneSaturatingBinaryABShiftUnMaskedRoundingMode;
    def "int_capstone_" # NAME # "_mask" : CapstoneSaturatingBinaryABShiftMaskedRoundingMode;
  }
  multiclass CapstoneRVVSlide {
    def "int_capstone_" # NAME : CapstoneRVVSlideUnMasked;
    def "int_capstone_" # NAME # "_mask" : CapstoneRVVSlideMasked;
  }
  multiclass CapstoneTernaryAAXA {
    def "int_capstone_" # NAME : CapstoneTernaryAAXAUnMasked;
    def "int_capstone_" # NAME # "_mask" : CapstoneTernaryAAXAMasked;
  }
  multiclass CapstoneTernaryAAXARoundingMode {
    def "int_capstone_" # NAME : CapstoneTernaryAAXAUnMaskedRoundingMode;
    def "int_capstone_" # NAME # "_mask" : CapstoneTernaryAAXAMaskedRoundingMode;
  }
  multiclass CapstoneCompare {
    def "int_capstone_" # NAME : CapstoneCompareUnMasked;
    def "int_capstone_" # NAME # "_mask" : CapstoneCompareMasked;
  }
  multiclass CapstoneClassify {
    def "int_capstone_" # NAME : CapstoneClassifyUnMasked;
    def "int_capstone_" # NAME # "_mask" : CapstoneClassifyMasked;
  }
  multiclass CapstoneTernaryWide {
    def "int_capstone_" # NAME : CapstoneTernaryWideUnMasked;
    def "int_capstone_" # NAME # "_mask" : CapstoneTernaryWideMasked;
  }
  multiclass CapstoneTernaryWideRoundingMode {
    def "int_capstone_" # NAME : CapstoneTernaryWideUnMaskedRoundingMode;
    def "int_capstone_" # NAME # "_mask" : CapstoneTernaryWideMaskedRoundingMode;
  }
  multiclass CapstoneReduction {
    def "int_capstone_" # NAME : CapstoneReductionUnMasked;
    def "int_capstone_" # NAME # "_mask" : CapstoneReductionMasked;
  }
  multiclass CapstoneReductionRoundingMode {
    def "int_capstone_" # NAME : CapstoneReductionUnMaskedRoundingMode;
    def "int_capstone_" # NAME # "_mask" : CapstoneReductionMaskedRoundingMode;
  }
  multiclass CapstoneMaskedUnarySOut {
    def "int_capstone_" # NAME : CapstoneMaskedUnarySOutUnMasked;
    def "int_capstone_" # NAME # "_mask" : CapstoneMaskedUnarySOutMasked;
  }
  multiclass CapstoneMaskedUnaryMOut {
    def "int_capstone_" # NAME : CapstoneUnaryUnMasked;
    def "int_capstone_" # NAME # "_mask" : CapstoneMaskedUnaryMOutMasked;
  }
  multiclass CapstoneConversion {
    def "int_capstone_" #NAME :CapstoneConversionUnMasked;
    def "int_capstone_" # NAME # "_mask" : CapstoneConversionMasked;
  }
  multiclass CapstoneConversionRoundingMode {
    def "int_capstone_" #NAME :CapstoneConversionUnMaskedRoundingMode;
    def "int_capstone_" # NAME # "_mask" : CapstoneConversionMaskedRoundingMode;
  }
  multiclass CapstoneUSSegLoad {
    def "int_capstone_" # NAME : CapstoneUSSegLoad;
    def "int_capstone_" # NAME # "_mask" : CapstoneUSSegLoadMasked;
  }
  multiclass CapstoneUSSegLoadFF {
    def "int_capstone_" # NAME : CapstoneUSSegLoadFF;
    def "int_capstone_" # NAME # "_mask" : CapstoneUSSegLoadFFMasked;
  }
  multiclass CapstoneSSegLoad {
    def "int_capstone_" # NAME : CapstoneSSegLoad;
    def "int_capstone_" # NAME # "_mask" : CapstoneSSegLoadMasked;
  }
  multiclass CapstoneISegLoad {
    def "int_capstone_" # NAME : CapstoneISegLoad;
    def "int_capstone_" # NAME # "_mask" : CapstoneISegLoadMasked;
  }
  multiclass CapstoneUSSegStore {
    def "int_capstone_" # NAME : CapstoneUSSegStore;
    def "int_capstone_" # NAME # "_mask" : CapstoneUSSegStoreMasked;
  }
  multiclass CapstoneSSegStore {
    def "int_capstone_" # NAME : CapstoneSSegStore;
    def "int_capstone_" # NAME # "_mask" : CapstoneSSegStoreMasked;
  }
  multiclass CapstoneISegStore {
    def "int_capstone_" # NAME : CapstoneISegStore;
    def "int_capstone_" # NAME # "_mask" : CapstoneISegStoreMasked;
  }

  //==-- Intrinsics to perform vector tuple subvector insertion/extraction --=//
  def int_capstone_tuple_insert
      : DefaultAttrsIntrinsic<[llvm_any_ty],
                              [LLVMMatchType<0>, llvm_anyvector_ty, llvm_i32_ty],
                              [IntrNoMem, IntrSpeculatable, ImmArg<ArgIndex<2>>]>;

  def int_capstone_tuple_extract
      : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                              [llvm_any_ty, llvm_i32_ty],
                              [IntrNoMem, IntrSpeculatable, ImmArg<ArgIndex<1>>]>;

  defm vle : CapstoneUSLoad;
  defm vleff : CapstoneUSLoadFF;
  defm vse : CapstoneUSStore;
  defm vlse: CapstoneSLoad;
  defm vsse: CapstoneSStore;
  defm vluxei : CapstoneILoad;
  defm vloxei : CapstoneILoad;
  defm vsoxei : CapstoneIStore;
  defm vsuxei : CapstoneIStore;

  def int_capstone_vlm : CapstoneUSMLoad;
  def int_capstone_vsm : CapstoneUSStore;

  defm vadd : CapstoneBinaryAAX;
  defm vsub : CapstoneBinaryAAX;
  defm vrsub : CapstoneBinaryAAX;

  defm vwaddu : CapstoneBinaryABX;
  defm vwadd : CapstoneBinaryABX;
  defm vwaddu_w : CapstoneBinaryAAX;
  defm vwadd_w : CapstoneBinaryAAX;
  defm vwsubu : CapstoneBinaryABX;
  defm vwsub : CapstoneBinaryABX;
  defm vwsubu_w : CapstoneBinaryAAX;
  defm vwsub_w : CapstoneBinaryAAX;

  defm vzext : CapstoneUnaryAB;
  defm vsext : CapstoneUnaryAB;

  defm vadc : CapstoneBinaryWithV0;
  defm vmadc_carry_in : CapstoneBinaryMaskOutWithV0;
  defm vmadc : CapstoneBinaryMaskOut;

  defm vsbc : CapstoneBinaryWithV0;
  defm vmsbc_borrow_in : CapstoneBinaryMaskOutWithV0;
  defm vmsbc : CapstoneBinaryMaskOut;

  defm vand : CapstoneBinaryAAX;
  defm vor : CapstoneBinaryAAX;
  defm vxor : CapstoneBinaryAAX;

  defm vsll : CapstoneBinaryAAShift;
  defm vsrl : CapstoneBinaryAAShift;
  defm vsra : CapstoneBinaryAAShift;

  defm vnsrl : CapstoneBinaryABShift;
  defm vnsra : CapstoneBinaryABShift;

  defm vmseq : CapstoneCompare;
  defm vmsne : CapstoneCompare;
  defm vmsltu : CapstoneCompare;
  defm vmslt : CapstoneCompare;
  defm vmsleu : CapstoneCompare;
  defm vmsle : CapstoneCompare;
  defm vmsgtu : CapstoneCompare;
  defm vmsgt : CapstoneCompare;
  defm vmsgeu : CapstoneCompare;
  defm vmsge : CapstoneCompare;

  defm vminu : CapstoneBinaryAAX;
  defm vmin : CapstoneBinaryAAX;
  defm vmaxu : CapstoneBinaryAAX;
  defm vmax : CapstoneBinaryAAX;

  defm vmul : CapstoneBinaryAAX;
  defm vmulh : CapstoneBinaryAAX;
  defm vmulhu : CapstoneBinaryAAX;
  defm vmulhsu : CapstoneBinaryAAX;

  defm vdivu : CapstoneBinaryAAX;
  defm vdiv : CapstoneBinaryAAX;
  defm vremu : CapstoneBinaryAAX;
  defm vrem : CapstoneBinaryAAX;

  defm vwmul : CapstoneBinaryABX;
  defm vwmulu : CapstoneBinaryABX;
  defm vwmulsu : CapstoneBinaryABX;

  defm vmacc : CapstoneTernaryAAXA;
  defm vnmsac : CapstoneTernaryAAXA;
  defm vmadd : CapstoneTernaryAAXA;
  defm vnmsub : CapstoneTernaryAAXA;

  defm vwmaccu  : CapstoneTernaryWide;
  defm vwmacc   : CapstoneTernaryWide;
  defm vwmaccus : CapstoneTernaryWide;
  defm vwmaccsu : CapstoneTernaryWide;

  defm vfadd : CapstoneBinaryAAXRoundingMode;
  defm vfsub : CapstoneBinaryAAXRoundingMode;
  defm vfrsub : CapstoneBinaryAAXRoundingMode;

  defm vfwadd : CapstoneBinaryABXRoundingMode;
  defm vfwsub : CapstoneBinaryABXRoundingMode;
  defm vfwadd_w : CapstoneBinaryAAXRoundingMode;
  defm vfwsub_w : CapstoneBinaryAAXRoundingMode;

  defm vsaddu : CapstoneSaturatingBinaryAAX;
  defm vsadd : CapstoneSaturatingBinaryAAX;
  defm vssubu : CapstoneSaturatingBinaryAAX;
  defm vssub : CapstoneSaturatingBinaryAAX;

  defm vmerge : CapstoneBinaryWithV0;

  // Output: (vector)
  // Input: (passthru, vector_in, vl)
  def int_capstone_vmv_v_v : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                                                [LLVMMatchType<0>,
                                                 LLVMMatchType<0>,
                                                 llvm_anyint_ty],
                                                [IntrNoMem]>, CapstoneVIntrinsic {
    let VLOperand = 2;
  }
  // Output: (vector)
  // Input: (passthru, scalar, vl)
  def int_capstone_vmv_v_x : DefaultAttrsIntrinsic<[llvm_anyint_ty],
                                                 [LLVMMatchType<0>,
                                                  LLVMVectorElementType<0>,
                                                  llvm_anyint_ty],
                                                 [IntrNoMem]>, CapstoneVIntrinsic {
    let VLOperand = 2;
  }
  // Output: (vector)
  // Input: (passthru, scalar, vl)
  def int_capstone_vfmv_v_f : DefaultAttrsIntrinsic<[llvm_anyfloat_ty],
                                                 [LLVMMatchType<0>,
                                                  LLVMVectorElementType<0>,
                                                  llvm_anyint_ty],
                                                 [IntrNoMem]>, CapstoneVIntrinsic {
    let VLOperand = 2;
  }

  def int_capstone_vmv_x_s : DefaultAttrsIntrinsic<[LLVMVectorElementType<0>],
                                                [llvm_anyint_ty],
                                                [IntrNoMem]>, CapstoneVIntrinsic;
  def int_capstone_vmv_s_x : DefaultAttrsIntrinsic<[llvm_anyint_ty],
                                                [LLVMMatchType<0>,
                                                 LLVMVectorElementType<0>,
                                                 llvm_anyint_ty],
                                                [IntrNoMem]>, CapstoneVIntrinsic {
    let VLOperand = 2;
  }

  def int_capstone_vfmv_f_s : DefaultAttrsIntrinsic<[LLVMVectorElementType<0>],
                                                 [llvm_anyfloat_ty],
                                                 [IntrNoMem]>, CapstoneVIntrinsic;
  def int_capstone_vfmv_s_f : DefaultAttrsIntrinsic<[llvm_anyfloat_ty],
                                                 [LLVMMatchType<0>,
                                                  LLVMVectorElementType<0>,
                                                  llvm_anyint_ty],
                                                 [IntrNoMem]>, CapstoneVIntrinsic {
    let VLOperand = 2;
  }

  defm vfmul : CapstoneBinaryAAXRoundingMode;
  defm vfdiv : CapstoneBinaryAAXRoundingMode;
  defm vfrdiv : CapstoneBinaryAAXRoundingMode;

  defm vfwmul : CapstoneBinaryABXRoundingMode;

  defm vfmacc : CapstoneTernaryAAXARoundingMode;
  defm vfnmacc : CapstoneTernaryAAXARoundingMode;
  defm vfmsac : CapstoneTernaryAAXARoundingMode;
  defm vfnmsac : CapstoneTernaryAAXARoundingMode;
  defm vfmadd : CapstoneTernaryAAXARoundingMode;
  defm vfnmadd : CapstoneTernaryAAXARoundingMode;
  defm vfmsub : CapstoneTernaryAAXARoundingMode;
  defm vfnmsub : CapstoneTernaryAAXARoundingMode;

  defm vfwmacc : CapstoneTernaryWideRoundingMode;
  defm vfwmaccbf16 : CapstoneTernaryWideRoundingMode;
  defm vfwnmacc : CapstoneTernaryWideRoundingMode;
  defm vfwmsac : CapstoneTernaryWideRoundingMode;
  defm vfwnmsac : CapstoneTernaryWideRoundingMode;

  defm vfsqrt : CapstoneUnaryAARoundingMode;
  defm vfrsqrt7 : CapstoneUnaryAA;
  defm vfrec7 : CapstoneUnaryAARoundingMode;

  defm vfmin : CapstoneBinaryAAX;
  defm vfmax : CapstoneBinaryAAX;

  defm vfsgnj : CapstoneBinaryAAX;
  defm vfsgnjn : CapstoneBinaryAAX;
  defm vfsgnjx : CapstoneBinaryAAX;

  defm vfclass : CapstoneClassify;

  defm vfmerge : CapstoneBinaryWithV0;

  defm vslideup : CapstoneRVVSlide;
  defm vslidedown : CapstoneRVVSlide;

  defm vslide1up : CapstoneBinaryAAX;
  defm vslide1down : CapstoneBinaryAAX;
  defm vfslide1up : CapstoneBinaryAAX;
  defm vfslide1down : CapstoneBinaryAAX;

  defm vrgather_vv : CapstoneRGatherVV;
  defm vrgather_vx : CapstoneRGatherVX;
  defm vrgatherei16_vv : CapstoneRGatherEI16VV;

  def int_capstone_vcompress : CapstoneCompress;

  defm vaaddu : CapstoneSaturatingBinaryAAXRoundingMode;
  defm vaadd : CapstoneSaturatingBinaryAAXRoundingMode;
  defm vasubu : CapstoneSaturatingBinaryAAXRoundingMode;
  defm vasub : CapstoneSaturatingBinaryAAXRoundingMode;

  defm vsmul : CapstoneSaturatingBinaryAAXRoundingMode;

  defm vssrl : CapstoneSaturatingBinaryAAShiftRoundingMode;
  defm vssra : CapstoneSaturatingBinaryAAShiftRoundingMode;

  defm vnclipu : CapstoneSaturatingBinaryABShiftRoundingMode;
  defm vnclip : CapstoneSaturatingBinaryABShiftRoundingMode;

  defm vmfeq : CapstoneCompare;
  defm vmfne : CapstoneCompare;
  defm vmflt : CapstoneCompare;
  defm vmfle : CapstoneCompare;
  defm vmfgt : CapstoneCompare;
  defm vmfge : CapstoneCompare;

  defm vredsum : CapstoneReduction;
  defm vredand : CapstoneReduction;
  defm vredor : CapstoneReduction;
  defm vredxor : CapstoneReduction;
  defm vredminu : CapstoneReduction;
  defm vredmin : CapstoneReduction;
  defm vredmaxu : CapstoneReduction;
  defm vredmax : CapstoneReduction;

  defm vwredsumu : CapstoneReduction;
  defm vwredsum : CapstoneReduction;

  defm vfredosum : CapstoneReductionRoundingMode;
  defm vfredusum : CapstoneReductionRoundingMode;
  defm vfredmin : CapstoneReduction;
  defm vfredmax : CapstoneReduction;

  defm vfwredusum : CapstoneReductionRoundingMode;
  defm vfwredosum : CapstoneReductionRoundingMode;

  def int_capstone_vmand: CapstoneBinaryAAAUnMasked;
  def int_capstone_vmnand: CapstoneBinaryAAAUnMasked;
  def int_capstone_vmandn: CapstoneBinaryAAAUnMasked;
  def int_capstone_vmxor: CapstoneBinaryAAAUnMasked;
  def int_capstone_vmor: CapstoneBinaryAAAUnMasked;
  def int_capstone_vmnor: CapstoneBinaryAAAUnMasked;
  def int_capstone_vmorn: CapstoneBinaryAAAUnMasked;
  def int_capstone_vmxnor: CapstoneBinaryAAAUnMasked;
  def int_capstone_vmclr : CapstoneNullaryIntrinsic;
  def int_capstone_vmset : CapstoneNullaryIntrinsic;

  defm vcpop : CapstoneMaskedUnarySOut;
  defm vfirst : CapstoneMaskedUnarySOut;
  defm vmsbf : CapstoneMaskedUnaryMOut;
  defm vmsof : CapstoneMaskedUnaryMOut;
  defm vmsif : CapstoneMaskedUnaryMOut;

  defm vfcvt_xu_f_v : CapstoneConversionRoundingMode;
  defm vfcvt_x_f_v : CapstoneConversionRoundingMode;
  defm vfcvt_rtz_xu_f_v : CapstoneConversion;
  defm vfcvt_rtz_x_f_v : CapstoneConversion;
  defm vfcvt_f_xu_v : CapstoneConversionRoundingMode;
  defm vfcvt_f_x_v : CapstoneConversionRoundingMode;

  defm vfwcvt_f_xu_v : CapstoneConversion;
  defm vfwcvt_f_x_v : CapstoneConversion;
  defm vfwcvt_xu_f_v : CapstoneConversionRoundingMode;
  defm vfwcvt_x_f_v : CapstoneConversionRoundingMode;
  defm vfwcvt_rtz_xu_f_v : CapstoneConversion;
  defm vfwcvt_rtz_x_f_v : CapstoneConversion;
  defm vfwcvt_f_f_v : CapstoneConversion;
  defm vfwcvtbf16_f_f_v : CapstoneConversion;

  defm vfncvt_f_xu_w : CapstoneConversionRoundingMode;
  defm vfncvt_f_x_w : CapstoneConversionRoundingMode;
  defm vfncvt_xu_f_w : CapstoneConversionRoundingMode;
  defm vfncvt_x_f_w : CapstoneConversionRoundingMode;
  defm vfncvt_rtz_xu_f_w : CapstoneConversion;
  defm vfncvt_rtz_x_f_w : CapstoneConversion;
  defm vfncvt_f_f_w : CapstoneConversionRoundingMode;
  defm vfncvtbf16_f_f_w : CapstoneConversionRoundingMode;
  defm vfncvt_rod_f_f_w : CapstoneConversion;

  // Output: (vector)
  // Input: (passthru, mask type input, vl)
  def int_capstone_viota
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                                [LLVMMatchType<0>,
                                 LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                                 llvm_anyint_ty],
                                [IntrNoMem]>, CapstoneVIntrinsic {
    let VLOperand = 2;
  }
  // Output: (vector)
  // Input: (maskedoff, mask type vector_in, mask, vl, policy)
  def int_capstone_viota_mask
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                                [LLVMMatchType<0>,
                                 LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                                 LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                                 llvm_anyint_ty, LLVMMatchType<1>],
                                [ImmArg<ArgIndex<4>>, IntrNoMem]>, CapstoneVIntrinsic {
    let VLOperand = 3;
  }
  // Output: (vector)
  // Input: (passthru, vl)
  def int_capstone_vid : CapstoneID;

  // Output: (vector)
  // Input: (maskedoff, mask, vl, policy)
  def int_capstone_vid_mask
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                                [LLVMMatchType<0>,
                                 LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                                 llvm_anyint_ty, LLVMMatchType<1>],
                                [ImmArg<ArgIndex<3>>, IntrNoMem]>, CapstoneVIntrinsic {
    let VLOperand = 2;
  }

  foreach nf = [2, 3, 4, 5, 6, 7, 8] in {
    defm vlseg # nf : CapstoneUSSegLoad;
    defm vlseg # nf # ff : CapstoneUSSegLoadFF;
    defm vlsseg # nf : CapstoneSSegLoad;
    defm vloxseg # nf : CapstoneISegLoad;
    defm vluxseg # nf : CapstoneISegLoad;
    defm vsseg # nf : CapstoneUSSegStore;
    defm vssseg # nf : CapstoneSSegStore;
    defm vsoxseg # nf : CapstoneISegStore;
    defm vsuxseg # nf : CapstoneISegStore;
  }

  // Segment loads/stores for fixed vectors.
  // Note: we only have the masked variants because CapstoneVectorPeephole
  // would lower any instructions with all-ones mask into unmasked version
  // anyway.
  foreach nf = [2, 3, 4, 5, 6, 7, 8] in {
    // Input: (pointer, mask, vl)
    def int_capstone_seg # nf # _load_mask
          : DefaultAttrsIntrinsic<!listconcat([llvm_anyvector_ty],
                                              !listsplat(LLVMMatchType<0>,
                                              !add(nf, -1))),
                                  [llvm_anyptr_ty, LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                                   llvm_anyint_ty],
                                  [NoCapture<ArgIndex<0>>, IntrReadMem]>;

    // Input: (pointer, stride, mask, vl)
    def int_capstone_sseg # nf # _load_mask
          : DefaultAttrsIntrinsic<!listconcat([llvm_anyvector_ty],
                                              !listsplat(LLVMMatchType<0>,
                                              !add(nf, -1))),
                                  [llvm_anyptr_ty, llvm_anyint_ty,
                                   LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                                   llvm_anyint_ty],
                                  [NoCapture<ArgIndex<0>>, IntrReadMem]>;

    // Input: (<stored values>..., pointer, mask, vl)
    def int_capstone_seg # nf # _store_mask
          : DefaultAttrsIntrinsic<[],
                                  !listconcat([llvm_anyvector_ty],
                                              !listsplat(LLVMMatchType<0>,
                                                          !add(nf, -1)),
                                              [llvm_anyptr_ty, LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                                               llvm_anyint_ty]),
                                  [NoCapture<ArgIndex<nf>>, IntrWriteMem]>;

    // Input: (<stored values>..., pointer, stride, mask, vl)
    def int_capstone_sseg # nf # _store_mask
          : DefaultAttrsIntrinsic<[],
                                  !listconcat([llvm_anyvector_ty],
                                              !listsplat(LLVMMatchType<0>,
                                                          !add(nf, -1)),
                                              [llvm_anyptr_ty, llvm_anyint_ty,
                                               LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                                               llvm_anyint_ty]),
                                  [NoCapture<ArgIndex<nf>>, IntrWriteMem]>;
  }

} // TargetPrefix = "capstone"

//===----------------------------------------------------------------------===//
// Scalar Cryptography
//
// These intrinsics will lower directly into the corresponding instructions
// added by the scalar cyptography extension, if the extension is present.

let TargetPrefix = "capstone" in {

class CapstoneScalarCryptoByteSelect32
    : DefaultAttrsIntrinsic<[llvm_i32_ty],
                            [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                            [IntrNoMem, IntrSpeculatable,
                             ImmArg<ArgIndex<2>>]>;

class CapstoneScalarCryptoGprGprIntrinsic32
    : DefaultAttrsIntrinsic<[llvm_i32_ty],
                            [llvm_i32_ty, llvm_i32_ty],
                            [IntrNoMem, IntrSpeculatable]>;

class CapstoneScalarCryptoGprGprIntrinsic64
    : DefaultAttrsIntrinsic<[llvm_i64_ty],
                            [llvm_i64_ty, llvm_i64_ty],
                            [IntrNoMem, IntrSpeculatable]>;

class CapstoneScalarCryptoGprIntrinsic32
    : DefaultAttrsIntrinsic<[llvm_i32_ty],
                            [llvm_i32_ty],
                            [IntrNoMem, IntrSpeculatable]>;

class CapstoneScalarCryptoGprIntrinsic64
    : DefaultAttrsIntrinsic<[llvm_i64_ty],
                            [llvm_i64_ty],
                            [IntrNoMem, IntrSpeculatable]>;

// Zknd
def int_capstone_aes32dsi  : CapstoneScalarCryptoByteSelect32,
                          ClangBuiltin<"__builtin_capstone_aes32dsi">;
def int_capstone_aes32dsmi : CapstoneScalarCryptoByteSelect32,
                          ClangBuiltin<"__builtin_capstone_aes32dsmi">;

def int_capstone_aes64ds   : CapstoneScalarCryptoGprGprIntrinsic64,
                          ClangBuiltin<"__builtin_capstone_aes64ds">;
def int_capstone_aes64dsm  : CapstoneScalarCryptoGprGprIntrinsic64,
                          ClangBuiltin<"__builtin_capstone_aes64dsm">;

def int_capstone_aes64im   : CapstoneScalarCryptoGprIntrinsic64,
                          ClangBuiltin<"__builtin_capstone_aes64im">;

// Zkne
def int_capstone_aes32esi  : CapstoneScalarCryptoByteSelect32,
                          ClangBuiltin<"__builtin_capstone_aes32esi">;
def int_capstone_aes32esmi : CapstoneScalarCryptoByteSelect32,
                          ClangBuiltin<"__builtin_capstone_aes32esmi">;

def int_capstone_aes64es   : CapstoneScalarCryptoGprGprIntrinsic64,
                          ClangBuiltin<"__builtin_capstone_aes64es">;
def int_capstone_aes64esm  : CapstoneScalarCryptoGprGprIntrinsic64,
                          ClangBuiltin<"__builtin_capstone_aes64esm">;

// Zknd & Zkne
def int_capstone_aes64ks2  : CapstoneScalarCryptoGprGprIntrinsic64,
                          ClangBuiltin<"__builtin_capstone_aes64ks2">;
def int_capstone_aes64ks1i : DefaultAttrsIntrinsic<[llvm_i64_ty],
                                                [llvm_i64_ty, llvm_i32_ty],
                                                [IntrNoMem, IntrSpeculatable,
                                                 ImmArg<ArgIndex<1>>]>,
                          ClangBuiltin<"__builtin_capstone_aes64ks1i">;

// Zknh
def int_capstone_sha256sig0 : CapstoneScalarCryptoGprIntrinsic32;
def int_capstone_sha256sig1 : CapstoneScalarCryptoGprIntrinsic32;
def int_capstone_sha256sum0 : CapstoneScalarCryptoGprIntrinsic32;
def int_capstone_sha256sum1 : CapstoneScalarCryptoGprIntrinsic32;

def int_capstone_sha512sig0l : CapstoneScalarCryptoGprGprIntrinsic32,
                            ClangBuiltin<"__builtin_capstone_sha512sig0l">;
def int_capstone_sha512sig0h : CapstoneScalarCryptoGprGprIntrinsic32,
                            ClangBuiltin<"__builtin_capstone_sha512sig0h">;
def int_capstone_sha512sig1l : CapstoneScalarCryptoGprGprIntrinsic32,
                            ClangBuiltin<"__builtin_capstone_sha512sig1l">;
def int_capstone_sha512sig1h : CapstoneScalarCryptoGprGprIntrinsic32,
                            ClangBuiltin<"__builtin_capstone_sha512sig1h">;
def int_capstone_sha512sum0r : CapstoneScalarCryptoGprGprIntrinsic32,
                            ClangBuiltin<"__builtin_capstone_sha512sum0r">;
def int_capstone_sha512sum1r : CapstoneScalarCryptoGprGprIntrinsic32,
                            ClangBuiltin<"__builtin_capstone_sha512sum1r">;

def int_capstone_sha512sig0 : CapstoneScalarCryptoGprIntrinsic64,
                           ClangBuiltin<"__builtin_capstone_sha512sig0">;
def int_capstone_sha512sig1 : CapstoneScalarCryptoGprIntrinsic64,
                           ClangBuiltin<"__builtin_capstone_sha512sig1">;
def int_capstone_sha512sum0 : CapstoneScalarCryptoGprIntrinsic64,
                           ClangBuiltin<"__builtin_capstone_sha512sum0">;
def int_capstone_sha512sum1 : CapstoneScalarCryptoGprIntrinsic64,
                           ClangBuiltin<"__builtin_capstone_sha512sum1">;

// Zksed
def int_capstone_sm4ks      : CapstoneScalarCryptoByteSelect32;
def int_capstone_sm4ed      : CapstoneScalarCryptoByteSelect32;

// Zksh
def int_capstone_sm3p0      : CapstoneScalarCryptoGprIntrinsic32;
def int_capstone_sm3p1      : CapstoneScalarCryptoGprIntrinsic32;
} // TargetPrefix = "capstone"

//===----------------------------------------------------------------------===//
// Vector Cryptography
//
// These intrinsics will lower directly into the corresponding instructions
// added by the vector cyptography extension, if the extension is present.
let TargetPrefix = "capstone" in {
  // Zvkb
  defm vandn             : CapstoneBinaryAAX;
  defm vbrev8            : CapstoneUnaryAA;
  defm vrev8             : CapstoneUnaryAA;
  defm vrol              : CapstoneBinaryAAX;
  defm vror              : CapstoneBinaryAAX;

  // Zvbb
  defm vbrev             : CapstoneUnaryAA;
  defm vclz              : CapstoneUnaryAA;
  defm vctz              : CapstoneUnaryAA;
  defm vcpopv            : CapstoneUnaryAA;
  defm vwsll             : CapstoneBinaryABX;

  // Zvbc
  defm vclmul            : CapstoneBinaryAAX;
  defm vclmulh           : CapstoneBinaryAAX;

  // Zvkg
  def int_capstone_vghsh    : CapstoneBinaryAAXUnMaskedZvk;
  def int_capstone_vgmul_vv : CapstoneUnaryAAUnMaskedZvk<IsVS=0>;

  // Zvkned
  defm vaesdf            : CapstoneUnaryAAUnMaskedZvk;
  defm vaesdm            : CapstoneUnaryAAUnMaskedZvk;
  defm vaesef            : CapstoneUnaryAAUnMaskedZvk;
  defm vaesem            : CapstoneUnaryAAUnMaskedZvk;
  def int_capstone_vaeskf1  : CapstoneBinaryAAXUnMasked<IsVI=1>;
  def int_capstone_vaeskf2  : CapstoneBinaryAAXUnMaskedZvk<IsVI=1>;
  defm vaesz             : CapstoneUnaryAAUnMaskedZvk<HasVV=0>;

  // Zvknha or Zvknhb
  def int_capstone_vsha2ch  : CapstoneBinaryAAXUnMaskedZvk;
  def int_capstone_vsha2cl  : CapstoneBinaryAAXUnMaskedZvk;
  def int_capstone_vsha2ms  : CapstoneBinaryAAXUnMaskedZvk;

  // Zvksed
  def int_capstone_vsm4k    : CapstoneBinaryAAXUnMasked<IsVI=1>;
  defm vsm4r             : CapstoneUnaryAAUnMaskedZvk;

  // Zvksh
  def int_capstone_vsm3c    : CapstoneBinaryAAXUnMaskedZvk<IsVI=1>;
  def int_capstone_vsm3me   : CapstoneBinaryAAXUnMasked;
} // TargetPrefix = "capstone"

//===----------------------------------------------------------------------===//
// Zvqdotq - Vector quad widening 4D Dot Product
//
// 8-bit Integer dot-product instructions performing the dot product between two
// 4-element vectors of 8-bit integer elements and accumulating it into a 32-bit
// integer accumulator.
let TargetPrefix = "capstone" in {
  // We use llvm_anyvector_ty and llvm_anyint_ty for future extensibility
  // purpose but only EEW=32 is defined for now
  // Input: (vector_in, vector_in, vector_in/scalar_in, vl, policy)
  class CapstoneVQDOTUnMasked<bit HasVV>
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                                [LLVMMatchType<0>, llvm_anyvector_ty,
                                 !if(HasVV, llvm_any_ty, llvm_anyint_ty),
                                 llvm_anyint_ty, LLVMMatchType<3>],
                                [ImmArg<ArgIndex<4>>, IntrNoMem]>,
                                CapstoneVIntrinsic {
    let ScalarOperand = 2;
    let VLOperand = 3;
  }
  // Input: (vector_in, vector_in, vector_in/scalar_in, mask, vl, policy)
  class CapstoneVQDOTMasked<bit HasVV>
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                                [LLVMMatchType<0>, llvm_anyvector_ty,
                                 !if(HasVV, llvm_any_ty, llvm_anyint_ty),
                                 LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                                 llvm_anyint_ty, LLVMMatchType<3>],
                                [ImmArg<ArgIndex<5>>, IntrNoMem]>,
                                CapstoneVIntrinsic {
    let ScalarOperand = 2;
    let VLOperand = 4;
  }

  multiclass CapstoneVQDOT<bit HasVV = 1> {
    def "int_capstone_" # NAME : CapstoneVQDOTUnMasked<HasVV=HasVV>;
    def "int_capstone_" # NAME # "_mask" : CapstoneVQDOTMasked<HasVV=HasVV>;
  }

  defm vqdot   : CapstoneVQDOT;
  defm vqdotu  : CapstoneVQDOT;
  defm vqdotsu : CapstoneVQDOT;
  defm vqdotus : CapstoneVQDOT<HasVV=0>;
} // TargetPrefix = "capstone"


// Zihintpause extensions
//===----------------------------------------------------------------------===//
let TargetPrefix = "capstone" in
def int_capstone_pause : DefaultAttrsIntrinsic<[], [], [IntrNoMem, IntrHasSideEffects]>;

// Vendor extensions
//===----------------------------------------------------------------------===//
include "llvm/IR/IntrinsicsCapstoneXTHead.td"
include "llvm/IR/IntrinsicsCapstoneXsf.td"
include "llvm/IR/IntrinsicsCapstoneXCV.td"
include "llvm/IR/IntrinsicsCapstoneXAndes.td"
