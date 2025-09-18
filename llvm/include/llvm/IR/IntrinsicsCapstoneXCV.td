//===- IntrinsicsCapstoneXCV.td - CORE-V intrinsics -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the CORE-V vendor intrinsics for Capstone.
//
//===----------------------------------------------------------------------===//

class CapstoneScalarCoreVBitManipGprGprIntrinsic
    : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
                            [IntrNoMem, IntrSpeculatable]>;

class CapstoneScalarCoreVBitManipGprIntrinsic
    : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty],
                            [IntrNoMem, IntrSpeculatable]>;

class CapstoneScalarCoreVAluGprIntrinsic
  : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty],
                          [IntrNoMem, IntrSpeculatable]>;

class CapstoneScalarCoreVAluGprGprIntrinsic
  : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
                          [IntrNoMem, IntrSpeculatable]>;

class CapstoneScalarCoreVAluGprGprGprIntrinsic
  : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                          [IntrNoMem, IntrSpeculatable]>;

class CapstoneScalarCoreVMacGprGprGprIntrinsic
  : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [IntrNoMem, IntrWillReturn, IntrSpeculatable]>;

class CapstoneScalarCoreVMacGprGPRImmIntrinsic
    : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrWillReturn, IntrSpeculatable, ImmArg<ArgIndex<2>>]>;

class CapstoneScalarCoreVMacGprGprGprImmIntrinsic
  : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [IntrNoMem, IntrWillReturn, IntrSpeculatable, ImmArg<ArgIndex<3>>]>;

let TargetPrefix = "capstone" in {
  def int_capstone_cv_bitmanip_extract : CapstoneScalarCoreVBitManipGprGprIntrinsic;
  def int_capstone_cv_bitmanip_extractu : CapstoneScalarCoreVBitManipGprGprIntrinsic;
  def int_capstone_cv_bitmanip_bclr : CapstoneScalarCoreVBitManipGprGprIntrinsic;
  def int_capstone_cv_bitmanip_bset : CapstoneScalarCoreVBitManipGprGprIntrinsic;

  def int_capstone_cv_bitmanip_insert
    : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                            [IntrNoMem, IntrSpeculatable]>;

  def int_capstone_cv_bitmanip_clb : CapstoneScalarCoreVBitManipGprIntrinsic;

  def int_capstone_cv_bitmanip_bitrev
    : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                            [IntrNoMem, IntrSpeculatable,
                            ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

  def int_capstone_cv_alu_clip   : CapstoneScalarCoreVAluGprGprIntrinsic;
  def int_capstone_cv_alu_clipu  : CapstoneScalarCoreVAluGprGprIntrinsic;
  def int_capstone_cv_alu_addN   : CapstoneScalarCoreVAluGprGprGprIntrinsic;
  def int_capstone_cv_alu_adduN  : CapstoneScalarCoreVAluGprGprGprIntrinsic;
  def int_capstone_cv_alu_addRN  : CapstoneScalarCoreVAluGprGprGprIntrinsic;
  def int_capstone_cv_alu_adduRN : CapstoneScalarCoreVAluGprGprGprIntrinsic;
  def int_capstone_cv_alu_subN   : CapstoneScalarCoreVAluGprGprGprIntrinsic;
  def int_capstone_cv_alu_subuN  : CapstoneScalarCoreVAluGprGprGprIntrinsic;
  def int_capstone_cv_alu_subRN  : CapstoneScalarCoreVAluGprGprGprIntrinsic;
  def int_capstone_cv_alu_subuRN : CapstoneScalarCoreVAluGprGprGprIntrinsic;

  def int_capstone_cv_mac_mac : CapstoneScalarCoreVMacGprGprGprIntrinsic;
  def int_capstone_cv_mac_msu : CapstoneScalarCoreVMacGprGprGprIntrinsic;

  def int_capstone_cv_mac_muluN    : CapstoneScalarCoreVMacGprGPRImmIntrinsic;
  def int_capstone_cv_mac_mulhhuN  : CapstoneScalarCoreVMacGprGPRImmIntrinsic;
  def int_capstone_cv_mac_mulsN    : CapstoneScalarCoreVMacGprGPRImmIntrinsic;
  def int_capstone_cv_mac_mulhhsN  : CapstoneScalarCoreVMacGprGPRImmIntrinsic;
  def int_capstone_cv_mac_muluRN   : CapstoneScalarCoreVMacGprGPRImmIntrinsic;
  def int_capstone_cv_mac_mulhhuRN : CapstoneScalarCoreVMacGprGPRImmIntrinsic;
  def int_capstone_cv_mac_mulsRN   : CapstoneScalarCoreVMacGprGPRImmIntrinsic;
  def int_capstone_cv_mac_mulhhsRN : CapstoneScalarCoreVMacGprGPRImmIntrinsic;

  def int_capstone_cv_mac_macuN    : CapstoneScalarCoreVMacGprGprGprImmIntrinsic;
  def int_capstone_cv_mac_machhuN  : CapstoneScalarCoreVMacGprGprGprImmIntrinsic;
  def int_capstone_cv_mac_macsN    : CapstoneScalarCoreVMacGprGprGprImmIntrinsic;
  def int_capstone_cv_mac_machhsN  : CapstoneScalarCoreVMacGprGprGprImmIntrinsic;
  def int_capstone_cv_mac_macuRN   : CapstoneScalarCoreVMacGprGprGprImmIntrinsic;
  def int_capstone_cv_mac_machhuRN : CapstoneScalarCoreVMacGprGprGprImmIntrinsic;
  def int_capstone_cv_mac_macsRN   : CapstoneScalarCoreVMacGprGprGprImmIntrinsic;
  def int_capstone_cv_mac_machhsRN : CapstoneScalarCoreVMacGprGprGprImmIntrinsic;
} // TargetPrefix = "capstone"
