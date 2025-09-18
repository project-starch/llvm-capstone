//===- IntrinsicsCapstoneXsf.td - SiFive intrinsics -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the SiFive vendor intrinsics for Capstone.
//
//===----------------------------------------------------------------------===//

let TargetPrefix = "capstone" in {
  // Output: (vector_out)
  // Input: (bit<27-26>, bit<24-20>, scalar_in, vl) or
  class CapstoneSFCustomVC_X<bit HasSE, bit ImmScalar>
        : Intrinsic<[llvm_anyvector_ty],
                    [llvm_anyint_ty, LLVMMatchType<1>, llvm_any_ty, llvm_anyint_ty],
                    !listconcat([IntrNoMem, ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>],    // bit<27-26> and bit<24-20>
                                !if(ImmScalar, [ImmArg<ArgIndex<2>>], []),                // ScalarOperand
                                !if(HasSE, [IntrHasSideEffects], []))>,
          CapstoneVIntrinsic {
    let ScalarOperand = !cond(ImmScalar: NoScalarOperand,
                              true: 2);
    let VLOperand = 3;
  }
  // Output: ()
  // Input: (bit<27-26>, bit<24-20>, bit<11-7>, scalar_in, sew, log2lmul, vl)
  class CapstoneSFCustomVC_X_WO_Suffix<bit ImmScalar>
        : Intrinsic<[],
                    [llvm_anyint_ty, LLVMMatchType<0>, LLVMMatchType<0>,
                     llvm_any_ty, LLVMMatchType<0>, LLVMMatchType<0>, llvm_anyint_ty],
                    !listconcat([IntrNoMem, IntrHasSideEffects, ImmArg<ArgIndex<0>>,
                                 ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>,
                                 ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>],    // bit<27-26>, bit<24-20>, bit<11-7>, sew, log2lmul
                                !if(ImmScalar, [ImmArg<ArgIndex<3>>], []))>,   // ScalarOperand
          CapstoneVIntrinsic {
    let ScalarOperand = !cond(ImmScalar: NoScalarOperand,
                              true: 3);
    let VLOperand = 6;
  }
  // Output: (vector_out) or ()
  // Input: (bit<27-26>, vector_in, vector_in/scalar_in, vl) or
  //        (bit<27-26>, bit<11-7>, vector_in, vector_in/scalar_in, vl)
  class CapstoneSFCustomVC_XV<bit HasDst, bit HasSE, bit ImmScalar>
        : Intrinsic<!if(HasDst, [llvm_anyvector_ty], []),
                    !listconcat(!if(HasDst, [llvm_anyint_ty, llvm_anyvector_ty],
                                            [llvm_anyint_ty, LLVMMatchType<0>, llvm_anyvector_ty]),
                                [llvm_any_ty, llvm_anyint_ty]),
                    !listconcat([IntrNoMem, ImmArg<ArgIndex<0>>],                        // bit<27-26>
                                !if(HasDst, [], [ImmArg<ArgIndex<1>>]),                  // Vd or bit<11-7>
                                !if(ImmScalar, !if(HasDst, [ImmArg<ArgIndex<2>>],
                                                           [ImmArg<ArgIndex<3>>]), []),  // ScalarOperand
                                !if(HasSE, [IntrHasSideEffects], []))>,
          CapstoneVIntrinsic {
    let ScalarOperand = !cond(ImmScalar: NoScalarOperand,
                              HasDst: 2,
                              true: 3);
    let VLOperand = !if(HasDst, 3, 4);
  }
  // Output: (vector_out) or ()
  // Input: (bit<27-26>, passthru, vector_in, vector_in/scalar_in, vl) or
  //        (bit<27-26>, vector_in, vector_in, vector_in/scalar_in, vl)
  class CapstoneSFCustomVC_XVV<bit HasDst, bit HasSE, bit ImmScalar>
        : Intrinsic<!if(HasDst, [llvm_anyvector_ty], []),
                    !listconcat(!if(HasDst, [llvm_anyint_ty, llvm_anyvector_ty, llvm_anyvector_ty],
                                            [llvm_anyint_ty, llvm_anyvector_ty, llvm_anyvector_ty]),
                                [llvm_any_ty, llvm_anyint_ty]),
                    !listconcat([IntrNoMem, ImmArg<ArgIndex<0>>],                        // bit<27-26>
                                !if(ImmScalar, [ImmArg<ArgIndex<3>>], []),               // ScalarOperand
                                !if(HasSE, [IntrHasSideEffects], []))>,
          CapstoneVIntrinsic {
    let ScalarOperand = !if(ImmScalar, NoScalarOperand, 3);
    let VLOperand = 4;
  }
  // Output: (wvector_out) or ()
  // Input: (bit<27-26>, passthru, vector_in, vector_in/scalar_in, vl) or
  //        (bit<27-26>, wvector_in, vector_in, vector_in/scalar_in, vl)
  class CapstoneSFCustomVC_XVW<bit HasDst, bit HasSE, bit ImmScalar>
        : Intrinsic<!if(HasDst, [llvm_anyvector_ty], []),
                    !listconcat(!if(HasDst, [llvm_anyint_ty, llvm_anyvector_ty, llvm_anyvector_ty],
                                            [llvm_anyint_ty, llvm_anyvector_ty, llvm_anyvector_ty]),
                                [llvm_any_ty, llvm_anyint_ty]),
                    !listconcat([IntrNoMem, ImmArg<ArgIndex<0>>],                        // bit<27-26>
                                !if(ImmScalar, [ImmArg<ArgIndex<3>>], []),               // ScalarOperand
                                !if(HasSE, [IntrHasSideEffects], []))>,
          CapstoneVIntrinsic {
    let ScalarOperand = !if(ImmScalar, NoScalarOperand, 3);
    let VLOperand = 4;
  }

  multiclass CapstoneSFCustomVC_X<list<string> type> {
    foreach t = type in {
      defvar ImmScalar = !eq(t, "i");
      def "int_capstone_sf_vc_" # t # "_se"   : CapstoneSFCustomVC_X_WO_Suffix<ImmScalar=ImmScalar>;
      def "int_capstone_sf_vc_v_" # t # "_se" : CapstoneSFCustomVC_X<HasSE=1, ImmScalar=ImmScalar>;
      def "int_capstone_sf_vc_v_" # t         : CapstoneSFCustomVC_X<HasSE=0, ImmScalar=ImmScalar>;
    }
  }

  multiclass CapstoneSFCustomVC_XV<list<string> type> {
    foreach t = type in {
      defvar ImmScalar = !eq(t, "i");
      def "int_capstone_sf_vc_" # t # "v_se"   : CapstoneSFCustomVC_XV<HasDst=0, HasSE=1, ImmScalar=ImmScalar>;
      def "int_capstone_sf_vc_v_" # t # "v_se" : CapstoneSFCustomVC_XV<HasDst=1, HasSE=1, ImmScalar=ImmScalar>;
      def "int_capstone_sf_vc_v_" # t # "v"    : CapstoneSFCustomVC_XV<HasDst=1, HasSE=0, ImmScalar=ImmScalar>;
    }
  }

  multiclass CapstoneSFCustomVC_XVV<list<string> type> {
    foreach t = type in {
      defvar ImmScalar = !eq(t, "i");
      def "int_capstone_sf_vc_" # t # "vv_se"   : CapstoneSFCustomVC_XVV<HasDst=0, HasSE=1, ImmScalar=ImmScalar>;
      def "int_capstone_sf_vc_v_" # t # "vv_se" : CapstoneSFCustomVC_XVV<HasDst=1, HasSE=1, ImmScalar=ImmScalar>;
      def "int_capstone_sf_vc_v_" # t # "vv"    : CapstoneSFCustomVC_XVV<HasDst=1, HasSE=0, ImmScalar=ImmScalar>;
    }
  }

  multiclass CapstoneSFCustomVC_XVW<list<string> type> {
    foreach t = type in {
      defvar ImmScalar = !eq(t, "i");
      def "int_capstone_sf_vc_" # t # "vw_se"   : CapstoneSFCustomVC_XVW<HasDst=0, HasSE=1, ImmScalar=ImmScalar>;
      def "int_capstone_sf_vc_v_" # t # "vw_se" : CapstoneSFCustomVC_XVW<HasDst=1, HasSE=1, ImmScalar=ImmScalar>;
      def "int_capstone_sf_vc_v_" # t # "vw"    : CapstoneSFCustomVC_XVW<HasDst=1, HasSE=0, ImmScalar=ImmScalar>;
    }
  }

  class CapstoneSFCustomVMACC
      : DefaultAttrsIntrinsic< [llvm_anyvector_ty],
                   [LLVMMatchType<0>, llvm_anyvector_ty, llvm_anyvector_ty,
                    llvm_anyint_ty, LLVMMatchType<3>],
                   [ImmArg<ArgIndex<4>>, IntrNoMem] >, CapstoneVIntrinsic {
    let VLOperand = 3;
  }

  // Input: (passthru, vector_in, scalar_in, frm, vl)
  class CapstoneSFCustomVFNRCLIPUnMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_anyfloat_ty, LLVMVectorElementType<1>,
                     llvm_anyint_ty, LLVMMatchType<2>],
                    [ImmArg<ArgIndex<3>>, IntrNoMem]>, CapstoneVIntrinsic {
    let VLOperand = 4;
  }

  // Input: (maskedoff, vector_in, vector_in/scalar_in, mask, frm, vl, policy)
  class CapstoneSFCustomVFNRCLIPMasked
       : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                   [LLVMMatchType<0>, llvm_anyfloat_ty, LLVMVectorElementType<1>,
                    LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyint_ty,
                    LLVMMatchType<2>, LLVMMatchType<2>],
                   [ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<6>>, IntrNoMem]>, CapstoneVIntrinsic {
    let VLOperand = 5;
  }

  multiclass CapstoneSFCustomVFNRCLIP {
    def NAME : CapstoneSFCustomVFNRCLIPUnMasked;
    def NAME # "_mask" : CapstoneSFCustomVFNRCLIPMasked;
  }

  defm "" : CapstoneSFCustomVC_X<["x", "i"]>;
  defm "" : CapstoneSFCustomVC_XV<["x", "i", "v", "f"]>;
  defm "" : CapstoneSFCustomVC_XVV<["x", "i", "v", "f"]>;
  defm "" : CapstoneSFCustomVC_XVW<["x", "i", "v", "f"]>;

  // XSfvqmaccdod
  def int_capstone_sf_vqmaccu_2x8x2  : CapstoneSFCustomVMACC;
  def int_capstone_sf_vqmacc_2x8x2   : CapstoneSFCustomVMACC;
  def int_capstone_sf_vqmaccus_2x8x2 : CapstoneSFCustomVMACC;
  def int_capstone_sf_vqmaccsu_2x8x2 : CapstoneSFCustomVMACC;

  // XSfvqmaccqoq
  def int_capstone_sf_vqmaccu_4x8x4  : CapstoneSFCustomVMACC;
  def int_capstone_sf_vqmacc_4x8x4   : CapstoneSFCustomVMACC;
  def int_capstone_sf_vqmaccus_4x8x4 : CapstoneSFCustomVMACC;
  def int_capstone_sf_vqmaccsu_4x8x4 : CapstoneSFCustomVMACC;

  // XSfvfwmaccqqq
  def int_capstone_sf_vfwmacc_4x4x4 : CapstoneSFCustomVMACC;

  // XSfvfnrclipxfqf
  defm int_capstone_sf_vfnrclip_x_f_qf : CapstoneSFCustomVFNRCLIP;
  defm int_capstone_sf_vfnrclip_xu_f_qf : CapstoneSFCustomVFNRCLIP;
} // TargetPrefix = "capstone"
