//===- IntrinsicsCapstone.td - Defines Capstone intrinsics -------*- tablegen -*-===//
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
// Definitions for all Capstone intrinsics.
//
// All Capstone intrinsics start with "llvm.capstone.".
let TargetPrefix = "capstone" in {
  /// Capstone_Intrinsic - Base class for the intrinsics exposed to Clang.
  class Capstone_Intrinsic<string suffix, list<LLVMType> ret_types,
                           list<LLVMType> param_types,
                           list<IntrinsicProperty> intr_props>
    : ClangBuiltin<!strconcat("__builtin_capstone_", suffix)>,
      DefaultAttrsIntrinsic<ret_types, param_types, intr_props>;

  def int_capstone_widening_smul :
    Capstone_Intrinsic<"widening_smul", [llvm_i32_ty], [llvm_i16_ty, llvm_i16_ty],
                    [IntrNoMem]>;
  def int_capstone_widening_umul :
    Capstone_Intrinsic<"widening_umul", [llvm_i32_ty], [llvm_i16_ty, llvm_i16_ty],
                    [IntrNoMem]>;
} // end TargetPrefix = "capstone"
