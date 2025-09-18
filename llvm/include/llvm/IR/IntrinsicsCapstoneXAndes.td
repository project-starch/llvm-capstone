//===- IntrinsicsCapstoneXAndes.td - Andes intrinsics ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the Andes vendor intrinsics for Capstone.
//
//===----------------------------------------------------------------------===//

let TargetPrefix = "capstone" in {
  // Andes Vector BFloat16 Conversion Extension
  def int_capstone_nds_vfwcvt_s_bf16 : CapstoneConversionUnMasked;
  def int_capstone_nds_vfncvt_bf16_s : CapstoneConversionUnMaskedRoundingMode;

  // Andes Vector INT4 Load Extension
  defm nds_vln  : CapstoneUSLoad;
  defm nds_vlnu : CapstoneUSLoad;

  // Andes Vector Packed FP16 Extension
  defm nds_vfpmadt : CapstoneBinaryAAXRoundingMode;
  defm nds_vfpmadb : CapstoneBinaryAAXRoundingMode;

  // Andes Vector Dot Product Extension
  defm nds_vd4dots  : CapstoneTernaryWide;
  defm nds_vd4dotu  : CapstoneTernaryWide;
  defm nds_vd4dotsu : CapstoneTernaryWide;
}
