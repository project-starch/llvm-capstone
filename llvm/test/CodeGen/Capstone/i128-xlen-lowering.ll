; RUN: llc -mtriple=capstone64 -mattr=+m -verify-machineinstrs < %s | FileCheck %s
;
; CoreMark bring-up regression coverage for the scalar i128 normalization rules
; that must collapse back into the xlen domain before instruction selection.

; CHECK-LABEL: and_i128_zext_small:
; CHECK: zext.b a0, a0
; CHECK: cjalr zero, 0(ra)
define i128 @and_i128_zext_small(i64 %x) {
entry:
  %wide = zext i64 %x to i128
  %masked = and i128 %wide, 255
  ret i128 %masked
}

; CHECK-LABEL: and_i128_zext_large:
; CHECK: lui [[MASK:a[0-9]+]], 16
; CHECK-NEXT: addi [[MASK]], [[MASK]], -1
; CHECK-NEXT: and a0, a0, [[MASK]]
; CHECK: cjalr zero, 0(ra)
define i128 @and_i128_zext_large(i64 %x) {
entry:
  %wide = zext i64 %x to i128
  %masked = and i128 %wide, 65535
  ret i128 %masked
}

; CHECK-LABEL: shl_i128_zext:
; CHECK: slli a0, a0, 4
; CHECK: cjalr zero, 0(ra)
define i128 @shl_i128_zext(i64 %x) {
entry:
  %wide = zext i64 %x to i128
  %shifted = shl i128 %wide, 4
  ret i128 %shifted
}

; CHECK-LABEL: mul_i128_zext:
; CHECK: li [[C:a[0-9]+]], 37
; CHECK-NEXT: mul a0, a0, [[C]]
; CHECK: cjalr zero, 0(ra)
define i128 @mul_i128_zext(i64 %x) {
entry:
  %wide = zext i64 %x to i128
  %mul = mul i128 %wide, 37
  ret i128 %mul
}

; CHECK-LABEL: gep_scaled_i64:
; CHECK: slli a1, a1, 4
; CHECK-NEXT: cincoffset a0, a0, a1
; CHECK: cjalr zero, 0(ra)
define ptr addrspace(200) @gep_scaled_i64(ptr addrspace(200) %p, i64 %idx) {
entry:
  %idx.wide = zext i64 %idx to i128
  %scaled = shl i128 %idx.wide, 4
  %q = getelementptr i8, ptr addrspace(200) %p, i128 %scaled
  ret ptr addrspace(200) %q
}
