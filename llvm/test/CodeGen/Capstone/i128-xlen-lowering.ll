; RUN: llc -mtriple=capstone64 -mattr=+m -verify-machineinstrs < %s | FileCheck %s
;
; CoreMark/BEEBS bring-up regression coverage for the scalar i128 normalization rules
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

@sink = addrspace(200) global ptr addrspace(200) null, align 16

; CHECK-LABEL: stack_field_addr:
; CHECK: cincoffsetimm [[ADDR:a[0-9]+]], {{(sp|s0)}},
; CHECK-NOT: addi [[ADDR]],
; CHECK: cincoffset [[GLOBAL:a[0-9]+]], gp,
; CHECK-NEXT: delin [[GLOBAL]]
; CHECK: stc [[ADDR]],
define void @stack_field_addr() addrspace(200) {
entry:
  %buf = alloca [256 x i8], align 16, addrspace(200)
  %field = getelementptr inbounds [256 x i8], ptr addrspace(200) %buf, i64 0, i64 186
  store ptr addrspace(200) %field, ptr addrspace(200) @sink, align 16
  ret void
}

; CHECK-LABEL: stack_cap_store:
; CHECK: cincoffsetimm [[HOLDER:a[0-9]+]], {{(sp|s0)}},
; CHECK: cincoffsetimm [[BUF:a[0-9]+]], {{(sp|s0)}},
; CHECK-NOT: addi [[HOLDER]],
; CHECK: stc [[BUF]], 16([[HOLDER]])
define void @stack_cap_store() addrspace(200) {
entry:
  %holder = alloca { i32, ptr addrspace(200) }, align 16, addrspace(200)
  %buf = alloca [32 x i8], align 16, addrspace(200)
  %buf0 = getelementptr inbounds [32 x i8], ptr addrspace(200) %buf, i64 0, i64 0
  %slot = getelementptr inbounds { i32, ptr addrspace(200) }, ptr addrspace(200) %holder, i64 0, i32 1
  store ptr addrspace(200) %buf0, ptr addrspace(200) %slot, align 16
  ret void
}



; CHECK-LABEL: shl_i128_sextload:
; CHECK: lw a0, 0(a0)
; CHECK-NEXT: slli a0, a0, 2
; CHECK: cjalr zero, 0(ra)
define i128 @shl_i128_sextload(ptr addrspace(200) %p) {
entry:
  %n = load i32, ptr addrspace(200) %p, align 4
  %wide = sext i32 %n to i128
  %shifted = shl i128 %wide, 2
  ret i128 %shifted
}

; CHECK-LABEL: or_i128_zextloads:
; CHECK: lwu a0, 0(a0)
; CHECK-NEXT: lwu a1, 0(a1)
; CHECK-NEXT: or a0, a0, a1
; CHECK: cjalr zero, 0(ra)
define i128 @or_i128_zextloads(ptr addrspace(200) %p, ptr addrspace(200) %q) {
entry:
  %a = load i32, ptr addrspace(200) %p, align 4
  %b = load i32, ptr addrspace(200) %q, align 4
  %aw = zext i32 %a to i128
  %bw = zext i32 %b to i128
  %r = or i128 %aw, %bw
  ret i128 %r
}

; CHECK-LABEL: xor_i128_sextloads:
; CHECK: lw a0, 0(a0)
; CHECK-NEXT: lw a1, 0(a1)
; CHECK-NEXT: xor a0, a0, a1
; CHECK: cjalr zero, 0(ra)
define i128 @xor_i128_sextloads(ptr addrspace(200) %p, ptr addrspace(200) %q) {
entry:
  %a = load i32, ptr addrspace(200) %p, align 4
  %b = load i32, ptr addrspace(200) %q, align 4
  %aw = sext i32 %a to i128
  %bw = sext i32 %b to i128
  %r = xor i128 %aw, %bw
  ret i128 %r
}
