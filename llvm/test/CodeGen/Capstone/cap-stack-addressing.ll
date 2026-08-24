; A stack address is a CAPABILITY, and the compiler must never form one with a
; plain integer add. `cincoffsetimm` derives the slot address from sp/s0 and keeps
; the tag and bounds; an `addi` on the same register would produce the right
; number and an unusable pointer. The store of that address must be an `stc`.
;
; Also covers a GEP whose index arrives wider than the index width: the scale
; happens at XLen and feeds cincoffset, rather than the pointer being rebuilt.
;
; These three came from i128-xlen-lowering.ll, which was deleted with the scalar
; i128 normalization rules it existed for. They are not about i128 -- they are
; about a stack slot staying a capability -- and had no other coverage.
;
; Pinned to -capstone-shrink-stack=false: this is about capability addressing,
; not stack narrowing (cap-shrink-{stack,dynalloca}.ll cover the narrowed path).
; RUN: llc -mtriple=capstone64 -mattr=+m -capstone-shrink-stack=false \
; RUN:   -verify-machineinstrs < %s | FileCheck %s

target datalayout = "e-m:e-pf200:128:128:128:64-p:64:64-i64:64-i128:128-n32:64-S128-A200-P200-G200"

@sink = addrspace(200) global ptr addrspace(200) null, align 16

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
