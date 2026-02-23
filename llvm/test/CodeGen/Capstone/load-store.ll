; RUN: llc < %s -mtriple=capstone64 | FileCheck %s

; CHECK-LABEL: store_cap:
; CHECK: stc a1, 0(a0)
; CHECK: cjalr zero, 0(ra)
define void @store_cap(ptr addrspace(200) %ptr, ptr addrspace(200) %val) {
entry:
  store ptr addrspace(200) %val, ptr addrspace(200) %ptr, align 16
  ret void
}

; CHECK-LABEL: load_cap:
; CHECK: ldc a0, 0(a0)
; CHECK: cjalr zero, 0(ra)
define ptr addrspace(200) @load_cap(ptr addrspace(200) %ptr) {
entry:
  %0 = load ptr addrspace(200), ptr addrspace(200) %ptr, align 16
  ret ptr addrspace(200) %0
}

; CHECK-LABEL: load_int:
; CHECK: lw a0, 0(a0)
; CHECK: cjalr zero, 0(ra)
define i32 @load_int(ptr addrspace(200) %ptr) {
entry:
  %0 = load i32, ptr addrspace(200) %ptr, align 4
  ret i32 %0
}