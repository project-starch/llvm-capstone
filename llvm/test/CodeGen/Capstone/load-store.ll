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

; CHECK-LABEL: store_int:
; CHECK: sw a1, 0(a0)
; CHECK: cjalr zero, 0(ra)
define void @store_int(ptr addrspace(200) %ptr, i32 %val) {
entry:
  store i32 %val, ptr addrspace(200) %ptr, align 4
  ret void
}

; CHECK-LABEL: load_int_offset:
; CHECK: lw a0, 16(a0)
; CHECK: cjalr zero, 0(ra)
define i32 @load_int_offset(ptr addrspace(200) %ptr) {
entry:
  %gep = getelementptr inbounds i32, ptr addrspace(200) %ptr, i64 4
  %0 = load i32, ptr addrspace(200) %gep, align 4
  ret i32 %0
}

; CHECK-LABEL: store_long_offset:
; CHECK: sd a1, 32(a0)
; CHECK: cjalr zero, 0(ra)
define void @store_long_offset(ptr addrspace(200) %ptr, i64 %val) {
entry:
  %gep = getelementptr inbounds i64, ptr addrspace(200) %ptr, i64 4
  store i64 %val, ptr addrspace(200) %gep, align 8
  ret void
}

; CHECK-LABEL: load_int_large_offset:
; CHECK: lui [[TMP:a[0-9]+]], 1
; CHECK-NEXT: cincoffset a0, a0, [[TMP]]
; CHECK-NEXT: lw a0, 0(a0)
; CHECK: cjalr zero, 0(ra)
define i32 @load_int_large_offset(ptr addrspace(200) %ptr) {
entry:
  %gep = getelementptr inbounds i8, ptr addrspace(200) %ptr, i64 4096
  %0 = load i32, ptr addrspace(200) %gep, align 4
  ret i32 %0
}
