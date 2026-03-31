; RUN: llc -mtriple=capstone64 -verify-machineinstrs -o - < %s | FileCheck %s

; CHECK-LABEL: test_ptr_pos:
; CHECK: lui [[HI:a[0-9]+]], 1
; CHECK: addi a0, [[HI]], 564
; CHECK: cjalr zero, 0(ra)
define ptr addrspace(200) @test_ptr_pos() {
entry:
  ret ptr addrspace(200) inttoptr (i128 4660 to ptr addrspace(200))
}

; CHECK-LABEL: test_ptr_neg:
; CHECK: li a0, -1
; CHECK: cjalr zero, 0(ra)
define ptr addrspace(200) @test_ptr_neg() {
entry:
  ret ptr addrspace(200) inttoptr (i128 -1 to ptr addrspace(200))
}

; CHECK-LABEL: test_gep_neg:
; CHECK: cincoffsetimm a0, a0, -1
; CHECK: cjalr zero, 0(ra)
define ptr addrspace(200) @test_gep_neg(ptr addrspace(200) %p) {
entry:
  %gep = getelementptr i8, ptr addrspace(200) %p, i128 -1
  ret ptr addrspace(200) %gep
}

; CHECK-LABEL: test_scalar_load_gep_neg:
; CHECK: lw a0, -1(a0)
; CHECK: cjalr zero, 0(ra)
define i32 @test_scalar_load_gep_neg(ptr addrspace(200) %p) {
entry:
  %gep = getelementptr i8, ptr addrspace(200) %p, i128 -1
  %v = load i32, ptr addrspace(200) %gep, align 4
  ret i32 %v
}

; CHECK-LABEL: test_cap_load_gep_neg:
; CHECK: ldc a0, -16(a0)
; CHECK: cjalr zero, 0(ra)
define ptr addrspace(200) @test_cap_load_gep_neg(ptr addrspace(200) %p) {
entry:
  %gep = getelementptr i8, ptr addrspace(200) %p, i128 -16
  %v = load ptr addrspace(200), ptr addrspace(200) %gep, align 16
  ret ptr addrspace(200) %v
}

