; RUN: llc -mtriple=capstone64 -verify-machineinstrs < %s | FileCheck %s

; CHECK-LABEL: test_imm:
; CHECK: cincoffsetimm a0, a0, 1
; CHECK: cjalr zero, 0(ra)
define ptr addrspace(200) @test_imm(ptr addrspace(200) %p) {
entry:
  %add.ptr = getelementptr inbounds i8, ptr addrspace(200) %p, i128 1
  ret ptr addrspace(200) %add.ptr
}

; CHECK-LABEL: test_reg:
; CHECK: cincoffset a0, a0, a1
; CHECK: cjalr zero, 0(ra)
define ptr addrspace(200) @test_reg(ptr addrspace(200) %p, i128 %offset) {
entry:
  %add.ptr = getelementptr inbounds i8, ptr addrspace(200) %p, i128 %offset
  ret ptr addrspace(200) %add.ptr
}

; Scaled-index GEP: index is i64, multiplied by element size → shl(sext, k).
; lowerADD must recognise shl(sext, k) as an integer offset so the capability
; ends up as rs1 in cincoffset.
; CHECK-LABEL: test_scaled_gep:
; CHECK: cincoffset a0, a0, a1
; CHECK: cjalr zero, 0(ra)
define ptr addrspace(200) @test_scaled_gep(ptr addrspace(200) %arr, i64 %i) {
  %p = getelementptr i32, ptr addrspace(200) %arr, i64 %i
  ret ptr addrspace(200) %p
}

; Two independent scaled GEPs in one function force the backend to emit two
; cincoffsets.  Both must have the capability in rs1, not the integer index.
; CHECK-LABEL: test_two_geps:
; CHECK-COUNT-2: cincoffset
define ptr addrspace(200) @test_two_geps(ptr addrspace(200) %arr, i64 %i, i64 %j) {
  %p1 = getelementptr i32, ptr addrspace(200) %arr, i64 %i
  %p2 = getelementptr i32, ptr addrspace(200) %arr, i64 %j
  ; Use both results so the compiler cannot CSE or eliminate one GEP.
  store i32 0, ptr addrspace(200) %p1
  store i32 0, ptr addrspace(200) %p2
  ret ptr addrspace(200) %p1
}

; Capability comes from a memory load (simulates spill-reload after a call).
; lowerADD must recognise ISD::LOAD i128 as a capability value so it is not
; placed in the integer-offset slot of cincoffset.
; CHECK-LABEL: test_cap_from_load:
; CHECK: cincoffset {{a[0-9]+}}, {{a[0-9]+}}, {{a[0-9]+}}
; CHECK: cjalr zero, 0(ra)
define ptr addrspace(200) @test_cap_from_load(ptr %arr_ptr, i64 %i) {
  %arr = load ptr addrspace(200), ptr %arr_ptr
  %p = getelementptr i32, ptr addrspace(200) %arr, i64 %i
  ret ptr addrspace(200) %p
}