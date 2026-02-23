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