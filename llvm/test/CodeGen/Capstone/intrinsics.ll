; RUN: llc -mtriple=capstone64 -verify-machineinstrs < %s | FileCheck %s

declare i64 @llvm.capstone.cap.get.tag.p200(ptr addrspace(200))
declare i64 @llvm.capstone.cap.get.base.p200(ptr addrspace(200))
declare ptr addrspace(200) @llvm.capstone.cap.shrink.p200(ptr addrspace(200), i128, i128)

; CHECK-LABEL: test_tag:
; CHECK: lcc a0, a0, 0
define i64 @test_tag(ptr addrspace(200) %p) {
entry:
  %0 = call i64 @llvm.capstone.cap.get.tag.p200(ptr addrspace(200) %p)
  ret i64 %0
}

; CHECK-LABEL: test_shrink:
; CHECK: shrink a0, a1, a2
define ptr addrspace(200) @test_shrink(ptr addrspace(200) %p, i128 %base, i128 %end) {
entry:
  %0 = call ptr addrspace(200) @llvm.capstone.cap.shrink.p200(ptr addrspace(200) %p, i128 %base, i128 %end)
  ret ptr addrspace(200) %0
}