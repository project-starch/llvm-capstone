; RUN: llc -mtriple=capstone64 -verify-machineinstrs < %s | FileCheck %s

; CHECK-LABEL: check_eq:
; CHECK: beq a0, a1, .LBB
define i64 @check_eq(ptr addrspace(200) %a, ptr addrspace(200) %b) {
entry:
  %cmp = icmp eq ptr addrspace(200) %a, %b
  %conv = zext i1 %cmp to i64
  ret i64 %conv
}

; CHECK-LABEL: check_null:
; CHECK: beqz a0, .LBB
define i64 @check_null(ptr addrspace(200) %a) {
entry:
  %cmp = icmp eq ptr addrspace(200) %a, null
  %conv = zext i1 %cmp to i64
  ret i64 %conv
}