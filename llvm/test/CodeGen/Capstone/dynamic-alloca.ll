; Pinned to -capstone-shrink-stack=false (default on since 2026-07-03): this test
; checks the base dynamic-alloca lowering, not stack narrowing; the narrowed path
; is covered by cap-shrink-dynalloca.ll.
; RUN: llc -mtriple=capstone64 -capstone-shrink-stack=false -verify-machineinstrs < %s | FileCheck %s

target triple = "capstone64"

define ptr addrspace(200) @vla(i64 %n) {
; CHECK-LABEL: vla:
; CHECK: addi a0, a0, 15
; CHECK-NEXT: andi a0, a0, -16
; CHECK-NEXT: neg a0, a0
; CHECK-NEXT: cincoffset a0, sp, a0
; CHECK-NEXT: movc sp, a0
entry:
  %p = alloca i8, i64 %n, align 16, addrspace(200)
  ret ptr addrspace(200) %p
}


