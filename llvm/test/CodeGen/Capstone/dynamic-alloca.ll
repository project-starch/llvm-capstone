; RUN: llc -mtriple=capstone64 -verify-machineinstrs < %s | FileCheck %s

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


