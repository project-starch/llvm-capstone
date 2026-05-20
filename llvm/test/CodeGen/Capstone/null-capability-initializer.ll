; RUN: llc -mtriple=capstone64 -filetype=asm -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=capstone64 -filetype=obj -verify-machineinstrs < %s -o %t.o

target datalayout = "e-m:e-p:64:128-p200:128:128:128-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"
target triple = "capstone64-unknown-unknown-elf"

%pair = type { ptr addrspace(200), ptr addrspace(200) }

@pair = internal addrspace(200) constant %pair {
  ptr addrspace(200) null,
  ptr addrspace(200) @pair
}, align 16

; CHECK-LABEL: pair:
; CHECK:       .zero 16
; CHECK-NEXT:  .quad pair
; CHECK-NEXT:  .zero 8

define dso_local ptr addrspace(200) @get_pair() addrspace(200) {
entry:
  ret ptr addrspace(200) @pair
}

