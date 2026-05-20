; RUN: llc -mtriple=capstone64 -filetype=asm -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=capstone64 -filetype=obj -verify-machineinstrs < %s -o %t.o

target datalayout = "e-m:e-p:64:128-p200:128:128:128-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"
target triple = "capstone64-unknown-unknown-elf"

%struct.bundle = type { ptr addrspace(200), ptr addrspace(200) }

@bundle = internal addrspace(200) constant %struct.bundle {
  ptr addrspace(200) @callee,
  ptr addrspace(200) @bundle
}, align 16

@llvm.compiler.used = appending addrspace(200) global [1 x ptr addrspace(200)] [
  ptr addrspace(200) @bundle
], section "llvm.metadata"

; CHECK-LABEL: get_bundle:
; CHECK:       auipc
; CHECK:       cincoffset
; CHECK-LABEL: bundle:
; CHECK:       .quad callee
; CHECK-NEXT:  .zero 8
; CHECK:       .quad bundle
; CHECK-NEXT:  .zero 8
; CHECK-NOT:   llvm.compiler.used

define internal void @callee() addrspace(200) {
entry:
  ret void
}

define dso_local ptr addrspace(200) @get_bundle() addrspace(200) {
entry:
  ret ptr addrspace(200) @bundle
}

