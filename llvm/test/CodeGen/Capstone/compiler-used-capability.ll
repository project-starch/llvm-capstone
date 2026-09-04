; RUN: llc -mtriple=capstone64 -filetype=asm -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=capstone64 -filetype=obj -verify-machineinstrs < %s -o %t.o
; RUN: llvm-readobj -r %t.o | FileCheck %s --check-prefix=OBJ

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


; Object view (measured 2026-09-04): the two symbol halves are R_64
; relocations at their slot offsets and nothing else is relocated in .rodata --
; in particular no relocation for the llvm.compiler.used array, which must not
; reach the object.  Type NAMES print as "Unknown" until C-37.
; OBJ: .rela.rodata {
; OBJ-NEXT: 0x0 {{R_Capstone_64|Unknown}} callee 0x0
; OBJ-NEXT: 0x10 {{R_Capstone_64|Unknown}} bundle 0x0
; OBJ-NEXT: }
; MUTATION (for the CHECK-NOT above): make the array an ordinary global --
; name it @llvm.compiler.used2, drop `appending` and the llvm.metadata
; section -- and its symbol reaches the .s, so `CHECK-NOT: llvm.compiler.used`
; fires (performed 2026-09-04).
