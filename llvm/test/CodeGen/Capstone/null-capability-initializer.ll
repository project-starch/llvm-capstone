; RUN: llc -mtriple=capstone64 -filetype=asm -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=capstone64 -filetype=obj -verify-machineinstrs < %s -o %t.o
; RUN: llvm-readobj -r %t.o | FileCheck %s --check-prefix=OBJ
; RUN: %llc_cap -O0 < %s -o /dev/null
; RUN: %llc_cap -O1 < %s -o /dev/null

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


; Object view (measured 2026-09-04): the null slot at .rodata+0 carries NO
; relocation -- the first and only .rela.rodata entry is the @pair half at
; 0x10.  Type NAMES print as "Unknown" until C-37.
; MUTATION: replace the null with `@pair` -> a relocation at 0x0 appears first
; and the 0x10 line fails (performed 2026-09-04).
; OBJ: .rela.rodata {
; OBJ-NEXT: 0x10 {{R_Capstone_64|Unknown}} pair 0x0
; OBJ-NEXT: }
