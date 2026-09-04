; RUN: llc -mtriple=capstone64 -filetype=asm -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=capstone64 -filetype=obj -verify-machineinstrs < %s -o %t.o
; RUN: llvm-readobj -r %t.o | FileCheck %s --check-prefix=OBJ

@bundle = internal addrspace(200) constant { ptr addrspace(200), ptr addrspace(200) } {
  ptr addrspace(200) @callee,
  ptr addrspace(200) @bundle
}, align 16

; CHECK-LABEL: bundle:
; CHECK:       .quad callee
; CHECK-NEXT:  .zero 8
; CHECK-NEXT:  .quad bundle
; CHECK-NEXT:  .zero 8

define dso_local ptr addrspace(200) @get_bundle() addrspace(200) {
entry:
  ret ptr addrspace(200) @bundle
}

define internal void @callee() addrspace(200) {
entry:
  ret void
}



; Object view (measured 2026-09-04): each symbol half is an R_64 relocation at
; its slot offset, the metadata halves carry none, and the module's
; __capstone_cap_init is registered as an ADD64/SUB64 pair.  Relocation type
; NAMES print as "Unknown" until C-37 (lib/Object/ELF.cpp has no EM_CAPSTONE
; case); the offsets and symbols are the pin.
; OBJ: .rela.rodata {
; OBJ-NEXT: 0x0 {{R_Capstone_64|Unknown}} callee 0x0
; OBJ-NEXT: 0x10 {{R_Capstone_64|Unknown}} bundle 0x0
; OBJ-NEXT: }
; OBJ: .rela.capstone_cap_init {
; OBJ-NEXT: 0x0 {{R_Capstone_ADD64|Unknown}} __capstone_cap_init 0x0
; OBJ-NEXT: 0x0 {{R_Capstone_SUB64|Unknown}} .L0
; OBJ-NEXT: }
