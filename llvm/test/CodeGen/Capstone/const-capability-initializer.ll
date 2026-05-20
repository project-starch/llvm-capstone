; RUN: llc -mtriple=capstone64 -filetype=asm -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=capstone64 -filetype=obj -verify-machineinstrs < %s -o %t.o

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


