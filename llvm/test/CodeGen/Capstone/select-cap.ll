; RUN: llc -mtriple=capstone64 -verify-machineinstrs -o - < %s | FileCheck %s

define ptr addrspace(200) @select_cap(i1 %cond,
                                      ptr addrspace(200) %a,
                                      ptr addrspace(200) %b) {
entry:
  %r = select i1 %cond, ptr addrspace(200) %a, ptr addrspace(200) %b
  ret ptr addrspace(200) %r
}

define ptr addrspace(200) @select_cap_null(i1 %cond,
                                           ptr addrspace(200) %a) {
entry:
  %r = select i1 %cond, ptr addrspace(200) %a, ptr addrspace(200) null
  ret ptr addrspace(200) %r
}

; CHECK-LABEL: select_cap:
; CHECK: andi [[COND:a[0-9]+]], a0, 1
; CHECK: movc a0, a1
; CHECK: bnez [[COND]],
; CHECK: movc a0, a2
; CHECK: cjalr zero, 0(ra)

; CHECK-LABEL: select_cap_null:
; CHECK: andi [[COND2:a[0-9]+]], a0, 1
; CHECK: movc a0, a1
; CHECK: bnez [[COND2]],
; CHECK: movc a0, zero
; CHECK: cjalr zero, 0(ra)

