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

; A capability select whose arms are two *non-null* constants. GlobalMerge +
; DAGCombine can sink constant offsets into an i128 select like this; the arms
; must be rematerialized into registers for the branch-based capability select.
; This used to ICE in lowerSELECT (isa<> on a null SDValue). The arms MUST be
; materialized as plain integers (li) -- an untagged integer value in a
; capability register -- and NOT via cincoffsetimm from the null register, which
; requires a tagged source and faults at runtime.
define void @select_cap_const_arms(ptr addrspace(200) %arg, i1 %c) {
entry:
  %p = select i1 %c, ptr addrspace(200) inttoptr (i64 32 to ptr addrspace(200)),
                     ptr addrspace(200) inttoptr (i64 16 to ptr addrspace(200))
  store ptr addrspace(200) %arg, ptr addrspace(200) %p
  ret void
}

; CHECK-LABEL: select_cap_const_arms:
; CHECK-NOT: cincoffsetimm {{[a-z0-9]+}}, zero,
; CHECK-DAG: li {{a[0-9]+}}, 16
; CHECK-DAG: li {{a[0-9]+}}, 32
; CHECK: cjalr zero, 0(ra)

