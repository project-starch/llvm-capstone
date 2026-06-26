; Object-granularity capability narrowing for data globals (C1).
; When -capstone-shrink-globals is on (default), materializing a SIZED data
; global narrows its capability to [&g, &g+sizeof(g)) via lcc(cursor)/add/shrink
; after the gp-relative cincoffset+delin. Functions, block addresses, constant
; pools and unsized/incomplete externs must NOT be narrowed (no known object
; size; narrowing a code pointer would break calls). The flag toggles it off.
;
; See CapstoneISelDAGToDAG.cpp selectLGA and design/capability-bounds-model.md.

; RUN: llc -mtriple=capstone64 -mattr=+m < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,SHRINK
; RUN: llc -mtriple=capstone64 -mattr=+m -capstone-shrink-globals=false < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,NOSHRINK

target datalayout = "e-m:e-pf200:128:128:128:64-p:64:64-i64:64-i128:128-n32:64-S128-A200-P200-G200"

@g = addrspace(200) global [16 x i8] zeroinitializer, align 1
@u = external addrspace(200) global [0 x i8]
declare void @ext_fn() addrspace(200)

; A sized data global is narrowed to its object when the flag is on.
; CHECK-LABEL: load_global:
; CHECK: cincoffset a0, gp, a0
; CHECK: delin a0
; SHRINK: lcc {{a[0-9]+}}, a0, 2
; SHRINK: shrink
; NOSHRINK-NOT: shrink
define i8 @load_global() addrspace(200) {
  %p = getelementptr [16 x i8], ptr addrspace(200) @g, i64 0, i64 3
  %v = load i8, ptr addrspace(200) %p
  ret i8 %v
}

; A function address has no object size and must never be narrowed.
; CHECK-LABEL: take_func:
; CHECK: cincoffset a0, gp, a0
; CHECK: delin a0
; CHECK-NOT: shrink
define ptr addrspace(200) @take_func() addrspace(200) {
  ret ptr addrspace(200) @ext_fn
}

; An unsized/incomplete extern ([0 x i8]) has unknown size and must not narrow.
; CHECK-LABEL: load_unsized:
; CHECK: cincoffset a0, gp, a0
; CHECK: delin a0
; CHECK-NOT: shrink
define i8 @load_unsized() addrspace(200) {
  %p = getelementptr [0 x i8], ptr addrspace(200) @u, i64 0, i64 3
  %v = load i8, ptr addrspace(200) %p
  ret i8 %v
}
