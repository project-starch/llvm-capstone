; A pointer DIFFERENCE with a constant element offset on one side, e.g.
; `p - (q + 1)`. DAGCombine reassociates it to `add(sub(p, q), -elemsize)`, where
; `sub(p, q)` is a SCALAR byte count. The `cincoffset` must land on the capability
; that the offset actually belongs to, never on the scalar difference:
; cscincoffsetimm faults on an untagged rs1.
;
; The file keeps its i128 name for the history that references it; the arithmetic
; is i64, which is what the front end emits. It used to be i128, and that is where
; the original bug lived: lowerADD treated every i128 add as capability+offset and
; could not tell the two apart, because one register class held both. It cannot
; arise now -- an offset applies to a c128 or it is integer arithmetic, and the
; type says which. That lowering is gone; this file is the lock on the property.
;
; RUN: llc -mtriple=capstone64 -mattr=+m -verify-machineinstrs < %s \
; RUN:   | FileCheck %s --implicit-check-not=__divti3

target datalayout = "e-m:e-pf200:128:128:128:64-p:64:64-i64:64-i128:128-n32:64-S128-A200-P200-G200"

%struct.SV = type { [4 x i64] }   ; 32-byte element, like Lua's StackValue

; The offset is on q, so q is loaded as a CAPABILITY (ldc) and cincoffsetimm
; applies to it. p is only ever read as an address, so a plain ld suffices.
; CHECK-LABEL: ptr_diff_q1:
; CHECK-DAG: ldc [[Q:a[0-9]+]], 16(a0)
; CHECK-DAG: ld [[P:a[0-9]+]], 0(a0)
; CHECK: cincoffsetimm [[Q]], [[Q]], 32
; CHECK: sub a0, [[P]], [[Q]]
; CHECK: srai a0, a0, 5
define i64 @ptr_diff_q1(ptr addrspace(200) %slots) addrspace(200) {
entry:
  %qp = getelementptr ptr addrspace(200), ptr addrspace(200) %slots, i64 1
  %p = load ptr addrspace(200), ptr addrspace(200) %slots
  %q = load ptr addrspace(200), ptr addrspace(200) %qp
  %add.ptr = getelementptr %struct.SV, ptr addrspace(200) %q, i64 1
  %lhs = ptrtoint ptr addrspace(200) %p to i64
  %rhs = ptrtoint ptr addrspace(200) %add.ptr to i64
  %sub = sub i64 %lhs, %rhs
  %div = sdiv exact i64 %sub, 32
  ret i64 %div
}

; The mirror image: the offset is on p this time.
; CHECK-LABEL: ptr_diff_p1:
; CHECK: cincoffsetimm {{a[0-9]+}}, {{a[0-9]+}}, 32
; CHECK: sub a0,
; CHECK: srai a0, a0, 5
define i64 @ptr_diff_p1(ptr addrspace(200) %slots) addrspace(200) {
entry:
  %qp = getelementptr ptr addrspace(200), ptr addrspace(200) %slots, i64 1
  %p = load ptr addrspace(200), ptr addrspace(200) %slots
  %q = load ptr addrspace(200), ptr addrspace(200) %qp
  %add.ptr = getelementptr %struct.SV, ptr addrspace(200) %p, i64 1
  %lhs = ptrtoint ptr addrspace(200) %add.ptr to i64
  %rhs = ptrtoint ptr addrspace(200) %q to i64
  %sub = sub i64 %lhs, %rhs
  %div = sdiv exact i64 %sub, 32
  ret i64 %div
}

; The control that keeps the two above honest: with NO constant offset there is
; nothing for a cincoffset to be, so neither pointer need be loaded as a
; capability at all. If this one grew a cincoffset, the checks above would be
; matching something incidental.
; CHECK-LABEL: ptr_diff_plain:
; CHECK-NOT: cincoffset
; CHECK: sub a0,
; CHECK: srai a0, a0, 5
; CHECK: cjalr zero, 0(ra)
define i64 @ptr_diff_plain(ptr addrspace(200) %slots) addrspace(200) {
entry:
  %qp = getelementptr ptr addrspace(200), ptr addrspace(200) %slots, i64 1
  %p = load ptr addrspace(200), ptr addrspace(200) %slots
  %q = load ptr addrspace(200), ptr addrspace(200) %qp
  %lhs = ptrtoint ptr addrspace(200) %p to i64
  %rhs = ptrtoint ptr addrspace(200) %q to i64
  %sub = sub i64 %lhs, %rhs
  %div = sdiv exact i64 %sub, 32
  ret i64 %div
}
