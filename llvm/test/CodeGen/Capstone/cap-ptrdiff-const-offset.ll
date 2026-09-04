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
; On the libcall property: every function here divides an i64 by 32, which is
; strength-reduced to `srai` and could never have been a libcall, so the
; implicit-check-not for __divti3 alone was VACUOUS (the C-26 class).  It is
; kept, widened to the 128-bit multiply helpers a 128-bit reassociation would
; reach for, and its ability to fire is guarded by the `div` control in
; ptr-diff-signed.ll on the same toolchain.  The property this file actually
; locks is the cincoffset placement, and that is controlled by @ptr_diff_plain.
;
; MUTATION: make %struct.SV 48 bytes ([6 x i64]) and divide by 48 -> the
; exact division is no longer a power of two, `srai a0, a0, 5` becomes a
; multiply-by-inverse sequence, and the srai checks fail (performed 2026-09-04).
;
; RUN: llc -mtriple=capstone64 -mattr=+m -verify-machineinstrs < %s \
; RUN:   | FileCheck %s --implicit-check-not=__divti3 --implicit-check-not=__muloti4 --implicit-check-not=__multi3
; RUN: %llc_cap -O0 < %s -o /dev/null
; RUN: %llc_cap -O1 < %s -o /dev/null

target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"

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
