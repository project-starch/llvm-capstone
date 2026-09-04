; The -O1 shapes that broke while i128 was the capability carrier -- C-21: a
; select of two i128 constants could not be selected; C-22: `c ? -1 : 7` lost
; its condition and returned 7; C-23: an i128 assembled from two halves kept
; only the low one.  Since the c128 split i128 is a plain integer that the
; generic legalizer expands to two i64, so each shape must produce exact
; two-register code and no libcall (no auipc anywhere).  Measured 2026-09-04 on
; the branch tools.  The repro folders are tests/compiler-repros/C2[123]-*.
;
; MUTATION: in @c23_store_assembled store %za instead of %v -> `sd a2, 8(a0)`
; becomes `sd zero, 8(a0)` and its CHECK-NEXT fails (performed 2026-09-04).
;
; RUN: llc -mtriple=capstone64 -mattr=+m -O1 -verify-machineinstrs < %s | FileCheck %s --implicit-check-not=auipc
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 -verify-machineinstrs < %s | FileCheck %s --implicit-check-not=auipc
; RUN: %llc_cap -O0 < %s -o /dev/null

; C-21: both arms constant.  -4 is all-ones above bit 1, so the high half is
; the condition mask and the low half is mask & -4.
; CHECK-LABEL: c21_select_two_consts:
; CHECK: # %bb.0:
; CHECK-NEXT: sext.w a0, a0
; CHECK-NEXT: seqz a1, a0
; CHECK-NEXT: addi a1, a1, -1
; CHECK-NEXT: andi a0, a1, -4
; CHECK-NEXT: cjalr zero, 0(ra)
define i128 @c21_select_two_consts(i32 %c) {
  %t = icmp ne i32 %c, 0
  %r = select i1 %t, i128 -4, i128 0
  ret i128 %r
}

; C-22: arms of mixed sign.  The condition must survive: high half = mask,
; low half = mask | 7.
; CHECK-LABEL: c22_mixed_sign_arms:
; CHECK: # %bb.0:
; CHECK-NEXT: sext.w a0, a0
; CHECK-NEXT: seqz a1, a0
; CHECK-NEXT: addi a1, a1, -1
; CHECK-NEXT: ori a0, a1, 7
; CHECK-NEXT: cjalr zero, 0(ra)
define i128 @c22_mixed_sign_arms(i32 %c) {
  %t = icmp ne i32 %c, 0
  %r = select i1 %t, i128 -1, i128 7
  ret i128 %r
}

; C-23: both halves of an assembled i128 reach memory.
; CHECK-LABEL: c23_store_assembled:
; CHECK: # %bb.0:
; CHECK-NEXT: sd a1, 0(a0)
; CHECK-NEXT: sd a2, 8(a0)
; CHECK-NEXT: cjalr zero, 0(ra)
define void @c23_store_assembled(ptr addrspace(200) %out, i64 %a, i64 %b) {
  %za = zext i64 %a to i128
  %zb = zext i64 %b to i128
  %sh = shl i128 %zb, 64
  %v = or i128 %za, %sh
  store i128 %v, ptr addrspace(200) %out
  ret void
}

; C-23: a full 128-bit compare reads both halves of x (a0/a1) against a/b (a2/a3).
; CHECK-LABEL: c23_eq_full:
; CHECK: # %bb.0:
; CHECK-NEXT: xor a1, a1, a3
; CHECK-NEXT: xor a0, a0, a2
; CHECK-NEXT: or a0, a0, a1
; CHECK-NEXT: seqz a0, a0
; CHECK-NEXT: cjalr zero, 0(ra)
define i32 @c23_eq_full(i128 %x, i64 %a, i64 %b) {
  %za = zext i64 %a to i128
  %zb = zext i64 %b to i128
  %sh = shl i128 %zb, 64
  %v = or i128 %za, %sh
  %e = icmp eq i128 %x, %v
  %r = zext i1 %e to i32
  ret i32 %r
}
