; A mixed zext/sext pair is not a widening multiply of one signedness, so the
; high half cannot be a single mulh or mulhu. Before the widening-multiply hook
; existed this reached the generic legaliser and asserted
;   "Unable to legalize non-vector shift"
; which is a compiler crash on valid IR. It now reports instead.
;
; This is a real limitation, not a fix: a genuine 128-bit right shift cannot be
; expressed on this target, because i128 is the capability carrier and only its
; low XLen bits are an integer. The point of the test is that the compiler says
; so rather than asserting.
;
; Plain `not`, NOT `not --crash`: a clean diagnostic exits non-zero and satisfies
; it, while the assertion this replaced aborts and does not. Without that
; distinction the test passed against the unfixed compiler too, which is to say it
; tested nothing.
; RUN: not llc -mtriple=capstone64 -mattr=+m < %s 2>&1 | FileCheck %s

target datalayout = "e-m:e-pf200:128:128:128:64-p:64:64-i64:64-i128:128-n32:64-S128-A200-P200-G200"

; CHECK: cannot lower a 128-bit right shift by >= XLen
define i64 @mixed_signedness(i64 %a, i64 %b) {
  %wa = zext i64 %a to i128
  %wb = sext i64 %b to i128
  %p = mul i128 %wa, %wb
  %hi = lshr i128 %p, 64
  %r = trunc i128 %hi to i64
  ret i64 %r
}
