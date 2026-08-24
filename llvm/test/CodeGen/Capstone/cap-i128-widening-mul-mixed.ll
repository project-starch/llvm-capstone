; The high half of a MIXED-signedness widening multiply: zext(a) * sext(b) >> 64.
; It is not a widening multiply of one signedness, so it is not a single mulh or
; mulhu -- that was the miscompile the old i128 matcher had to be taught to
; refuse. It is mulhsu, which this target does not select, so it arrives as the
; identity that defines mulhsu:
;
;     mulhsu(b, a) == mulhu(a, b) + a * (b >> 63)
;
; The correction term is the whole point of this file. A bare `mulhu` as the
; entire answer is the wrong code that used to be possible.
;
; THIS FILE ALSO RECORDS A LIFTED LIMITATION. It used to assert that the compiler
; REPORTED here, on the grounds that "a genuine 128-bit right shift cannot be
; expressed on this target, because i128 is the capability carrier and only its
; low XLen bits are an integer". That premise is gone: a capability is c128, so
; i128 is an ordinary illegal integer type and the generic legalizer expands it
; exactly as it does on any other RV64 target. Before the change this input made
; the compiler abort with "Unable to legalize non-vector shift".
;
; Cross-checked against upstream riscv64 on the same IR, which picks the single
; `mulhsu a0, a1, a0`. Ours is four instructions for the same value -- a missed
; selection, not a wrong answer.
;
; RUN: llc -mtriple=capstone64 -mattr=+m -verify-machineinstrs < %s | FileCheck %s

target datalayout = "e-m:e-pf200:128:128:128:64-p:64:64-i64:64-i128:128-n32:64-S128-A200-P200-G200"

; CHECK-LABEL: mixed_signedness:
; CHECK-DAG: srai [[SIGN:a[0-9]+]], a1, 63
; CHECK-DAG: mulhu [[HI:a[0-9]+]], a0, a1
; CHECK: mul [[CORR:a[0-9]+]], a0, [[SIGN]]
; CHECK: add a0, [[HI]], [[CORR]]
; CHECK: cjalr zero, 0(ra)
define i64 @mixed_signedness(i64 %a, i64 %b) {
  %wa = zext i64 %a to i128
  %wb = sext i64 %b to i128
  %p = mul i128 %wa, %wb
  %hi = lshr i128 %p, 64
  %r = trunc i128 %hi to i64
  ret i64 %r
}

; The single-signedness controls, which is what keeps the sequence above readable
; as "mixed": both of these ARE one instruction, so the correction term in
; mixed_signedness is genuinely about the mixing and not about this target being
; unable to emit a widening multiply.
; CHECK-LABEL: both_unsigned:
; CHECK: mulhu a0, a0, a1
; CHECK-NEXT: cjalr zero, 0(ra)
define i64 @both_unsigned(i64 %a, i64 %b) {
  %wa = zext i64 %a to i128
  %wb = zext i64 %b to i128
  %p = mul i128 %wa, %wb
  %hi = lshr i128 %p, 64
  %r = trunc i128 %hi to i64
  ret i64 %r
}

; CHECK-LABEL: both_signed:
; CHECK: mulh a0, a0, a1
; CHECK-NEXT: cjalr zero, 0(ra)
define i64 @both_signed(i64 %a, i64 %b) {
  %wa = sext i64 %a to i128
  %wb = sext i64 %b to i128
  %p = mul i128 %wa, %wb
  %hi = ashr i128 %p, 64
  %r = trunc i128 %hi to i64
  ret i64 %r
}
