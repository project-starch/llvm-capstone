; REGRESSION for a MISCOMPILE, and it needs a file of its own.
;
; 2^64-1 fits XLen as an UNSIGNED value and truncates to -1. Paired with a
; SIGN_EXTEND the widening-multiply matcher used to accept it and emit
;     mulh a0, a0, -1
; which is the high word of a * -1, for IR that asks for the high word of a
; multiply by a large POSITIVE number. For a = 1 that returns 0xFFFFFFFFFFFFFFFF
; where the answer is 0. Silently wrong code, not a crash.
;
; The constant must now mean the same number after truncation as it does as an
; i128, under the signedness the multiply is lowered with, so this shape is no
; longer foldable and is reported instead.
;
; ITS OWN FILE ON PURPOSE: report_fatal_error ends the whole compilation, so in a
; file with any other unlowerable function this one would never be reached and the
; test would pass against the unfixed compiler. It did, which is how this note came
; to be written.
;
; RUN: not llc -mtriple=capstone64 -mattr=+m < %s 2>&1 | FileCheck %s

target datalayout = "e-m:e-pf200:128:128:128:64-p:64:64-i64:64-i128:128-n32:64-S128-A200-P200-G200"

; CHECK: cannot lower a 128-bit right shift by >= XLen
define i64 @sext_times_big_unsigned_const(i64 %a) {
  %wa = sext i64 %a to i128
  %p  = mul i128 %wa, 18446744073709551615
  %hi = lshr i128 %p, 64
  %r  = trunc i128 %hi to i64
  ret i64 %r
}
