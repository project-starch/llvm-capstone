; RUN: llc -mtriple=capstone64-unknown-elf -O2 < %s | FileCheck %s
;
; Issue C-2: "Cannot select: i128 = or / = xor" with MIXED extends.
;
; lowerScalarI128Logical computes the operation in XLen and re-extends. That is
; exact only while the i128 carrier's high half is an extension of its low half.
; Mixed extends break the invariant: for sext(a) OR zext(b) the true 128-bit high
; half is sign(a), which is NOT a function of the low-half result, so re-extending
; the narrow result under either rule would MISCOMPILE. Bailing is correct.
;
; The recoverable case, and the only one this widens to: when the sign-extended
; operand is known non-negative, its sign extension and a zero extension are the
; same bits, so both operands agree and the invariant holds. Below, the `and`
; masks make the sign bit provably zero.
;
; The genuinely unrepresentable case is NOT covered and must keep bailing -- do
; not "fix" it by picking an extension rule.

; CHECK-LABEL: mixed_or_known_nonneg:
define i128 @mixed_or_known_nonneg(i64 %a, i64 %b) {
  %m = and i64 %a, 2147483647
  %sa = sext i64 %m to i128
  %zb = zext i64 %b to i128
  %r = or i128 %sa, %zb
  ret i128 %r
}

; CHECK-LABEL: mixed_xor_known_nonneg:
define i128 @mixed_xor_known_nonneg(i64 %a, i64 %b) {
  %m = and i64 %a, 65535
  %sa = sext i64 %m to i128
  %zb = zext i64 %b to i128
  %r = xor i128 %sa, %zb
  ret i128 %r
}
