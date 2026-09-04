; Issue C-2: "Cannot select: i128 = or / = xor" with MIXED extends.
;
; While i128 was the capability carrier, lowerScalarI128Logical computed the
; operation in XLen and re-extended, which is exact only while the high half is
; an extension of the low half; sext(a) OR zext(b) broke that and the backend
; had to bail.  Since the c128 split i128 is a plain integer expanded to two
; i64 by the generic legalizer, so these shapes must produce the exact two-
; register form: the masked operand's high half is provably zero, so the
; result's high half is `li a1, 0` and the low half is the plain 64-bit op.
; Measured 2026-09-04 on the branch tools; the header above records the
; history that the file name still carries.
;
; RUN: llc -mtriple=capstone64-unknown-elf -O2 < %s | FileCheck %s
; RUN: %llc_cap -O0 < %s -o /dev/null
; RUN: %llc_cap -O1 < %s -o /dev/null

; CHECK-LABEL: mixed_or_known_nonneg:
; CHECK: # %bb.0:
; CHECK-NEXT: lui a2, 524288
; CHECK-NEXT: addiw a2, a2, -1
; CHECK-NEXT: and a0, a0, a2
; CHECK-NEXT: or a0, a0, a1
; CHECK-NEXT: li a1, 0
; CHECK-NEXT: cjalr zero, 0(ra)
define i128 @mixed_or_known_nonneg(i64 %a, i64 %b) {
  %m = and i64 %a, 2147483647
  %sa = sext i64 %m to i128
  %zb = zext i64 %b to i128
  %r = or i128 %sa, %zb
  ret i128 %r
}

; CHECK-LABEL: mixed_xor_known_nonneg:
; CHECK: # %bb.0:
; CHECK-NEXT: lui a2, 16
; CHECK-NEXT: addi a2, a2, -1
; CHECK-NEXT: and a0, a0, a2
; CHECK-NEXT: xor a0, a0, a1
; CHECK-NEXT: li a1, 0
; CHECK-NEXT: cjalr zero, 0(ra)
define i128 @mixed_xor_known_nonneg(i64 %a, i64 %b) {
  %m = and i64 %a, 65535
  %sa = sext i64 %m to i128
  %zb = zext i64 %b to i128
  %r = xor i128 %sa, %zb
  ret i128 %r
}
