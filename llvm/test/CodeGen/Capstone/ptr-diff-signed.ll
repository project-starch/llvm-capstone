; Verify signed element scaling after subtracting two capability cursors.
; An exact signed division by a power of two must use SRA after the i128
; pointer-difference carrier is narrowed to XLEN. SRL is wrong for negative
; differences. Genuine logical shifts must remain SRL.
;
; RUN: llc -mtriple=capstone64 -mattr=+m -verify-machineinstrs < %s \
; RUN:   | FileCheck %s

target datalayout = "e-m:e-pf200:128:128:128:64-p:64:64-i64:64-i128:128-n32:64-S128-A200-P200-G200"

; CHECK-LABEL: ptrdiff_signed_positive:
; CHECK-DAG: lcc [[POS_LHS:a[0-9]+]], a0, 2
; CHECK-DAG: lcc [[POS_RHS:a[0-9]+]], a1, 2
; CHECK: sub [[POS_DIFF:a[0-9]+]], [[POS_LHS]], [[POS_RHS]]
; CHECK-NEXT: srai a0, [[POS_DIFF]], 2
; CHECK-NOT: srli
; CHECK: cjalr zero, 0(ra)
define i64 @ptrdiff_signed_positive(ptr addrspace(200) %high,
                                    ptr addrspace(200) %low) {
  %hi = ptrtoint ptr addrspace(200) %high to i128
  %lo = ptrtoint ptr addrspace(200) %low to i128
  %bytes = sub i128 %hi, %lo
  %elements = sdiv exact i128 %bytes, 4
  %result = trunc i128 %elements to i64
  ret i64 %result
}

; Reverse the subtraction operands to represent the negative-result path.
; CHECK-LABEL: ptrdiff_signed_negative:
; CHECK-DAG: lcc [[NEG_HIGH:a[0-9]+]], a0, 2
; CHECK-DAG: lcc [[NEG_LOW:a[0-9]+]], a1, 2
; CHECK: sub [[NEG_DIFF:a[0-9]+]], [[NEG_LOW]], [[NEG_HIGH]]
; CHECK-NEXT: srai a0, [[NEG_DIFF]], 2
; CHECK-NOT: srli
; CHECK: cjalr zero, 0(ra)
define i64 @ptrdiff_signed_negative(ptr addrspace(200) %high,
                                    ptr addrspace(200) %low) {
  %hi = ptrtoint ptr addrspace(200) %high to i128
  %lo = ptrtoint ptr addrspace(200) %low to i128
  %bytes = sub i128 %lo, %hi
  %elements = sdiv exact i128 %bytes, 4
  %result = trunc i128 %elements to i64
  ret i64 %result
}

; A 12-byte element uses exact division by 4 followed by multiplication by the
; modular inverse of 3. Its signed power-of-two stage must still be SRA; this
; must not collapse to a lone shift.
; CHECK-LABEL: ptrdiff_signed_size12:
; CHECK: sub [[TWELVE_DIFF:a[0-9]+]],
; CHECK: srai [[TWELVE_SCALED:a[0-9]+]], [[TWELVE_DIFF]], 2
; CHECK: mul a0, [[TWELVE_SCALED]],
; CHECK: cjalr zero, 0(ra)
define i64 @ptrdiff_signed_size12(ptr addrspace(200) %p,
                                  ptr addrspace(200) %q) {
  %pi = ptrtoint ptr addrspace(200) %p to i128
  %qi = ptrtoint ptr addrspace(200) %q to i128
  %bytes = sub i128 %pi, %qi
  %elements = sdiv exact i128 %bytes, 12
  %result = trunc i128 %elements to i64
  ret i64 %result
}

; An exact logical shift of a zero-extended scalar is not signed division and
; must remain SRL.
; CHECK-LABEL: exact_logical_shift:
; CHECK: srli a0, a0, 2
; CHECK-NOT: srai
; CHECK: cjalr zero, 0(ra)
define i64 @exact_logical_shift(i64 %value) {
  %wide = zext i64 %value to i128
  %shifted = lshr exact i128 %wide, 2
  %result = trunc i128 %shifted to i64
  ret i64 %result
}
