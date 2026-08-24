; Verify signed element scaling after subtracting two capability addresses.
; An exact signed division by a power of two must use SRA. SRL is wrong for
; negative differences. Genuine logical shifts must remain SRL.
;
; The difference is computed at ADDRESS width. That is what the front end emits
; (`ptrdiff_t` is 64-bit here) and what the architecture supports: a capability
; is 128 bits, but the address it carries is 64, so pointer arithmetic that ran
; at capability width used to reach an illegal i128 and come out as a __divti3
; call. Reading the address itself costs no instruction -- ptrtoint is
; EXTRACT_SUBREG on sub_cap_addr, and X is the low half of C.
;
; RUN: llc -mtriple=capstone64 -mattr=+m -verify-machineinstrs < %s \
; RUN:   | FileCheck %s

target datalayout = "e-m:e-pf200:128:128:128:64-p:64:64-i64:64-i128:128-n32:64-S128-A200-P200-G200"

; CHECK-LABEL: ptrdiff_signed_positive:
; CHECK-NOT: __divti3
; CHECK: sub [[POS_DIFF:a[0-9]+]], a0, a1
; CHECK-NEXT: srai a0, [[POS_DIFF]], 2
; CHECK-NOT: srli
; CHECK: cjalr zero, 0(ra)
define i64 @ptrdiff_signed_positive(ptr addrspace(200) %high,
                                    ptr addrspace(200) %low) {
  %hi = ptrtoint ptr addrspace(200) %high to i64
  %lo = ptrtoint ptr addrspace(200) %low to i64
  %bytes = sub i64 %hi, %lo
  %elements = sdiv exact i64 %bytes, 4
  ret i64 %elements
}

; Reverse the subtraction operands to represent the negative-result path.
; CHECK-LABEL: ptrdiff_signed_negative:
; CHECK-NOT: __divti3
; CHECK: sub [[NEG_DIFF:a[0-9]+]], a1, a0
; CHECK-NEXT: srai a0, [[NEG_DIFF]], 2
; CHECK-NOT: srli
; CHECK: cjalr zero, 0(ra)
define i64 @ptrdiff_signed_negative(ptr addrspace(200) %high,
                                    ptr addrspace(200) %low) {
  %hi = ptrtoint ptr addrspace(200) %high to i64
  %lo = ptrtoint ptr addrspace(200) %low to i64
  %bytes = sub i64 %lo, %hi
  %elements = sdiv exact i64 %bytes, 4
  ret i64 %elements
}

; A genuine LOGICAL shift of an address must stay SRL -- the control that keeps
; the CHECK-NOTs above honest. Without it "never emit srli" would be satisfied
; by a backend that cannot emit srli at all.
; CHECK-LABEL: addr_logical_shift:
; CHECK: srli a0, a0, 2
; CHECK-NOT: srai
; CHECK: cjalr zero, 0(ra)
define i64 @addr_logical_shift(ptr addrspace(200) %p) {
  %a = ptrtoint ptr addrspace(200) %p to i64
  %s = lshr i64 %a, 2
  ret i64 %s
}
