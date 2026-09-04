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
; C-26 (registry): the per-function "no __divti3" checks this file used to carry
; were VACUOUS -- every function computes at i64, where a divide by 4 was never
; going to be a libcall, so the check could not fail.  The libcall property is
; now a file-level implicit-check-not, and @sdiv_nonexact_var is the control
; that proves the backend CAN emit division (a `div`): without it, "no libcall"
; would be satisfied by a backend that cannot divide at all.  The srai/srli
; halves were always controlled by @addr_logical_shift.
;
; MUTATION: drop `-mattr=+m` from the RUN line -> @sdiv_nonexact_var can no
; longer use `div` and emits `call __divdi3`: its `div` check fails first
; (performed 2026-09-04, line of the `div` check) and the implicit-check-not
; rejects the libcall that replaced it; the srai checks are unaffected, which
; shows the two properties are guarded independently.
;
; RUN: llc -mtriple=capstone64 -mattr=+m -verify-machineinstrs < %s \
; RUN:   | FileCheck %s --implicit-check-not=__divti3 --implicit-check-not=__divdi3 --implicit-check-not=__moddi3
; RUN: %llc_cap -O0 < %s -o /dev/null
; RUN: %llc_cap -O1 < %s -o /dev/null
; NOTE: reading a capability's ADDRESS is `mv rd, rs` (addi rd, rs, 0), NOT
; `lcc rd, rs, 2`. Same value -- the plain regfile slot holds the cursor
; (RTL ex_stage.sv:463-479; QEMU cap.h union aliases scalar onto bounds.cursor)
; -- but the plain read is TOTAL, whereas lcc selector 2 TRAPS on an untagged
; operand and a NULL pointer is untagged. That was C-19: `p != 0 || q != 0`
; folds to `(addr(p)|addr(q)) != 0` and the first null killed the domain.

target datalayout = "e-m:e-pf200:128:128:128:64-p:64:64-i64:64-i128:128-n32:64-S128-A200-P200-G200"

; CHECK-LABEL: ptrdiff_signed_positive:
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

; CONTROL for the libcall property: a non-exact division by a variable must be
; a real `div` -- the backend CAN divide, so when the functions above show no
; libcall it is because the exact-power-of-two case was strength-reduced, not
; because division is unavailable.  Drop -mattr=+m and this becomes a
; `call __divdi3`, which the implicit-check-not rejects.
; CHECK-LABEL: sdiv_nonexact_var:
; CHECK: div a0, a0, a1
; CHECK: cjalr zero, 0(ra)
define i64 @sdiv_nonexact_var(i64 %a, i64 %b) {
  %q = sdiv i64 %a, %b
  ret i64 %q
}
