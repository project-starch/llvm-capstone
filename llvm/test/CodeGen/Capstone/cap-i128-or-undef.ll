; RUN: llc -mtriple=capstone64 -mattr=+m -verify-machineinstrs < %s | FileCheck %s
;
; Regression: "Cannot select: i128 = or X, undef:i128".
;
; A pointer difference divided by a non-power-of-two element size --
; `(p - q) / sizeof(T)`, sizeof(T) == 48 here -- lowers to `sdiv exact i128`,
; which the middle-end rewrites into a multiply by the 128-bit modular inverse
; of the odd part (48 = 16 * 3, so a multiply by inverse-of-3). That i128
; multiply by a full-width constant does not fit lowerScalarI128Mul, so it falls
; through to the generic long-multiply expansion `or(zext(lo), shl(hi, XLEN))`.
;
; The `shl i128 hi, 64` high partial product was then mis-lowered: the scalar
; i128 shift lowering narrows to XLEN and re-extends, and for a shift amount
; >= XLEN it emitted `shl i64 X, 64`, which SelectionDAG folds to UNDEF. The
; re-extension propagated that undef, leaving `or(zext(lo), undef)` with no
; instruction-selection pattern -- and, worse, an undef where the low XLEN bits
; must be zero. The fix handles shift amounts >= XLEN up front: SHL yields a
; defined zero low half (every source bit shifts past XLEN), so the OR collapses
; to the correct low half. The whole thing must become a plain sub + shift + mul.

; CHECK-LABEL: ptrdiff_div_nonpow2:
; CHECK: sub a0, a0, a1
; CHECK: srai a0, a0, 4
; CHECK: mul a0, a0, a1
; CHECK-NOT: or {{[a-z]}}
; CHECK: cjalr zero, 0(ra)
define i64 @ptrdiff_div_nonpow2(ptr addrspace(200) %p, ptr addrspace(200) %q) {
entry:
  %pi = ptrtoint ptr addrspace(200) %p to i128
  %qi = ptrtoint ptr addrspace(200) %q to i128
  %d = sub i128 %pi, %qi
  %div = sdiv exact i128 %d, 48
  %r = trunc i128 %div to i64
  ret i64 %r
}

; The exact shape that ltable.c's findindex() hits: the quotient is truncated to
; i32 and stored, so only the low bits are demanded and the shift stays logical.
; CHECK-LABEL: ptrdiff_div_nonpow2_store:
; CHECK: sub a0, a0, a1
; CHECK: srli a0, a0, 4
; CHECK: mul a0, a0, a1
; CHECK-NOT: or {{[a-z]}}
; The store base is copied with movc, not addi: a memory base has to keep its
; tag. That copy exists because the ordinary store instructions still type their
; base operand GPR while the pointer is GPCR; retyping them removes it.
; CHECK: sw a0, 0({{a[0-9]+}})
define void @ptrdiff_div_nonpow2_store(ptr addrspace(200) %p, ptr addrspace(200) %q, ptr addrspace(200) %out) {
entry:
  %pi = ptrtoint ptr addrspace(200) %p to i128
  %qi = ptrtoint ptr addrspace(200) %q to i128
  %d = sub i128 %pi, %qi
  %div = sdiv exact i128 %d, 48
  %r = trunc i128 %div to i32
  store i32 %r, ptr addrspace(200) %out, align 4
  ret void
}
