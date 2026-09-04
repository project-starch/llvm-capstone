; A pointer difference divided by a NON-power-of-two element size --
; `(p - q) / sizeof(T)`, sizeof(T) == 48 here. The middle end rewrites the exact
; division into a multiply by the modular inverse of the odd part, which must be
; strength-reduced inline. It must NOT become a __divti3 call: a freestanding
; domain has no compiler-rt, and the division is by a constant.
;
; The file keeps its i128 name for the history that references it, but the
; arithmetic is i64 -- which is what the front end emits, because ptrdiff_t is
; 64-bit here. It used to be i128, and the original crash it guards ("Cannot
; select: i128 = or X, undef:i128") came from that: the 128-bit multiply by a
; full-width constant fell out of the target's i128 multiply lowering and reached
; ISel unlowered. There is no such lowering any more -- i128 is an ordinary
; illegal type -- and the arithmetic never leaves XLen in the first place.
;
; On the libcall property: with the arithmetic at i64 nothing here could ever
; have produced __muloti4, so that implicit-check-not was VACUOUS (the C-26
; class).  The 128-bit multiply helpers are still the right things to forbid --
; a widening multiply is exactly where a 128-bit reassociation would reach for
; them -- so three controls follow at the end of the file: a plain i64 multiply
; and the two widening shapes (high half and low half), each pinned to the
; single `mul`/`mulhu` it must be.  They prove the backend multiplies inline;
; the negatives then mean something.
;
; MUTATION: drop `-mattr=+m` from the RUN line -> the controls can no longer
; use mul/mulhu, the multiplies become __muldi3/__multi3 calls, and both the
; pinned bodies and the implicit-check-nots fail (performed 2026-09-04).
;
; RUN: llc -mtriple=capstone64 -mattr=+m -verify-machineinstrs < %s \
; RUN:   | FileCheck %s --implicit-check-not=__divti3 --implicit-check-not=__muloti4 --implicit-check-not=__multi3 --implicit-check-not=__muldi3
; RUN: %llc_cap -O0 < %s -o /dev/null
; RUN: %llc_cap -O1 < %s -o /dev/null

target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"

; CHECK-LABEL: ptrdiff_div_nonpow2:
; CHECK: sub a0, a0, a1
; CHECK: srai a0, a0, 4
; CHECK: mul a0, a0, {{a[0-9]+}}
; CHECK: cjalr zero, 0(ra)
define i64 @ptrdiff_div_nonpow2(ptr addrspace(200) %p, ptr addrspace(200) %q) {
entry:
  %pi = ptrtoint ptr addrspace(200) %p to i64
  %qi = ptrtoint ptr addrspace(200) %q to i64
  %d = sub i64 %pi, %qi
  %div = sdiv exact i64 %d, 48
  ret i64 %div
}

; The same difference narrowed to i32 on the way into a store. Only the low half
; is demanded, so the arithmetic shift becomes a logical one -- that is a valid
; narrowing, not a lost sign.
; CHECK-LABEL: ptrdiff_div_nonpow2_store:
; CHECK: sub a0, a0, a1
; CHECK: srli a0, a0, 4
; CHECK: mul a0, a0, {{a[0-9]+}}
; CHECK: sw a0, 0(a2)
define void @ptrdiff_div_nonpow2_store(ptr addrspace(200) %p, ptr addrspace(200) %q,
                                       ptr addrspace(200) %out) {
entry:
  %pi = ptrtoint ptr addrspace(200) %p to i64
  %qi = ptrtoint ptr addrspace(200) %q to i64
  %d = sub i64 %pi, %qi
  %div = sdiv exact i64 %d, 48
  %r = trunc i64 %div to i32
  store i32 %r, ptr addrspace(200) %out, align 4
  ret void
}

; CONTROLS for the libcall negatives: the backend multiplies inline, so a
; missing helper call above means strength reduction, not an inability to
; multiply.  Each body is exactly one instruction.
; CHECK-LABEL: mul_var:
; CHECK: # %bb.0:
; CHECK-NEXT: mul a0, a0, a1
; CHECK-NEXT: cjalr zero, 0(ra)
define i64 @mul_var(i64 %a, i64 %b) {
  %m = mul i64 %a, %b
  ret i64 %m
}

; The high half of a 64x64->128 product is `mulhu`, never __multi3: this is the
; one shape in the file that actually reaches 128 bits.
; CHECK-LABEL: mulhu_widen:
; CHECK: # %bb.0:
; CHECK-NEXT: mulhu a0, a0, a1
; CHECK-NEXT: cjalr zero, 0(ra)
define i64 @mulhu_widen(i64 %a, i64 %b) {
  %za = zext i64 %a to i128
  %zb = zext i64 %b to i128
  %m = mul i128 %za, %zb
  %hi = lshr i128 %m, 64
  %r = trunc i128 %hi to i64
  ret i64 %r
}

; The low half of the same product is a plain `mul`; the widening is folded away.
; CHECK-LABEL: mullo_widen:
; CHECK: # %bb.0:
; CHECK-NEXT: mul a0, a0, a1
; CHECK-NEXT: cjalr zero, 0(ra)
define i64 @mullo_widen(i64 %a, i64 %b) {
  %za = zext i64 %a to i128
  %zb = zext i64 %b to i128
  %m = mul i128 %za, %zb
  %r = trunc i128 %m to i64
  ret i64 %r
}
