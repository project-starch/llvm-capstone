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
; RUN: llc -mtriple=capstone64 -mattr=+m -verify-machineinstrs < %s \
; RUN:   | FileCheck %s --implicit-check-not=__divti3 --implicit-check-not=__muloti4

target datalayout = "e-m:e-pf200:128:128:128:64-p:64:64-i64:64-i128:128-n32:64-S128-A200-P200-G200"

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
