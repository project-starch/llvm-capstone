; C-43: an anonymous constant POOL is unreachable under the gp-free/gp-captable ABI and would fault
; out of bounds on silicon (gp is bounded to the capability table; an .LCPI pool is not a
; GlobalVariable, so it gets no cap-table slot). What normally PREVENTS the fault is
; useConstantPoolForLargeInts returning false under the ABI -- the constant is materialised inline.
; The C-43 guard is a backstop for the case where that avoidance is bypassed; the hidden
; -capstone-gpfree-constant-pools knob bypasses it precisely so this guard has a positive control.
; See docs/ref/ISSUES.md C-43.
;
; Arm (b) -- CONTROL proving the trigger exists: in the default ABI this expensive 64-bit constant
; is pooled, and the pool address is even formed from gp (cincoffset a0, gp, a0), which is exactly
; what breaks once gp is bounded. Without this arm, arm (a) could pass by no pool ever forming.
; RUN: llc -mtriple=capstone64 -mattr=+m < %s | FileCheck --check-prefix=POOL %s
;
; Arm (c) -- the AVOIDANCE that ships: under gp-captable alone the constant is materialised inline,
; no pool, and the guard stays silent (it is a backstop, not the mechanism).
; RUN: llc -mtriple=capstone64 -mattr=+m -capstone-gp-captable < %s 2>&1 | FileCheck --check-prefix=INLINE %s
;
; Arm (a) -- the GUARD: bypass the avoidance with the knob, and the compile fails with the located
; C-43 diagnostic instead of emitting an image that faults on silicon.
; RUN: not llc -mtriple=capstone64 -mattr=+m -capstone-gp-captable -capstone-gpfree-constant-pools < %s 2>&1 | FileCheck --check-prefix=C43 %s

; POOL: .LCPI0_0:
; POOL: .quad -3750763034362895579
; POOL-LABEL: big:
; POOL: cincoffset a0, gp, a0

; INLINE-NOT: C-43
; INLINE-NOT: .LCPI
; INLINE-LABEL: big:
; INLINE: lui a0,
; INLINE: ret

; C43: error: {{.*}}constant-pool data has no cap-table slot under the gp-free/gp-captable ABI (C-43)

define i64 @big() {
  ret i64 -3750763034362895579
}
