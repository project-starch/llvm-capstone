; A direct call's target capability must be NON-LINEAR (C-46).
;
; The target is gp moved to the callee. A bare `cincoffset rd, gp, off` is
; LINEAR and pure, so MachineCSE merges the targets of several calls to one
; callee into a single value, and once register allocation copies it -- `movc`,
; which CONSUMES a linear source -- the original register is null and the next
; `cjalr` through it faults (cause 24). Observed in a -O2 domain: `movc s8, s11`
; then `cjalr ra, 0(s11)`. The target is now built like a global's base, as
; PseudoCapGlobalBase: cincoffset and delin as one instruction, so no linear
; value is ever shared.
;
; MUTATION: build the target with a bare CIncOffset in selectCall again -> the
; CHECK-NEXT delin line fails (performed 2026-09-24, against dev's compiler).
;
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=capstone64 -mattr=+m -O0 -verify-machineinstrs < %s | FileCheck %s --check-prefix=O0

@total = internal addrspace(200) global i32 0, align 4

define internal void @f(i32 %x) addrspace(200) noinline {
  %t = load i32, ptr addrspace(200) @total
  %s = add i32 %t, %x
  store i32 %s, ptr addrspace(200) @total
  ret void
}

; Three calls to one callee share one target register: it is delin'd once, at
; its definition, and every cjalr goes through the non-linear result.
; CHECK-LABEL: g:
; CHECK:       cincoffset [[T:[a-z0-9]+]], gp, {{[a-z0-9]+}}
; CHECK-NEXT:  delin [[T]]
; CHECK:       cjalr ra, 0([[T]])
; CHECK:       cjalr ra, 0([[T]])
; At -O0 the target is spilled across the next call (stc/ldc): a copy, harmless
; only because it is already non-linear.
; O0-LABEL:    g:
; O0:          cincoffset [[T:[a-z0-9]+]], gp, {{[a-z0-9]+}}
; O0-NEXT:     delin [[T]]
; O0:          cjalr ra, 0([[T]])
define i32 @g(i32 %n) addrspace(200) {
entry:
  call addrspace(200) void @f(i32 1)
  call addrspace(200) void @f(i32 2)
  %c = icmp sgt i32 %n, 0
  br i1 %c, label %loop, label %done
loop:
  %i = phi i32 [ 0, %entry ], [ %i1, %loop ]
  call addrspace(200) void @f(i32 %i)
  %i1 = add i32 %i, 1
  %e = icmp eq i32 %i1, %n
  br i1 %e, label %done, label %loop
done:
  %r = load i32, ptr addrspace(200) @total
  ret i32 %r
}
