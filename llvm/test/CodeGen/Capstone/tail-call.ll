; A tail call must be lowered as a JUMP, not as a call.  Today it is not:
; CapstoneISD::TAIL is routed into selectCall (CapstoneISelDAGToDAG.cpp, the
; CALL/TAIL case), which always builds PseudoCALLIndirect.  The callee is
; therefore entered with `cjalr ra` and the caller has NO epilogue and NO
; return after it -- control falls off the end of the function into whatever
; symbol follows.  Verified 2026-09-04 on `long g(long x){return f(x+1);}` at
; -O1.  Masked since June by -fno-optimize-sibling-calls in the CoreMark and
; SQLite build scripts.  PseudoTAILIndirect (-> `cjalr zero, 0(rs1)`) exists
; and is never selected.
;
; Correct lowering: restore callee-saves if any, pop the frame, then
; `cjalr zero, 0(target)`.  This file pins that shape.  It is XFAIL until the
; fix lands; when it lands lit reports XPASS and the marker comes off.
;
; MUTATION: change `tail call` to `call` in @tail_direct -> `CHECK-NOT: cjalr ra`
; in that function fires.  @not_tail is the standing control: it MUST contain
; `cjalr ra` followed by a real return, which proves the negative check can
; distinguish the two shapes.
;
; RUN: llc < %s -mtriple=capstone64 -mattr=+m -O1 -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -mtriple=capstone64 -mattr=+m -O2 -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -mtriple=capstone64 -mattr=+m -O2 -verify-machineinstrs -capstone-gp-free | FileCheck %s --check-prefix=GPFREE
; XFAIL: *

declare i64 @callee(i64)

; A direct sibling call: materialise the target, drop the frame, jump.
; CHECK-LABEL: tail_direct:
; CHECK-NOT: cjalr ra
; CHECK: cjalr zero, 0(a{{[0-9]+}})
; GPFREE-LABEL: tail_direct:
; GPFREE-NOT: jalr ra
; GPFREE-NOT: cjalr ra
; GPFREE: {{j(alr zero, 0\(a[0-9]+\)|r a[0-9]+)}}
define i64 @tail_direct(i64 %x) {
  %y = add i64 %x, 1
  %r = tail call i64 @callee(i64 %y)
  ret i64 %r
}

; An indirect sibling call through a capability function pointer.
; CHECK-LABEL: tail_indirect:
; CHECK-NOT: cjalr ra
; CHECK: cjalr zero, 0(a{{[0-9]+}})
define i64 @tail_indirect(ptr addrspace(200) %f, i64 %x) {
  %r = tail call i64 %f(i64 %x)
  ret i64 %r
}

; A sibling call after a real call: ra was spilled for the first call and must
; be RELOADED, and the frame popped, before the jump -- the jump returns to our
; caller with our caller's ra.
; CHECK-LABEL: tail_after_call:
; CHECK: cjalr ra, 0(a{{[0-9]+}})
; CHECK: ldc ra, {{[0-9]+}}(sp)
; CHECK: cincoffsetimm sp, sp, {{[0-9]+}}
; CHECK-NEXT: cjalr zero, 0(a{{[0-9]+}})
define i64 @tail_after_call(i64 %x) {
  %a = call i64 @callee(i64 %x)
  %b = add i64 %a, %x
  %r = tail call i64 @callee(i64 %b)
  ret i64 %r
}

; CONTROL -- not a tail call: the result is used after the call, so this MUST
; be `cjalr ra` and MUST end with a real return.  If this function ever shows
; `cjalr zero, 0(a...)` the backend has started mis-selecting ordinary calls.
; CHECK-LABEL: not_tail:
; CHECK: cjalr ra, 0(a{{[0-9]+}})
; CHECK: addi a0, a0, 1
; CHECK: cjalr zero, 0(ra)
define i64 @not_tail(i64 %x) {
  %r = call i64 @callee(i64 %x)
  %s = add i64 %r, 1
  ret i64 %s
}
