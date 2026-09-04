; Tier 4.3 of the validation plan: DROP is `hasSideEffects = 1` with no memory
; flags, and the question was whether a load could be scheduled across it and
; read through a dropped capability.  Measured 2026-09-05 at -O2: it cannot --
; an instruction with unmodeled side effects is a barrier for every memory
; operation in the scheduling DAG, and the two loads of %q below are neither
; merged nor moved past the drop of %p (the second load is re-issued after it,
; which is the conservative and correct reading).  This file pins that order so
; a change to DROP's flags that relaxes it is visible.
;
; MUTATION: swap the two IR lines `%d = call ... drop` and `%v = load` in
; @load_after_drop -> the emitted order becomes ld, ld, drop and the adjacent
; drop-then-ld pair below fails (performed 2026-09-05).  (Prose must not spell
; a directive token; FileCheck reads every line.)
;
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 -verify-machineinstrs < %s | FileCheck %s
; RUN: %llc_cap -O0 < %s -o /dev/null
; RUN: %llc_cap -O1 < %s -o /dev/null

declare ptr addrspace(200) @llvm.capstone.cap.drop.p200(ptr addrspace(200))

; CHECK-LABEL: load_after_drop:
; CHECK: ld a2, 0(a1)
; CHECK-NEXT: drop a0
; CHECK-NEXT: ld a0, 0(a1)
; CHECK-NEXT: add a0, a0, a2
; CHECK-NEXT: cjalr zero, 0(ra)
define i64 @load_after_drop(ptr addrspace(200) %p, ptr addrspace(200) %q) {
  %v0 = load i64, ptr addrspace(200) %q
  %d = call ptr addrspace(200) @llvm.capstone.cap.drop.p200(ptr addrspace(200) %p)
  %v = load i64, ptr addrspace(200) %q
  %s = add i64 %v, %v0
  ret i64 %s
}

; CONTROL: a load that precedes the drop in the IR stays before it.
; CHECK-LABEL: load_before_drop:
; CHECK: ld a1, 0(a1)
; CHECK-NEXT: drop a0
; CHECK-NEXT: mv a0, a1
; CHECK-NEXT: cjalr zero, 0(ra)
define i64 @load_before_drop(ptr addrspace(200) %p, ptr addrspace(200) %q) {
  %v = load i64, ptr addrspace(200) %q
  %d = call ptr addrspace(200) @llvm.capstone.cap.drop.p200(ptr addrspace(200) %p)
  ret i64 %v
}
