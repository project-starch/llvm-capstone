; Object-granularity capability narrowing for DYNAMIC (runtime-sized) allocas
; (C1, stack slice, opt-in via -capstone-shrink-stack). A dynamic alloca lowers
; through ISD::DYNAMIC_STACKALLOC and never reaches a FrameIndex, so the
; fixed-object helper (narrowToFrameObjectBounds) does not cover it. Instead,
; lowerDYNAMIC_STACKALLOC narrows the pointer *returned to the program* to the
; freshly allocated region [addr, addr+size) via lcc(cursor)/add/shrink, while
; the real stack pointer (sp/X2) keeps the broad bounds it needs for further
; allocations. Default off -> no narrowing.
;
; See CapstoneISelLowering.cpp (lowerDYNAMIC_STACKALLOC) and
; CapstoneISelDAGToDAG.cpp (CapstoneShrinkStack, shared flag).

; RUN: llc -mtriple=capstone64 -mattr=+m -capstone-shrink-stack=true < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,SHRINK
; RUN: llc -mtriple=capstone64 -mattr=+m < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,NOSHRINK

target datalayout = "e-m:e-pf200:128:128:128:64-p:64:64-i64:64-i128:128-n32:64-S128-A200-P200-G200"

; A runtime-sized alloca. With the flag on, the returned pointer is narrowed to
; the allocated region (lcc cursor, add size, shrink), and the sp update uses the
; un-narrowed base (movc sp, <broad>). With the flag off, no shrink is emitted.
; CHECK-LABEL: dynalloca:
; The freshly allocated base is formed by offsetting sp by -size.
; CHECK: cincoffset [[BASE:a[0-9]+]], sp, {{a[0-9]+}}
; SHRINK: lcc {{a[0-9]+}}, [[BASE]], 2
; SHRINK: shrink
; The real stack pointer keeps the un-narrowed (broad) base capability.
; SHRINK: movc sp, [[BASE]]
; NOSHRINK-NOT: shrink
define i8 @dynalloca(i64 %n) addrspace(200) {
  %p = alloca i8, i64 %n, align 16, addrspace(200)
  store volatile i8 7, ptr addrspace(200) %p
  %v = load volatile i8, ptr addrspace(200) %p
  ret i8 %v
}

; Two dynamic allocas in one function: each returned pointer is narrowed
; independently, so two shrinks are emitted when the flag is on and none off.
; CHECK-LABEL: dynalloca_two:
; SHRINK: shrink
; SHRINK: shrink
; NOSHRINK-NOT: shrink
define i8 @dynalloca_two(i64 %n, i64 %m) addrspace(200) {
  %p = alloca i8, i64 %n, align 16, addrspace(200)
  %q = alloca i8, i64 %m, align 16, addrspace(200)
  store volatile i8 1, ptr addrspace(200) %p
  store volatile i8 2, ptr addrspace(200) %q
  %v = load volatile i8, ptr addrspace(200) %p
  ret i8 %v
}
