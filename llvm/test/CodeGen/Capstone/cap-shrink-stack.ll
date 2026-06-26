; Object-granularity capability narrowing for address-taken stack objects (C1,
; stack slice, opt-in via -capstone-shrink-stack). When on, materializing the
; address of a whole stack object (a bare FrameIndex) narrows its capability to
; [&obj, &obj+size) via lcc(cursor)/add/shrink. Default off -> no narrowing.
;
; See CapstoneISelDAGToDAG.cpp (ISD::FrameIndex) and design/capability-bounds-model.md.

; RUN: llc -mtriple=capstone64 -mattr=+m -capstone-shrink-stack=true < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,SHRINK
; RUN: llc -mtriple=capstone64 -mattr=+m < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,NOSHRINK

target datalayout = "e-m:e-pf200:128:128:128:64-p:64:64-i64:64-i128:128-n32:64-S128-A200-P200-G200"

; A 64-byte local indexed by a runtime value: &buf (bare FrameIndex) is narrowed
; to its object when the flag is on, and left at the broad stack bounds when off.
; CHECK-LABEL: stack_idx:
; CHECK: cincoffsetimm {{a[0-9]+}}, sp, 0
; SHRINK: lcc {{a[0-9]+}}, {{a[0-9]+}}, 2
; SHRINK: shrink
; NOSHRINK-NOT: shrink
define i8 @stack_idx(i64 %i) addrspace(200) {
  %buf = alloca [64 x i8], align 1, addrspace(200)
  %p = getelementptr [64 x i8], ptr addrspace(200) %buf, i64 0, i64 %i
  %v = load i8, ptr addrspace(200) %p
  ret i8 %v
}
