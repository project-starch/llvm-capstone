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
; RUN: llc -mtriple=capstone64 -mattr=+m -capstone-shrink-stack=false < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,NOSHRINK
; RUN: %llc_cap -O0 < %s -o /dev/null
; RUN: %llc_cap -O1 < %s -o /dev/null
; NOTE: reading a capability's ADDRESS is `mv rd, rs` (addi rd, rs, 0), NOT
; `lcc rd, rs, 2`. Same value -- the plain regfile slot holds the cursor
; (RTL ex_stage.sv:463-479; QEMU cap.h union aliases scalar onto bounds.cursor)
; -- but the plain read is TOTAL, whereas lcc selector 2 TRAPS on an untagged
; operand and a NULL pointer is untagged. That was C-19: `p != 0 || q != 0`
; folds to `(addr(p)|addr(q)) != 0` and the first null killed the domain.

target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"

; A runtime-sized alloca. With the flag on, the returned pointer is narrowed to
; the allocated region (lcc cursor, add size, shrink), and the sp update uses the
; un-narrowed base (movc sp, <broad>). With the flag off, no shrink is emitted.
; CHECK-LABEL: dynalloca:
; The freshly allocated base is formed by offsetting sp by -size.
; CHECK: cincoffset [[BASE:a[0-9]+]], sp, {{a[0-9]+}}
; SHRINK: mv {{a[0-9]+}}, [[BASE]]
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

; Memory-sourced (loaded) alloca size. lowerDynamicAllocaSizeToXLen must
; materialize a size that comes from a load (an i32 -> i128 extending load in
; PureCap lowering) into an XLen register instead of erroring out
; ("Unsupported dynamic alloca size expression"). This case previously failed to
; compile at all; here it must compile both ways and still narrow when on.
; CHECK-LABEL: dynalloca_memsize:
; SHRINK: shrink
; NOSHRINK-NOT: shrink
define i8 @dynalloca_memsize(ptr addrspace(200) %np) addrspace(200) {
  %n32 = load volatile i32, ptr addrspace(200) %np, align 4
  %n = zext i32 %n32 to i64
  %p = alloca i8, i64 %n, align 16, addrspace(200)
  store volatile i8 5, ptr addrspace(200) %p
  %v = load volatile i8, ptr addrspace(200) %p
  ret i8 %v
}

; A zero-byte dynamic alloca (alloca(0); a count of 0 for an XALLOCAVEC).
; SHRINK refuses an empty range (base >= end raises Illegal operand value), so
; with narrowing on, the size is raised to one stack-alignment granule before
; sp moves: the result is a fresh, non-empty region below the old sp, never an
; empty one and never one that covers the object below it. Without narrowing
; nothing changes.
declare void @use(ptr addrspace(200)) addrspace(200)

; A runtime size: after rounding to 16, a size below 16 (that is, 0) becomes 16,
; and that clamped size both moves sp and ends the narrowed range.
; CHECK-LABEL: dynalloca_maybe_zero:
; SHRINK: andi [[SZ:a[0-9]+]], {{a[0-9]+}}, -16
; SHRINK-NEXT: li [[MIN:a[0-9]+]], 16
; SHRINK-NEXT: bltu [[MIN]], [[SZ]],
; SHRINK: li [[SZ]], 16
; SHRINK: neg [[NEG:a[0-9]+]], [[SZ]]
; SHRINK-NEXT: cincoffset [[BASE:a[0-9]+]], sp, [[NEG]]
; SHRINK-NEXT: mv [[ADDR:a[0-9]+]], [[BASE]]
; SHRINK-NEXT: add [[END:a[0-9]+]], [[ADDR]], [[SZ]]
; SHRINK: shrink {{a[0-9]+}}, [[ADDR]], [[END]]
; NOSHRINK-NOT: bltu
; NOSHRINK-NOT: shrink
; CHECK: cjalr ra
define void @dynalloca_maybe_zero(i64 %n) addrspace(200) {
  %p = alloca i8, i64 %n, align 16, addrspace(200)
  call addrspace(200) void @use(ptr addrspace(200) %p)
  ret void
}

; A constant zero size outside the entry block stays dynamic. Before the fix
; this was `shrink rd, rX, rX`, an empty range, which traps.
; CHECK-LABEL: dynalloca_const_zero:
; SHRINK: cincoffsetimm [[BASE:a[0-9]+]], sp, -16
; SHRINK-NEXT: mv [[ADDR:a[0-9]+]], [[BASE]]
; SHRINK-NEXT: addi [[END:a[0-9]+]], [[ADDR]], 16
; SHRINK: shrink {{a[0-9]+}}, [[ADDR]], [[END]]
; SHRINK: movc sp, [[BASE]]
; NOSHRINK: cincoffsetimm {{a[0-9]+}}, sp, 0
; NOSHRINK-NOT: shrink
; CHECK: cjalr ra
define void @dynalloca_const_zero() addrspace(200) {
entry:
  br label %body
body:
  %p = alloca i8, i64 0, align 16, addrspace(200)
  call addrspace(200) void @use(ptr addrspace(200) %p)
  ret void
}
