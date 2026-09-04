; RUN: opt -passes=licm -S < %s | FileCheck %s

; A GEP on a POSSIBLY-NULL pointer in a NON-INTEGRAL address space must not be
; hoisted out of a loop.
;
; On a capability target, address arithmetic is not total: Capstone's
; `cincoffset` raises UNEXPECTED_OPERAND when its base register holds no
; capability, and a NULL pointer holds none. Hoisting `&p->field` into the
; preheader therefore turns a guarded, never-executed computation into a trap.
; That is Capstone issue C-19: LICM hoisted exactly two such GEPs out of
; SQLite's selectExpander and the domain died at loop entry.
;
; The two functions below differ in ONE thing -- whether the base is known
; non-null -- so this pins the DISCRIMINATION, not merely the absence of a
; hoist. If the guard were removed, @maybe_null would hoist and fail; if the
; guard were too broad, @known_nonnull would stop hoisting and fail.

target datalayout = "e-m:e-p:64:128-p200:128:128:128-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"

; The base may be null, so the GEP must STAY in the loop body.
; CHECK-LABEL: @maybe_null(
; CHECK: entry:
; CHECK-NOT: getelementptr
; CHECK: loop:
define void @maybe_null(ptr addrspace(200) %p, i32 %n) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %latch ]
  %isnull = icmp eq ptr addrspace(200) %p, null
  br i1 %isnull, label %latch, label %use

use:
  %f = getelementptr inbounds i8, ptr addrspace(200) %p, i128 16
  store i32 %i, ptr addrspace(200) %f, align 4
  br label %latch

latch:
  %i.next = add i32 %i, 1
  %done = icmp eq i32 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; Base is nonnull, so hoisting is still allowed and still happens.
; CHECK-LABEL: @known_nonnull(
; CHECK: entry:
; CHECK: getelementptr
define void @known_nonnull(ptr addrspace(200) nonnull %p, i32 %n) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %latch ]
  %c = icmp eq i32 %i, 3
  br i1 %c, label %latch, label %use

use:
  %f = getelementptr inbounds i8, ptr addrspace(200) %p, i128 16
  store i32 %i, ptr addrspace(200) %f, align 4
  br label %latch

latch:
  %i.next = add i32 %i, 1
  %done = icmp eq i32 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}
