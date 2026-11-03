; RUN: opt -passes=indvars -S %s | FileCheck %s
;
; The inner loop only runs at i=4096, computing b and b+1. With bounds
; [65536,65552), both are valid. Exit expansion must not introduce an outer
; capability IV starting at b-4095=61441: that lies below the representable
; window (lower edge 63488), and incrementing it never restores its tag.

target datalayout = "e-p:64:64-p200:128:128:128:64-i64:64-n32:64-ni:200"
; CHECK-LABEL: @nested(
; CHECK: entry:
; CHECK-NOT: getelementptr
; CHECK: outer:
; CHECK-NOT: phi ptr addrspace(200)
; CHECK: [[OFFSET:%.*]] = phi i64
; CHECK: [[CURSOR:%.*]] = getelementptr i8, ptr addrspace(200) %b, i64 [[OFFSET]]
; CHECK: inner.preheader:
; CHECK: inner.exit:
; CHECK: load volatile i8, ptr addrspace(200) [[CURSOR]]
define void @nested(ptr addrspace(200) %b, ptr addrspace(200) %out) {
entry:
  br label %outer
outer:
  %i = phi i64 [0, %entry], [%inext, %latch]
  %take = icmp eq i64 %i, 4096
  br i1 %take, label %inner.preheader, label %latch
inner.preheader:
  br label %inner
inner:
  %j = phi i64 [0, %inner.preheader], [%jnext, %inner]
  %sum = add i64 %i, %j
  %idx = add i64 %sum, -4096
  %p = getelementptr i8, ptr addrspace(200) %b, i64 %idx
  %jnext = add nuw nsw i64 %j, 1
  %jdone = icmp eq i64 %jnext, 2
  br i1 %jdone, label %inner.exit, label %inner
inner.exit:
  %last = phi ptr addrspace(200) [%p, %inner]
  %v = load volatile i8, ptr addrspace(200) %last
  br label %latch
latch:
  %inext = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %inext, 4097
  br i1 %done, label %exit, label %outer
exit:
  ret void
}

; CHECK-LABEL: @nested_integral(
; CHECK: entry:
; CHECK-NEXT: [[START:%.*]] = getelementptr i8, ptr %b, i64 -4095
; CHECK: outer:
; CHECK-NEXT: [[IV:%.*]] = phi ptr [ {{.*}} ], [ [[START]], %entry ]
; CHECK: inner.exit:
; CHECK-NEXT: %v = load volatile i8, ptr [[IV]]
define void @nested_integral(ptr %b, ptr %out) {
entry:
  br label %outer
outer:
  %i = phi i64 [0, %entry], [%inext, %latch]
  %take = icmp eq i64 %i, 4096
  br i1 %take, label %inner.preheader, label %latch
inner.preheader:
  br label %inner
inner:
  %j = phi i64 [0, %inner.preheader], [%jnext, %inner]
  %sum = add i64 %i, %j
  %idx = add i64 %sum, -4096
  %p = getelementptr i8, ptr %b, i64 %idx
  %jnext = add nuw nsw i64 %j, 1
  %jdone = icmp eq i64 %jnext, 2
  br i1 %jdone, label %inner.exit, label %inner
inner.exit:
  %last = phi ptr [%p, %inner]
  %v = load volatile i8, ptr %last
  br label %latch
latch:
  %inext = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %inext, 4097
  br i1 %done, label %exit, label %outer
exit:
  ret void
}

; An enclosing-loop load is not an add-recurrence. SCEV expansion must still
; keep its offset with the invariant constant. With tagged b=0, off=2,
; take=true, the original inner cursors are 0 and 1. Splitting out b-1 first
; crosses phys56; adding off afterward cannot restore the tag.
; CHECK-LABEL: @unknown_outer(
; CHECK: entry:
; CHECK-NOT: getelementptr
; CHECK: outer:
; CHECK: %off = load volatile i64
; CHECK-NOT: getelementptr
; CHECK: inner.exit:
; CHECK: [[IDX:%.*]] = add i64 %off, -1
; CHECK-NEXT: [[P:%.*]] = getelementptr i8, ptr addrspace(200) %b, i64 [[IDX]]
; CHECK-NEXT: %v = load volatile i8, ptr addrspace(200) [[P]]
define void @unknown_outer(ptr addrspace(200) %b, ptr addrspace(200) %src, i1 %take) {
entry:
  br label %outer
outer:
  %i = phi i64 [0, %entry], [%inext, %latch]
  %off = load volatile i64, ptr addrspace(200) %src
  br i1 %take, label %inner.preheader, label %latch
inner.preheader:
  br label %inner
inner:
  %j = phi i64 [0, %inner.preheader], [%jnext, %inner]
  %sum = add i64 %off, %j
  %idx = add i64 %sum, -2
  %p = getelementptr i8, ptr addrspace(200) %b, i64 %idx
  %jnext = add nuw nsw i64 %j, 1
  %jdone = icmp eq i64 %jnext, 2
  br i1 %jdone, label %inner.exit, label %inner
inner.exit:
  %last = phi ptr addrspace(200) [%p, %inner]
  %v = load volatile i8, ptr addrspace(200) %last
  br label %latch
latch:
  %inext = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %inext, 2
  br i1 %done, label %exit, label %outer
exit:
  ret void
}
