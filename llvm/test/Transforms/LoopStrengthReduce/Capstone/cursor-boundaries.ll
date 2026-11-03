; RUN: opt -mtriple=capstone64 -passes=loop-reduce -S %s | FileCheck %s
;
; Generic LSR must not invent capability recurrences. The specialized
; accessed-anchor reduction is covered by lsr-capability-diamond-iv.ll.

target datalayout = "e-p:64:64-p200:128:128:128:64-i64:64-n32:64-ni:200"
target triple = "capstone64"

; With b=65536 and n=3, both stores use b. Generic LSR steps a capability
; by -2^63 on the skipped iteration, losing its tag under sv39 or phys56.
; CHECK-LABEL: @conditional(
; CHECK-NOT: phi ptr addrspace(200)
; CHECK: use:
; CHECK: getelementptr i8, ptr addrspace(200) %b,
; CHECK-NOT: getelementptr i8, ptr addrspace(200) {{.*}}, i64 -9223372036854775808
; CHECK: ret void
define void @conditional(ptr addrspace(200) nonnull %b, i64 %n) {
entry:
  br label %loop
loop:
  %i = phi i64 [0, %entry], [%next, %latch]
  %odd = and i64 %i, 1
  %skip = icmp ne i64 %odd, 0
  br i1 %skip, label %latch, label %use
use:
  %idx = shl i64 %i, 63
  %p = getelementptr i8, ptr addrspace(200) %b, i64 %idx
  store volatile i8 1, ptr addrspace(200) %p
  br label %latch
latch:
  %next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %next, %n
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

; CHECK-LABEL: @conditional_integral(
; CHECK: phi ptr
; CHECK: getelementptr i8, ptr {{.*}}, i64 -9223372036854775808
; CHECK: ret void
define void @conditional_integral(ptr nonnull %b, i64 %n) {
entry:
  br label %loop
loop:
  %i = phi i64 [0, %entry], [%next, %latch]
  %odd = and i64 %i, 1
  %skip = icmp ne i64 %odd, 0
  br i1 %skip, label %latch, label %use
use:
  %idx = shl i64 %i, 63
  %p = getelementptr i8, ptr %b, i64 %idx
  store volatile i8 1, ptr %p
  br label %latch
latch:
  %next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %next, %n
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

; p=2, n=1, bounds [0,16): p-8 is refused by phys56. The old normalization
; lost the tag before the first load and also returned an untagged pointer.
; CHECK-LABEL: @live_out_phys56(
; CHECK-NOT: getelementptr i8, ptr addrspace(200) %p, i64 -8
; CHECK: for.body:
; CHECK: [[IV:%.*]] = phi ptr addrspace(200) [ %p, %for.body.preheader ], [ [[NEXT:%.*]], %for.body ]
; CHECK: load i16, ptr addrspace(200) [[IV]]
; CHECK: [[NEXT]] = getelementptr i8, ptr addrspace(200) [[IV]], i64 8
; CHECK: for.end.loopexit:
; CHECK-NEXT: [[LIVE:%.*]] = phi ptr addrspace(200) [ [[IV]], %for.body ]
; CHECK: [[LAST:%.*]] = phi ptr addrspace(200) [ %p, %entry ], [ [[LIVE]], %for.end.loopexit ]
; CHECK: ret ptr addrspace(200) [[LAST]]
define ptr addrspace(200) @live_out_phys56(ptr addrspace(200) %p, i64 %n) addrspace(200) {
entry:
  %nonnull = icmp ne ptr addrspace(200) %p, null
  %positive = icmp sgt i64 %n, 0
  %enter = and i1 %nonnull, %positive
  br i1 %enter, label %for.body, label %for.end

for.body:
  %i = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %idx = shl nsw i64 %i, 3
  %a0 = getelementptr inbounds nuw i8, ptr addrspace(200) %p, i64 %idx
  %v0 = load i16, ptr addrspace(200) %a0, align 2
  %n0 = xor i16 %v0, -1
  store i16 %n0, ptr addrspace(200) %a0, align 2
  %inc = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %inc, %n
  br i1 %done, label %for.end, label %for.body

for.end:
  %last = phi ptr addrspace(200) [ %p, %entry ], [ %a0, %for.body ]
  ret ptr addrspace(200) %last
}
; p=end=2: the first load succeeds in the original before exiting. Its
; comparison must not normalize the pointer IV onto the refused p-8.
; CHECK-LABEL: @cmp_end_8(
; CHECK-NOT: getelementptr i8, ptr addrspace(200) %p, i64 -8
; CHECK: loop:
; CHECK: [[IV:%.*]] = phi ptr addrspace(200) [ %p, %entry ], [ [[NEXT:%.*]], %loop ]
; CHECK: load i16, ptr addrspace(200) [[IV]]
; CHECK: icmp eq ptr addrspace(200) [[IV]], %end
; CHECK: [[NEXT]] = getelementptr i8, ptr addrspace(200) [[IV]], i64 8
; CHECK: ret i16
define i16 @cmp_end_8(ptr addrspace(200) nonnull %p, ptr addrspace(200) %end) addrspace(200) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %acc = phi i16 [ 0, %entry ], [ %acc.next, %loop ]
  %idx = shl nsw i64 %i, 3
  %a = getelementptr inbounds nuw i8, ptr addrspace(200) %p, i64 %idx
  %v = load i16, ptr addrspace(200) %a, align 2
  %acc.next = add i16 %acc, %v
  %i.next = add nuw nsw i64 %i, 1
  %done = icmp eq ptr addrspace(200) %a, %end
  br i1 %done, label %exit, label %loop

exit:
  ret i16 %acc.next
}
