; RUN: opt -passes=indvars -S %s | FileCheck %s
; RUN: opt -mtriple=capstone64 -passes='default<O2>' -S %s | FileCheck %s
; REQUIRES: capstone-registered-target
;
; With bounds [65536,65552) and t=4096, the original only forms b+4*j and b
; on the entered iteration. A pointer recurrence starting at b-16384 loses
; its tag below the window's lower edge (63488) and cannot recover it.
; replaceIVUserWithLoopInvariant uses a non-canonical SCEV expander.
;
; CHECK-LABEL: @inv_user(
; CHECK: entry:
; CHECK-NOT: getelementptr
; CHECK: outer:
; CHECK-NOT: phi ptr addrspace(200)
; CHECK: [[OFFSET:%.*]] = phi i64
; CHECK: [[CURSOR:%.*]] = getelementptr i8, ptr addrspace(200) %b, i64 [[OFFSET]]
; CHECK: inner.preheader:
; CHECK: inner:
; CHECK: load volatile i32, ptr addrspace(200) [[CURSOR]]
; CHECK: latch:
; CHECK-NOT: getelementptr
; CHECK: ret void

target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"
target triple = "capstone64-unknown-linux-gnu"
; flags[i] != 0 only for i == t (t = 4096 at run time); m >= 1.
; Original cursors per entered inner loop: p = b + 4*(i-t+j), q = p - 4*j.
; At i = t: p = b + 4j, q = b. Nothing is formed on skipped outer iterations.
define void @inv_user(ptr addrspace(200) %b, ptr addrspace(200) %flags, i64 %n, i64 %t, i64 %m) {
entry:
  br label %outer
outer:
  %i = phi i64 [0, %entry], [%inext, %latch]
  %fp = getelementptr i8, ptr addrspace(200) %flags, i64 %i
  %fl = load volatile i8, ptr addrspace(200) %fp
  %take = icmp ne i8 %fl, 0
  br i1 %take, label %inner.preheader, label %latch
inner.preheader:
  %ib = sub i64 %i, %t
  br label %inner
inner:
  %j = phi i64 [0, %inner.preheader], [%jnext, %inner]
  %sum = add i64 %ib, %j
  %p = getelementptr i32, ptr addrspace(200) %b, i64 %sum
  store volatile i32 0, ptr addrspace(200) %p
  %j4 = mul i64 %j, -4
  %q = getelementptr i8, ptr addrspace(200) %p, i64 %j4
  %v = load volatile i32, ptr addrspace(200) %q
  %jnext = add nuw nsw i64 %j, 1
  %jdone = icmp eq i64 %jnext, %m
  br i1 %jdone, label %inner.exit, label %inner
inner.exit:
  br label %latch
latch:
  %inext = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %inext, %n
  br i1 %done, label %exit, label %outer
exit:
  ret void
}
