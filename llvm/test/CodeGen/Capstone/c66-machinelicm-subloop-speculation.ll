; C-66: MachineLICM decides whether a block runs on every iteration once per
; block and keeps the answer, but it asks for more than one loop. When an
; instruction cannot leave the outermost loop -- its operand is defined there --
; HoistOutOfLoop tries each subloop's preheader in turn and gets the answer it
; computed for the outermost loop. Here %body runs on every iteration of the
; outer loop that reaches its exit (the only exit is behind it) but not on every
; iteration of the inner one: the inner loop leaves through %skip when %p is
; NULL. Keyed by block alone, the cache said "guaranteed" and the increment of
; %p went into the inner preheader, %outer, above that test. CIncOffset of NULL
; traps. C-58's own hook asked per loop and did not do this; C-66 moved the
; question into MachineLICM's cached test, so the cache is now keyed by loop.
; Control: the increment in the inner header, which does run on every inner
; iteration, still goes to the inner preheader.
; RUN: llc -mtriple=capstone64 -mattr=+m -verify-machineinstrs -stop-after=early-machinelicm < %s | FileCheck %s
; RUN: %llc_cap -O0 < %s -o /dev/null
; RUN: %llc_cap -O2 < %s -o /dev/null

; CHECK-LABEL: name: subloop_guarded
; CHECK: bb.{{[0-9]+}}.outer:
; CHECK-NOT: CIncOffsetImm
; CHECK: bb.{{[0-9]+}}.body:
; CHECK-NOT: {{^ *bb\.[0-9]+}}
; CHECK: CIncOffsetImm %{{[0-9]+}}, 48

; CHECK-LABEL: name: subloop_every_iteration
; CHECK: bb.{{[0-9]+}}.outer:
; CHECK-NOT: {{^ *bb\.[0-9]+}}
; CHECK: CIncOffsetImm %{{[0-9]+}}, 48

define void @subloop_guarded(ptr addrspace(200) %base, i64 %n, i64 %m, ptr addrspace(200) %out) addrspace(200) {
entry:
  br label %outer
outer:
  %i = phi i64 [ 0, %entry ], [ %i.next, %skip ], [ %i.next2, %after ]
  %slot = getelementptr ptr addrspace(200), ptr addrspace(200) %base, i64 %i
  %p = load ptr addrspace(200), ptr addrspace(200) %slot, align 16
  br label %ihead
ihead:
  %j = phi i64 [ 0, %outer ], [ %j.next, %ilatch ]
  %isnull = icmp eq ptr addrspace(200) %p, null
  br i1 %isnull, label %skip, label %body
body:
  %g = getelementptr inbounds i8, ptr addrspace(200) %p, i64 48
  store ptr addrspace(200) %g, ptr addrspace(200) %out, align 16
  br label %ilatch
ilatch:
  %j.next = add i64 %j, 1
  %c2 = icmp slt i64 %j.next, %m
  br i1 %c2, label %ihead, label %after
skip:
  %i.next = add i64 %i, 1
  br label %outer
after:
  %i.next2 = add i64 %i, 1
  %c3 = icmp slt i64 %i.next2, %n
  br i1 %c3, label %outer, label %exit
exit:
  ret void
}

define void @subloop_every_iteration(ptr addrspace(200) %base, i64 %n, i64 %m, ptr addrspace(200) %out) addrspace(200) {
entry:
  br label %outer
outer:
  %i = phi i64 [ 0, %entry ], [ %i.next, %after ]
  %slot = getelementptr ptr addrspace(200), ptr addrspace(200) %base, i64 %i
  %p = load ptr addrspace(200), ptr addrspace(200) %slot, align 16
  br label %ihead
ihead:
  %j = phi i64 [ 0, %outer ], [ %j.next, %ihead ]
  %g = getelementptr inbounds i8, ptr addrspace(200) %p, i64 48
  store ptr addrspace(200) %g, ptr addrspace(200) %out, align 16
  %j.next = add i64 %j, 1
  %c2 = icmp slt i64 %j.next, %m
  br i1 %c2, label %ihead, label %after
after:
  %i.next = add i64 %i, 1
  %c3 = icmp slt i64 %i.next, %n
  br i1 %c3, label %outer, label %exit
exit:
  ret void
}
