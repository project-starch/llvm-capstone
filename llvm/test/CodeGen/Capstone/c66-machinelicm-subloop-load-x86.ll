; REQUIRES: x86-registered-target
; C-66, the target-independent half: MachineLICM's speculation cache, keyed by
; block alone, answered a subloop with what it had computed for the outermost
; loop, and that is not Capstone's to own. A load is the instruction the cache
; was written for. %body runs on every iteration of the outer loop that reaches
; its exit but not on every iteration of the inner loop, which leaves through
; %skip when %p is NULL; the load of %p+48 went into the inner preheader, %outer,
; above that test, and reads address 48 on x86 whenever a slot is NULL. It lives
; here rather than under X86/ because this fork's lit gate runs this directory,
; and the fix is in a file the fork shares with upstream.
; Control: the same load in the inner header runs on every inner iteration and
; still goes to the inner preheader.
; The stale answer also cut the other way. In @subloop_outer_exits_first the outer
; loop can leave before the inner loop starts, so the load is not guaranteed for
; the outer loop; that "no" was handed to the inner loop, whose header it runs
; on every iteration, and the load stayed in %ihead. Keyed by loop, it goes to
; the inner preheader, %pre.
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -stop-after=early-machinelicm < %s | FileCheck %s

; CHECK-LABEL: name: subloop_guarded_load
; CHECK: bb.{{[0-9]+}}.outer:
; CHECK-NOT: MOV64rm %{{[0-9]+}}, 1, $noreg, 48
; CHECK: bb.{{[0-9]+}}.body:
; CHECK-NOT: {{^ *bb\.[0-9]+}}
; CHECK: MOV64rm %{{[0-9]+}}, 1, $noreg, 48

; CHECK-LABEL: name: subloop_every_iteration_load
; CHECK: bb.{{[0-9]+}}.outer:
; CHECK-NOT: {{^ *bb\.[0-9]+}}
; CHECK: MOV64rm %{{[0-9]+}}, 1, $noreg, 48

; CHECK-LABEL: name: subloop_outer_exits_first
; CHECK: bb.{{[0-9]+}}.pre:
; CHECK-NOT: {{^ *bb\.[0-9]+}}
; CHECK: MOV64rm %{{[0-9]+}}, 1, $noreg, 48

define i64 @subloop_guarded_load(ptr %base, i64 %n, i64 %m) {
entry:
  br label %outer
outer:
  %i = phi i64 [ 0, %entry ], [ %i.next, %skip ], [ %i.next2, %after ]
  %acc.o = phi i64 [ 0, %entry ], [ %acc.o, %skip ], [ %acc.l, %after ]
  %slot = getelementptr ptr, ptr %base, i64 %i
  %p = load ptr, ptr %slot, align 8
  br label %ihead
ihead:
  %j = phi i64 [ 0, %outer ], [ %j.next, %ilatch ]
  %acc = phi i64 [ %acc.o, %outer ], [ %acc.l, %ilatch ]
  %isnull = icmp eq ptr %p, null
  br i1 %isnull, label %skip, label %body
body:
  %g = getelementptr inbounds i8, ptr %p, i64 48
  %v = load i64, ptr %g, align 8
  %v2 = mul i64 %v, %v
  %acc.b = add i64 %acc, %v2
  br label %ilatch
ilatch:
  %acc.l = phi i64 [ %acc.b, %body ]
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
  ret i64 %acc.l
}

define i64 @subloop_every_iteration_load(ptr %base, i64 %n, i64 %m) {
entry:
  br label %outer
outer:
  %i = phi i64 [ 0, %entry ], [ %i.next, %after ]
  %acc.o = phi i64 [ 0, %entry ], [ %acc.b, %after ]
  %slot = getelementptr ptr, ptr %base, i64 %i
  %p = load ptr, ptr %slot, align 8
  br label %ihead
ihead:
  %j = phi i64 [ 0, %outer ], [ %j.next, %ihead ]
  %acc = phi i64 [ %acc.o, %outer ], [ %acc.b, %ihead ]
  %g = getelementptr inbounds i8, ptr %p, i64 48
  %v = load i64, ptr %g, align 8
  %v2 = mul i64 %v, %v
  %acc.b = add i64 %acc, %v2
  %j.next = add i64 %j, 1
  %c2 = icmp slt i64 %j.next, %m
  br i1 %c2, label %ihead, label %after
after:
  %i.next = add i64 %i, 1
  %c3 = icmp slt i64 %i.next, %n
  br i1 %c3, label %outer, label %exit
exit:
  ret i64 %acc.b
}

define i64 @subloop_outer_exits_first(ptr %base, ptr %flag, i64 %n, i64 %m) {
entry:
  br label %outer
outer:
  %i = phi i64 [ 0, %entry ], [ %i.next, %after ]
  %acc.o = phi i64 [ 0, %entry ], [ %acc.b, %after ]
  %fs = getelementptr i64, ptr %flag, i64 %i
  %f = load i64, ptr %fs, align 8
  %stop = icmp eq i64 %f, 0
  br i1 %stop, label %exit, label %pre
pre:
  %slot = getelementptr ptr, ptr %base, i64 %i
  %p = load ptr, ptr %slot, align 8
  br label %ihead
ihead:
  %j = phi i64 [ 0, %pre ], [ %j.next, %ihead ]
  %acc = phi i64 [ %acc.o, %pre ], [ %acc.b, %ihead ]
  %g = getelementptr inbounds i8, ptr %p, i64 48
  %v = load i64, ptr %g, align 8
  %v2 = mul i64 %v, %v
  %acc.b = add i64 %acc, %v2
  %j.next = add i64 %j, 1
  %c2 = icmp slt i64 %j.next, %m
  br i1 %c2, label %ihead, label %after
after:
  %i.next = add i64 %i, 1
  %c3 = icmp slt i64 %i.next, %n
  br i1 %c3, label %outer, label %exit
exit:
  %r = phi i64 [ %acc.o, %outer ], [ %acc.b, %after ]
  ret i64 %r
}
