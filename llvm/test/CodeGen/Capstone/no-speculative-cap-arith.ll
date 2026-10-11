; C-58: capability arithmetic must not be hoisted out of a block that does not
; run on every iteration. CIncOffset on an untagged value -- NULL -- traps, so
; forming &p->items above the test that p is non-null turns a correct program
; into one that faults. MachineLICM only held loads to that rule; CPython's
; argument parser had &kwnames->ob_item moved above `if (kwnames)` and every
; call without keywords trapped. The shape here is that one: the address is
; invariant, formed in the guarded preheader of an inner loop, and early
; MachineLICM would hoist it out of the outer loop, above the guard.
; Control: the same address formed unconditionally is still hoisted.
; RUN: llc -mtriple=capstone64 -mattr=+m -verify-machineinstrs -stop-after=early-machinelicm < %s | FileCheck %s
; RUN: %llc_cap -O0 < %s -o /dev/null
; RUN: %llc_cap -O2 < %s -o /dev/null

; CHECK-LABEL: name: guarded
; CHECK: bb.{{[0-9]+}}.inner.ph:
; CHECK-NOT: {{^ *bb\.[0-9]+}}
; CHECK: CIncOffsetImm %{{[0-9]+}}, 48
; CHECK-LABEL: name: control_unguarded
; CHECK: bb.0.entry:
; CHECK-NOT: {{^ *bb\.[0-9]+}}
; CHECK: CIncOffsetImm %{{[0-9]+}}, 48

define void @guarded(ptr addrspace(200) %p, i64 %n, i64 %m, ptr addrspace(200) %out) addrspace(200) {
entry:
  br label %outer
outer:
  %j = phi i64 [ 0, %entry ], [ %j.next, %outer.latch ]
  %isnull = icmp eq ptr addrspace(200) %p, null
  br i1 %isnull, label %outer.latch, label %inner.ph
inner.ph:
  %items = getelementptr i8, ptr addrspace(200) %p, i64 48
  br label %inner
inner:
  %i = phi i64 [ 0, %inner.ph ], [ %i.next, %inner ]
  %slot = getelementptr ptr addrspace(200), ptr addrspace(200) %items, i64 %i
  %v = load ptr addrspace(200), ptr addrspace(200) %slot, align 16
  store ptr addrspace(200) %v, ptr addrspace(200) %out, align 16
  %i.next = add i64 %i, 1
  %idone = icmp eq i64 %i.next, %n
  br i1 %idone, label %outer.latch, label %inner
outer.latch:
  %j.next = add i64 %j, 1
  %odone = icmp eq i64 %j.next, %m
  br i1 %odone, label %exit, label %outer
exit:
  ret void
}

define void @control_unguarded(ptr addrspace(200) %p, i64 %n, i64 %m, ptr addrspace(200) %out) addrspace(200) {
entry:
  br label %outer
outer:
  %j = phi i64 [ 0, %entry ], [ %j.next, %outer.latch ]
  br label %inner.ph
inner.ph:
  %items = getelementptr i8, ptr addrspace(200) %p, i64 48
  br label %inner
inner:
  %i = phi i64 [ 0, %inner.ph ], [ %i.next, %inner ]
  %slot = getelementptr ptr addrspace(200), ptr addrspace(200) %items, i64 %i
  %v = load ptr addrspace(200), ptr addrspace(200) %slot, align 16
  store ptr addrspace(200) %v, ptr addrspace(200) %out, align 16
  %i.next = add i64 %i, 1
  %idone = icmp eq i64 %i.next, %n
  br i1 %idone, label %outer.latch, label %inner
outer.latch:
  %j.next = add i64 %j, 1
  %odone = icmp eq i64 %j.next, %m
  br i1 %odone, label %exit, label %outer
exit:
  ret void
}
