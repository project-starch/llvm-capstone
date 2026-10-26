; RUN: %llc_cap -O2 < %s | FileCheck %s
; RUN: %llc_cap -O1 < %s -o /dev/null
; RUN: %llc_cap -O2 -stop-after=loop-term-fold < %s | FileCheck %s --check-prefix=LTF
; RUN: %llc_cap -O2 < %s | FileCheck %s --check-prefix=NOZERO --implicit-check-not='cincoffset{{(imm)?}} {{[a-z0-9]+}}, zero'
;
; The loop body of a counted array walk steps a capability register by an
; immediate and folds the element offsets into the accesses; it no longer
; rebuilds the address from the integer index with a shift and a register
; cincoffset per iteration. The preheader may still scale the trip count to
; compute the end pointer. This holds because the access runs on every
; iteration of the loop, with a null check before the loop or without one.
;
; The companion IR test, with the shapes that must NOT get a pointer induction
; variable, is Transforms/LoopStrengthReduce/Capstone/capability-pointer-iv.ll.

; CHECK-LABEL: guarded:
; CHECK:       [[LOOP:\.LBB[0-9_]+]]: # %for.body
; CHECK-NOT:     slli
; CHECK-NOT:     cincoffset {{[a-z0-9]+}}, {{[a-z0-9]+}}, {{[a-z0-9]+}}
; CHECK:         cincoffsetimm {{[a-z0-9]+}}, {{[a-z0-9]+}}, 8
; CHECK-NOT:     slli
; CHECK-NOT:     cincoffset {{[a-z0-9]+}}, {{[a-z0-9]+}}, {{[a-z0-9]+}}
; CHECK:         bne {{.*}}[[LOOP]]
define void @guarded(ptr addrspace(200) %p, i64 %n) addrspace(200) {
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
  %a1 = getelementptr inbounds nuw i8, ptr addrspace(200) %a0, i64 2
  %v1 = load i16, ptr addrspace(200) %a1, align 2
  %n1 = xor i16 %v1, -1
  store i16 %n1, ptr addrspace(200) %a1, align 2
  %inc = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %inc, %n
  br i1 %done, label %for.end, label %for.body

for.end:
  ret void
}

; CHECK-LABEL: every_iteration:
; CHECK:       [[LOOP:\.LBB[0-9_]+]]: # %for.body
; CHECK-NOT:     slli
; CHECK-NOT:     cincoffset {{[a-z0-9]+}}, {{[a-z0-9]+}}, {{[a-z0-9]+}}
; CHECK:         cincoffsetimm {{[a-z0-9]+}}, {{[a-z0-9]+}}, 8
; CHECK-NOT:     slli
; CHECK-NOT:     cincoffset {{[a-z0-9]+}}, {{[a-z0-9]+}}, {{[a-z0-9]+}}
; CHECK:         bne {{.*}}[[LOOP]]
define void @every_iteration(ptr addrspace(200) %p, i64 %n) addrspace(200) {
entry:
  %positive = icmp sgt i64 %n, 0
  br i1 %positive, label %for.body, label %for.end

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
  ret void
}

; The C-58 nest with inbounds walks: LSR forms the inner pointer phi, and
; loop-term-fold, which runs next and replaces the counter's exit test with a
; comparison against the walk's end pointer, forms that end pointer in the
; inner preheader -- below the outer loop's null check -- not in entry, where
; its expander used to hoist an invariant value.
; LTF-LABEL: define void @nested_inbounds(
; LTF:       entry:
; LTF-NOT:     getelementptr
; LTF:       inner.ph:
; LTF:         getelementptr i8, ptr addrspace(200) %p, i64
; LTF:       inner:
; LTF:         icmp eq ptr addrspace(200)
; LTF:         ret void
define void @nested_inbounds(ptr addrspace(200) %p, i64 %n, i64 %m, ptr addrspace(200) %out) addrspace(200) {
entry:
  br label %outer

outer:
  %j = phi i64 [ 0, %entry ], [ %j.next, %outer.latch ]
  %isnull = icmp eq ptr addrspace(200) %p, null
  br i1 %isnull, label %outer.latch, label %inner.ph

inner.ph:
  br label %inner

inner:
  %i = phi i64 [ 0, %inner.ph ], [ %i.next, %inner ]
  %items = getelementptr inbounds i8, ptr addrspace(200) %p, i64 48
  %slot = getelementptr inbounds ptr addrspace(200), ptr addrspace(200) %items, i64 %i
  %v = load ptr addrspace(200), ptr addrspace(200) %slot, align 16
  store ptr addrspace(200) %v, ptr addrspace(200) %out, align 16
  %i.next = add nuw nsw i64 %i, 1
  %idone = icmp eq i64 %i.next, %n
  br i1 %idone, label %outer.latch, label %inner

outer.latch:
  %j.next = add i64 %j, 1
  %odone = icmp eq i64 %j.next, %m
  br i1 %odone, label %exit, label %outer

exit:
  ret void
}

; An exit test that compares the walked address with an end pointer: LSR's
; ICmpZero rewrite turns it into a count-down compared through a null-based
; byte GEP (C-40's shape), which the backend lowers as integer arithmetic --
; never a cincoffset whose base is the zero register (the implicit check of
; the NOZERO run covers every function here).
; NOZERO-LABEL: cmp_end:
; NOZERO:         cjalr zero, 0(ra)
define i16 @cmp_end(ptr addrspace(200) %p, i64 %n) addrspace(200) {
entry:
  %nonnull = icmp ne ptr addrspace(200) %p, null
  br i1 %nonnull, label %loop, label %exit

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %acc = phi i16 [ 0, %entry ], [ %acc.next, %loop ]
  %idx = shl nsw i64 %i, 3
  %a = getelementptr inbounds nuw i8, ptr addrspace(200) %p, i64 %idx
  %v = load i16, ptr addrspace(200) %a, align 2
  %acc.next = add i16 %acc, %v
  %nidx = shl nsw i64 %n, 3
  %end = getelementptr inbounds nuw i8, ptr addrspace(200) %p, i64 %nidx
  %i.next = add nuw nsw i64 %i, 1
  %done = icmp eq ptr addrspace(200) %a, %end
  br i1 %done, label %exit, label %loop

exit:
  %r = phi i16 [ 0, %entry ], [ %acc.next, %loop ]
  ret i16 %r
}
