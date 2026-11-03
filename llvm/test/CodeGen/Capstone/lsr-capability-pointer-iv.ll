; RUN: %llc_cap -O2 < %s | FileCheck %s
; RUN: %llc_cap -O1 < %s -o /dev/null
; RUN: %llc_cap -O2 -stop-after=loop-term-fold < %s | FileCheck %s --check-prefix=LTF
; RUN: %llc_cap -O2 < %s | FileCheck %s --check-prefix=NOZERO --implicit-check-not='cincoffset{{(imm)?}} {{[a-z0-9]+}}, zero'
;
; Generic LSR shares the integer byte offset and retains a capability
; increment from the original base at each access. It must not introduce
; a normalized or conditionally unused capability recurrence.
;
; The companion IR test is
; Transforms/LoopStrengthReduce/Capstone/capability-pointer-iv.ll.

; CHECK-LABEL: guarded:
; CHECK:       [[LOOP:\.LBB[0-9_]+]]: # %for.body
; CHECK-NOT:     slli
; CHECK:         cincoffset {{[a-z0-9]+}}, a0, [[OFFSET:[a-z0-9]+]]
; CHECK:         addi [[OFFSET]], [[OFFSET]], 8
; CHECK-NOT:     slli
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
; CHECK:         cincoffset {{[a-z0-9]+}}, a0, [[OFFSET:[a-z0-9]+]]
; CHECK:         addi [[OFFSET]], [[OFFSET]], 8
; CHECK-NOT:     slli
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

; The guarded nested walk keeps its own GEPs below the null check. Without
; a pointer recurrence, loop-term-fold retains the integer exit condition.
; LTF-LABEL: define void @nested_inbounds(
; LTF:       entry:
; LTF-NOT:     getelementptr
; LTF:       inner:
; LTF-NOT:     phi ptr addrspace(200)
; LTF:         %items = getelementptr inbounds i8, ptr addrspace(200) %p, i64 48
; LTF:         %slot = getelementptr inbounds ptr addrspace(200), ptr addrspace(200) %items, i64 %i
; LTF:         icmp eq i64
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

; Pointer comparisons must not lower to a capability increment from zero.
; The NOZERO run covers every function in this file.
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
