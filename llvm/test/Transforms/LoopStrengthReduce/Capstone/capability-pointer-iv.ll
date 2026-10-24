; RUN: opt -mtriple=capstone64 -passes=loop-reduce -S < %s | FileCheck %s --implicit-check-not=ptrtoint --implicit-check-not=inttoptr
;
; A loop that walks a capability-typed array gets a pointer induction variable
; (a `ptr addrspace(200)` phi stepped by a GEP) when the loop runs the address
; computation on every iteration anyway, so that stepping the capability
; cannot fault where the original program would not have. Otherwise the
; address keeps being recomputed from the integer index.
;
; On this target address arithmetic on a capability can trap (cincoffset on a
; base that holds no capability -- sealed or uninitialised; issue C-19), so
; the fork's isSafeToSpeculativelyExecute refuses a GEP on a base not known
; to be non-null on its own, and IVUsers asks it without a context: a null
; check before the loop would prove non-nullness, not that the base is a
; capability, and a GEP the original skips must not be stepped on its behalf.
; A GEP the loop is guaranteed to execute on every iteration (in the header,
; behind only instructions that transfer execution) qualifies without a
; proof. A walk whose start moves with an enclosing loop is left as it is:
; forming it would carry that outer recurrence in an induction variable of
; the enclosing loop, outside its checks, or in a value picked by SCEV
; identity, which is blind to poison. And LSR's expander never hoists a
; non-integral pointer expression above the preheader of the loop it
; reduces, so an invariant start stays below the checks of the enclosing
; loops. Without any of this every capability GEP whose base was not non-null
; on its own (an argument, a loaded pointer -- anything but an alloca, a
; global or a nonnull parameter) was refused, and every such indexed loop
; paid a shift and a cincoffset per access and per iteration (538.imagick_r:
; 6.7M of 103M attempts; 544.nab_r: 711M of 7850M).
;
; The address space is non-integral, so no ptrtoint/inttoptr may appear
; anywhere in the output (the implicit checks above).

target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"

; The access runs on every iteration (the null check before the loop proves
; nothing about the capability and is not what admits it): a pointer phi,
; stepped by 8, and the accesses at +0 and +2 are folded onto it.
; CHECK-LABEL: @guarded(
; CHECK:       for.body:
; CHECK:         [[IV:%[a-z0-9.]+]] = phi ptr addrspace(200) [ %{{[a-z0-9.]+}}, %{{[a-z0-9.]+}} ], [ [[STEP:%[a-z0-9.]+]], %for.body ]
; CHECK-NOT:     shl
; CHECK:         [[STEP]] = getelementptr i8, ptr addrspace(200) [[IV]], i64 8
; CHECK:         ret void
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

; The access is conditional inside the loop (behind the flag test), under a
; null check that dominates the loop: non-nullness does not prove the base
; is a capability (a sealed one is non-null), and a GEP the original skips
; must not be stepped on its behalf -- so %p keeps being indexed from the
; integer counter. The flags walk, which runs on every iteration, gets its
; pointer phi.
; CHECK-LABEL: @guarded_conditional(
; CHECK-NOT:     phi ptr addrspace(200) [ %p,
; CHECK:         getelementptr inbounds nuw i8, ptr addrspace(200) %p, i64
; CHECK-NOT:     phi ptr addrspace(200) [ %p,
; CHECK:         ret
define void @guarded_conditional(ptr addrspace(200) %p, ptr addrspace(200) %flags, i64 %n) addrspace(200) {
entry:
  %nonnull = icmp ne ptr addrspace(200) %p, null
  %positive = icmp sgt i64 %n, 0
  %enter = and i1 %nonnull, %positive
  br i1 %enter, label %for.body, label %for.end

for.body:
  %i = phi i64 [ %inc, %for.inc ], [ 0, %entry ]
  %f = getelementptr inbounds nuw i8, ptr addrspace(200) %flags, i64 %i
  %flag = load i8, ptr addrspace(200) %f, align 1
  %take = icmp ne i8 %flag, 0
  br i1 %take, label %if.then, label %for.inc

if.then:
  %idx = shl nsw i64 %i, 3
  %a0 = getelementptr inbounds nuw i8, ptr addrspace(200) %p, i64 %idx
  %v0 = load i16, ptr addrspace(200) %a0, align 2
  %n0 = xor i16 %v0, -1
  store i16 %n0, ptr addrspace(200) %a0, align 2
  br label %for.inc

for.inc:
  %inc = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %inc, %n
  br i1 %done, label %for.end, label %for.body

for.end:
  ret void
}

; A nonnull parameter is known non-null without any context, so this walk was
; already reduced before this change: a control that the change keeps it.
; CHECK-LABEL: @attributed(
; CHECK:       for.body:
; CHECK:         [[IV:%[a-z0-9.]+]] = phi ptr addrspace(200) [ %p, %{{[a-z0-9.]+}} ], [ [[STEP:%[a-z0-9.]+]], %for.body ]
; CHECK-NOT:     shl
; CHECK:         [[STEP]] = getelementptr i8, ptr addrspace(200) [[IV]], i64 8
; CHECK:         ret void
define void @attributed(ptr addrspace(200) nonnull %p, i64 %n) addrspace(200) {
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

; No null check, but the access sits in the loop header behind nothing that
; could leave: every iteration, the first included, computes the address, so
; a null %p faults on the first iteration either way and the capability may
; be stepped. This is the shape of a function's own parameter walked in a
; counted loop, which the callers checked (ImageMagick's HorizontalFilter,
; nab's mme34).
; CHECK-LABEL: @every_iteration(
; CHECK:       for.body:
; CHECK:         [[IV:%[a-z0-9.]+]] = phi ptr addrspace(200) [ %p, %{{[a-z0-9.]+}} ], [ [[STEP:%[a-z0-9.]+]], %for.body ]
; CHECK-NOT:     shl
; CHECK:         [[STEP]] = getelementptr i8, ptr addrspace(200) [[IV]], i64 8
; CHECK:         ret void
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

; The inner loop of a nest, entered only past a null check in the outer loop,
; walking the parameter from +48 (the shape of issue C-58): the walk gets its
; pointer phi, and its start, invariant in both loops, is formed in the inner
; preheader -- below the check -- not in the entry block.
; CHECK-LABEL: @nested_guard(
; CHECK:       entry:
; CHECK-NOT:     getelementptr
; CHECK:       inner.ph:
; CHECK-NEXT:    [[START:%[a-z0-9.]+]] = getelementptr i8, ptr addrspace(200) %p, i64 48
; CHECK:       inner:
; CHECK:         [[IV:%[a-z0-9.]+]] = phi ptr addrspace(200) [ [[STEP:%[a-z0-9.]+]], %inner ], [ [[START]], %inner.ph ]
; CHECK:         [[STEP]] = getelementptr i8, ptr addrspace(200) [[IV]], i64 16
; CHECK:         ret void
define void @nested_guard(ptr addrspace(200) %p, i64 %n, i64 %m, ptr addrspace(200) %out) addrspace(200) {
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
  %items = getelementptr i8, ptr addrspace(200) %p, i64 48
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

; The C-58 nest with a base that moves with the outer loop: the inner walk's
; start would be a recurrence of the outer loop, which this change does not
; form (the expander would carry it in an induction variable of the outer
; loop, outside the check, or pick a same-SCEV value by identity, blind to
; poison), so the inner loop keeps its integer index and recomputes.
; CHECK-LABEL: @nested_moving(
; CHECK:       inner:
; CHECK-NEXT:    %i = phi i64
; CHECK:         %slot = getelementptr ptr addrspace(200), ptr addrspace(200) %items, i64 %i
; CHECK:         load ptr addrspace(200), ptr addrspace(200) %slot
; CHECK:         ret void
define void @nested_moving(ptr addrspace(200) %p, i64 %n, i64 %m, ptr addrspace(200) %out) addrspace(200) {
entry:
  br label %outer

outer:
  %j = phi i64 [ 0, %entry ], [ %j.next, %outer.latch ]
  %isnull = icmp eq ptr addrspace(200) %p, null
  br i1 %isnull, label %outer.latch, label %inner.ph

inner.ph:
  %row = shl i64 %j, 4
  %off = add i64 %row, 48
  %items = getelementptr i8, ptr addrspace(200) %p, i64 %off
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

; The GEP in the header ahead of an early exit: every iteration, the first
; included, computes the address before the exit test, so the walk gets its
; phi; its start is %p itself (no arithmetic before the loop) and its step
; sits in the latch, which only an iteration past the exit reaches.
; CHECK-LABEL: @header_exit(
; CHECK:       loop.preheader:
; CHECK-NEXT:    br label %loop
; CHECK:       loop:
; CHECK:         [[IV:%[a-z0-9.]+]] = phi ptr addrspace(200) [ %p, %loop.preheader ], [ [[STEP:%[a-z0-9.]+]], %latch ]
; CHECK:         load i16, ptr addrspace(200) [[IV]]
; CHECK:       latch:
; CHECK:         [[STEP]] = getelementptr i8, ptr addrspace(200) [[IV]], i64 8
; CHECK:         ret void
define void @header_exit(ptr addrspace(200) %p, ptr addrspace(200) %stop, i64 %n) addrspace(200) {
entry:
  %positive = icmp sgt i64 %n, 0
  br i1 %positive, label %loop, label %exit

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  %idx = shl nsw i64 %i, 3
  %a = getelementptr inbounds nuw i8, ptr addrspace(200) %p, i64 %idx
  %v = load i16, ptr addrspace(200) %a, align 2
  %leave = icmp eq i16 %v, 0
  br i1 %leave, label %exit, label %latch

latch:
  %w = xor i16 %v, -1
  store i16 %w, ptr addrspace(200) %a, align 2
  %i.next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; A walk the source already expresses as a pointer phi, with an exit ahead of
; the step and no check on the base: left as it is (the same output as before
; this change), the step staying in the latch and nothing formed before the loop.
; CHECK-LABEL: @phi_walk(
; CHECK:       loop.preheader:
; CHECK-NEXT:    br label %loop
; CHECK:       loop:
; CHECK:         %q = phi ptr addrspace(200) [ %q.next, %latch ], [ %p, %loop.preheader ]
; CHECK:       latch:
; CHECK:         %q.next = getelementptr inbounds nuw i8, ptr addrspace(200) %q, i64 8
; CHECK:         ret void
define void @phi_walk(ptr addrspace(200) %p, ptr addrspace(200) %stop, i64 %n) addrspace(200) {
entry:
  %positive = icmp sgt i64 %n, 0
  br i1 %positive, label %loop, label %exit

loop:
  %q = phi ptr addrspace(200) [ %p, %entry ], [ %q.next, %latch ]
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  %flag = load volatile i8, ptr addrspace(200) %stop, align 1
  %leave = icmp ne i8 %flag, 0
  br i1 %leave, label %exit, label %latch

latch:
  %v = load i16, ptr addrspace(200) %q, align 2
  %w = xor i16 %v, -1
  store i16 %w, ptr addrspace(200) %q, align 2
  %q.next = getelementptr inbounds nuw i8, ptr addrspace(200) %q, i64 8
  %i.next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; Three deep, checked in the middle loop, with a base that moves with the
; outer loop: as nested_moving, the inner loop is left as it is.
; CHECK-LABEL: @three_deep(
; CHECK:       inner:
; CHECK-NEXT:    %i = phi i64
; CHECK:         %slot = getelementptr i8, ptr addrspace(200) %items, i64 %i
; CHECK:         load i8, ptr addrspace(200) %slot
; CHECK:         ret void
define void @three_deep(ptr addrspace(200) %p, i64 %n, i64 %m, i64 %o, ptr addrspace(200) %out) addrspace(200) {
entry:
  br label %outer

outer:
  %j = phi i64 [ 0, %entry ], [ %j.next, %outer.latch ]
  br label %mid

mid:
  %k = phi i64 [ 0, %outer ], [ %k.next, %mid.latch ]
  %isnull = icmp eq ptr addrspace(200) %p, null
  br i1 %isnull, label %mid.latch, label %inner.ph

inner.ph:
  %row = shl i64 %j, 6
  %col = shl i64 %k, 4
  %o1 = add i64 %row, %col
  %off = add i64 %o1, 48
  %items = getelementptr i8, ptr addrspace(200) %p, i64 %off
  br label %inner

inner:
  %i = phi i64 [ 0, %inner.ph ], [ %i.next, %inner ]
  %slot = getelementptr i8, ptr addrspace(200) %items, i64 %i
  %v = load i8, ptr addrspace(200) %slot, align 1
  store i8 %v, ptr addrspace(200) %out, align 1
  %i.next = add i64 %i, 1
  %idone = icmp eq i64 %i.next, %n
  br i1 %idone, label %mid.latch, label %inner

mid.latch:
  %k.next = add i64 %k, 1
  %kdone = icmp eq i64 %k.next, %m
  br i1 %kdone, label %outer.latch, label %mid

outer.latch:
  %j.next = add i64 %j, 1
  %jdone = icmp eq i64 %j.next, %o
  br i1 %jdone, label %exit, label %outer

exit:
  ret void
}

; The outer loop walks rows through its own pointer phi and the inner loop
; walks a row: the inner walk's start is the outer recurrence, so it is left
; as it is -- the program's row phi, its inbounds step, and the inner index.
; CHECK-LABEL: @nested_phi_row(
; CHECK:         %row = phi ptr addrspace(200) [ %p, %entry ], [ %row.next, %outer.latch ]
; CHECK:       inner:
; CHECK-NEXT:    %i = phi i64
; CHECK:         %slot = getelementptr i8, ptr addrspace(200) %row, i64 %i
; CHECK:         load i8, ptr addrspace(200) %slot
; CHECK:         %row.next = getelementptr inbounds i8, ptr addrspace(200) %row, i64 16
; CHECK:         ret void
define void @nested_phi_row(ptr addrspace(200) %p, i64 %n, i64 %m, ptr addrspace(200) %out) addrspace(200) {
entry:
  br label %outer

outer:
  %row = phi ptr addrspace(200) [ %p, %entry ], [ %row.next, %outer.latch ]
  %j = phi i64 [ 0, %entry ], [ %j.next, %outer.latch ]
  %isnull = icmp eq ptr addrspace(200) %row, null
  br i1 %isnull, label %outer.latch, label %inner.ph

inner.ph:
  br label %inner

inner:
  %i = phi i64 [ 0, %inner.ph ], [ %i.next, %inner ]
  %slot = getelementptr i8, ptr addrspace(200) %row, i64 %i
  %v = load i8, ptr addrspace(200) %slot, align 1
  store i8 %v, ptr addrspace(200) %out, align 1
  %i.next = add i64 %i, 1
  %idone = icmp eq i64 %i.next, %n
  br i1 %idone, label %outer.latch, label %inner

outer.latch:
  %row.next = getelementptr inbounds i8, ptr addrspace(200) %row, i64 16
  %j.next = add i64 %j, 1
  %odone = icmp eq i64 %j.next, %m
  br i1 %odone, label %exit, label %outer

exit:
  ret void
}

; The row nest with the row phi fed through an inbounds detour, poison where
; the inner walk's base %p + 16*j is not, and with the same SCEV as it: left
; as it is, so nothing can start the walk from the phi.
; CHECK-LABEL: @nested_phi_poison(
; CHECK:         %far = getelementptr inbounds i8, ptr addrspace(200) %p, i64 4096
; CHECK:         %row = phi ptr addrspace(200) [ %back, %entry ], [ %row.next, %outer.latch ]
; CHECK:       inner:
; CHECK-NEXT:    %i = phi i64
; CHECK:         %slot = getelementptr i8, ptr addrspace(200) %base, i64 %i
; CHECK:         load i8, ptr addrspace(200) %slot
; CHECK:         ret void
define void @nested_phi_poison(ptr addrspace(200) %p, i64 %n, i64 %m, ptr addrspace(200) %out) addrspace(200) {
entry:
  %far = getelementptr inbounds i8, ptr addrspace(200) %p, i64 4096
  %back = getelementptr i8, ptr addrspace(200) %far, i64 -4096
  br label %outer

outer:
  %row = phi ptr addrspace(200) [ %back, %entry ], [ %row.next, %outer.latch ]
  %j = phi i64 [ 0, %entry ], [ %j.next, %outer.latch ]
  %fr = freeze ptr addrspace(200) %row
  store ptr addrspace(200) %fr, ptr addrspace(200) %out, align 16
  %j16 = mul i64 %j, 16
  %base = getelementptr i8, ptr addrspace(200) %p, i64 %j16
  %isnull = icmp eq ptr addrspace(200) %p, null
  br i1 %isnull, label %outer.latch, label %inner.ph

inner.ph:
  br label %inner

inner:
  %i = phi i64 [ 0, %inner.ph ], [ %i.next, %inner ]
  %slot = getelementptr i8, ptr addrspace(200) %base, i64 %i
  %v = load i8, ptr addrspace(200) %slot, align 1
  store i8 %v, ptr addrspace(200) %out, align 1
  %i.next = add i64 %i, 1
  %idone = icmp eq i64 %i.next, %n
  br i1 %idone, label %outer.latch, label %inner

outer.latch:
  %row.next = getelementptr inbounds i8, ptr addrspace(200) %row, i64 16
  %j.next = add i64 %j, 1
  %odone = icmp eq i64 %j.next, %m
  br i1 %odone, label %exit, label %outer

exit:
  ret void
}

; The row nest with a poison integer phi of the program whose SCEV is the
; outer recurrence's offset: left as it is, so nothing can pick that phi as
; the walk's offset.
; CHECK-LABEL: @nested_int_phi(
; CHECK:         %bad = phi i64 [ %z, %entry ], [ %bad.next, %outer.latch ]
; CHECK:       inner:
; CHECK-NEXT:    %i = phi i64
; CHECK:         %slot = getelementptr i8, ptr addrspace(200) %row, i64 %i
; CHECK:         load i8, ptr addrspace(200) %slot
; CHECK:         ret void
define void @nested_int_phi(ptr addrspace(200) %p, i64 %n, i64 %m, ptr addrspace(200) %out, i64 %x) addrspace(200) {
entry:
  %t = add nsw i64 %x, 1
  %z = sub i64 %t, %t
  br label %outer

outer:
  %row = phi ptr addrspace(200) [ %p, %entry ], [ %row.next, %outer.latch ]
  %bad = phi i64 [ %z, %entry ], [ %bad.next, %outer.latch ]
  %j = phi i64 [ 0, %entry ], [ %j.next, %outer.latch ]
  %f = freeze i64 %bad
  store i64 %f, ptr addrspace(200) %out, align 8
  %isnull = icmp eq ptr addrspace(200) %row, null
  br i1 %isnull, label %outer.latch, label %inner.ph

inner.ph:
  br label %inner

inner:
  %i = phi i64 [ 0, %inner.ph ], [ %i.next, %inner ]
  %slot = getelementptr i8, ptr addrspace(200) %row, i64 %i
  %v = load i8, ptr addrspace(200) %slot, align 1
  store i8 %v, ptr addrspace(200) %out, align 1
  %i.next = add i64 %i, 1
  %idone = icmp eq i64 %i.next, %n
  br i1 %idone, label %outer.latch, label %inner

outer.latch:
  %row.next = getelementptr inbounds i8, ptr addrspace(200) %row, i64 16
  %bad.next = add i64 %bad, 16
  %j.next = add i64 %j, 1
  %odone = icmp eq i64 %j.next, %m
  br i1 %odone, label %exit, label %outer

exit:
  ret void
}

; An exit test that compares the walked address with an end pointer: the walk
; gets its pointer phi, and LSR rewrites the comparison into a count-down
; compared through a null-based byte GEP, which the backend lowers as integer
; arithmetic (C-40; the CodeGen test checks that no cincoffset on the zero
; register comes of it).
; CHECK-LABEL: @cmp_end(
; CHECK:       loop:
; CHECK:         [[IV:%[a-z0-9.]+]] = phi ptr addrspace(200) [ %p, %loop.preheader ], [ [[STEP:%[a-z0-9.]+]], %loop ]
; CHECK:         load i16, ptr addrspace(200) [[IV]], align 2
; CHECK:         [[END:%[a-z0-9.]+]] = getelementptr i8, ptr addrspace(200) null, i64 %
; CHECK:         [[STEP]] = getelementptr i8, ptr addrspace(200) [[IV]], i64 8
; CHECK:         icmp eq ptr addrspace(200) [[END]], null
; CHECK:         ret i16
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

; The nab shape: the inner loop starts where the outer counter stands
; (j = i + 1) and walks x[3*j]. Its start is a recurrence of the outer loop,
; so the capability address stays a plain use of its index -- which LSR
; still strength-reduces as an integer, carried by an integer recurrence of
; the outer loop (integer arithmetic cannot fault) -- and the base stays %x.
; CHECK-LABEL: @nested_index_start(
; CHECK-NOT:     phi ptr addrspace(200)
; CHECK:       inner:
; CHECK-NOT:     phi ptr addrspace(200)
; CHECK:         %slot = getelementptr inbounds double, ptr addrspace(200) %x, i64 %
; CHECK:         load double, ptr addrspace(200) %slot
; CHECK-NOT:     phi ptr addrspace(200)
; CHECK:         ret void
define void @nested_index_start(ptr addrspace(200) %x, i64 %n, ptr addrspace(200) %out) addrspace(200) {
entry:
  br label %outer

outer:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  %i.next = add nuw nsw i64 %i, 1
  %enter = icmp slt i64 %i.next, %n
  br i1 %enter, label %inner.ph, label %outer.latch

inner.ph:
  br label %inner

inner:
  %j = phi i64 [ %i.next, %inner.ph ], [ %j.next, %inner ]
  %j3 = mul nsw i64 %j, 3
  %slot = getelementptr inbounds double, ptr addrspace(200) %x, i64 %j3
  %v = load double, ptr addrspace(200) %slot, align 8
  store double %v, ptr addrspace(200) %out, align 8
  %j.next = add nuw nsw i64 %j, 1
  %jdone = icmp eq i64 %j.next, %n
  br i1 %jdone, label %outer.latch, label %inner

outer.latch:
  %odone = icmp eq i64 %i.next, %n
  br i1 %odone, label %exit, label %outer

exit:
  ret void
}

; A pointer phi of the program whose start moves with the outer loop is left
; as it is (its start is a recurrence of the outer loop; before this change
; LSR could re-form that start as an induction variable of the outer loop);
; the inner loop's counter is still reduced.
; CHECK-LABEL: @ptr_phi_outer(
; CHECK:       outer:
; CHECK-NEXT:    %j = phi i64
; CHECK-NEXT:    %j16 = mul i64 %j, 16
; CHECK-NEXT:    %row = getelementptr i8, ptr addrspace(200) %p, i64 %j16
; CHECK:       inner:
; CHECK:         %q = phi ptr addrspace(200) [ %row, %inner.ph ], [ %q.next, %inner ]
; CHECK:         load i8, ptr addrspace(200) %q, align 1
; CHECK:         %q.next = getelementptr i8, ptr addrspace(200) %q, i64 1
; CHECK:         ret void
define void @ptr_phi_outer(ptr addrspace(200) %p, i64 %n, i64 %m, ptr addrspace(200) %out) addrspace(200) {
entry:
  br label %outer

outer:
  %j = phi i64 [ 0, %entry ], [ %j.next, %outer.latch ]
  %j16 = mul i64 %j, 16
  %row = getelementptr i8, ptr addrspace(200) %p, i64 %j16
  %isnull = icmp eq ptr addrspace(200) %p, null
  br i1 %isnull, label %outer.latch, label %inner.ph

inner.ph:
  br label %inner

inner:
  %q = phi ptr addrspace(200) [ %row, %inner.ph ], [ %q.next, %inner ]
  %i = phi i64 [ 0, %inner.ph ], [ %i.next, %inner ]
  %v = load i8, ptr addrspace(200) %q, align 1
  %q2 = getelementptr i8, ptr addrspace(200) %q, i64 4
  %w = load i8, ptr addrspace(200) %q2, align 1
  %s = add i8 %v, %w
  store i8 %s, ptr addrspace(200) %out, align 1
  %q.next = getelementptr i8, ptr addrspace(200) %q, i64 1
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

; The null check is inside the loop body: it proves nothing at the preheader,
; the access does not run on every iteration, and stepping the capability in
; the latch would be arithmetic on a %p the original program only touches
; behind that check -- the line C-19 draws. The address stays inside the
; guarded block, derived from %p.
; CHECK-LABEL: @checked_inside(
; CHECK-NOT:     getelementptr
; CHECK:       for.body:
; CHECK-NOT:     phi ptr addrspace(200)
; CHECK:       if.then:
; CHECK:         getelementptr inbounds nuw i8, ptr addrspace(200) %p, i64 %{{[a-z0-9.]+}}
; CHECK-NOT:     phi ptr addrspace(200)
; CHECK:         ret void
define void @checked_inside(ptr addrspace(200) %p, i64 %n) addrspace(200) {
entry:
  %positive = icmp sgt i64 %n, 0
  br i1 %positive, label %for.body, label %for.end

for.body:
  %i = phi i64 [ %inc, %for.inc ], [ 0, %entry ]
  %nonnull = icmp ne ptr addrspace(200) %p, null
  br i1 %nonnull, label %if.then, label %for.inc

if.then:
  %idx = shl nsw i64 %i, 3
  %a0 = getelementptr inbounds nuw i8, ptr addrspace(200) %p, i64 %idx
  %v0 = load i16, ptr addrspace(200) %a0, align 2
  %n0 = xor i16 %v0, -1
  store i16 %n0, ptr addrspace(200) %a0, align 2
  br label %for.inc

for.inc:
  %inc = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %inc, %n
  br i1 %done, label %for.end, label %for.body

for.end:
  ret void
}

; An exit before the access: the first iteration may leave without computing
; the address, and a start in the preheader would be arithmetic on %p where
; the original program did none.
; CHECK-LABEL: @early_exit(
; CHECK-NOT:     getelementptr
; CHECK:       for.body:
; CHECK-NOT:     phi ptr addrspace(200)
; CHECK:         getelementptr inbounds nuw i8, ptr addrspace(200) %p, i64 %{{[a-z0-9.]+}}
; CHECK-NOT:     phi ptr addrspace(200)
; CHECK:         ret void
define void @early_exit(ptr addrspace(200) %p, ptr addrspace(200) %stop, i64 %n) addrspace(200) {
entry:
  %positive = icmp sgt i64 %n, 0
  br i1 %positive, label %for.body, label %for.end

for.body:
  %i = phi i64 [ %inc, %for.inc ], [ 0, %entry ]
  %flag = load volatile i8, ptr addrspace(200) %stop, align 1
  %leave = icmp ne i8 %flag, 0
  br i1 %leave, label %for.end, label %for.inc

for.inc:
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

; A call before the access that may not return (exit, longjmp; nounwind, so
; only its returning is in doubt): the first iteration may end inside it
; without computing the address, so the capability is not stepped. The walk
; starts at +2, so a start in the preheader would be real arithmetic on a %p
; the original program never touched, where it exits normally.
; CHECK-LABEL: @call_before(
; CHECK-NOT:     getelementptr
; CHECK:       for.body:
; CHECK-NOT:     phi ptr addrspace(200)
; CHECK:         call addrspace(200) void @may_not_return
; CHECK:         getelementptr inbounds nuw i8, ptr addrspace(200) %p, i64 %{{[a-z0-9.]+}}
; CHECK-NOT:     phi ptr addrspace(200)
; CHECK:         ret void
define void @call_before(ptr addrspace(200) %p, i64 %n) addrspace(200) {
entry:
  %positive = icmp sgt i64 %n, 0
  br i1 %positive, label %for.body, label %for.end

for.body:
  %i = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  call addrspace(200) void @may_not_return(i64 %i)
  %idx = shl nsw i64 %i, 3
  %off = add nsw i64 %idx, 2
  %a0 = getelementptr inbounds nuw i8, ptr addrspace(200) %p, i64 %off
  %v0 = load i16, ptr addrspace(200) %a0, align 2
  %n0 = xor i16 %v0, -1
  store i16 %n0, ptr addrspace(200) %a0, align 2
  %inc = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %inc, %n
  br i1 %done, label %for.end, label %for.body

for.end:
  ret void
}

declare void @may_not_return(i64) addrspace(200) nounwind

; The last address is live out of the loop: the walk gets its phi (LSR may
; normalize its start so that the stepped value is the last address), and the
; value handed on is that stepped capability, derived from %p (no cast).
; CHECK-LABEL: @live_out(
; CHECK:       for.body:
; CHECK:         [[IV:%[a-z0-9.]+]] = phi ptr addrspace(200) [ %{{[a-z0-9.]+}}, %{{[a-z0-9.]+}} ], [ [[STEP:%[a-z0-9.]+]], %for.body ]
; CHECK:         [[STEP]] = getelementptr i8, ptr addrspace(200) [[IV]], i64 8
; CHECK:         [[OUT:%[a-z0-9.]+]] = phi ptr addrspace(200) [ [[STEP]], %for.body ]
; CHECK:       for.end:
; CHECK:         [[LAST:%[a-z0-9.]+]] = phi ptr addrspace(200) [ %p, %entry ], [ [[OUT]], %{{[a-z0-9.]+}} ]
; CHECK:         ret ptr addrspace(200) [[LAST]]
define ptr addrspace(200) @live_out(ptr addrspace(200) %p, i64 %n) addrspace(200) {
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

; ---------------------------------------------------------------------------
; The window. A capability moved outside the range its compressed bounds can
; represent loses its tag (R-60). The representable window starts at least
; 2 KiB below the base (2^(E+11), decompress_bounds in ariane_pkg.sv) and
; ends at least about 4 KiB above the end, so a cursor LSR forms where the
; program formed none must stay within 2 KiB of an address the program does
; form. A use after the loop reads the post-incremented value, so LSR expresses
; it on an IV whose start is the first address less one step and hands on the
; stepped capability. With a step of a few hundred bytes that start is inside
; the window around the object; with 4 KiB, 64 KiB or a variable step it need
; not be, and the value handed on would have lost its tag where the program's
; had not. Such a value stays computed by the program (the loop's own accesses
; are reduced as before).
; ---------------------------------------------------------------------------

; A step of 2 KiB is the bound: the value after the loop is the stepped
; capability less one step, both inside the window.
; CHECK-LABEL: @live_out_2k(
; CHECK-NOT:   getelementptr i8, ptr addrspace(200) %p, i64 -
; CHECK:       for.body:
; CHECK:         [[IV:%[a-z0-9.]+]] = phi ptr addrspace(200) [ %p, %{{[a-z0-9.]+}} ], [ [[STEP:%[a-z0-9.]+]], %for.body ]
; CHECK:         [[STEP]] = getelementptr i8, ptr addrspace(200) [[IV]], i64 2048
; CHECK:         getelementptr i8, ptr addrspace(200) [[STEP]], i64 -2048
; CHECK:         ret ptr addrspace(200)
define ptr addrspace(200) @live_out_2k(ptr addrspace(200) %p, i64 %n) addrspace(200) {
entry:
  %nonnull = icmp ne ptr addrspace(200) %p, null
  %positive = icmp sgt i64 %n, 0
  %enter = and i1 %nonnull, %positive
  br i1 %enter, label %for.body, label %for.end

for.body:
  %i = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %idx = shl nsw i64 %i, 11
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

; Above the bound no capability at p-4096 is formed (an object of 4 to 8 KiB
; has less than 4 KiB below its base in the window); the value after the loop
; is the program's own address.
; CHECK-LABEL: @live_out_4k(
; CHECK-NOT:   getelementptr i8, ptr addrspace(200) %p, i64 -
; CHECK:       for.body:
; CHECK:         [[IV:%[a-z0-9.]+]] = phi ptr addrspace(200) [ %p, %{{[a-z0-9.]+}} ], [ [[STEP:%[a-z0-9.]+]], %for.body ]
; CHECK:         [[A0:%[a-z0-9.]+]] = getelementptr inbounds nuw i8, ptr addrspace(200) %p, i64 %idx
; CHECK:         [[STEP]] = getelementptr i8, ptr addrspace(200) [[IV]], i64 4096
; CHECK-NOT:     i64 -4096
; CHECK:         phi ptr addrspace(200) [ [[A0]], %for.body ]
; CHECK:         ret ptr addrspace(200)
define ptr addrspace(200) @live_out_4k(ptr addrspace(200) %p, i64 %n) addrspace(200) {
entry:
  %nonnull = icmp ne ptr addrspace(200) %p, null
  %positive = icmp sgt i64 %n, 0
  %enter = and i1 %nonnull, %positive
  br i1 %enter, label %for.body, label %for.end

for.body:
  %i = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %idx = shl nsw i64 %i, 12
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

; The same with a 64 KiB step.
; CHECK-LABEL: @live_out_64k(
; CHECK-NOT:   getelementptr i8, ptr addrspace(200) %p, i64 -
; CHECK:       for.body:
; CHECK:         [[A0:%[a-z0-9.]+]] = getelementptr inbounds nuw i8, ptr addrspace(200) %p, i64 %idx
; CHECK-NOT:     i64 -65536
; CHECK:         phi ptr addrspace(200) [ [[A0]], %for.body ]
; CHECK:         ret ptr addrspace(200)
define ptr addrspace(200) @live_out_64k(ptr addrspace(200) %p, i64 %n) addrspace(200) {
entry:
  %nonnull = icmp ne ptr addrspace(200) %p, null
  %positive = icmp sgt i64 %n, 0
  %enter = and i1 %nonnull, %positive
  br i1 %enter, label %for.body, label %for.end

for.body:
  %i = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %idx = shl nsw i64 %i, 16
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

; A step known only at run time: nothing bounds the start, so the value after
; the loop is the program's own address.
; CHECK-LABEL: @live_out_variable(
; CHECK-NOT:   getelementptr i8, ptr addrspace(200) %p,
; CHECK:       for.body:
; CHECK:         [[A0:%[a-z0-9.]+]] = getelementptr inbounds i8, ptr addrspace(200) %p, i64 %idx
; CHECK:         phi ptr addrspace(200) [ [[A0]], %for.body ]
; CHECK:         ret ptr addrspace(200)
define ptr addrspace(200) @live_out_variable(ptr addrspace(200) %p, i64 %n, i64 %s) addrspace(200) {
entry:
  %nonnull = icmp ne ptr addrspace(200) %p, null
  %positive = icmp sgt i64 %n, 0
  %enter = and i1 %nonnull, %positive
  br i1 %enter, label %for.body, label %for.end

for.body:
  %i = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %idx = mul nsw i64 %i, %s
  %a0 = getelementptr inbounds i8, ptr addrspace(200) %p, i64 %idx
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

; A walk downwards by 4 KiB, last address live out: the program's address too.
; CHECK-LABEL: @live_out_down_4k(
; CHECK:       for.body:
; CHECK:         [[A0:%[a-z0-9.]+]] = getelementptr inbounds nuw i8, ptr addrspace(200) %p, i64 %idx
; CHECK:         getelementptr i8, ptr addrspace(200) %{{[a-z0-9.]+}}, i64 -4096
; CHECK:         phi ptr addrspace(200) [ [[A0]], %for.body ]
; CHECK:         ret ptr addrspace(200)
define ptr addrspace(200) @live_out_down_4k(ptr addrspace(200) %p, i64 %n) addrspace(200) {
entry:
  %nonnull = icmp ne ptr addrspace(200) %p, null
  %positive = icmp sgt i64 %n, 0
  %enter = and i1 %nonnull, %positive
  br i1 %enter, label %for.body, label %for.end

for.body:
  %i = phi i64 [ %dec, %for.body ], [ %n, %entry ]
  %k = add nsw i64 %i, -1
  %idx = shl nsw i64 %k, 12
  %a0 = getelementptr inbounds nuw i8, ptr addrspace(200) %p, i64 %idx
  %v0 = load i16, ptr addrspace(200) %a0, align 2
  %n0 = xor i16 %v0, -1
  store i16 %n0, ptr addrspace(200) %a0, align 2
  %dec = add nsw i64 %i, -1
  %done = icmp eq i64 %dec, 0
  br i1 %done, label %for.end, label %for.body

for.end:
  %last = phi ptr addrspace(200) [ %p, %entry ], [ %a0, %for.body ]
  ret ptr addrspace(200) %last
}

; A constant offset in the index is the program's own address, not a bias:
; a[i+K] starts the walk where the first access is, and no capability below
; the base (-65536) is formed.
; CHECK-LABEL: @plus_k(
; CHECK:       for.body.preheader:
; CHECK:         getelementptr i8, ptr addrspace(200) %p, i64 65536
; CHECK-NOT:     i64 -65536
; CHECK:         ret void
define void @plus_k(ptr addrspace(200) %p, i64 %n) addrspace(200) {
entry:
  %positive = icmp sgt i64 %n, 0
  br i1 %positive, label %for.body, label %for.end

for.body:
  %i = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %idx = add nsw i64 %i, 65536
  %a0 = getelementptr inbounds i8, ptr addrspace(200) %p, i64 %idx
  %v0 = load i8, ptr addrspace(200) %a0, align 1
  %n0 = xor i8 %v0, -1
  store i8 %n0, ptr addrspace(200) %a0, align 1
  %inc = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %inc, %n
  br i1 %done, label %for.end, label %for.body

for.end:
  ret void
}

; a[i] and a[i-K] together (the loop runs from K): the walk starts at the base,
; and the second access is a positive offset from it, not a capability below
; the base.
; CHECK-LABEL: @minus_k(
; CHECK-NOT:   i64 -65536
; CHECK:       for.body:
; CHECK:         [[IV:%[a-z0-9.]+]] = phi ptr addrspace(200) [ %p, %{{[a-z0-9.]+}} ], [ %{{[a-z0-9.]+}}, %for.body ]
; CHECK-NOT:     i64 -65536
; CHECK:         ret void
define void @minus_k(ptr addrspace(200) %p, i64 %n) addrspace(200) {
entry:
  %positive = icmp sgt i64 %n, 65536
  br i1 %positive, label %for.body, label %for.end

for.body:
  %i = phi i64 [ %inc, %for.body ], [ 65536, %entry ]
  %a0 = getelementptr inbounds i8, ptr addrspace(200) %p, i64 %i
  %v0 = load i8, ptr addrspace(200) %a0, align 1
  %j = add nsw i64 %i, -65536
  %a1 = getelementptr inbounds i8, ptr addrspace(200) %p, i64 %j
  store i8 %v0, ptr addrspace(200) %a1, align 1
  %inc = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %inc, %n
  br i1 %done, label %for.end, label %for.body

for.end:
  ret void
}

; A reversed walk: the IV starts at the last element and never goes below the
; base, so no constant offset of four or more digits below it appears.
; CHECK-LABEL: @reversed(
; CHECK:       for.body:
; CHECK:         phi ptr addrspace(200)
; CHECK:         getelementptr i8, ptr addrspace(200) %{{[a-z0-9.]+}}, i64 -8
; CHECK-NOT:     i64 -{{[0-9][0-9][0-9][0-9]+}}
; CHECK:         ret void
define void @reversed(ptr addrspace(200) %p, i64 %n) addrspace(200) {
entry:
  %positive = icmp sgt i64 %n, 0
  br i1 %positive, label %for.body, label %for.end

for.body:
  %i = phi i64 [ %dec, %for.body ], [ %n, %entry ]
  %k = add nsw i64 %i, -1
  %idx = shl nsw i64 %k, 3
  %a0 = getelementptr inbounds i8, ptr addrspace(200) %p, i64 %idx
  %v0 = load i64, ptr addrspace(200) %a0, align 8
  %n0 = xor i64 %v0, -1
  store i64 %n0, ptr addrspace(200) %a0, align 8
  %dec = add nsw i64 %i, -1
  %done = icmp eq i64 %dec, 0
  br i1 %done, label %for.end, label %for.body

for.end:
  ret void
}

; The loop-terminating comparison of a walked pointer against a bound held in an
; independent argument can be expressed on the post-incremented value, whose
; normalized start is the original start less one step. As for a value handed on
; out of the loop, that start must stay in the window around the object: where the
; step is larger than 2 KiB, or not a constant, the comparison keeps the
; pre-increment value instead.

; A step of 8 bytes keeps the post-increment comparison: its start is a few bytes
; below the object, inside the window.
; CHECK-LABEL: @cmp_end_8(
; CHECK:         getelementptr i8, ptr addrspace(200) %p, i64 -8
; CHECK:       loop:
; CHECK:         [[IV:%[a-z0-9.]+]] = phi ptr addrspace(200) [ [[NEXT:%[a-z0-9.]+]], %loop ]
; CHECK:         [[NEXT]] = getelementptr i8, ptr addrspace(200) [[IV]], i64 8
; CHECK:         icmp eq ptr addrspace(200) [[NEXT]], %end
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

; 2048 bytes is the bound: no capability at p-2048 is formed in the preheader, and
; the comparison keeps its post-increment form.
; CHECK-LABEL: @cmp_end_2k(
; CHECK-NOT:   getelementptr i8, ptr addrspace(200) %p, i64 -
; CHECK:       loop:
; CHECK:         [[AHEAD:%[a-z0-9.]+]] = getelementptr i8, ptr addrspace(200) %lsr.iv, i64 2048
; CHECK:         [[BACK:%[a-z0-9.]+]] = getelementptr i8, ptr addrspace(200) [[AHEAD]], i64 -2048
; CHECK:         icmp eq ptr addrspace(200) [[BACK]], %end
define i16 @cmp_end_2k(ptr addrspace(200) nonnull %p, ptr addrspace(200) %end) addrspace(200) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %acc = phi i16 [ 0, %entry ], [ %acc.next, %loop ]
  %idx = shl nsw i64 %i, 11
  %a = getelementptr inbounds nuw i8, ptr addrspace(200) %p, i64 %idx
  %v = load i16, ptr addrspace(200) %a, align 2
  %acc.next = add i16 %acc, %v
  %i.next = add nuw nsw i64 %i, 1
  %done = icmp eq ptr addrspace(200) %a, %end
  br i1 %done, label %exit, label %loop

exit:
  ret i16 %acc.next
}

; Above the bound, 2056 bytes: the comparison uses the pre-increment value.
; CHECK-LABEL: @cmp_end_2056(
; CHECK-NOT:   getelementptr i8, ptr addrspace(200) %p, i64 -
; CHECK:       loop:
; CHECK:         [[IV:%[a-z0-9.]+]] = phi ptr addrspace(200) [ [[STEP:%[a-z0-9.]+]], %loop ], [ %p, %entry ]
; CHECK:         icmp eq ptr addrspace(200) [[IV]], %end
; CHECK:         [[STEP]] = getelementptr i8, ptr addrspace(200) [[IV]], i64 2056
define i16 @cmp_end_2056(ptr addrspace(200) nonnull %p, ptr addrspace(200) %end) addrspace(200) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %acc = phi i16 [ 0, %entry ], [ %acc.next, %loop ]
  %idx = mul nsw i64 %i, 2056
  %a = getelementptr inbounds nuw i8, ptr addrspace(200) %p, i64 %idx
  %v = load i16, ptr addrspace(200) %a, align 2
  %acc.next = add i16 %acc, %v
  %i.next = add nuw nsw i64 %i, 1
  %done = icmp eq ptr addrspace(200) %a, %end
  br i1 %done, label %exit, label %loop

exit:
  ret i16 %acc.next
}

; A 4 KiB step: an object of 4 to 8 KiB has less than 4 KiB below its base in the window, so p-4096 is not formed.
; CHECK-LABEL: @cmp_end_4k(
; CHECK-NOT:   getelementptr i8, ptr addrspace(200) %p, i64 -
; CHECK:       loop:
; CHECK:         [[IV:%[a-z0-9.]+]] = phi ptr addrspace(200) [ [[STEP:%[a-z0-9.]+]], %loop ], [ %p, %entry ]
; CHECK:         icmp eq ptr addrspace(200) [[IV]], %end
; CHECK:         [[STEP]] = getelementptr i8, ptr addrspace(200) [[IV]], i64 4096
define i16 @cmp_end_4k(ptr addrspace(200) nonnull %p, ptr addrspace(200) %end) addrspace(200) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %acc = phi i16 [ 0, %entry ], [ %acc.next, %loop ]
  %idx = shl nsw i64 %i, 12
  %a = getelementptr inbounds nuw i8, ptr addrspace(200) %p, i64 %idx
  %v = load i16, ptr addrspace(200) %a, align 2
  %acc.next = add i16 %acc, %v
  %i.next = add nuw nsw i64 %i, 1
  %done = icmp eq ptr addrspace(200) %a, %end
  br i1 %done, label %exit, label %loop

exit:
  ret i16 %acc.next
}

; The same with a 64 KiB step.
; CHECK-LABEL: @cmp_end_64k(
; CHECK-NOT:   getelementptr i8, ptr addrspace(200) %p, i64 -
; CHECK:       loop:
; CHECK:         [[IV:%[a-z0-9.]+]] = phi ptr addrspace(200) [ [[STEP:%[a-z0-9.]+]], %loop ], [ %p, %entry ]
; CHECK:         icmp eq ptr addrspace(200) [[IV]], %end
; CHECK:         [[STEP]] = getelementptr i8, ptr addrspace(200) [[IV]], i64 65536
define i16 @cmp_end_64k(ptr addrspace(200) nonnull %p, ptr addrspace(200) %end) addrspace(200) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %acc = phi i16 [ 0, %entry ], [ %acc.next, %loop ]
  %idx = shl nsw i64 %i, 16
  %a = getelementptr inbounds nuw i8, ptr addrspace(200) %p, i64 %idx
  %v = load i16, ptr addrspace(200) %a, align 2
  %acc.next = add i16 %acc, %v
  %i.next = add nuw nsw i64 %i, 1
  %done = icmp eq ptr addrspace(200) %a, %end
  br i1 %done, label %exit, label %loop

exit:
  ret i16 %acc.next
}

; A walk downward by 4 KiB.
; CHECK-LABEL: @cmp_end_down_4k(
; CHECK-NOT:   getelementptr i8, ptr addrspace(200) %p, i64 -
; CHECK:       loop:
; CHECK:         [[IV:%[a-z0-9.]+]] = phi ptr addrspace(200) [ [[STEP:%[a-z0-9.]+]], %loop ], [ %p, %entry ]
; CHECK:         icmp eq ptr addrspace(200) [[IV]], %end
; CHECK:         [[STEP]] = getelementptr i8, ptr addrspace(200) [[IV]], i64 -4096
define i16 @cmp_end_down_4k(ptr addrspace(200) nonnull %p, ptr addrspace(200) %end) addrspace(200) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %acc = phi i16 [ 0, %entry ], [ %acc.next, %loop ]
  %idx = mul nsw i64 %i, -4096
  %a = getelementptr inbounds nuw i8, ptr addrspace(200) %p, i64 %idx
  %v = load i16, ptr addrspace(200) %a, align 2
  %acc.next = add i16 %acc, %v
  %i.next = add nuw nsw i64 %i, 1
  %done = icmp eq ptr addrspace(200) %a, %end
  br i1 %done, label %exit, label %loop

exit:
  ret i16 %acc.next
}

; A run-time step.
; CHECK-LABEL: @cmp_end_variable(
; CHECK-NOT:   getelementptr i8, ptr addrspace(200) %p, i64 -
; CHECK:       loop:
; CHECK:         [[IV:%[a-z0-9.]+]] = phi ptr addrspace(200) [ [[STEP:%[a-z0-9.]+]], %loop ], [ %p, %entry ]
; CHECK:         icmp eq ptr addrspace(200) [[IV]], %end
; CHECK:         [[STEP]] = getelementptr i8, ptr addrspace(200) [[IV]], i64 %s
define i16 @cmp_end_variable(ptr addrspace(200) nonnull %p, ptr addrspace(200) %end, i64 %s) addrspace(200) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %acc = phi i16 [ 0, %entry ], [ %acc.next, %loop ]
  %idx = mul nsw i64 %i, %s
  %a = getelementptr inbounds nuw i8, ptr addrspace(200) %p, i64 %idx
  %v = load i16, ptr addrspace(200) %a, align 2
  %acc.next = add i16 %acc, %v
  %i.next = add nuw nsw i64 %i, 1
  %done = icmp eq ptr addrspace(200) %a, %end
  br i1 %done, label %exit, label %loop

exit:
  ret i16 %acc.next
}

