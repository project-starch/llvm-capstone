; RUN: opt -mtriple=capstone64 -passes=loop-reduce -verify-each -S %s | FileCheck %s --implicit-check-not=ptrtoint --implicit-check-not=inttoptr
; RUN: llc -mtriple=capstone64 -mattr=+m -verify-machineinstrs -O2 < %s | FileCheck %s --check-prefix=ASM
;
; Compare-only endpoints may lose their tag. They never seed an accessed
; recurrence, and the post-increment cursor is only accessed on the backedge.
; For the descending loop at b=0, n=1 under phys56, the load is at b and the
; endpoint is b-8: it is compared, never dereferenced or stepped back to b.

target datalayout = "e-p:64:64-p200:128:128:128:64-i64:64-n32:64-ni:200"

; CHECK-LABEL: @ascending(
; CHECK: [[END:%.*]] = getelementptr i64, ptr addrspace(200) %b,
; CHECK: loop:
; CHECK-NOT: phi i64
; CHECK: [[IV:%.*]] = phi ptr addrspace(200)
; CHECK: load volatile i64, ptr addrspace(200) [[IV]]
; CHECK: [[NEXT:%.*]] = getelementptr i8, ptr addrspace(200) [[IV]], i64 8
; CHECK: icmp ne ptr addrspace(200) [[NEXT]], [[END]]
; ASM-LABEL: ascending:
; ASM: [[LOOP:.LBB[0-9_]+]]:
; ASM-NOT: addi
; ASM: ld zero, 0([[P:[a-z0-9]+]])
; ASM-NEXT: cincoffsetimm [[P]], [[P]], 8
; ASM-NEXT: bne [[P]], {{[a-z0-9]+}}, [[LOOP]]
define void @ascending(ptr addrspace(200) %b, i32 %n) {
entry:
  %positive = icmp sgt i32 %n, 0
  br i1 %positive, label %preheader, label %exit
preheader:
  %limit = zext i32 %n to i64
  br label %loop
loop:
  %i = phi i64 [0, %preheader], [%next, %loop]
  %p = getelementptr inbounds i64, ptr addrspace(200) %b, i64 %i
  %v = load volatile i64, ptr addrspace(200) %p
  %next = add nuw nsw i64 %i, 1
  %more = icmp ult i64 %next, %limit
  br i1 %more, label %loop, label %exit
exit:
  ret void
}

; CHECK-LABEL: @descending(
; CHECK: [[END:%.*]] = getelementptr i64, ptr addrspace(200) %b, i64 -1
; CHECK: loop:
; CHECK-NOT: phi i64
; CHECK: [[IV:%.*]] = phi ptr addrspace(200)
; CHECK: load volatile i64, ptr addrspace(200) [[IV]]
; CHECK: [[NEXT:%.*]] = getelementptr i8, ptr addrspace(200) [[IV]], i64 -8
; CHECK: icmp eq ptr addrspace(200) [[NEXT]], [[END]]
; ASM-LABEL: descending:
; ASM: [[LOOP:.LBB[0-9_]+]]:
; ASM-NOT: addi
; ASM: ld zero, 0([[P:[a-z0-9]+]])
; ASM-NEXT: cincoffsetimm [[P]], [[P]], -8
; ASM-NEXT: bne [[P]], {{[a-z0-9]+}}, [[LOOP]]
define void @descending(ptr addrspace(200) %b, i32 %n) {
entry:
  %positive = icmp sgt i32 %n, 0
  br i1 %positive, label %preheader, label %exit
preheader:
  %limit = zext i32 %n to i64
  %start = add i64 %limit, -1
  br label %loop
loop:
  %i = phi i64 [%start, %preheader], [%next, %loop]
  %p = getelementptr inbounds i64, ptr addrspace(200) %b, i64 %i
  %v = load volatile i64, ptr addrspace(200) %p
  %next = add nsw i64 %i, -1
  %done = icmp eq i64 %i, 0
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

; An unbounded i64 trip count can exceed the period of an eight-byte step.
; CHECK-LABEL: @unbounded(
; CHECK-NOT: lsr.cap.end
; CHECK: phi i64
; CHECK: icmp {{eq|ne}} i64
; CHECK-NOT: lsr.cap.end
; CHECK: ret void
define void @unbounded(ptr addrspace(200) %b, i64 %n) {
entry:
  br label %loop
loop:
  %i = phi i64 [0, %entry], [%next, %loop]
  %p = getelementptr inbounds i64, ptr addrspace(200) %b, i64 %i
  %v = load volatile i64, ptr addrspace(200) %p
  %next = add i64 %i, 1
  %done = icmp eq i64 %next, %n
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

; A byte step of 2^63 repeats every two iterations. Three comparisons would
; hit the endpoint on the first iteration, before the integer loop exits.
; CHECK-LABEL: @repeating_cursor(
; CHECK-NOT: lsr.cap.end
; CHECK: icmp {{eq|ne}} i64
; CHECK-NOT: lsr.cap.end
; CHECK: ret void
define void @repeating_cursor(ptr addrspace(200) %b) {
entry:
  br label %loop
loop:
  %i = phi i64 [0, %entry], [%next, %loop]
  %offset = mul i64 %i, -9223372036854775808
  %p = getelementptr inbounds i8, ptr addrspace(200) %b, i64 %offset
  %v = load volatile i8, ptr addrspace(200) %p
  %next = add i64 %i, 1
  %done = icmp eq i64 %next, 3
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

; Conditional accesses cannot establish an accessed recurrence, even if the
; integer exit count is small and exact.
; CHECK-LABEL: @conditional(
; CHECK-NOT: phi ptr addrspace(200)
; CHECK-NOT: lsr.cap.end
; CHECK: ret void
define void @conditional(ptr addrspace(200) %b, i1 %take) {
entry:
  br label %loop
loop:
  %i = phi i64 [0, %entry], [%next, %latch]
  br i1 %take, label %access, label %latch
access:
  %p = getelementptr inbounds i64, ptr addrspace(200) %b, i64 %i
  %v = load volatile i64, ptr addrspace(200) %p
  br label %latch
latch:
  %next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %next, 4
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

; Replacing all bases first permits removal of a shared copy-loop counter.
; CHECK-LABEL: @copy_offset(
; CHECK: loop:
; CHECK-NOT: phi i64
; CHECK: [[SRC:%.*]] = phi ptr addrspace(200)
; CHECK: [[DST:%.*]] = phi ptr addrspace(200)
; CHECK: load i32, ptr addrspace(200) [[SRC]]
; CHECK: store i32 {{.*}}, ptr addrspace(200) [[DST]]
; CHECK: icmp eq ptr addrspace(200)
; ASM-LABEL: copy_offset:
; ASM: [[LOOP:.LBB[0-9_]+]]:
; ASM-NOT: addi
; ASM: lw
; ASM: sw
; ASM-NOT: addi
; ASM: bne {{.*}}, [[LOOP]]
define void @copy_offset(ptr addrspace(200) %src, ptr addrspace(200) %dst, i64 %k, i32 %n) {
entry:
  %positive = icmp sgt i32 %n, 0
  br i1 %positive, label %preheader, label %exit
preheader:
  %limit = zext i32 %n to i64
  br label %loop
loop:
  %i = phi i64 [0, %preheader], [%next, %loop]
  %sp = getelementptr inbounds i32, ptr addrspace(200) %src, i64 %i
  %v = load i32, ptr addrspace(200) %sp
  %ki = add i64 %k, %i
  %dp = getelementptr inbounds i32, ptr addrspace(200) %dst, i64 %ki
  store i32 %v, ptr addrspace(200) %dp
  %next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %next, %limit
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

; Do not add an endpoint when the counter remains needed for a stored value.
; CHECK-LABEL: @counter_value(
; CHECK-NOT: lsr.cap.end
; CHECK: phi i64
; CHECK: icmp {{eq|ne}} i64
; CHECK-NOT: lsr.cap.end
; CHECK: ret void
define void @counter_value(ptr addrspace(200) %b) {
entry:
  br label %loop
loop:
  %i = phi i64 [0, %entry], [%next, %loop]
  %p = getelementptr inbounds i64, ptr addrspace(200) %b, i64 %i
  store i64 %i, ptr addrspace(200) %p
  %next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %next, 100
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

; A different exit can determine the trip count; leave that control flow alone.
; CHECK-LABEL: @early_exit(
; CHECK-NOT: lsr.cap.end
; CHECK: icmp {{eq|ne}} i64
; CHECK-NOT: lsr.cap.end
; CHECK: ret void
define void @early_exit(ptr addrspace(200) %b, i1 %stop) {
entry:
  br label %loop
loop:
  %i = phi i64 [0, %entry], [%next, %latch]
  %p = getelementptr inbounds i64, ptr addrspace(200) %b, i64 %i
  %v = load volatile i64, ptr addrspace(200) %p
  br i1 %stop, label %exit, label %latch
latch:
  %next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %next, 100
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

; A 2^40-byte element repeats its address after 2^24 steps. Exactly 2^24
; iterations (a backedge count of 2^24 - 1) stay below that period.
; CHECK-LABEL: @bound_ok(
; CHECK: [[END:%.*]] = getelementptr [1099511627776 x i8], ptr addrspace(200) %b, i64 16777216
; CHECK: loop:
; CHECK-NOT: phi i64
; CHECK: [[IV:%.*]] = phi ptr addrspace(200)
; CHECK: [[NEXT:%.*]] = getelementptr i8, ptr addrspace(200) [[IV]], i64 1099511627776
; CHECK: icmp eq ptr addrspace(200) [[NEXT]], [[END]]
define void @bound_ok(ptr addrspace(200) %b) {
entry:
  br label %loop
loop:
  %i = phi i64 [0, %entry], [%next, %loop]
  %p = getelementptr inbounds [1099511627776 x i8], ptr addrspace(200) %b, i64 %i
  %v = load volatile i8, ptr addrspace(200) %p
  %next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %next, 16777216
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

; One more iteration reaches the period: the cursor after 2^24 steps equals
; the first one, so a cursor compare could exit one iteration early.
; CHECK-LABEL: @bound_bad(
; CHECK-NOT: lsr.cap.end
; CHECK: icmp {{eq|ne}} i64
; CHECK-NOT: lsr.cap.end
; CHECK: ret void
define void @bound_bad(ptr addrspace(200) %b) {
entry:
  br label %loop
loop:
  %i = phi i64 [0, %entry], [%next, %loop]
  %p = getelementptr inbounds [1099511627776 x i8], ptr addrspace(200) %b, i64 %i
  %v = load volatile i8, ptr addrspace(200) %p
  %next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %next, 16777217
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

; CHECK-LABEL: @bound_edge_bad(
; CHECK-NOT: lsr.cap.end
; CHECK: icmp {{eq|ne}} i64
; CHECK-NOT: lsr.cap.end
; CHECK: ret void
define void @bound_edge_bad(ptr addrspace(200) %b) {
entry:
  br label %loop
loop:
  %i = phi i64 [0, %entry], [%next, %loop]
  %p = getelementptr inbounds [1099511627776 x i8], ptr addrspace(200) %b, i64 %i
  %v = load volatile i8, ptr addrspace(200) %p
  %next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %next, 16777218
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

; The counter returned after the loop is a live-out of the exit test's slice
; (an LCSSA phi, not a stored value), so the integer test stays.
; CHECK-LABEL: @liveout(
; CHECK-NOT: lsr.cap.end
; CHECK: phi i64
; CHECK: icmp eq i64
; CHECK-NOT: lsr.cap.end
; CHECK: ret i64
define i64 @liveout(ptr addrspace(200) %b, i32 %n) {
entry:
  %positive = icmp sgt i32 %n, 0
  br i1 %positive, label %preheader, label %exit
preheader:
  %limit = zext i32 %n to i64
  br label %loop
loop:
  %i = phi i64 [0, %preheader], [%next, %loop]
  %p = getelementptr inbounds i64, ptr addrspace(200) %b, i64 %i
  %v = load volatile i64, ptr addrspace(200) %p
  %next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %next, %limit
  br i1 %done, label %exit.loop, label %loop
exit.loop:
  ret i64 %next
exit:
  ret i64 0
}

; The first candidate (a 2^40-byte element, rejected by the period guard) must
; not stop the search: the byte-stride second candidate supplies the endpoint,
; derived from its own base. The i32 count bounds the byte-stride loop only.
; CHECK-LABEL: @two(
; CHECK: [[END:%.*]] = getelementptr i8, ptr addrspace(200) %b, i64
; CHECK: loop:
; CHECK-NOT: phi i64
; CHECK: [[A:%.*]] = phi ptr addrspace(200)
; CHECK: [[B:%.*]] = phi ptr addrspace(200)
; CHECK: [[NEXTB:%.*]] = getelementptr i8, ptr addrspace(200) [[B]], i64 1
; CHECK: icmp eq ptr addrspace(200) [[NEXTB]], [[END]]
define void @two(ptr addrspace(200) %a, ptr addrspace(200) %b, i32 %n) {
entry:
  %positive = icmp sgt i32 %n, 0
  br i1 %positive, label %preheader, label %exit
preheader:
  %limit = zext i32 %n to i64
  br label %loop
loop:
  %i = phi i64 [0, %preheader], [%next, %loop]
  %p = getelementptr inbounds [1099511627776 x i8], ptr addrspace(200) %a, i64 %i
  %v = load volatile i8, ptr addrspace(200) %p
  %q = getelementptr inbounds i8, ptr addrspace(200) %b, i64 %i
  %w = load volatile i8, ptr addrspace(200) %q
  %next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %next, %limit
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

; The signed exit predicate is not handled by the later loop-term fold.
; The unconditional load still proves an accessed anchor around the call.
; CHECK-LABEL: @descending_call(
; CHECK: [[END:%.*]] = getelementptr ptr addrspace(200), ptr addrspace(200) %b, i64 -1
; CHECK: loop:
; CHECK-NOT: phi i64
; CHECK: [[IV:%.*]] = phi ptr addrspace(200)
; CHECK: load ptr addrspace(200), ptr addrspace(200) [[IV]]
; CHECK: latch:
; CHECK: [[NEXT:%.*]] = getelementptr i8, ptr addrspace(200) [[IV]], i64 -16
; CHECK: icmp ne ptr addrspace(200) [[NEXT]], [[END]]
; ASM-LABEL: descending_call:
; ASM: cincoffsetimm [[CUR:s[0-9]+]], [[CUR]], -16
; ASM-NEXT: beq [[CUR]], {{s[0-9]+}},
define void @descending_call(ptr addrspace(200) %b, i32 %n) {
entry:
  %positive = icmp sgt i32 %n, 0
  br i1 %positive, label %preheader, label %exit
preheader:
  %limit = zext i32 %n to i64
  %start = add i64 %limit, -1
  br label %loop
loop:
  %i = phi i64 [%start, %preheader], [%next, %latch]
  %p = getelementptr inbounds ptr addrspace(200), ptr addrspace(200) %b, i64 %i
  %v = load ptr addrspace(200), ptr addrspace(200) %p
  %present = icmp ne ptr addrspace(200) %v, null
  br i1 %present, label %call, label %latch
call:
  call void @consume(ptr addrspace(200) %v)
  br label %latch
latch:
  %next = add nsw i64 %i, -1
  %more = icmp sgt i64 %i, 0
  br i1 %more, label %loop, label %exit
exit:
  ret void
}

declare void @consume(ptr addrspace(200))
