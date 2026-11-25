; RUN: opt -mtriple=capstone64 -passes=loop-reduce -verify-each -S %s | FileCheck %s --check-prefix=OFF
; RUN: opt -mtriple=capstone64 -passes=loop-reduce -lsr-capability-exit-runtime-end -verify-each -S %s | FileCheck %s --check-prefix=ON
;
; -lsr-capability-exit-runtime-end moves a loop's exit test onto its capability IV even when the max backedge count
; is not below the cursor period, 2^(64 - ctz(byte step)). The endpoint is chosen in the preheader:
; - the real one when the count is below the period;
; - otherwise lsr.cap.never, the start cursor plus one byte, which no cursor of an even step reaches.
;
; With the second endpoint the loop leaves only by trapping, as the original must. Its anchor is accessed on every
; iteration that reaches the test (the candidate filter), and accessed cursors lie in [0, 2^38) by the format: a
; reference whose cursor is at or above 2^38 is written as an untagged integer and traps at its access (capstone-qemu
; target/riscv/cap.h:199-233, sublet-reference-model.md section 2.1; bounds are not limited). A count at or above the
; period visits a full cycle of cursors, so one lies at or above 2^38. Below the period the exits coincide.
;
; cursor-exit.ll pins @unbounded, @repeating_cursor and @bound_bad as keeping the integer test. That is still the
; default. The guard was conservative, not wrong: with only a static bound, the one safe answer for a count that
; may reach the period was the counter. Endpoint-by-inbounds alone was ruled out (the semantics owner, 2026-11-25).
; It would make @repeating_cursor exit after one iteration where the original traps at its second load. The ON
; copies of those three below check the run-time endpoint instead.

target datalayout = "e-p:64:64-p200:128:128:128:64-i64:64-n32:64-ni:200"

; OFF-LABEL: @unbounded(
; OFF-NOT:   lsr.cap.end
; OFF:       icmp {{eq|ne}} i64
; ON-LABEL:  @unbounded(
; ON:        [[END:%.*]] = getelementptr i64, ptr addrspace(200) %b, i64 %n
; ON:        [[FITS:%.*]] = icmp ult i64 {{%.*}}, 2305843009213693952
; ON:        [[NEVER:%.*]] = getelementptr i8, ptr addrspace(200) %b, i64 1
; ON:        [[SEL:%.*]] = select i1 [[FITS]], ptr addrspace(200) [[END]], ptr addrspace(200) [[NEVER]]
; ON:        loop:
; ON-NOT:    phi i64
; ON:        [[IV:%.*]] = phi ptr addrspace(200)
; ON:        load volatile i64, ptr addrspace(200) [[IV]]
; ON:        [[NEXT:%.*]] = getelementptr i8, ptr addrspace(200) [[IV]], i64 8
; ON:        icmp eq ptr addrspace(200) [[NEXT]], [[SEL]]
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

; A count of 2 against a period of 2: the endpoint b + 3 * 2^63 = b + 2^63 is the second cursor, so a cursor test
; with the real endpoint would exit after the first iteration, while the original traps at b + 2^63. The run-time
; choice is constant here: the unreachable endpoint, so the loop runs into the same trap.
; ON-LABEL:  @repeating_cursor(
; ON:        [[NEVER:%.*]] = getelementptr i8, ptr addrspace(200) %b, i64 1
; ON:        [[SEL:%.*]] = select i1 false, ptr addrspace(200) {{%.*}}, ptr addrspace(200) [[NEVER]]
; ON:        icmp eq ptr addrspace(200) {{%.*}}, [[SEL]]
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

; One iteration past the period (2^24 steps of 2^40 bytes): the same constant choice.
; ON-LABEL:  @bound_bad(
; ON:        [[NEVER:%.*]] = getelementptr i8, ptr addrspace(200) %b, i64 1
; ON:        [[SEL:%.*]] = select i1 false, ptr addrspace(200) {{%.*}}, ptr addrspace(200) [[NEVER]]
; ON:        icmp eq ptr addrspace(200) {{%.*}}, [[SEL]]
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

; A count the static bound already covers keeps the plain endpoint: no run-time choice.
; ON-LABEL:  @bounded(
; ON-NOT:    lsr.cap.fits
; ON-NOT:    lsr.cap.never
; ON:        icmp {{eq|ne}} ptr addrspace(200)
define void @bounded(ptr addrspace(200) %b, i32 %n) {
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

; Negative control: the same unbounded loop with its only access under a condition, so no access dominates the
; latch. There is no accessed recurrence, and the integer test stays under the option.
; ON-LABEL:  @conditional_access(
; ON-NOT:    lsr.cap.end
; ON:        icmp eq i64
; ON-NOT:    lsr.cap.end
; ON:        ret void
define void @conditional_access(ptr addrspace(200) %b, i64 %n, i1 %take) {
entry:
  br label %loop
loop:
  %i = phi i64 [0, %entry], [%next, %latch]
  %p = getelementptr inbounds i64, ptr addrspace(200) %b, i64 %i
  br i1 %take, label %access, label %latch
access:
  %v = load volatile i64, ptr addrspace(200) %p
  br label %latch
latch:
  %next = add i64 %i, 1
  %done = icmp eq i64 %next, %n
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

; Negative control: an anchor without inbounds is no candidate either, so the integer test stays.
; ON-LABEL:  @plain_gep(
; ON-NOT:    lsr.cap.end
; ON:        icmp eq i64
; ON-NOT:    lsr.cap.end
; ON:        ret void
define void @plain_gep(ptr addrspace(200) %b, i64 %n) {
entry:
  br label %loop
loop:
  %i = phi i64 [0, %entry], [%next, %loop]
  %p = getelementptr i64, ptr addrspace(200) %b, i64 %i
  %v = load volatile i64, ptr addrspace(200) %p
  %next = add i64 %i, 1
  %done = icmp eq i64 %next, %n
  br i1 %done, label %exit, label %loop
exit:
  ret void
}
