; RUN: opt -mtriple=capstone64 -passes=loop-reduce -verify-each -S %s | FileCheck %s --check-prefix=OFF
; RUN: opt -mtriple=capstone64 -passes=loop-reduce -lsr-capability-exit-inbounds -verify-each -S %s | FileCheck %s --check-prefix=ON
;
; -lsr-capability-exit-inbounds lets an inbounds anchor, accessed on every iteration, stand in for the
; max-backedge-count bound that replaceCapabilityLatchTest otherwise needs. That bound guards against
; cursors that repeat within the trip count (cursor-exit.ll's @unbounded, @repeating_cursor and @bound_bad).
; Every executed inbounds anchor lies in one allocated object, smaller than 2^63 bytes, so the compared
; cursors cannot repeat. The default keeps every integer test that cursor-exit.ll pins.
;
; imagick's HorizontalFilter has this shape: for (i = 0; i < n; i++) over contribution[i], with n an
; unrestricted i64.

target datalayout = "e-p:64:64-p200:128:128:128:64-i64:64-n32:64-ni:200"

; OFF-LABEL: @unbounded_inbounds(
; OFF-NOT:   lsr.cap.end
; OFF:       icmp {{eq|ne}} i64
; ON-LABEL:  @unbounded_inbounds(
; ON:        [[END:%.*]] = getelementptr i64, ptr addrspace(200) %b,
; ON:        loop:
; ON-NOT:    phi i64
; ON:        [[IV:%.*]] = phi ptr addrspace(200)
; ON:        load volatile i64, ptr addrspace(200) [[IV]]
; ON:        [[NEXT:%.*]] = getelementptr i8, ptr addrspace(200) [[IV]], i64 8
; ON:        icmp {{eq|ne}} ptr addrspace(200) [[NEXT]], [[END]]
define void @unbounded_inbounds(ptr addrspace(200) %b, i64 %n) {
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

; Without inbounds on the anchor nothing bounds the cursors: the integer test stays under the option too.
; OFF-LABEL: @unbounded_plain_gep(
; OFF-NOT:   lsr.cap.end
; OFF:       icmp {{eq|ne}} i64
; ON-LABEL:  @unbounded_plain_gep(
; ON-NOT:    lsr.cap.end
; ON:        icmp {{eq|ne}} i64
; ON-NOT:    lsr.cap.end
; ON:        ret void
define void @unbounded_plain_gep(ptr addrspace(200) %b, i64 %n) {
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

; An anchor that is not accessed on every iteration is no candidate at all: nothing changes under the option.
; ON-LABEL:  @conditional_inbounds(
; ON-NOT:    lsr.cap.end
; ON:        ret void
define void @conditional_inbounds(ptr addrspace(200) %b, i64 %n, i1 %take) {
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
