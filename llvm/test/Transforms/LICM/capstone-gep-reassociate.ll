; RUN: opt -passes=licm -S %s | FileCheck %s
;
; b=2^38-2, off=2, n=0: one iteration accesses b-1+2=2^38-1.
; Swapping the offsets first forms b+2=2^38, refused by sv39.
; The original intermediate b-1 and final cursor both retain their tags.

target datalayout = "e-p:64:64-p200:128:128:128:64-i64:64-n32:64-ni:200"
; CHECK-LABEL: @swap(
; CHECK: entry:
; CHECK-NOT: getelementptr
; CHECK: loop:
; CHECK: %a = getelementptr i8, ptr addrspace(200) %b, i64 %i
; CHECK-NEXT: %p = getelementptr i8, ptr addrspace(200) %a, i64 %off
; CHECK-NEXT: %v = load volatile i8, ptr addrspace(200) %p
define i8 @swap(ptr addrspace(200) %b, i64 %off, i64 %n) {
entry:
  br label %loop
loop:
  %i = phi i64 [ -1, %entry ], [ %inc, %loop ]
  %a = getelementptr i8, ptr addrspace(200) %b, i64 %i
  %p = getelementptr i8, ptr addrspace(200) %a, i64 %off
  %v = load volatile i8, ptr addrspace(200) %p
  %inc = add i64 %i, 1
  %done = icmp eq i64 %inc, %n
  br i1 %done, label %exit, label %loop
exit:
  ret i8 %v
}

; CHECK-LABEL: @swap_integral(
; CHECK: entry:
; CHECK-NEXT: [[INV:%.*]] = getelementptr i8, ptr %b, i64 %off
; CHECK: loop:
; CHECK: [[P:%.*]] = getelementptr i8, ptr [[INV]], i64 %i
; CHECK-NEXT: %v = load volatile i8, ptr [[P]]
define i8 @swap_integral(ptr %b, i64 %off, i64 %n) {
entry:
  br label %loop
loop:
  %i = phi i64 [ -1, %entry ], [ %inc, %loop ]
  %a = getelementptr i8, ptr %b, i64 %i
  %p = getelementptr i8, ptr %a, i64 %off
  %v = load volatile i8, ptr %p
  %inc = add i64 %i, 1
  %done = icmp eq i64 %inc, %n
  br i1 %done, label %exit, label %loop
exit:
  ret i8 %v
}

; CHECK-LABEL: @swap_vector(
; CHECK: entry:
; CHECK-NOT: getelementptr
; CHECK: loop:
; CHECK: %a = getelementptr i8, <2 x ptr addrspace(200)> %b, i64 %i
; CHECK-NEXT: %p = getelementptr i8, <2 x ptr addrspace(200)> %a, i64 %off
; CHECK-NEXT: %one = extractelement <2 x ptr addrspace(200)> %p, i64 0
; CHECK-NEXT: %v = load volatile i8, ptr addrspace(200) %one
define i8 @swap_vector(<2 x ptr addrspace(200)> %b, i64 %off, i64 %n) {
entry:
  br label %loop
loop:
  %i = phi i64 [ -1, %entry ], [ %inc, %loop ]
  %a = getelementptr i8, <2 x ptr addrspace(200)> %b, i64 %i
  %p = getelementptr i8, <2 x ptr addrspace(200)> %a, i64 %off
  %one = extractelement <2 x ptr addrspace(200)> %p, i64 0
  %v = load volatile i8, ptr addrspace(200) %one
  %inc = add i64 %i, 1
  %done = icmp eq i64 %inc, %n
  br i1 %done, label %exit, label %loop
exit:
  ret i8 %v
}
