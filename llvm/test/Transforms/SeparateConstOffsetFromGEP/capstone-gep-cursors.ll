; RUN: opt -mtriple=capstone64 -passes=separate-const-offset-from-gep -S %s | FileCheck %s
; RUN: opt -mtriple=capstone64 -passes='separate-const-offset-from-gep<lower-gep>' -S %s | FileCheck %s --check-prefix=LOWER
; REQUIRES: capstone-registered-target
;
; b=2^38-16, n=16, cond=true: the original accesses 2^38-1. Extracting
; the -1 first creates a cursor at 2^38, refused by sv39. This optional pass
; is not in the default Capstone pipeline, but must obey the same rule.

target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"
target triple = "capstone64-unknown-linux-gnu"
; CHECK-LABEL: @whole_cap(
; CHECK: %i = add i64 %n, -1
; CHECK-NEXT: %p = getelementptr i8, ptr addrspace(200) %b, i64 %i
; CHECK-NOT: getelementptr
; CHECK: load i8, ptr addrspace(200) %p
; LOWER-LABEL: @whole_cap(
; LOWER: %i = add i64 %n, -1
; LOWER-NEXT: %p = getelementptr i8, ptr addrspace(200) %b, i64 %i
; LOWER-NOT: getelementptr
; LOWER: load i8, ptr addrspace(200) %p
define i8 @whole_cap(ptr addrspace(200) %b, i64 %n, i1 %cond) {
entry:
  %i = add i64 %n, -1
  %p = getelementptr i8, ptr addrspace(200) %b, i64 %i
  br i1 %cond, label %use, label %end
use:
  %v = load i8, ptr addrspace(200) %p
  ret i8 %v
end:
  ret i8 0
}

; CHECK-LABEL: @whole_int(
; CHECK: [[VAR:%.*]] = getelementptr i8, ptr %b, i64 %n
; CHECK-NEXT: [[FINAL:%.*]] = getelementptr i8, ptr [[VAR]], i64 -1
; CHECK: load i8, ptr [[FINAL]]
; LOWER-LABEL: @whole_int(
; LOWER: ret i8
define i8 @whole_int(ptr %b, i64 %n, i1 %cond) {
entry:
  %i = add i64 %n, -1
  %p = getelementptr i8, ptr %b, i64 %i
  br i1 %cond, label %use, label %end
use:
  %v = load i8, ptr %p
  ret i8 %v
end:
  ret i8 0
}
