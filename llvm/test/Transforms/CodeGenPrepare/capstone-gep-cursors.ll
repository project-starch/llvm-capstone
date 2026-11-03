; RUN: opt -mtriple=capstone64 -passes='require<profile-summary>,function(codegenprepare)' -S %s | FileCheck %s
; REQUIRES: capstone-registered-target
;
; Positive offsets permit sharing the smallest offset: the new cursor lies
; between the original base and every final cursor. Speculation is a separate
; restriction: every path must reach a group GEP or an access through the base,
; unless the base satisfies the existing non-null speculation rule.

target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"
target triple = "capstone64-unknown-linux-gnu"
; CHECK-LABEL: @split(
; CHECK: entry:
; CHECK: [[BASE:%.*]] = getelementptr i8, ptr addrspace(200) %b, i64 4096
; CHECK: left:
; CHECK: load i8, ptr addrspace(200) [[BASE]]
; CHECK: right:
; CHECK: getelementptr i8, ptr addrspace(200) [[BASE]], i64 1
; CHECK: ret i8
define i8 @split(ptr addrspace(200) %b, i1 %take) {
entry:
  br i1 %take, label %left, label %right
left:
  %p = getelementptr i8, ptr addrspace(200) %b, i64 4096
  br label %loadleft
loadleft:
  %v = load i8, ptr addrspace(200) %p
  ret i8 %v
right:
  %q = getelementptr i8, ptr addrspace(200) %b, i64 4097
  br label %loadright
loadright:
  %w = load i8, ptr addrspace(200) %q
  ret i8 %w
}

; CHECK-LABEL: @split_integral(
; CHECK: entry:
; CHECK-NEXT: [[BASE:%.*]] = getelementptr i8, ptr %b, i64 4096
; CHECK: right:
; CHECK: getelementptr i8, ptr [[BASE]], i64 1
define i8 @split_integral(ptr %b, i1 %take) {
entry:
  br i1 %take, label %left, label %right
left:
  %p = getelementptr i8, ptr %b, i64 4096
  br label %loadleft
loadleft:
  %v = load i8, ptr %p
  ret i8 %v
right:
  %q = getelementptr i8, ptr %b, i64 4097
  br label %loadright
loadright:
  %w = load i8, ptr %q
  ret i8 %w
}

; CHECK-LABEL: @split_nonnull(
; CHECK: entry:
; CHECK-NEXT: [[BASE:%.*]] = getelementptr i8, ptr addrspace(200) %b, i64 4096
; CHECK: right:
; CHECK: getelementptr i8, ptr addrspace(200) [[BASE]], i64 1
define i8 @split_nonnull(ptr addrspace(200) nonnull %b, i1 %take) {
entry:
  br i1 %take, label %left, label %right
left:
  %p = getelementptr i8, ptr addrspace(200) %b, i64 4096
  br label %loadleft
loadleft:
  %v = load i8, ptr addrspace(200) %p
  ret i8 %v
right:
  %q = getelementptr i8, ptr addrspace(200) %b, i64 4097
  br label %loadright
loadright:
  %w = load i8, ptr addrspace(200) %q
  ret i8 %w
}


; The original first GEP dominates the group. No non-null attribute is needed,
; and the shared cursor must stay below the guard.
; CHECK-LABEL: @guarded_group(
; CHECK: entry:
; CHECK-NOT: getelementptr
; CHECK: use:
; CHECK: [[BASE:%.*]] = getelementptr i8, ptr addrspace(200) %b, i64 4096
; CHECK: getelementptr i8, ptr addrspace(200) [[BASE]], i64 8
; CHECK-NOT: i64 4104
; CHECK: ret i64
define i64 @guarded_group(ptr addrspace(200) %b, i1 %take) {
entry:
  br i1 %take, label %use, label %exit
use:
  %p = getelementptr i8, ptr addrspace(200) %b, i64 4096
  %q = getelementptr i8, ptr addrspace(200) %b, i64 4104
  br label %loads
loads:
  %v = load i64, ptr addrspace(200) %p
  %w = load i64, ptr addrspace(200) %q
  %sum = add i64 %v, %w
  ret i64 %sum
exit:
  ret i64 0
}

; A negative minimum would not lie between b and the right-hand cursor.
; These candidates must never enter a group, even with a non-null base.
; CHECK-LABEL: @negative_group(
; CHECK: entry:
; CHECK-NOT: getelementptr
; CHECK: left:
; CHECK: getelementptr i8, ptr addrspace(200) %b, i64 -4000
; CHECK: right:
; CHECK: getelementptr i8, ptr addrspace(200) %b, i64 -2100
; CHECK: ret i64
define i64 @negative_group(ptr addrspace(200) nonnull %b, i1 %take) {
entry:
  br i1 %take, label %left, label %right
left:
  %p = getelementptr i8, ptr addrspace(200) %b, i64 -4000
  br label %loadleft
loadleft:
  %v = load i64, ptr addrspace(200) %p
  ret i64 %v
right:
  %q = getelementptr i8, ptr addrspace(200) %b, i64 -2100
  br label %loadright
loadright:
  %w = load i64, ptr addrspace(200) %q
  ret i64 %w
}
