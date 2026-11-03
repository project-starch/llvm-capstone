; RUN: opt -mtriple=capstone64 -passes='require<profile-summary>,function(codegenprepare)' -S %s | FileCheck %s
; REQUIRES: capstone-registered-target
;
; The left cursor is not formed or accessed on the right path. It must not
; become a shared capability base in entry. This pins the conservative cursor
; rule, without claiming that these positive offsets alone cause tag loss.

target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"
target triple = "capstone64-unknown-linux-gnu"
; CHECK-LABEL: @split(
; CHECK: entry:
; CHECK-NOT: getelementptr
; CHECK: left:
; CHECK: getelementptr i8, ptr addrspace(200) %b, i64 4096
; CHECK: right:
; CHECK: getelementptr i8, ptr addrspace(200) %b, i64 4097
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
