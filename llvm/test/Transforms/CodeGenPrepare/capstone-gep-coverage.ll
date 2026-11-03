; RUN: opt -mtriple=capstone64 -passes='require<profile-summary>,function(codegenprepare)' -S %s | FileCheck %s
; REQUIRES: capstone-registered-target

target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"
target triple = "capstone64-unknown-linux-gnu"

; The LBM diamond: both arms access the base. No nonnull attribute or
; dominating group GEP is needed to place the shared base above the branch.
; CHECK-LABEL: @arms_accessed(
; CHECK: body:
; CHECK: [[BASE:%.*]] = getelementptr i8, ptr addrspace(200) %p, i64 15904
; CHECK: a:
; CHECK: getelementptr i8, ptr addrspace(200) [[BASE]], i64 104
; CHECK: getelementptr i8, ptr addrspace(200) [[BASE]], i64 312
; CHECK: b:
; CHECK: getelementptr i8, ptr addrspace(200) [[BASE]], i64 104
; CHECK: store double %x, ptr addrspace(200) [[BASE]]
define void @arms_accessed(ptr addrspace(200) %p, i1 %c, double %x) {
body:
  br i1 %c, label %a, label %b
a:
  store double %x, ptr addrspace(200) %p
  %a1 = getelementptr i8, ptr addrspace(200) %p, i64 16008
  store double %x, ptr addrspace(200) %a1
  %a2 = getelementptr i8, ptr addrspace(200) %p, i64 16216
  store double %x, ptr addrspace(200) %a2
  br label %end
b:
  store double %x, ptr addrspace(200) %p
  %b1 = getelementptr i8, ptr addrspace(200) %p, i64 16008
  store double %x, ptr addrspace(200) %b1
  %b2 = getelementptr i8, ptr addrspace(200) %p, i64 15904
  store double %x, ptr addrspace(200) %b2
  br label %end
end:
  ret void
}

; The outer bypass cannot acquire arithmetic. The inner diamond is covered
; by its group GEPs, so sharing BELOW the guard is safe.
; CHECK-LABEL: @one_arm(
; CHECK: body:
; CHECK-NOT: getelementptr
; CHECK: a:
; CHECK: [[BASE:%.*]] = getelementptr i8, ptr addrspace(200) %p, i64 16008
; CHECK: a1b:
; CHECK: store double %x, ptr addrspace(200) [[BASE]]
; CHECK: a2b:
; CHECK: getelementptr i8, ptr addrspace(200) [[BASE]], i64 208
define void @one_arm(ptr addrspace(200) %p, i1 %c, i1 %d, double %x) {
body:
  br i1 %c, label %a, label %end
a:
  br i1 %d, label %a1b, label %a2b
a1b:
  %a1 = getelementptr i8, ptr addrspace(200) %p, i64 16008
  store double %x, ptr addrspace(200) %a1
  br label %end
a2b:
  %a2 = getelementptr i8, ptr addrspace(200) %p, i64 16216
  store double %x, ptr addrspace(200) %a2
  br label %end
end:
  ret void
}

; An uncovered path through the nearest common dominator must be refused.
; CHECK-LABEL: @uncovered_exit(
; CHECK-NOT: splitgep
; CHECK: ret void
define void @uncovered_exit(ptr addrspace(200) %p, i1 %c, i1 %d) {
entry:
  br i1 %c, label %a, label %b
a:
  %p1 = getelementptr i8, ptr addrspace(200) %p, i64 4096
  store i8 0, ptr addrspace(200) %p1
  br label %exit
b:
  br i1 %d, label %use, label %exit
use:
  %p2 = getelementptr i8, ptr addrspace(200) %p, i64 4104
  store i8 0, ptr addrspace(200) %p2
  br label %exit
exit:
  ret void
}

; A cycle without an access cannot bootstrap its own coverage proof.
; CHECK-LABEL: @uncovered_cycle(
; CHECK-NOT: splitgep
; CHECK: ret void
define void @uncovered_cycle(ptr addrspace(200) %p, i1 %c, i1 %d) {
entry:
  br i1 %c, label %a, label %spin
a:
  %p1 = getelementptr i8, ptr addrspace(200) %p, i64 4096
  store i8 0, ptr addrspace(200) %p1
  br label %exit
spin:
  br i1 %d, label %spin, label %b
b:
  %p2 = getelementptr i8, ptr addrspace(200) %p, i64 4104
  store i8 0, ptr addrspace(200) %p2
  br label %exit
exit:
  ret void
}

declare void @may_leave()

; A call before the covering use may unwind or never return.
; CHECK-LABEL: @uncovered_call(
; CHECK-NOT: splitgep
; CHECK: ret void
define void @uncovered_call(ptr addrspace(200) %p, i1 %c) {
entry:
  br i1 %c, label %a, label %b
a:
  call void @may_leave()
  %p1 = getelementptr i8, ptr addrspace(200) %p, i64 4096
  store i8 0, ptr addrspace(200) %p1
  br label %exit
b:
  %p2 = getelementptr i8, ptr addrspace(200) %p, i64 4104
  store i8 0, ptr addrspace(200) %p2
  br label %exit
exit:
  ret void
}

; Coverage can come from a direct base access on a path with no group GEP.
; CHECK-LABEL: @base_only_arm(
; CHECK: entry:
; CHECK: [[BASE:%.*]] = getelementptr i8, ptr addrspace(200) %p, i64 4096
; CHECK: a:
; CHECK: store i8 0, ptr addrspace(200) [[BASE]]
; CHECK: b:
; CHECK: getelementptr i8, ptr addrspace(200) [[BASE]], i64 8
define void @base_only_arm(ptr addrspace(200) %p, i1 %c, i1 %d) {
entry:
  br i1 %c, label %a, label %other
a:
  %p1 = getelementptr i8, ptr addrspace(200) %p, i64 4096
  store i8 0, ptr addrspace(200) %p1
  br label %exit
other:
  br i1 %d, label %b, label %base
b:
  %p2 = getelementptr i8, ptr addrspace(200) %p, i64 4104
  store i8 0, ptr addrspace(200) %p2
  br label %exit
base:
  store double 0.0, ptr addrspace(200) %p
  br label %exit
exit:
  ret void
}

; A capability store is allowed to initialize an UNINIT base. It must not
; justify speculating cursor arithmetic on the path containing only that store.
; CHECK-LABEL: @initializing_store(
; CHECK-NOT: splitgep
; CHECK: ret void
define void @initializing_store(ptr addrspace(200) %p, i1 %c, i1 %d) {
entry:
  br i1 %c, label %a, label %other
a:
  %p1 = getelementptr i8, ptr addrspace(200) %p, i64 4096
  store i8 0, ptr addrspace(200) %p1
  br label %exit
other:
  br i1 %d, label %b, label %base
b:
  %p2 = getelementptr i8, ptr addrspace(200) %p, i64 4104
  store i8 0, ptr addrspace(200) %p2
  br label %exit
base:
  store ptr addrspace(200) null, ptr addrspace(200) %p
  br label %exit
exit:
  ret void
}
