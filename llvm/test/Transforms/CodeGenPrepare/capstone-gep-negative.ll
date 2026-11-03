; RUN: opt -mtriple=capstone64 -passes='require<profile-summary>,function(codegenprepare)' -S %s | FileCheck %s
; RUN: opt -mtriple=capstone64 -passes='require<profile-summary>,function(codegenprepare)' %s | llc -mtriple=capstone64 -mattr=+m,+f,+d -verify-machineinstrs | FileCheck %s --check-prefix=ASM
; REQUIRES: capstone-registered-target

target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"
target triple = "capstone64-unknown-linux-gnu"

; LBM_handleInOutFlow's negative residual cluster. The far offset cannot
; become a shared intermediate for the closer access. Use the nearest to zero.
; CHECK-LABEL: @negative_cluster(
; CHECK: [[BASE:%.*]] = getelementptr i8, ptr addrspace(200) %p, i64 -15999896
; CHECK: getelementptr i8, ptr addrspace(200) [[BASE]], i64 -208
; CHECK: load double, ptr addrspace(200) [[BASE]]
; CHECK-NOT: getelementptr
; CHECK: ret double
; ASM-LABEL: negative_cluster:
; ASM: cincoffset [[BASE:[a-z][a-z0-9]*]], a0,
; ASM-NOT: cincoffset
; ASM-DAG: fld {{[a-z][a-z0-9]*}}, -208([[BASE]])
; ASM-DAG: fld {{[a-z][a-z0-9]*}}, 0([[BASE]])
; ASM-NOT: cincoffset
; ASM: cjalr zero, 0(ra)
define double @negative_cluster(ptr addrspace(200) %p) {
entry:
  %far = getelementptr i8, ptr addrspace(200) %p, i64 -16000104
  %near = getelementptr i8, ptr addrspace(200) %p, i64 -15999896
  %x = load double, ptr addrspace(200) %far
  %y = load double, ptr addrspace(200) %near
  %sum = fadd double %x, %y
  ret double %sum
}

; Even with coverage on both arms, never share a negative base with a
; positive access. Both directions begin at the original p.
; CHECK-LABEL: @mixed_signs(
; CHECK-DAG: [[NEG:%.*]] = getelementptr i8, ptr addrspace(200) %p, i64 -65536
; CHECK-DAG: [[POS:%.*]] = getelementptr i8, ptr addrspace(200) %p, i64 4800
; CHECK: left:
; CHECK: load i64, ptr addrspace(200) [[NEG]]
; CHECK: right:
; CHECK: load i64, ptr addrspace(200) [[POS]]
define i64 @mixed_signs(ptr addrspace(200) %p, i1 %take) {
entry:
  br i1 %take, label %left, label %right
left:
  %a = getelementptr i8, ptr addrspace(200) %p, i64 -65536
  %x = load i64, ptr addrspace(200) %a
  ret i64 %x
right:
  %b = getelementptr i8, ptr addrspace(200) %p, i64 4800
  %y = load i64, ptr addrspace(200) %b
  ret i64 %y
}

; A negative cluster still needs placement coverage.
; CHECK-LABEL: @negative_bypass(
; CHECK-NOT: splitgep
; CHECK: ret void
define void @negative_bypass(ptr addrspace(200) %p, i1 %c, i1 %d) {
entry:
  br i1 %c, label %a, label %other
a:
  %p1 = getelementptr i8, ptr addrspace(200) %p, i64 -4096
  store i8 0, ptr addrspace(200) %p1
  br label %exit
other:
  br i1 %d, label %b, label %exit
b:
  %p2 = getelementptr i8, ptr addrspace(200) %p, i64 -4104
  store i8 0, ptr addrspace(200) %p2
  br label %exit
exit:
  ret void
}

; Signed minimum needs no negation and cannot be subtracted from a positive
; group base. It must start a separate group before offset subtraction.
; CHECK-LABEL: @signed_minimum(
; CHECK-DAG: getelementptr i8, ptr addrspace(200) %p, i64 -9223372036854775808
; CHECK-DAG: getelementptr i8, ptr addrspace(200) %p, i64 4096
; CHECK: ret i64
define i64 @signed_minimum(ptr addrspace(200) %p) {
entry:
  %a = getelementptr i8, ptr addrspace(200) %p, i64 -9223372036854775808
  %b = getelementptr i8, ptr addrspace(200) %p, i64 4096
  %x = load i64, ptr addrspace(200) %a
  %y = load i64, ptr addrspace(200) %b
  %sum = add i64 %x, %y
  ret i64 %sum
}

; Integral pointers retain the original candidate policy.
; CHECK-LABEL: @negative_integral(
; CHECK-NOT: splitgep
; CHECK: getelementptr i8, ptr %p, i64 -4096
; CHECK: getelementptr i8, ptr %p, i64 -4104
; CHECK: ret i64
define i64 @negative_integral(ptr %p) {
entry:
  %a = getelementptr i8, ptr %p, i64 -4096
  %b = getelementptr i8, ptr %p, i64 -4104
  %x = load i64, ptr %a
  %y = load i64, ptr %b
  %sum = add i64 %x, %y
  ret i64 %sum
}
