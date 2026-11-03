; RUN: opt -mtriple=capstone64 -passes='default<O2>' %s -o %t.bc
; RUN: %llc_cap -O2 < %t.bc | FileCheck %s
; RUN: %llc_cap -O0 < %s | FileCheck %s --check-prefix=MULTI
;
; Check the surviving cursors after both IR optimization and instruction
; selection. The multi-index witness uses b=2^38-2, n=1, k=-4095: the
; complete offset is 1, but the partial offset 4096 crosses sv39.
;
; CHECK-LABEL: multi_cap:
; CHECK-NOT: cincoffset
; CHECK: slli [[S:[a-z0-9]+]], a1, 12
; CHECK-NEXT: add [[O:[a-z0-9]+]], [[S]], a2
; CHECK-NEXT: cincoffset [[P:[a-z0-9]+]], a0, [[O]]
; CHECK-NEXT: lbu {{[a-z0-9]+}}, 0([[P]])
; CHECK-NOT: cincoffset
; CHECK: cjalr
; MULTI-LABEL: multi_cap:
; MULTI-NOT: cincoffset
; MULTI: add [[O:[a-z0-9]+]], {{[a-z0-9]+}}, a2
; MULTI-NEXT: cincoffset [[P:[a-z0-9]+]], a0, [[O]]
; MULTI-NEXT: lbu {{[a-z0-9]+}}, 0([[P]])
; MULTI-NOT: cincoffset
; MULTI: cjalr
;
; CHECK-LABEL: canonical_cap:
; CHECK-NOT: cincoffset
; CHECK: slli [[S:[a-z0-9]+]], a1, 2
; CHECK-NEXT: addi [[O:[a-z0-9]+]], [[S]], -3
; CHECK-NEXT: cincoffset [[P:[a-z0-9]+]], a0, [[O]]
; CHECK-NEXT: lbu {{[a-z0-9]+}}, 0([[P]])
; CHECK-NOT: cincoffset
; CHECK: cjalr
;
; CHECK-LABEL: swap:
; CHECK-NOT: cincoffset
; CHECK: li [[I:[a-z0-9]+]], -1
; CHECK: [[LOOP:.LBB[0-9_]+]]:
; CHECK-NEXT: # =>This Inner Loop Header: Depth=1
; CHECK-NEXT: cincoffset [[A:[a-z0-9]+]], a0, [[I]]
; CHECK-NEXT: cincoffset [[P:[a-z0-9]+]], [[A]], a1
; CHECK-NEXT: lbu {{[a-z0-9]+}}, 0([[P]])
; CHECK: bne {{.*}}, [[LOOP]]

target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"
target triple = "capstone64-unknown-linux-gnu"
; author's InstCombine multi_cap: b=2^38-2, n=1, k=-4095; only cursor 2^38-1.
define i8 @multi_cap(ptr addrspace(200) %b, i64 %n, i64 %k) {
  %p = getelementptr [4096 x i8], ptr addrspace(200) %b, i64 %n, i64 %k
  %v = load i8, ptr addrspace(200) %p
  ret i8 %v
}
; author's canonical_cap: b=2, n=1; cursors 3, 3.
define i8 @canonical_cap(ptr addrspace(200) %b, i64 %n) {
  %a = getelementptr i8, ptr addrspace(200) %b, i64 1
  %i = add i64 %n, -1
  %p = getelementptr i32, ptr addrspace(200) %a, i64 %i
  %v = load i8, ptr addrspace(200) %p
  ret i8 %v
}
; author's constants_cap: b=3, n=-1, k=0; cursors 4, 0, 0, 1.
define i8 @constants_cap(ptr addrspace(200) %b, i64 %n, i64 %k) {
  %a = getelementptr i8, ptr addrspace(200) %b, i64 1
  %c = getelementptr i32, ptr addrspace(200) %a, i64 %n
  %p = getelementptr i16, ptr addrspace(200) %c, i64 %k
  %q = getelementptr i8, ptr addrspace(200) %p, i64 1
  %v = load i8, ptr addrspace(200) %q
  ret i8 %v
}
; author's LICM swap: b=2^38-2, off=2, i=-1.
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
