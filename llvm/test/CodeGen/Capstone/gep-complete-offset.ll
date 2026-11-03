; RUN: %llc_cap -O2 < %s | FileCheck %s
; RUN: %llc_cap -O0 < %s | FileCheck %s --check-prefix=O0
;
; A single GEP applies its entire byte offset at once, including struct
; fields. Index arithmetic must stay integer arithmetic until that point.
; CHECK-LABEL: addidx:
; CHECK: slli [[S:[a-z0-9]+]], a1, 2
; CHECK-NEXT: addi [[O:[a-z0-9]+]], [[S]], 32
; CHECK-NEXT: cincoffset [[P:[a-z0-9]+]], a0, [[O]]
; CHECK-NEXT: lw {{[a-z0-9]+}}, 0([[P]])
; CHECK-NOT: cincoffset
; CHECK: cjalr
; CHECK-LABEL: field:
; CHECK-NOT: cincoffset
; CHECK: addi [[O:[a-z0-9]+]], {{[a-z0-9]+}}, 16
; CHECK-NEXT: cincoffset [[P:[a-z0-9]+]], a0, [[O]]
; CHECK-NEXT: ld {{[a-z0-9]+}}, 0([[P]])
; CHECK-NOT: cincoffset
; CHECK: cjalr
; O0-LABEL: field:
; O0-NOT: cincoffset
; O0: addi [[O:[a-z0-9]+]], {{[a-z0-9]+}}, 16
; O0-NEXT: cincoffset [[P:[a-z0-9]+]], a0, [[O]]
; O0-NEXT: ld {{[a-z0-9]+}}, 0([[P]])
; O0-NOT: cincoffset
; O0: cjalr

target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"
target triple = "capstone64-unknown-linux-gnu"
; Original cursors: b+(x+8)*4 only. A split forms b+4x first.
define i32 @addidx(ptr addrspace(200) %b, i64 %x) {
  %i = add i64 %x, 8
  %p = getelementptr i32, ptr addrspace(200) %b, i64 %i
  %v = load i32, ptr addrspace(200) %p
  ret i32 %v
}
; Original cursors: b+32, then b+32+4x. A reassociation forms b+4x first.
define i32 @constfirst(ptr addrspace(200) %b, i64 %x) {
  %a = getelementptr i8, ptr addrspace(200) %b, i64 32
  %p = getelementptr i32, ptr addrspace(200) %a, i64 %x
  %v = load i32, ptr addrspace(200) %p
  ret i32 %v
}
; Multi-index: original single cursor b + 4096n + k.
define i8 @multi(ptr addrspace(200) %b, i64 %n, i64 %k) {
  %p = getelementptr [4096 x i8], ptr addrspace(200) %b, i64 %n, i64 %k
  %v = load i8, ptr addrspace(200) %p
  ret i8 %v
}
; Struct field after variable index: original single cursor b + 24n + 16.
%s = type { i64, i64, i64 }
define i64 @field(ptr addrspace(200) %b, i64 %n) {
  %p = getelementptr %s, ptr addrspace(200) %b, i64 %n, i32 2
  %v = load i64, ptr addrspace(200) %p
  ret i64 %v
}
