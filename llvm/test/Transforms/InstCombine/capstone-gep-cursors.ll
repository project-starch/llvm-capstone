; RUN: opt -passes=instcombine -S %s | FileCheck %s
;
; Capability cursors must not change at surviving intermediate GEPs. Losing
; a tag at an address-policy boundary is irreversible even if a later GEP
; returns to an address in bounds. Ordinary pointers retain the rewrites.

target datalayout = "e-p:64:64-p200:128:128:128:64-ni:200"

; b = 2^38-2, n = 1, k = -4095: the original cursor is 2^38-1.
; Splitting at the array index first forms 2^38+4094, refused by sv39.
define i8 @multi_cap(ptr addrspace(200) %b, i64 %n, i64 %k) {
; CHECK-LABEL: @multi_cap(
; CHECK-NEXT:  %p = getelementptr [4096 x i8], ptr addrspace(200) %b, i64 %n, i64 %k
; CHECK-NEXT:  %v = load i8, ptr addrspace(200) %p, align 1
; CHECK-NEXT:  ret i8 %v
  %p = getelementptr [4096 x i8], ptr addrspace(200) %b, i64 %n, i64 %k
  %v = load i8, ptr addrspace(200) %p
  ret i8 %v
}

define ptr @multi_int(ptr %b, i64 %n, i64 %k) {
; CHECK-LABEL: @multi_int(
; CHECK-NEXT:  [[FRONT:%.*]] = getelementptr [4096 x i8], ptr %b, i64 %n
; CHECK-NEXT:  %p = getelementptr i8, ptr [[FRONT]], i64 %k
; CHECK-NEXT:  ret ptr %p
  %p = getelementptr [4096 x i8], ptr %b, i64 %n, i64 %k
  ret ptr %p
}

; The same restriction applies independently to each vector lane.
define <2 x ptr addrspace(200)> @multi_vector(<2 x ptr addrspace(200)> %b, <2 x i64> %n, <2 x i64> %k) {
; CHECK-LABEL: @multi_vector(
; CHECK-NEXT:  %p = getelementptr [4096 x i8], <2 x ptr addrspace(200)> %b, <2 x i64> %n, <2 x i64> %k
; CHECK-NEXT:  ret <2 x ptr addrspace(200)> %p
  %p = getelementptr [4096 x i8], <2 x ptr addrspace(200)> %b, <2 x i64> %n, <2 x i64> %k
  ret <2 x ptr addrspace(200)> %p
}

; b = 2, n = 1 in an object [0,16): both original cursors are 3.
; Moving -4 into the first GEP forms b-3 = 2^64-1, refused by phys56.
define i8 @canonical_cap(ptr addrspace(200) %b, i64 %n) {
; CHECK-LABEL: @canonical_cap(
; CHECK-NEXT:  %a = getelementptr i8, ptr addrspace(200) %b, i64 1
; CHECK-NEXT:  %i = add i64 %n, -1
; CHECK-NEXT:  %p = getelementptr i32, ptr addrspace(200) %a, i64 %i
; CHECK-NEXT:  %v = load i8, ptr addrspace(200) %p, align 1
; CHECK-NEXT:  ret i8 %v
  %a = getelementptr i8, ptr addrspace(200) %b, i64 1
  %i = add i64 %n, -1
  %p = getelementptr i32, ptr addrspace(200) %a, i64 %i
  %v = load i8, ptr addrspace(200) %p
  ret i8 %v
}

define ptr @canonical_int(ptr %b, i64 %n) {
; CHECK-LABEL: @canonical_int(
; CHECK-NEXT:  [[BASE:%.*]] = getelementptr i8, ptr %b, i64 -3
; CHECK-NEXT:  %p = getelementptr i32, ptr [[BASE]], i64 %n
; CHECK-NEXT:  ret ptr %p
  %a = getelementptr i8, ptr %b, i64 1
  %i = add i64 %n, -1
  %p = getelementptr i32, ptr %a, i64 %i
  ret ptr %p
}

; Cancelling the constant removes an intermediate without changing the result.
define ptr addrspace(200) @canonical_cancel(ptr addrspace(200) %b, i64 %n) {
; CHECK-LABEL: @canonical_cancel(
; CHECK-NEXT:  %p = getelementptr i32, ptr addrspace(200) %b, i64 %n
; CHECK-NEXT:  ret ptr addrspace(200) %p
  %a = getelementptr i8, ptr addrspace(200) %b, i64 4
  %i = add i64 %n, -1
  %p = getelementptr i32, ptr addrspace(200) %a, i64 %i
  ret ptr addrspace(200) %p
}

; b = 3, n = -1, k = 0 in [0,16): original cursors are 4, 0, 0, 1.
; Moving the first +1 to the end changes the middle cursors to 2^64-1.
define i8 @constants_cap(ptr addrspace(200) %b, i64 %n, i64 %k) {
; CHECK-LABEL: @constants_cap(
; CHECK-NEXT:  %a = getelementptr i8, ptr addrspace(200) %b, i64 1
; CHECK-NEXT:  %c = getelementptr i32, ptr addrspace(200) %a, i64 %n
; CHECK-NEXT:  %p = getelementptr i16, ptr addrspace(200) %c, i64 %k
; CHECK-NEXT:  %q = getelementptr i8, ptr addrspace(200) %p, i64 1
; CHECK-NEXT:  %v = load i8, ptr addrspace(200) %q, align 1
; CHECK-NEXT:  ret i8 %v
  %a = getelementptr i8, ptr addrspace(200) %b, i64 1
  %c = getelementptr i32, ptr addrspace(200) %a, i64 %n
  %p = getelementptr i16, ptr addrspace(200) %c, i64 %k
  %q = getelementptr i8, ptr addrspace(200) %p, i64 1
  %v = load i8, ptr addrspace(200) %q
  ret i8 %v
}

define ptr @constants_int(ptr %b, i64 %n, i64 %k) {
; CHECK-LABEL: @constants_int(
; CHECK-NEXT:  %c = getelementptr i32, ptr %b, i64 %n
; CHECK-NEXT:  %p = getelementptr i16, ptr %c, i64 %k
; CHECK-NEXT:  %q = getelementptr i8, ptr %p, i64 2
; CHECK-NEXT:  ret ptr %q
  %a = getelementptr i8, ptr %b, i64 1
  %c = getelementptr i32, ptr %a, i64 %n
  %p = getelementptr i16, ptr %c, i64 %k
  %q = getelementptr i8, ptr %p, i64 1
  ret ptr %q
}

; Merging adjacent offsets only removes the inner cursor and remains legal.
define ptr addrspace(200) @merge_cap(ptr addrspace(200) %b) {
; CHECK-LABEL: @merge_cap(
; CHECK-NEXT:  %p = getelementptr i8, ptr addrspace(200) %b, i64 3
; CHECK-NEXT:  ret ptr addrspace(200) %p
  %a = getelementptr i8, ptr addrspace(200) %b, i64 1
  %p = getelementptr i8, ptr addrspace(200) %a, i64 2
  ret ptr addrspace(200) %p
}
