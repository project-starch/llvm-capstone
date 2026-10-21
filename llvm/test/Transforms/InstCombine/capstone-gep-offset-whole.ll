; RUN: opt -passes=instcombine -S %s | FileCheck %s
;
; A capability's cursor is representable only within a window around its bounds, and R-60 clears the tag of an
; offset whose result leaves it. InstCombine's single-index GEP rewrites move part of an index into a second GEP,
; so the first GEP is a pointer the source never held: musl's fmt_fp (vfprintf.c, `d = r + 1 + ((j +
; 9*LDBL_MAX_EXP)/9 - LDBL_MAX_EXP)`) became (r + 4*q) - 0xfffc, about 64 KiB past a 7 KiB array, and faulted
; with cause 24 at every rounded %f. For non-integral (addrspace 200) pointers the offset stays whole; the
; integral copies of the same IR still split, so integral code is unchanged.

target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"

; The fmt_fp shape: gep (gep r, 1), sext(q - 16384) -- a sext'd nsw add of a constant.
define i32 @fmt_fp_cap(ptr addrspace(200) %r, i32 %j) {
; The source's own form survives: r + 1 (inside the array), then the whole index q - 16384. No GEP takes the
; variable part alone, and no -65532 GEP follows it.
; CHECK-LABEL: @fmt_fp_cap(
; CHECK:       [[R1:%.*]] = getelementptr inbounds nuw i8, ptr addrspace(200) %r, i64 4
; CHECK:       [[SUB:%.*]] = add nsw i32 {{%.*}}, -16384
; CHECK-NEXT:  [[IDX:%.*]] = sext i32 [[SUB]] to i64
; CHECK-NEXT:  [[P:%.*]] = getelementptr inbounds i32, ptr addrspace(200) [[R1]], i64 [[IDX]]
; CHECK-NEXT:  load i32, ptr addrspace(200) [[P]]
  %add.ptr = getelementptr inbounds i32, ptr addrspace(200) %r, i64 1
  %add = add nsw i32 %j, 147456
  %div = sdiv i32 %add, 9
  %sub = sub nsw i32 %div, 16384
  %idx = sext i32 %sub to i64
  %p = getelementptr inbounds i32, ptr addrspace(200) %add.ptr, i64 %idx
  %v = load i32, ptr addrspace(200) %p, align 4
  ret i32 %v
}

; The plain add shape: gep p, (x + y).
define i32 @add_index_cap(ptr addrspace(200) %r, i64 %x, i64 %y) {
; CHECK-LABEL: @add_index_cap(
; CHECK:       getelementptr
; CHECK-NOT:   getelementptr
; CHECK:       load i32, ptr addrspace(200)
  %i = add i64 %x, %y
  %p = getelementptr i32, ptr addrspace(200) %r, i64 %i
  %v = load i32, ptr addrspace(200) %p, align 4
  ret i32 %v
}

; Controls: the same IR on integral pointers still splits into two GEPs.
define i32 @fmt_fp_int(ptr %r, i32 %j) {
; CHECK-LABEL: @fmt_fp_int(
; CHECK:       [[A:%.*]] = getelementptr i32, ptr %r, i64
; CHECK-NEXT:  getelementptr i8, ptr [[A]], i64 -65532
  %add.ptr = getelementptr inbounds i32, ptr %r, i64 1
  %add = add nsw i32 %j, 147456
  %div = sdiv i32 %add, 9
  %sub = sub nsw i32 %div, 16384
  %idx = sext i32 %sub to i64
  %p = getelementptr inbounds i32, ptr %add.ptr, i64 %idx
  %v = load i32, ptr %p, align 4
  ret i32 %v
}

define i32 @add_index_int(ptr %r, i64 %x, i64 %y) {
; CHECK-LABEL: @add_index_int(
; CHECK:       [[A:%.*]] = getelementptr i32, ptr %r, i64 %x
; CHECK-NEXT:  getelementptr i32, ptr [[A]], i64 %y
  %i = add i64 %x, %y
  %p = getelementptr i32, ptr %r, i64 %i
  %v = load i32, ptr %p, align 4
  ret i32 %v
}
