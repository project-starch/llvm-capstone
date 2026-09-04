; A dynamic alloca's size operand reaches lowerDynamicStackAlloc as XLen
; whatever type the IR gave it: i16/i32 are promoted, i128 is expanded and
; truncated by the generic legalizer, and an i128 expression (here a mul) is
; computed on its low half.  So lowerDynamicAllocaSizeToXLen never sees a node
; it cannot fold, and the "Unsupported dynamic alloca size expression" fatal
; route (CapstoneISelLowering.cpp) is unreachable from IR -- this file is the
; pin recorded for it in lit-coverage-unreachable.txt.  Each function must
; adjust sp by the (negated) size with a capability increment.  Measured
; 2026-09-04 on the branch tools.
;
; RUN: llc -mtriple=capstone64 -mattr=+m -verify-machineinstrs < %s | FileCheck %s
; RUN: %llc_cap -O0 < %s -o /dev/null
; RUN: %llc_cap -O1 < %s -o /dev/null

target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"

; CHECK-LABEL: size_i32:
; CHECK: neg a1, a0
; CHECK-NEXT: cincoffset a1, sp, a1
define ptr addrspace(200) @size_i32(i32 %n) {
  %a = alloca i8, i32 %n, addrspace(200)
  ret ptr addrspace(200) %a
}

; CHECK-LABEL: size_i16:
; CHECK: neg a1, a0
; CHECK-NEXT: cincoffset a1, sp, a1
define ptr addrspace(200) @size_i16(i16 %n) {
  %a = alloca i8, i16 %n, addrspace(200)
  ret ptr addrspace(200) %a
}

; CHECK-LABEL: size_i128_arg:
; CHECK: neg a1, a0
; CHECK-NEXT: cincoffset a1, sp, a1
define ptr addrspace(200) @size_i128_arg(i128 %n) {
  %a = alloca i8, i128 %n, addrspace(200)
  ret ptr addrspace(200) %a
}

; CHECK-LABEL: size_i128_load:
; CHECK: neg a1, a0
; CHECK-NEXT: cincoffset a1, sp, a1
define ptr addrspace(200) @size_i128_load(ptr addrspace(200) %p) {
  %n = load i128, ptr addrspace(200) %p
  %a = alloca i8, i128 %n, addrspace(200)
  ret ptr addrspace(200) %a
}

; CHECK-LABEL: size_i128_mul:
; CHECK: mul a0, a0, a1
; CHECK: neg a1, a0
; CHECK-NEXT: cincoffset a1, sp, a1
define ptr addrspace(200) @size_i128_mul(i64 %x, i64 %y) {
  %zx = zext i64 %x to i128
  %zy = zext i64 %y to i128
  %n = mul i128 %zx, %zy
  %a = alloca i8, i128 %n, addrspace(200)
  ret ptr addrspace(200) %a
}
