; A constant GEP in a global initializer is folded with an APInt whose width
; accumulateConstantOffset asserts must be the INDEX width. On a fat-pointer
; target that is narrower than the pointer, so taking the pointer width aborts.

; RUN: llc -mtriple=capstone64 -filetype=asm -verify-machineinstrs < %s | FileCheck %s

target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"
target triple = "capstone64-unknown-elf"

@arr = addrspace(200) global [8 x i32] zeroinitializer, align 4

; CHECK-LABEL: elem3:
; CHECK: arr+12
@elem3 = addrspace(200) global ptr addrspace(200) getelementptr inbounds ([8 x i32], ptr addrspace(200) @arr, i64 0, i64 3), align 16

; A negative index must stay a subtraction, not wrap through the metadata bits.
; CHECK-LABEL: before:
; CHECK: arr-8
@before = addrspace(200) global ptr addrspace(200) getelementptr inbounds ([8 x i32], ptr addrspace(200) @arr, i64 0, i64 -2), align 16
