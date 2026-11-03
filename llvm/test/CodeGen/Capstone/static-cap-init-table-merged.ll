target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"
@a = internal addrspace(200) global i32 1
@b = internal addrspace(200) global i32 2
@p = addrspace(200) global [2 x ptr addrspace(200)] [ptr addrspace(200) @a, ptr addrspace(200) @b]
define i32 @f() addrspace(200) {
 %x=load i32,ptr addrspace(200) @a
 %y=load i32,ptr addrspace(200) @b
 %z=add i32 %x,%y
 ret i32 %z
}

; RUN: llc -mtriple=capstone64 -capstone-gp-free -capstone-image-gp -capstone-enable-global-merge=true -verify-machineinstrs < %s | FileCheck %s
; The original constructor participates in GlobalMerge. The table must describe
; the final eight-byte container, not the two original four-byte objects.
; CHECK: .section .capstone_cap_relocs
; CHECK: [[T:.L__capstone_cap_init_records]]:
; CHECK: .word .L_MergedGlobals-[[T]]
; CHECK-NEXT: .word 8
; CHECK-NEXT: .word 0
; CHECK: .word .L_MergedGlobals-([[T]]+16)
; CHECK-NEXT: .word 8
; CHECK-NEXT: .word 4
