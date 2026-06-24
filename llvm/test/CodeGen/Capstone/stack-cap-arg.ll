; A capability (addrspace-200 pointer) argument passed on the stack (the 9th+
; argument, beyond a0-a7) must be stored to its stack slot with a capability
; store (stc) at an sp-relative offset, so its tag is preserved. The bug this
; guards against computed the slot address with an integer `addi` on the stack
; pointer capability, which strips the tag and leaves the callee's argument
; untagged (faulting on first dereference) -- the RV8 norx failure.
;
; RUN: llc -mtriple=capstone64 -mattr=+m < %s | FileCheck %s

target datalayout = "e-m:e-pf200:128:128:128:64-p:64:64-i64:64-i128:128-n32:64-S128-A200-P200-G200"

declare void @callee(i64, i64, i64, i64, i64, i64, i64, i64,
                     ptr addrspace(200), ptr addrspace(200))

; CHECK-LABEL: caller:
; The two stack-passed capability args are stored with stc at sp-relative
; offsets (tag-preserving), with no integer addi computing the slot address.
; CHECK-DAG: stc a0, 0(sp)
; CHECK-DAG: stc a1, 16(sp)
; CHECK-NOT: addi {{a[0-9]+}}, sp,
define void @caller(ptr addrspace(200) %p8, ptr addrspace(200) %p9) addrspace(200) {
  call void @callee(i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0,
                    ptr addrspace(200) %p8, ptr addrspace(200) %p9)
  ret void
}
