; RUN: llc -mtriple=capstone64 -capstone-gp-captable -verify-machineinstrs < %s | FileCheck %s
;
; Under gp-captable, gp is bounded to the cap table, so a code address must not
; be derived from it: a function's slot holds the raw pc-relative address, which
; is fetched through pcc. An alias of a function is a code symbol too. When the
; capability-global initializer began storing alias slots (C-75), a function
; alias's slot went through `scc gp` and got cap-table bounds with a cursor far
; outside them.

target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"

define dso_local void @fn() addrspace(200) {
  ret void
}
@fn_alias = weak dso_local alias void (), ptr addrspace(200) @fn

@table = dso_local addrspace(200) global [2 x ptr addrspace(200)] [
  ptr addrspace(200) @fn,
  ptr addrspace(200) @fn_alias
], align 16

; CHECK-LABEL: __capstone_cap_init:
; CHECK-NOT:   scc
; CHECK:       auipc {{a[0-9]+}}, %pcrel_hi(fn_alias)
; CHECK-NOT:   scc
; CHECK:       stc {{a[0-9]+}}, 16(a{{[0-9]+}})
; CHECK:       ret
