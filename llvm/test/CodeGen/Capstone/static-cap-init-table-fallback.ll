; RUN: llc -mtriple=capstone64 -capstone-gp-free -capstone-image-gp -capstone-enable-global-merge=false -verify-machineinstrs < %s | FileCheck %s
;
; Object sizes and cursor addends that do not fit the compact record keep their
; ordinary constructor stores. A normal leaf in the same module uses the table.
target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"
@big = external addrspace(200) global [4294967296 x i8]
@small = addrspace(200) global [16 x i8] zeroinitializer
@ptrs = addrspace(200) global [3 x ptr addrspace(200)] [
  ptr addrspace(200) @big,
  ptr addrspace(200) getelementptr (i8, ptr addrspace(200) @small, i64 2147483648),
  ptr addrspace(200) @small
]
; CHECK-LABEL: __capstone_cap_init:
; CHECK-COUNT-2: stc
; CHECK: __capstone_cap_init_table_v1
; CHECK: .section .capstone_cap_relocs
; CHECK: .word small-
; CHECK-NEXT: .word 16
; CHECK-NEXT: .word 0
; CHECK: .size .L__capstone_cap_init_records, 16

; RUN: llc -mtriple=capstone64 -capstone-gp-free -capstone-image-gp -capstone-cap-init-limit=1 < %s | FileCheck %s --check-prefix=LIMIT
; LIMIT-LABEL: __capstone_cap_init:
; LIMIT: stc
; LIMIT-NOT: __capstone_cap_init_table_v1
; LIMIT-NOT: .capstone_cap_relocs
