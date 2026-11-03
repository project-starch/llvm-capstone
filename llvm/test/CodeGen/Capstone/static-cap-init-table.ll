; RUN: llc -mtriple=capstone64 -capstone-gp-free -capstone-image-gp -capstone-enable-global-merge=false -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=capstone64 -capstone-gp-free -capstone-image-gp -capstone-cap-init-table=false < %s | FileCheck %s --check-prefix=LEGACY
; RUN: llc -mtriple=capstone64 -capstone-gp-free -capstone-image-gp -capstone-shrink-globals=false -capstone-enable-global-merge=false < %s | FileCheck %s --check-prefix=BROAD
; RUN: llc -mtriple=capstone64 -capstone-gp-free -capstone-image-gp -capstone-enable-global-merge=false -filetype=obj < %s -o %t
;
; The table retains nested slot offsets, signed interior offsets, object bounds
; and broad image authority for functions, labels, aliases and incomplete data.
; Null slots need no record. A tagged TLS template remains rejected separately.

target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"

@data = addrspace(200) global [32 x i8] zeroinitializer, align 16
@alias = weak alias [32 x i8], ptr addrspace(200) @data
@unknown = external addrspace(200) global [0 x i8]
@slots = addrspace(200) constant {i64, [7 x ptr addrspace(200)]} {i64 7, [7 x ptr addrspace(200)] [
  ptr addrspace(200) @data,
  ptr addrspace(200) getelementptr (i8, ptr addrspace(200) @data, i64 -3),
  ptr addrspace(200) @alias,
  ptr addrspace(200) @unknown,
  ptr addrspace(200) @code,
  ptr addrspace(200) blockaddress(@code, %label),
  ptr addrspace(200) null
]}, align 16

define void @code() addrspace(200) {
entry:
  br label %label
label:
  ret void
}

; CHECK-LABEL: __capstone_cap_init:
; CHECK-NOT: stc
; CHECK: __capstone_cap_init_table_v1
; CHECK: .section .capstone_cap_relocs
; CHECK: [[T:.L__capstone_cap_init_records[0-9]*]]:
; CHECK-NEXT: .word slots-[[T]]+16
; CHECK-NEXT: .word data-[[T]]
; CHECK-NEXT: .word 32
; CHECK-NEXT: .word 0
; CHECK-NEXT: .word slots+32-([[T]]+16)
; CHECK-NEXT: .word data-([[T]]+16)
; CHECK-NEXT: .word 32
; CHECK-NEXT: .word 4294967293
; CHECK-NEXT: .word slots+48-([[T]]+32)
; CHECK-NEXT: .word alias-([[T]]+32)
; CHECK-NEXT: .word 0
; CHECK-NEXT: .word 0
; CHECK-NEXT: .word slots+64-([[T]]+48)
; CHECK-NEXT: .word unknown-([[T]]+48)
; CHECK-NEXT: .word 0
; CHECK-NEXT: .word 0
; CHECK-NEXT: .word slots+80-([[T]]+64)
; CHECK-NEXT: .word code-([[T]]+64)
; CHECK-NEXT: .word 0
; CHECK-NEXT: .word 0
; CHECK-NEXT: .word slots+96-([[T]]+80)
; CHECK-NEXT: .word {{.*}}-([[T]]+80)
; CHECK-NEXT: .word 0
; CHECK-NEXT: .word 0
; CHECK-NEXT: .size [[T]], 96
;
; LEGACY-LABEL: __capstone_cap_init:
; LEGACY: stc
; LEGACY-NOT: __capstone_cap_init_table_v1
;
; BROAD: .section .capstone_cap_relocs
; BROAD: .word data-
; BROAD-NEXT: .word 0
; BROAD-NEXT: .word 0
