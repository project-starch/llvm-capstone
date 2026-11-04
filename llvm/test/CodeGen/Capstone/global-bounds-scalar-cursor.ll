; RUN: llc -mtriple=capstone64 -mattr=+m -capstone-gp-free -capstone-image-gp -verify-machineinstrs -O2 < %s | FileCheck %s --check-prefix=ABS --implicit-check-not="mv "
; RUN: llc -mtriple=capstone64 -mattr=+m -capstone-gp-free -capstone-image-gp -relocation-model=pic -verify-machineinstrs -O2 < %s | FileCheck %s --check-prefix=ABS --implicit-check-not="mv "
; RUN: llc -mtriple=capstone64 -mattr=+m -verify-machineinstrs -O2 < %s | FileCheck %s --check-prefix=REL
; RUN: llc -mtriple=capstone64 -mattr=+m -capstone-gp-free -capstone-image-gp -verify-machineinstrs -O0 < %s -o /dev/null
;
; SCC sets the cursor to its already untagged PC-relative integer operand.
; Reuse that integer for SHRINK's base and end without changing either
; capability operation. The relative-cursor ABI must still extract the cursor.

target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"

@g = addrspace(200) global [4 x i64] zeroinitializer, align 8

; ABS-LABEL: global_load:
; ABS: li [[SIZE:a[0-9]+]], 32
; ABS: auipc [[HI:a[0-9]+]],
; ABS-NEXT: addi [[ADDR:a[0-9]+]], [[HI]],
; ABS-DAG: scc [[CAP:a[0-9]+]], gp, [[ADDR]]
; ABS-DAG: add [[END:a[0-9]+]], [[ADDR]], [[SIZE]]
; ABS: shrink [[CAP]], [[ADDR]], [[END]]
; ABS: ld a0, 0([[CAP]])
; REL-LABEL: global_load:
; REL: cincoffset [[CAP:a[0-9]+]], gp,
; REL: delin [[CAP]]
; REL: mv [[ADDR:a[0-9]+]], [[CAP]]
; REL: shrink [[CAP]], [[ADDR]],
define i64 @global_load() addrspace(200) {
  %v = load i64, ptr addrspace(200) @g
  ret i64 %v
}

; Narrowing remains before a variable index, including out-of-bounds indices.
; ABS-LABEL: global_index:
; ABS: scc
; ABS: shrink [[CAP:a[0-9]+]],
; ABS: cincoffset [[PTR:a[0-9]+]], [[CAP]],
; ABS: ld a0, 0([[PTR]])
define i64 @global_index(i64 %i) addrspace(200) {
  %p = getelementptr [4 x i64], ptr addrspace(200) @g, i64 0, i64 %i
  %v = load i64, ptr addrspace(200) %p
  ret i64 %v
}

; Escaping pointers still carry exactly the object's narrowed authority.
; ABS-LABEL: global_address:
; ABS: scc
; ABS: shrink a0,
; ABS: ret
define ptr addrspace(200) @global_address() addrspace(200) {
  ret ptr addrspace(200) @g
}
