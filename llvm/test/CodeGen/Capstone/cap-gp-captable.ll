; gp-captable global ABI (-capstone-gp-captable), Stage 1: access-side lowering.
;
; Default (capability ABI, flag off): a global is addressed by deriving its
; capability from gp with `cincoffset gp, <pcrel VA>` (+ delin + shrink) -- which
; needs gp to be a data cap over the code image, unusable on the RTL.
;
; With -capstone-gp-captable: gp is the base of a per-global capability table (a
; data cap the entry glue derives from sp/cscratch). Global i's data capability
; lives at gp[i] (byte offset i*16); it is loaded with `ldc rd, i*16(gp)` and used
; directly as the base pointer. This is the model proven to run on captype-fixed
; CVA6 (tests/runtime-qemu/gp-free-domain/start-gpfree-captable.S). The flag
; defaults off, so ordinary codegen is byte-identical.
;
; See CapstoneISelLowering.cpp lowerGlobalAddress, CapstoneISelDAGToDAG.cpp
; getGpCaptableIndex, and plans/gp-captable-codegen-plan.md.

; RUN: llc -mtriple=capstone64 -mattr=+m < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,CAP
; RUN: llc -mtriple=capstone64 -mattr=+m -capstone-gp-captable < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,CT

target datalayout = "e-m:e-pf200:128:128:128:64-p:64:64-i64:64-i128:128-n32:64-S128-A200-P200-G200"

@g0 = addrspace(200) global i32 0, align 4
@g1 = addrspace(200) global i32 7, align 4

; Global index 1 -> cap-table slot at 1*16 = 16(gp).
; CHECK-LABEL: load_g1:
; CT:      ldc a0, 16(gp)
; CT-NEXT: lw a0, 0(a0)
; CT-NOT:  scc
; CAP:     cincoffset a0, gp, a0
; CAP:     lw a0, 0(a0)
define i32 @load_g1() {
  %v = load i32, ptr addrspace(200) @g1, align 4
  ret i32 %v
}

; Global index 0 -> cap-table slot at 0(gp).
; CHECK-LABEL: store_g0:
; CT:      ldc a1, 0(gp)
; CT-NEXT: sw a0, 0(a1)
; CT-NOT:  scc
; CAP:     cincoffset a1, gp, a1
; CAP:     sw a0, 0(a1)
define void @store_g0(i32 %v) {
  store i32 %v, ptr addrspace(200) @g0, align 4
  ret void
}

; Stage 2: the `.capstone_gp_table` descriptor the entry glue reads to build the
; runtime cap-table. One record per global in index order: {size, align, init_off}
; (init_off = 0 for a zero global, else PC-relative to its image template). Emitted
; only under the flag.
; CAP-NOT: .capstone_gp_table
; CT:      .section .capstone_gp_table
; CT:      .quad 2
; g0 (index 0): i32 zeroinitializer -> size 4, align 4, init_off 0.
; CT:      .quad 4
; CT-NEXT: .quad 4
; CT-NEXT: .quad 0
; g1 (index 1): i32 7 -> size 4, align 4, PC-relative offset to the template.
; CT:      .quad 4
; CT-NEXT: .quad 4
; CT:      .quad g1-{{.*}}
