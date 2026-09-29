; RUN: llc -mtriple=capstone64 -verify-machineinstrs < %s | FileCheck %s
;
; A table of label addresses (GNU labels-as-values; an interpreter's direct-
; threaded dispatch table, as in mruby's VM) is initialized at run time like a
; table of function pointers. In the static image each slot holds the label's
; LINK-time address; the domain is loaded at a runtime base and processes no
; relocations, so an indirect branch through that value fetched from a stale
; address (mruby: instruction access fault, pc = tval = the link-time label).
; The capability-global initializer now materializes each label pc-relatively,
; derived from gp and tagged, and stores it over the slot.
;
; The dispatch itself is unchanged: the slot is loaded as a capability and
; branched through.

target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"

@run.table = internal addrspace(200) constant [2 x ptr addrspace(200)] [
  ptr addrspace(200) blockaddress(@run, %op_add),
  ptr addrspace(200) blockaddress(@run, %op_ret)
], align 16

; CHECK-LABEL: run:
; CHECK:       ldc [[T:a[0-9]+]], 0(a{{[0-9]+}})
; CHECK:       jr [[T]]
; CHECK:       .Ltmp[[ADD:[0-9]+]]: # Block address taken
; CHECK:       .Ltmp[[RET:[0-9]+]]: # Block address taken

; CHECK-LABEL: __capstone_cap_init:
; CHECK:       auipc [[TB:a[0-9]+]], %pcrel_hi(run.table)
; CHECK-DAG:   auipc [[L0:a[0-9]+]], %pcrel_hi(.Ltmp[[ADD]])
; CHECK-DAG:   auipc [[L1:a[0-9]+]], %pcrel_hi(.Ltmp[[RET]])
; CHECK-DAG:   cincoffset [[L0]], gp, [[L0]]
; CHECK-DAG:   cincoffset [[L1]], gp, [[L1]]
; CHECK-DAG:   stc [[L0]], 0(a{{[0-9]+}})
; CHECK-DAG:   stc [[L1]], 16(a{{[0-9]+}})
; CHECK:       cjalr zero, 0(ra)

; The static bytes remain the link-time addresses; the stores above replace them.
; CHECK-LABEL: run.table:
; CHECK-NEXT:  .quad .Ltmp[[ADD]]
; CHECK-NEXT:  .zero 8
; CHECK-NEXT:  .quad .Ltmp[[RET]]

define dso_local signext i32 @run(ptr addrspace(200) %code) addrspace(200) {
entry:
  %op0 = load i8, ptr addrspace(200) %code, align 1
  %i0 = zext i8 %op0 to i64
  %s0 = getelementptr inbounds [2 x ptr addrspace(200)], ptr addrspace(200) @run.table, i64 0, i64 %i0
  %t0 = load ptr addrspace(200), ptr addrspace(200) %s0, align 16
  indirectbr ptr addrspace(200) %t0, [label %op_add, label %op_ret]

op_add:
  %acc = phi i32 [ 0, %entry ], [ %acc1, %op_add ]
  %pc = phi ptr addrspace(200) [ %code, %entry ], [ %pc1, %op_add ]
  %acc1 = add i32 %acc, 1
  %pc1 = getelementptr inbounds i8, ptr addrspace(200) %pc, i64 1
  %op1 = load i8, ptr addrspace(200) %pc1, align 1
  %i1 = zext i8 %op1 to i64
  %s1 = getelementptr inbounds [2 x ptr addrspace(200)], ptr addrspace(200) @run.table, i64 0, i64 %i1
  %t1 = load ptr addrspace(200), ptr addrspace(200) %s1, align 16
  indirectbr ptr addrspace(200) %t1, [label %op_add, label %op_ret]

op_ret:
  %r = phi i32 [ 0, %entry ], [ %acc1, %op_add ]
  ret i32 %r
}
