; RUN: llc -mtriple=capstone64 -verify-machineinstrs < %s | FileCheck %s
;
; A table of alias addresses is initialized at run time like a table of global
; addresses. musl's fork() keeps one: atfork_locks holds &__atexit_lockptr and
; nine more, each a weak alias of one null lock pointer that a strong
; definition elsewhere may replace. The capability-global initializer took a
; slot only when it referenced a variable, a function or a label, so an alias
; slot kept its link-time address without a tag; with a second thread running,
; fork() walked the table and faulted (cause 24) loading through it. The
; initializer now stores the alias as it stores any global, pc-relatively and
; tagged, and the link resolves the name as it resolves the static relocation.

target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"

; CHECK-LABEL: __capstone_cap_init:
; CHECK:       auipc [[T:a[0-9]+]], %pcrel_hi(locks)
; CHECK:       auipc [[W:a[0-9]+]], %pcrel_hi(weak_lockptr)
; CHECK-NEXT:  addi [[W]], [[W]], %pcrel_lo(.Lpcrel_hi{{[0-9]+}})
; CHECK-NEXT:  cincoffset [[W]], gp, [[W]]
; CHECK-NEXT:  delin [[W]]
; CHECK-DAG:   stc {{a[0-9]+}}, 0([[T]])
; CHECK-DAG:   stc [[W]], 16([[T]])
; CHECK:       cjalr zero, 0(ra)

; The static bytes remain the link-time addresses; the stores above replace them.
; CHECK-LABEL: locks:
; CHECK-NEXT:  .quad real_lockptr
; CHECK-NEXT:  .zero 8
; CHECK-NEXT:  .quad weak_lockptr

@lock = internal addrspace(200) global i32 0, align 4
@dummy_lockptr = internal addrspace(200) constant ptr addrspace(200) null, align 16
@real_lockptr = dso_local addrspace(200) constant ptr addrspace(200) @lock, align 16
@weak_lockptr = weak dso_local alias ptr addrspace(200), ptr addrspace(200) @dummy_lockptr

@locks = internal addrspace(200) constant [2 x ptr addrspace(200)] [
  ptr addrspace(200) @real_lockptr,
  ptr addrspace(200) @weak_lockptr
], align 16

define dso_local ptr addrspace(200) @lock_at(i64 %i) addrspace(200) {
entry:
  %slot = getelementptr inbounds [2 x ptr addrspace(200)], ptr addrspace(200) @locks, i64 0, i64 %i
  %p = load ptr addrspace(200), ptr addrspace(200) %slot, align 16
  %q = load ptr addrspace(200), ptr addrspace(200) %p, align 16
  ret ptr addrspace(200) %q
}
