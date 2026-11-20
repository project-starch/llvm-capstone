; C-74: an i8 or i16 cmpxchg/atomicrmw is widened to its aligned 32-bit word.
; A global bounded to itself overran that word when its size was not a multiple
; of 4 or it was not 4-aligned (a lone byte faulted at -O0). Globals targeted by
; a sub-word atomic are now padded to a multiple of 4 and 4-aligned; a global
; with no such atomic is unchanged.
; RUN: llc -mtriple=capstone64 -mattr=+a,+m -O0 < %s | FileCheck %s
; RUN: llc -mtriple=capstone64 -mattr=+a,+m -O2 < %s | FileCheck %s --check-prefix=O2

@lone = internal addrspace(200) global i8 0, align 1
@five = internal addrspace(200) global [5 x i8] zeroinitializer, align 1
@eight = internal addrspace(200) global [8 x i8] zeroinitializer, align 1
@half = internal addrspace(200) global i16 0, align 2
@plain = internal addrspace(200) global i8 0, align 1

; CHECK-DAG: .comm lone,4,4
; CHECK-DAG: .comm five,8,4
; CHECK-DAG: .comm eight,8,4
; CHECK-DAG: .comm half,4,4
; The global with no sub-word atomic keeps its size and alignment.
; CHECK-DAG: .comm plain,1,1
; O2-NOT: lone,1
; O2-NOT: five,5

define i8 @cas_lone(i8 %e) addrspace(200) {
  %r = cmpxchg ptr addrspace(200) @lone, i8 %e, i8 1 seq_cst seq_cst
  %v = extractvalue { i8, i1 } %r, 0
  ret i8 %v
}

define i8 @cas_five_last(i8 %e) addrspace(200) {
  %p = getelementptr inbounds [5 x i8], ptr addrspace(200) @five, i64 0, i64 4
  %r = cmpxchg ptr addrspace(200) %p, i8 %e, i8 1 seq_cst seq_cst
  %v = extractvalue { i8, i1 } %r, 0
  ret i8 %v
}

define i8 @add_eight(i8 %v) addrspace(200) {
  %p = getelementptr inbounds [8 x i8], ptr addrspace(200) @eight, i64 0, i64 4
  %r = atomicrmw add ptr addrspace(200) %p, i8 %v seq_cst
  ret i8 %r
}

define i16 @xchg_half(i16 %v) addrspace(200) {
  %r = atomicrmw xchg ptr addrspace(200) @half, i16 %v seq_cst
  ret i16 %r
}

define i8 @touch_plain() addrspace(200) {
  %v = load i8, ptr addrspace(200) @plain
  %w = add i8 %v, 1
  store i8 %w, ptr addrspace(200) @plain
  ret i8 %v
}
