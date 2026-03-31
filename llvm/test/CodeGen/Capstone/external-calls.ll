; RUN: llc < %s -mtriple=capstone64 | FileCheck %s

; Compiler-generated ExternalSymbol callees now reuse the same capability
; materialization path as direct symbol calls: materialize a PC-relative offset,
; derive a callable capability from gp, then perform cjalr.

declare void @ext_nop()

; CHECK-LABEL: call_external_symbol_like:
; CHECK: auipc [[HI:a[0-9]+]], %pcrel_hi(ext_nop)
; CHECK: addi [[OFF:a[0-9]+]], [[HI]], %pcrel_lo
; CHECK: cincoffset [[CAP:a[0-9]+]], gp, [[OFF]]
; CHECK: cjalr ra, 0([[CAP]])
define void @call_external_symbol_like() {
entry:
  call void @ext_nop()
  ret void
}

