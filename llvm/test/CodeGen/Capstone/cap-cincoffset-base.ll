; cscincoffset requires the tagged capability in the base (rs1) position; the
; integer index must be the offset (rs2). ISD::ADD i128 is commutative, so a
; `cap + int` may reach instruction selection with the operands in either order
; (e.g. a capability reloaded from a spill slot as a raw i128 load facing a
; scaled integer index that DAGCombine has already reassociated ahead of it).
; CapstoneTargetLowering canonicalizes custom-lowered adds, and selectCIncOffset
; applies the same predicate-based canonicalization for adds that reach ISel
; directly, so the reloaded capability is always the base. If a raw i128 load
; were misclassified as an integer offset (or the capability were left in the
; offset position), cscincoffset would get an untagged integer base and the
; runtime would fault on the first dereference.
;
; The reversed-operand path that only arises from post-legalize combines is
; exercised end to end by the SQLite domain (design/sqlite-gap5-
; cscincoffset-operand-order.md); this test locks in the operand-role classifier
; that both canonicalization sites share.
;
; See CapstoneISelDAGToDAG.cpp (selectCIncOffset) and CapstoneISelLowering.cpp
; (isCapstoneIntegerOffset / isCapstoneCapabilityValue).

; RUN: llc -mtriple=capstone64 -mattr=+m -verify-machineinstrs < %s | FileCheck %s

target datalayout = "e-m:e-pf200:128:128:128:64-p:64:64-i64:64-i128:128-n32:64-S128-A200-P200-G200"

; A capability loaded from memory (raw i128 ldc) indexed by a scaled runtime
; integer: the reloaded capability must be the cincoffset base, the index the
; offset.
; CHECK-LABEL: load_cap_scaled_index:
; CHECK: ldc [[CAP:a[0-9]+]], 0(a0)
; CHECK: cincoffset {{a[0-9]+}}, [[CAP]], {{a[0-9]+}}
; CHECK: lw a0, 0(a0)
define i32 @load_cap_scaled_index(ptr addrspace(200) %pp, i64 %i) addrspace(200) {
  %cap = load ptr addrspace(200), ptr addrspace(200) %pp, align 16
  %idx = shl i64 %i, 2
  %g = getelementptr i8, ptr addrspace(200) %cap, i64 %idx
  %v = load i32, ptr addrspace(200) %g, align 4
  ret i32 %v
}

; Plain cap + zero-extended 32-bit index: the capability stays the base and the
; widened integer is the offset.
; CHECK-LABEL: cap_plus_zext_index:
; CHECK: ldc [[CAP2:a[0-9]+]], 0(a0)
; CHECK: cincoffset {{a[0-9]+}}, [[CAP2]], {{a[0-9]+}}
; CHECK: lbu a0, 0(a0)
define i8 @cap_plus_zext_index(ptr addrspace(200) %pp, i32 %i) addrspace(200) {
  %cap = load ptr addrspace(200), ptr addrspace(200) %pp, align 16
  %idx = zext i32 %i to i64
  %g = getelementptr i8, ptr addrspace(200) %cap, i64 %idx
  %v = load i8, ptr addrspace(200) %g, align 1
  ret i8 %v
}
