; Fuzz finding F-01 (llvm-stress, every seed, 2026-09-04): a variable-index
; extractelement or insertelement on a vector wider than the legal width is split
; by the type legalizer through a stack temporary, and
; TargetLowering::getVectorSubVecPointer zero-extends the INDEX to the POINTER's
; value type -- which on this target is the capability type c128, not an integer
; -- so SelectionDAG::getNode asserts "Invalid ZERO_EXTEND!" and llc dies at
; every optimisation level.  Reduced by capstone/tests/reduce.sh from
; llvm-stress seed 1 (240 -> 8 lines).  Reachable from C through a vector
; extension with a variable subscript; no project corpus does that, which is
; why no suite caught it.  Registry ID C-39.
;
; XFAIL: *
; RUN: llc -mtriple=capstone64 -mattr=+m -O0 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 -verify-machineinstrs < %s | FileCheck %s
; RUN: %llc_cap -O1 < %s -o /dev/null

target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"

; CHECK-LABEL: extract_var:
; CHECK: cjalr zero, 0(ra)
define i64 @extract_var(i32 %i) addrspace(200) {
  %e = extractelement <8 x i64> zeroinitializer, i32 %i
  ret i64 %e
}

; CHECK-LABEL: insert_var:
; CHECK: cjalr zero, 0(ra)
define void @insert_var(ptr addrspace(200) %p, i64 %x, i32 %i) addrspace(200) {
  %v = load <8 x i64>, ptr addrspace(200) %p
  %w = insertelement <8 x i64> %v, i64 %x, i32 %i
  store <8 x i64> %w, ptr addrspace(200) %p
  ret void
}

; The VALUE arm: "does not assert" is satisfied by a wrong-but-legal index as much
; as by the right one.  Eight distinct constants, a variable index: the emitted
; addressing must clamp the index to the vector (andi 7), scale it by the element
; size (slli 3), apply it to the stack temporary with a capability increment, and
; load through the result.  An index computed in the wrong width or scale changes
; the slli/andi and fails here.  (Exact lines pinned once the fix lands.)
; CHECK-LABEL: extract_known:
; CHECK-DAG: andi {{a[0-9]+}}, {{a[0-9]+}}, 7
; CHECK-DAG: slli {{a[0-9]+}}, {{a[0-9]+}}, 3
; CHECK: cincoffset {{a[0-9]+}}, {{(sp|s0|a[0-9]+)}}, {{a[0-9]+}}
; CHECK: ld a0, 0({{a[0-9]+}})
; CHECK: cjalr zero, 0(ra)
define i64 @extract_known(i32 %i) addrspace(200) {
  %e = extractelement <8 x i64> <i64 10, i64 11, i64 12, i64 13, i64 14, i64 15, i64 16, i64 17>, i32 %i
  ret i64 %e
}
