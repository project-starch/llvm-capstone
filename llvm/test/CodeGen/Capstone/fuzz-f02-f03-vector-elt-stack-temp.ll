; F-02 / F-03 (tests/fuzz/findings/F02-vector-elt-load-recreated, F03-vector-elt-store-recreated):
; a variable-index extractelement / insertelement goes through a stack temporary. The element
; load (and the store twin) is created with an unknown-stack pointer info, addrspace 200 and no
; value; when the index folds to a constant the pointer becomes a bare FrameIndex and getLoad
; re-infers a fixed-stack pointer info, whose address space came from the target machine's
; getAddressSpaceForPseudoSourceKind -- 0 by default. The DAG's CSE key includes that address
; space, so DAGCombiner's alignment refinement, which assumes getExtLoad hands back the SAME
; node, got a fresh one and asserted `NewLoad.getNode() == N` (F-02) / `NewStore.getNode() == N`
; (F-03) at -O2 (llvm-stress seeds 32 and 155, 2026-09-04). Red first: both functions crashed
; llc -O2 before the fix; -O0 always compiled them. The fix has two halves and needs both:
; CapstoneTargetMachine::getAddressSpaceForPseudoSourceKind returns the capability address
; space for stack pseudo sources, and InferPointerInfo (SelectionDAG.cpp) gives a value-less
; address-space-0 pointer info on a capability pointer that address space before inferring --
; the hook alone moved the mismatch to loads created with MachinePointerInfo() (vararg.ll
; asserted the other way round at -O1).
;
; RUN: llc -mtriple=capstone64 -O2 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=capstone64 -O0 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=capstone64 -O2 -verify-machineinstrs -filetype=obj < %s -o /dev/null
; RUN: %llc_cap -O1 < %s -o /dev/null

target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"
target triple = "capstone64"

; CHECK-LABEL: f02_extract_variable_index:
; CHECK: cjalr zero, 0(ra)
define i8 @f02_extract_variable_index(i32 %0) addrspace(200) {
BB:
  %E6 = extractelement <1 x i32> zeroinitializer, i32 %0
  %E13 = extractelement <4 x i8> zeroinitializer, i32 %E6
  ret i8 %E13
}

; CHECK-LABEL: f03_insert_variable_index:
; CHECK: cjalr zero, 0(ra)
define <8 x i8> @f03_insert_variable_index(i32 %0) addrspace(200) {
BB:
  %E35 = extractelement <1 x i32> zeroinitializer, i32 %0
  %I37 = insertelement <8 x i8> zeroinitializer, i8 1, i32 %E35
  ret <8 x i8> %I37
}

; The un-folded shape: a genuinely variable index, so the pointer stays FrameIndex + idx and the
; stack temporary is read at run time. Must compile at every level, with tables of the right
; address space in the memory operands (the MIR would show `addrspace 200` on the stack access).
; CHECK-LABEL: f02_extract_runtime_index:
; CHECK: cjalr zero, 0(ra)
define i8 @f02_extract_runtime_index(<4 x i8> %v, i32 %i) addrspace(200) {
  %e = extractelement <4 x i8> %v, i32 %i
  ret i8 %e
}
