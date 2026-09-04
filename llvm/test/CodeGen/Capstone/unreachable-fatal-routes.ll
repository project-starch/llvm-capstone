; Three fatal routes in CapstoneISelDAGToDAG.cpp are recorded as UNREACHABLE in
; capstone/tests/lit-coverage-unreachable.txt, and this file is the test that
; entry names.  Each part below is the input that would have to reach the
; route, and the check shows why it never does.
;
; 1. "Folded load/store displacement must fit in signed 64-bits": a GEP index
;    is truncated to the 64-bit index width by IR semantics before selection
;    ever sees it.  An i128 index of exactly 2^64 is therefore offset 0, and the
;    load folds to `ld a0, 0(a0)` -- no displacement overflow is representable.
; 2./3. "TIGHTEN immediate must be a constant" / "CCSRRW CSR must be a
;    constant": both intrinsics declare that operand ImmArg, so the IR verifier
;    rejects a non-immediate before the DAG is built.  (From C, `_Constant` in
;    the builtin prototype rejects it earlier still: see
;    clang/test/Sema/capstone-tighten-nonconst.c.)
;
; MUTATION: in gep.ll change the index to i128 8 -> the fold becomes
; `ld a0, 8(a0)` and the pinned `ld a0, 0(a0)` fails; in immarg.ll replace
; %n with `i64 7` -> the verifier accepts the module and `not opt` fails the
; RUN line (performed 2026-09-04 on scratch copies).
;
; RUN: split-file %s %t
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 -verify-machineinstrs < %t/gep.ll | FileCheck %s --check-prefix=GEP
; RUN: not opt -passes=verify -S < %t/immarg.ll 2>&1 | FileCheck %s --check-prefix=IMMARG

;--- gep.ll
; GEP-LABEL: gep_2pow64:
; GEP: ld a0, 0(a0)
define i64 @gep_2pow64(ptr addrspace(200) %p) {
  %g = getelementptr i8, ptr addrspace(200) %p, i128 18446744073709551616
  %v = load i64, ptr addrspace(200) %g
  ret i64 %v
}

;--- immarg.ll
declare ptr addrspace(200) @llvm.capstone.cap.tighten.p200(ptr addrspace(200), i64)
; IMMARG: immarg operand has non-immediate parameter
define ptr addrspace(200) @tighten_nonconst(ptr addrspace(200) %p, i64 %n) {
  %r = call ptr addrspace(200) @llvm.capstone.cap.tighten.p200(ptr addrspace(200) %p, i64 %n)
  ret ptr addrspace(200) %r
}
