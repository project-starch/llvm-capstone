; C-40: address arithmetic on the NULL capability must not become a cincoffset
; whose base is the zero register.  On this ISA cincoffset raises
; UNEXPECTED_OPERAND when rs1 holds no capability, and null holds none, so
; `cincoffset rd, zero, rs2` faults on every execution.  The middle end and the
; codegen pipeline create exactly that shape: Loop Strength Reduction rewrites a
; pointer loop's exit test `a == aLast` into `(gep i8, null, %lsr.iv) == null`
; (SCEVExpander's expansion for a non-integral address space), which is what
; took every -O1 and -O2 SQLite SLT twin down at its first loop on 2026-09-04
; (sqlite3WhereClauseClear, pc ...9d4c, cause 24), while -O0 agreed with native.
; The right lowering is the integer one: null + x can never be a usable
; capability, so materialise x as an untagged value (the inttoptr route) and let
; comparisons see its cursor.  Registry ID C-40 (requested).
;
; MUTATION: the negative check is the --implicit-check-not below, and the thing
; that makes it fire is the selector without its null-base guard: on the
; pre-fix compiler (branch state before 2026-09-04, when this file was XFAIL)
; every function here emitted `cincoffset(imm) aN, zero, ...` and the check
; failed on each -- that red run is the performed demonstration.  An input-side
; mutation was tried and does NOT fire: an i128 GEP index on the null base
; (the auditor's residual) is truncated to i64 before selection at -O0 and -O2,
; so no zero-base cincoffset appears; recorded so nobody re-tries it.
; RUN: llc -mtriple=capstone64 -mattr=+m -O0 -verify-machineinstrs < %s | FileCheck %s --implicit-check-not='cincoffset{{(imm)?}} {{a[0-9]+}}, zero'
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 -verify-machineinstrs < %s | FileCheck %s --implicit-check-not='cincoffset{{(imm)?}} {{a[0-9]+}}, zero'
; RUN: %llc_cap -O1 < %s -o /dev/null

; The post-LSR shape: the comparison reads the cursor, which is %iv.
; CHECK-LABEL: null_gep_cmp:
; CHECK: seqz a0, a0
; CHECK: cjalr zero, 0(ra)
define i1 @null_gep_cmp(i64 %iv) addrspace(200) {
  %p = getelementptr i8, ptr addrspace(200) null, i64 %iv
  %c = icmp eq ptr addrspace(200) %p, null
  ret i1 %c
}

; A null-based pointer handed on: an untagged capability whose cursor is %iv.
; CHECK-LABEL: null_gep_value:
; CHECK: cjalr zero, 0(ra)
define ptr addrspace(200) @null_gep_value(i64 %iv) addrspace(200) {
  %p = getelementptr i8, ptr addrspace(200) null, i64 %iv
  ret ptr addrspace(200) %p
}

; The constant form (select-cap.ll guards the same instruction from another route).
; CHECK-LABEL: null_gep_const:
; CHECK: li {{a[0-9]+}}, 32
; CHECK: cjalr zero, 0(ra)
define ptr addrspace(200) @null_gep_const() addrspace(200) {
  %p = getelementptr i8, ptr addrspace(200) null, i64 32
  ret ptr addrspace(200) %p
}
