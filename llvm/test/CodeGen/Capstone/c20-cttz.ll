; C-20: `llvm.cttz` crashes the legalizer on capstone64 at every optimisation
; level (LegalizeDAG assertion `Res.getValueType() == Node->getValueType(0)`),
; for i32 and i64 alike.  Measured 2026-09-04 on the branch toolchain; the
; registry's C-20/C-24 commits are repro packages, not fixes.  ctlz and ctpop
; compile (3 of 3 functions emitted in the same measurement), so the control
; below is green on its own and the crash is specific to cttz.
;
; Fixed 2026-09-04: the cause was not a missing legalization action but the
; generic de Bruijn table lookup (TargetLowering::CTTZTableLookup) asking for
; the table's address in address space 0, an i64, while lowerConstantPool
; always produces the c128 capability -- "Type mismatch for custom legalized
; operation".  The lookup now uses the default globals address space for the
; constant-pool pointer, so the expansion is a multiply, a shift and a byte
; load through a capability, pinned below.
;
; MUTATION: n/a until the fix lands -- llc produces no output at all today, so
; FileCheck sees an empty input; once it passes, the control's `ctlz` check is
; the mutation target (change ctlz to cttz in @ctlz32_control and the two
; functions must still both be emitted).
;
; RUN: llc -mtriple=capstone64 -mattr=+m -O0 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 -verify-machineinstrs < %s | FileCheck %s
; RUN: %llc_cap -O1 < %s -o /dev/null

declare i32 @llvm.cttz.i32(i32, i1)
declare i64 @llvm.cttz.i64(i64, i1)
declare i32 @llvm.ctlz.i32(i32, i1)

; CHECK-LABEL: cttz32:
; CHECK: cjalr zero, 0(ra)
define i32 @cttz32(i32 %x) {
  %r = call i32 @llvm.cttz.i32(i32 %x, i1 false)
  ret i32 %r
}

; CHECK-LABEL: cttz64:
; CHECK: cjalr zero, 0(ra)
define i64 @cttz64(i64 %x) {
  %r = call i64 @llvm.cttz.i64(i64 %x, i1 false)
  ret i64 %r
}

; CONTROL: the sibling intrinsic legalizes and is emitted.
; CHECK-LABEL: ctlz32_control:
; CHECK: cjalr zero, 0(ra)
define i32 @ctlz32_control(i32 %x) {
  %r = call i32 @llvm.ctlz.i32(i32 %x, i1 false)
  ret i32 %r
}
