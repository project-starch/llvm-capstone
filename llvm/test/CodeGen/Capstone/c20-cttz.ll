; C-20: `llvm.cttz` crashes the legalizer on capstone64 at every optimisation
; level (LegalizeDAG assertion `Res.getValueType() == Node->getValueType(0)`),
; for i32 and i64 alike.  Measured 2026-09-04 on the branch toolchain; the
; registry's C-20/C-24 commits are repro packages, not fixes.  ctlz and ctpop
; compile (3 of 3 functions emitted in the same measurement), so the control
; below is green on its own and the crash is specific to cttz.
;
; XFAIL until the CTTZ legalization actions the RISCV copy lost when GPR became
; 64-bit-only are restored (Tier 5); then lit reports XPASS and the marker
; comes off.  The expected shape after the fix is an inline expansion or an
; `__builtin_ctz`-class sequence -- pinned loosely here as "the function is
; emitted and returns", which is exactly what a crash cannot satisfy.
;
; MUTATION: n/a until the fix lands -- llc produces no output at all today, so
; FileCheck sees an empty input; once it passes, the control's `ctlz` check is
; the mutation target (change ctlz to cttz in @ctlz32_control and the two
; functions must still both be emitted).
;
; RUN: llc -mtriple=capstone64 -mattr=+m -O0 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 -verify-machineinstrs < %s | FileCheck %s
; RUN: %llc_cap -O1 < %s -o /dev/null
; XFAIL: *

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
