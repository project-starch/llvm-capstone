; WHAT THIS TEST ACTUALLY COVERS -- CORRECTED 2026-08-19 after an adversarial audit.
;
; IT DOES NOT EXERCISE lowerScalarI128Shift's widening-multiply hook. Measured, not
; argued: with breakpoints on the function entry, the guard and the MULH emit, this file
; produces NO hits at all, and the pre-legalize DAG dump shows why --
;
;     Initial selection DAG:            t7: i128 = mul t5, t6
;                                       t10: i128 = srl t7, Constant:i64<64>
;     Optimized lowered selection DAG:  t15: i64 = mulhu t2, t4
;
; DAGCombiner::combineShiftToMULH (DAGCombiner.cpp:10717, from visitSRL:11326) consumes
; the shape BEFORE legalisation. It has no requirement that the wide type be illegal --
; only isOperationLegalOrCustom(MULHU, i64), which holds here. So the claim that "i128 is
; legal, therefore that combine never runs" is FALSE, and these CHECKs pass with or
; without the Capstone hook.
;
; `smulh` below settles it independently: it uses `ashr`, i.e. ISD::SRA, which the hook
; cannot match under any circumstances, so its `mulh` provably comes from the combiner.
;
; THE TEST IS STILL WORTH KEEPING, as a LOCK ON THE COMBINER PATH: if combineShiftToMULH
; ever stops firing for these shapes, they would fall through to legalisation and abort,
; and this file catches that. It must simply not be read as coverage of the hook.
;
; The hook is GONE (2026-08-24, with the i128 capability carrier). Its emit path never
; had a known reaching input, and the file that exercised its reject path --
; cap-i128-widening-mul-const-signedness.ll -- went with it. This one stays, because what
; it locks is the COMBINER, which is unaffected.
;
; The HIGH HALF of a widening multiply, which on this target arrives as a real
; 128-bit shift and used to crash the compiler.
;
; DAGCombiner rewrites srl(mul(zext a, zext b), 64) into MULHU before legalisation.
; It USED to be that i128 was legal here -- it was the capability carrier -- and the
; original crash was blamed on that: the shift was thought to survive to legalisation,
; where SelectionDAGLegalize::ExpandNode asserted "Unable to legalize non-vector shift".
; The audit above showed the combine fires either way. A capability is c128 now, so
; i128 is illegal like everywhere else, and this shape is doubly covered.
;
; Nothing has to write __int128 to reach this: division by a constant is strength
; reduced to mulhu(x, magic) >> s, which is how lib/oofatfs's f_mkfs crashed the
; compiler and blocked MICROPY_VFS in a Capstone domain.
;
; RUN: llc -mtriple=capstone64 -mattr=+m < %s | FileCheck %s

target datalayout = "e-m:e-pf200:128:128:128:64-p:64:64-i64:64-i128:128-n32:64-S128-A200-P200-G200"

; CHECK-LABEL: umulh:
; CHECK: mulhu
; CHECK-NOT: call
define i64 @umulh(i64 %a, i64 %b) {
  %wa = zext i64 %a to i128
  %wb = zext i64 %b to i128
  %p = mul i128 %wa, %wb
  %hi = lshr i128 %p, 64
  %r = trunc i128 %hi to i64
  ret i64 %r
}

; The signed form must pick mulh, not mulhu; getting that backwards would be a
; miscompile rather than a crash, which is why it is pinned separately.
; CHECK-LABEL: smulh:
; CHECK: mulh {{.*}}
; CHECK-NOT: mulhu
define i64 @smulh(i64 %a, i64 %b) {
  %wa = sext i64 %a to i128
  %wb = sext i64 %b to i128
  %p = mul i128 %wa, %wb
  %hi = ashr i128 %p, 64
  %r = trunc i128 %hi to i64
  ret i64 %r
}

; One extended operand and one constant that fits XLen, which is the shape a
; udiv-by-constant expansion actually produces.
; CHECK-LABEL: umulh_const:
; CHECK: mulhu
define i64 @umulh_const(i64 %a) {
  %wa = zext i64 %a to i128
  %p = mul i128 %wa, 1148256711715503
  %hi = lshr i128 %p, 64
  %r = trunc i128 %hi to i64
  ret i64 %r
}

; A SIGNED widening multiply by a constant that really is representable in XLen as a
; signed value. This must still fold to mulh -- the fix for the miscompile below
; tightened the constant check and must not have thrown this away too.
; CHECK-LABEL: smulh_small_const:
; CHECK: mulh
define i64 @smulh_small_const(i64 %a) {
  %wa = sext i64 %a to i128
  %p  = mul i128 %wa, 1000003
  %hi = lshr i128 %p, 64
  %r  = trunc i128 %hi to i64
  ret i64 %r
}
