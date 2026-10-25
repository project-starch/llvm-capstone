; RUN: llc -mtriple=capstone64-unknown-elf -mattr=+m,+a,+f,+d -target-abi lp64 \
; RUN:   -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=capstone64-unknown-elf -mattr=+m,+a,+f,+d -target-abi lp64 \
; RUN:   -verify-machineinstrs < %s | FileCheck %s --check-prefix=NOADD
;
; C-80: a select between two FP constants that are not legal immediates is
; combined into one load from a two-entry constant pool at
; pool + select(c, 0, 8) (DAGCombiner::convertSelectOfFPConstantsToLoadOffset).
; The pool address has getConstantPoolPtrTy's capability type and the offset is
; applied with getMemBasePlusOffset, so the entry is reached by cincoffset from
; the gp-derived pool capability. Before the fix the pool node was typed i64,
; the custom lowering returned a capability, and an assertions build aborted
; with "Type mismatch for custom legalized operation" (a release build crashed
; in DAGCombiner::visitADD on the integer ADD). The float pair are legal
; immediates and never reach the constant pool; they are the control.
; Reduced from SPEC CPU2017 544.nab_r (sff.c, ephi) built with +f,+d.

target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"

; One pool, false value first: offset 8 selects the NaN.
; CHECK-LABEL: .LCPI0_0:
; CHECK-NEXT:  .quad 0x3ff8000000000000
; CHECK-NEXT:  .quad 0x7ff8000000000000
; CHECK-LABEL: nan_or_one:
; CHECK-DAG:   flt.d [[C:a[0-9]+]],
; CHECK-DAG:   %pcrel_hi(.LCPI0_0)
; CHECK:       slli [[OFF:a[0-9]+]], [[C]], 3
; CHECK:       cincoffset [[P:a[0-9]+]], gp,
; CHECK:       cincoffset [[E:a[0-9]+]], [[P]], [[OFF]]
; CHECK:       ld {{a[0-9]+}}, 0([[E]])
; CHECK:       cjalr zero, 0(ra)
; The whole function, address construction included, has no integer add.
; NOADD-LABEL: nan_or_one:
; NOADD-NOT:   add {{a[0-9]+}}, {{a[0-9]+}}, {{a[0-9]+}}
; NOADD:       cjalr zero, 0(ra)
define double @nan_or_one(double %x) addrspace(200) {
  %neg = fcmp olt double %x, 0.000000e+00
  %r = select i1 %neg, double 0x7FF8000000000000, double 1.500000e+00
  ret double %r
}

; CHECK-NOT:   .LCPI1_
; CHECK-LABEL: two_floats:
; CHECK:       flt.s
; CHECK-NOT:   %pcrel_hi
; CHECK:       cjalr zero, 0(ra)
define float @two_floats(float %x) addrspace(200) {
  %neg = fcmp olt float %x, 0.000000e+00
  %r = select i1 %neg, float 3.250000e+00, float 7.500000e+00
  ret float %r
}
