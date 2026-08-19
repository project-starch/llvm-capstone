; A select whose RESULT ARMS are i128 constants. On capstone64 i128 is the capability
; carrier, so an ordinary ternary on __int128 is a 128-bit select, and a NEGATIVE arm
; reached ISel unselectable at every optimisation level:
;
;   Cannot select: i128 = CapstoneISD::SELECT_CC ... Constant:i128<-4>, Constant:i128<0>
;
; The materialisation path tested a constant arm with getActiveBits(), which rejects
; every negative value: as an i128, -4 is 0xFFFF..FFFC and has 128 active bits. The
; rule is now the one getI128NumericValueOrFatal already uses for this question --
; representable in XLen under one signedness or the other -- and the arm is extended
; back the way it came so the materialised constant EQUALS the one it replaces.
;
; Found in jerry-core's ecma_op_object_find_own at -Os. Repro folder:
; capstone/tests/compiler-repros/C21-i128-select-of-constants/
;
; RUN: llc -mtriple=capstone64 -mattr=+m < %s | FileCheck %s

target datalayout = "e-m:e-pf200:128:128:128:64-p:64:64-i64:64-i128:128-n32:64-S128-A200-P200-G200"

; The regression. Without the fix llc aborts and NOTHING reaches FileCheck, so every
; check below fails -- which is what makes this a test rather than a description.
; CHECK-LABEL: neg_const_arm:
; CHECK: beqz
; CHECK: li{{.*}}, -4
define i128 @neg_const_arm(i32 %c) {
  %t = icmp ne i32 %c, 0
  %r = select i1 %t, i128 -4, i128 0
  ret i128 %r
}

; Both arms negative, so neither is the zero shortcut and both take the new path.
; CHECK-LABEL: two_neg_arms:
; CHECK: li{{.*}}, -9
define i128 @two_neg_arms(i32 %c) {
  %t = icmp ne i32 %c, 0
  %r = select i1 %t, i128 -9, i128 -3
  ret i128 %r
}

; A value that fits XLen UNSIGNED but not signed: 2^63. It must survive as itself.
; Lowered as a shift of the condition rather than a branch, which is fine -- the point
; is that it compiles and the constant is not mangled.
; CHECK-LABEL: large_unsigned_arm:
; CHECK: slli{{.*}}, 63
define i128 @large_unsigned_arm(i32 %c) {
  %t = icmp ne i32 %c, 0
  %r = select i1 %t, i128 9223372036854775808, i128 0
  ret i128 %r
}
