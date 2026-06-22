; Verify that floating-point operations lower to named soft-float libcalls on
; Capstone, instead of aborting in TargetLowering::makeLibCall with
; "unsupported library call operation" (the Capstone runtime libcall-name table
; was previously empty), and that fp128 constants are loaded from the constant
; pool (ldc) rather than materialized as an unforgeable 128-bit capability.
;
; RUN: llc -mtriple=capstone64 -mattr=+m < %s | FileCheck %s

; A double divide must become a __divdf3 call (doubles are soft-float here).
; CHECK-LABEL: ddiv:
; CHECK: __divdf3
define double @ddiv(double %a, double %b) {
  %r = fdiv double %a, %b
  ret double %r
}

; An fp128 (long double) add must become a __addtf3 call.
; CHECK-LABEL: qadd:
; CHECK: __addtf3
define fp128 @qadd(fp128 %a, fp128 %b) {
  %r = fadd fp128 %a, %b
  ret fp128 %r
}

; An fp128 constant operand is softened to a 128-bit value; it must be loaded
; from the constant pool with a capability load (ldc), not forged as an
; immediate. The add itself still lowers to __addtf3.
; CHECK-LABEL: qconst:
; CHECK-DAG: ldc
; CHECK-DAG: __addtf3
define fp128 @qconst(fp128 %a) {
  %r = fadd fp128 %a, 0xL00000000000000004000900000000000
  ret fp128 %r
}
