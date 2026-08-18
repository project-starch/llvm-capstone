; RUN: llc -mtriple=capstone64 -filetype=asm -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=capstone64 -filetype=obj -verify-machineinstrs < %s -o %t.o

; An INTEGER stored in a capability-sized slot, i.e. the absolute-value sibling of
; const-capability-initializer.ll. It reaches AsmPrinter as an inttoptr ConstantExpr rather than a
; ConstantInt, so emitGlobalConstantLargeInt does not handle it and emitValue would ask the streamer
; for a 16-byte integer, which asserts. Found by MicroPython's MP_ROM_INT, an integer stored in a
; union whose other member is an object pointer; a const table mixes both forms in one array, which
; is what the second global here reproduces.

@int_only = internal addrspace(200) constant ptr addrspace(200) inttoptr (i128 14 to ptr addrspace(200)), align 16

; CHECK-LABEL: int_only:
; CHECK:       .quad 14
; CHECK-NEXT:  .zero 8

@mixed = internal addrspace(200) constant [3 x ptr addrspace(200)] [
  ptr addrspace(200) inttoptr (i128 14 to ptr addrspace(200)),
  ptr addrspace(200) @sym,
  ptr addrspace(200) inttoptr (i128 1 to ptr addrspace(200))
], align 16

; CHECK-LABEL: mixed:
; CHECK:       .quad 14
; CHECK-NEXT:  .zero 8
; CHECK-NEXT:  .quad sym
; CHECK-NEXT:  .zero 8
; CHECK-NEXT:  .quad 1
; CHECK-NEXT:  .zero 8

; The high half is a ZERO extension, not a sign extension. This value arrives as a negative int64
; while its true 128-bit value is positive, so a sign-extending implementation would write an
; all-ones high word here and this line is what catches it.
@boundary = internal addrspace(200) constant ptr addrspace(200) inttoptr (i128 9223372036854775808 to ptr addrspace(200)), align 16

; CHECK-LABEL: boundary:
; CHECK:       .quad -9223372036854775808
; CHECK-NEXT:  .zero 8

@sym = internal addrspace(200) constant i64 7

define ptr addrspace(200) @get() addrspace(200) {
  ret ptr addrspace(200) @mixed
}
