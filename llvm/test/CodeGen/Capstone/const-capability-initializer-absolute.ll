; RUN: llc -mtriple=capstone64 -filetype=asm -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=capstone64 -filetype=obj -verify-machineinstrs < %s -o %t.o
; RUN: llvm-readobj -r %t.o | FileCheck %s --check-prefix=OBJ
; RUN: llvm-objdump -s -j .rodata %t.o | FileCheck %s --check-prefix=BYTES
; RUN: %llc_cap -O0 < %s -o /dev/null
; RUN: %llc_cap -O1 < %s -o /dev/null

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

; Object view (measured 2026-09-04).  Only the @sym slot of @mixed (.rodata
; +0x20) is relocated; the three integer slots and @boundary carry none, and
; the bytes show the boundary value ZERO-extended: low quad 0x8000000000000000,
; high quad 0.  Type NAMES print as "Unknown" until C-37.
; MUTATION: replace the first `inttoptr (i128 14 ...)` of @mixed with `@sym`
; -> a relocation at 0x10 appears before the 0x20 line and it fails
; (performed 2026-09-04).
; OBJ: .rela.rodata {
; OBJ-NEXT: 0x20 {{R_Capstone_64|Unknown}} sym 0x0
; OBJ-NEXT: }
; BYTES: 0000 0e000000 00000000 00000000 00000000
; BYTES-NEXT: 0010 0e000000 00000000 00000000 00000000
; BYTES-NEXT: 0020 00000000 00000000 00000000 00000000
; BYTES-NEXT: 0030 01000000 00000000 00000000 00000000
; BYTES-NEXT: 0040 00000000 00000080 00000000 00000000
; BYTES-NEXT: 0050 07000000 00000000
