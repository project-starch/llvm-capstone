; A constant one-field struct holding a capability (`static const struct { char
; *p; } h = {"abc"}`) keeps its .gct record although CapstoneCapGlobalInit
; clears its constant flag (the pass writes the holder at startup, so it is
; placed with the writable data). The emitter's struct records are limited to
; holders that are constant in the source; the pass records that as the
; "capstone-cap-init-const" attribute.
;
; RUN: llc -mtriple=capstone64 -mattr=+m < %s | FileCheck %s

@.str = private unnamed_addr addrspace(200) constant [4 x i8] c"abc\00", align 1
@holder = dso_local addrspace(200) constant { ptr addrspace(200) } { ptr addrspace(200) @.str }, align 16

define dso_local ptr addrspace(200) @get() addrspace(200) {
  %v = load ptr addrspace(200), ptr addrspace(200) @holder, align 16
  ret ptr addrspace(200) %v
}

; CHECK:      .data
; CHECK-NOT:  .section
; CHECK:      holder:
; CHECK:      .section .gct
; CHECK:      __llvm_static_cap_gct_begin:
; Header: 'SCAP' magic, version 1, reserved 0.
; CHECK-NEXT: .word 1346454355
; CHECK-NEXT: .half 1
; CHECK-NEXT: .half 0
; Object count = 2 (the holder + the string), slot count = 1.
; CHECK-NEXT: .word 2
; CHECK-NEXT: .word 1
