; Verify the static-capability GCT (.gct) metadata is emitted for an *array* of
; addrspace(200) capability pointers (e.g. `const char *tbl[]`), not just a
; one-field struct.  Capability-typed fields in file-scope statics do not carry
; tags in the static image, so the backend emits a descriptor of the holder
; object plus one slot per element (and a target object per distinct string) for
; a runtime materialization step to consume.  Without array support a string
; table like BEEBS dtoa's `nums[]` produced no GCT records and its pointers
; loaded untagged.
;
; RUN: llc -mtriple=capstone64 -mattr=+m < %s | FileCheck %s

target datalayout = "e-m:e-pf200:128:128:128:64-p:64:64-i64:64-i128:128-n32:64-S128-A200-P200-G200"

@.str = private unnamed_addr addrspace(200) constant [4 x i8] c"abc\00", align 1
@.str.1 = private unnamed_addr addrspace(200) constant [3 x i8] c"de\00", align 1
@tbl = dso_local addrspace(200) constant [2 x ptr addrspace(200)]
    [ptr addrspace(200) @.str, ptr addrspace(200) @.str.1], align 16

; CHECK: .section .gct
; CHECK: __llvm_static_cap_gct_begin:
; Header: 'SCAP' magic, version 1, reserved 0.
; CHECK-NEXT: .word 1346454355
; CHECK-NEXT: .half 1
; CHECK-NEXT: .half 0
; Object count = 3 (one holder array + two string targets), slot count = 2.
; CHECK-NEXT: .word 3
; CHECK-NEXT: .word 2
; Template bytes = 32 (2 x 16-byte holder slots) + 4 ("abc\0") + 3 ("de\0") = 39.
; CHECK-NEXT: .word 39
; CHECK-NEXT: .word 24
; CHECK-NEXT: .word 40
; Holder object: id 0, size 32, align 16, template offset 0, first slot 0, 2 slots.
; CHECK-NEXT: .word 0
; CHECK-NEXT: .word 32
; CHECK-NEXT: .word 16
; CHECK-NEXT: .word 0
; CHECK-NEXT: .word 0
; CHECK-NEXT: .word 2
