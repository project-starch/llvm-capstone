; RUN: llc -mtriple=capstone64 -filetype=asm -verify-machineinstrs < %s | FileCheck %s

; A const pointer table whose entry points at an EXTERN object. collectStaticCapReducedObject
; guards the holder against having no initializer but not the target it points at, and a
; declaration has none, so asking for it asserted. Reached from MicroPython's obj.c, which only
; got this far once the selection failures ahead of it were fixed.
;
; The table is simply not reducible here (only a byte-array target is), so the right answer is to
; decline it, not to inspect it.

@extern_target = external addrspace(200) global [4 x i8]
@bytes = internal addrspace(200) constant [4 x i8] c"abc\00"

@tbl_extern = internal addrspace(200) constant [1 x ptr addrspace(200)] [
  ptr addrspace(200) @extern_target
], align 16

@tbl_local = internal addrspace(200) constant [1 x ptr addrspace(200)] [
  ptr addrspace(200) @bytes
], align 16

; CHECK-LABEL: tbl_extern:
; CHECK:       .quad extern_target
; CHECK-NEXT:  .zero 8

define ptr addrspace(200) @get_extern() addrspace(200) {
  ret ptr addrspace(200) @tbl_extern
}

define ptr addrspace(200) @get_local() addrspace(200) {
  ret ptr addrspace(200) @tbl_local
}
