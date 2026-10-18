; A constant global whose initializer holds the address of another global or
; function (`const char *const tbl[] = {...}`, `void (*const ops[])(void)`) is
; written at run time by __capstone_cap_init: a tag cannot be part of the static
; image. Such a holder must not be placed in read-only data, where an image
; mapped with W^X page permissions would fault on the initializer's store. The
; pass therefore clears its constant flag, and the holder is placed with the
; writable data. A constant without capability slots stays read-only.
;
; RUN: llc -mtriple=capstone64 -mattr=+m < %s | FileCheck %s
; RUN: llc -mtriple=capstone64 -mattr=+m -data-sections < %s | FileCheck %s --check-prefix=SECT

@.s0 = private addrspace(200) constant [3 x i8] c"ab\00"
@.s1 = private addrspace(200) constant [3 x i8] c"cd\00"
@names = dso_local addrspace(200) constant [2 x ptr addrspace(200)]
    [ptr addrspace(200) @.s0, ptr addrspace(200) @.s1], align 16
@ops = dso_local addrspace(200) constant [1 x ptr addrspace(200)]
    [ptr addrspace(200) @f], align 16
@scalars = dso_local addrspace(200) constant [2 x i64] [i64 1, i64 2], align 8

define dso_local void @f() addrspace(200) {
  ret void
}

define dso_local ptr addrspace(200) @get(i64 %i) addrspace(200) {
  %p = getelementptr inbounds [2 x ptr addrspace(200)], ptr addrspace(200) @names, i64 0, i64 %i
  %v = load ptr addrspace(200), ptr addrspace(200) %p, align 16
  ret ptr addrspace(200) %v
}

; CHECK-LABEL: __capstone_cap_init:
; CHECK:      stc
; CHECK:      .type names,@object
; CHECK-NEXT: .data
; CHECK-NOT:  .section
; CHECK:      names:
; CHECK-NOT:  .section
; CHECK:      ops:
; CHECK:      .section .rodata,"a",@progbits
; CHECK-NOT:  .section
; CHECK:      scalars:

; SECT: .section .data.names,"aw"
; SECT: .section .data.ops,"aw"
; SECT: .section .rodata.scalars,"a"
