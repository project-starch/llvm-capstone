; Bitwise arithmetic on a CAPABILITY, which is what C that goes through uintptr_t
; and back turns into: align a pointer down, steal a low bit as a flag, hash two
; pointers. The address is read and the operation happens at XLen.
;
; Bare pointer parameters may be LINEAR even when nonnull. The provenance
; pass therefore leaves their round trips untagged. Both pass-on and pass-off
; runs pin the scalar form; known NONLIN recovery has positive controls in
; recover-provenance.ll.
;
; RUN: llc -mtriple=capstone64 -filetype=asm -verify-machineinstrs < %s \
; RUN:   | FileCheck %s --implicit-check-not=init --implicit-check-not=scc \
; RUN:       --implicit-check-not=movc --implicit-check-not=cincoffset
; RUN: llc -mtriple=capstone64 -filetype=asm -verify-machineinstrs \
; RUN:     -capstone-recover-provenance=false < %s \
; RUN:   | FileCheck %s --implicit-check-not=init --implicit-check-not=scc \
; RUN:       --implicit-check-not=movc --implicit-check-not=cincoffset
; RUN: %llc_cap -O0 < %s -o /dev/null
; RUN: %llc_cap -O1 < %s -o /dev/null

target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"

; CHECK-LABEL: align_down:
; CHECK: andi a0, a0, -32
; CHECK-NEXT: cjalr zero, 0(ra)
define ptr addrspace(200) @align_down(ptr addrspace(200) nonnull %p) addrspace(200) {
  %i = ptrtoint ptr addrspace(200) %p to i64
  %and = and i64 %i, -32
  %r = inttoptr i64 %and to ptr addrspace(200)
  ret ptr addrspace(200) %r
}

; CHECK-LABEL: clear_flag_bit:
; CHECK: andi a0, a0, -2
; CHECK-NEXT: cjalr zero, 0(ra)
define ptr addrspace(200) @clear_flag_bit(ptr addrspace(200) nonnull %p) addrspace(200) {
  %i = ptrtoint ptr addrspace(200) %p to i64
  %and = and i64 %i, -2
  %r = inttoptr i64 %and to ptr addrspace(200)
  ret ptr addrspace(200) %r
}

; Each address read is one integer write (C-31, `mv`), then the xor.  A second
; `mv a0, a0` follows the xor: the combiner re-forms the xor of two truncates as
; a truncate of the xor, and that truncate is read the same way -- harmless, one
; redundant self-move, noted rather than pinned.
; CHECK-LABEL: hash_two:
; CHECK: xor a0, a0, a1
; CHECK: cjalr zero, 0(ra)
define i64 @hash_two(ptr addrspace(200) %a, ptr addrspace(200) %b) addrspace(200) {
  %x = ptrtoint ptr addrspace(200) %a to i64
  %y = ptrtoint ptr addrspace(200) %b to i64
  %h = xor i64 %x, %y
  %p = inttoptr i64 %h to ptr addrspace(200)
  %r = ptrtoint ptr addrspace(200) %p to i64
  ret i64 %r
}

; The same arithmetic on a pointer that may be null: left as the address
; computation it was, since moving a base that holds no capability would trap at
; the cincoffset instead of returning an address the caller may never use.
; CHECK-LABEL: align_down_maybe_null:
; CHECK: andi a0, a0, -32
; CHECK-NEXT: cjalr zero, 0(ra)
define ptr addrspace(200) @align_down_maybe_null(ptr addrspace(200) %p) addrspace(200) {
  %i = ptrtoint ptr addrspace(200) %p to i64
  %and = and i64 %i, -32
  %r = inttoptr i64 %and to ptr addrspace(200)
  ret ptr addrspace(200) %r
}
