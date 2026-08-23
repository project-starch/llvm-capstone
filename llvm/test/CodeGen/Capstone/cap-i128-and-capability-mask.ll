; RUN: llc -mtriple=capstone64 -filetype=asm -verify-machineinstrs < %s | FileCheck %s

; Bitwise arithmetic on a CAPABILITY, which is what C that goes through uintptr_t and back turns
; into: align a pointer down, steal a low bit as a flag, hash two pointers. The address is read with
; the same scalar move a pointer difference uses (ptr-diff-signed.ll), the operation happens at XLen,
; and the result is untagged -- which is what the source asked for, since a value built out of
; uintptr_t bits cannot carry a tag.
;
; The read used to be `lcc rd, rs, 2`. Since a capability became c128, ptrtoint is a TRUNCATE to the
; index width, selected as PseudoTRUNC_CAP; the low half of the register IS the cursor, so the two
; read the same thing and the move is one instruction instead of two.
;
; The narrow form is not avoidable by writing better C: expressing the align-down as
; `p - (p & (N-1))` to stay in the pointer domain is folded straight back into `p & ~(N-1)` by
; DAGCombiner, which is why this is lowered rather than diagnosed. All three shapes come from
; MicroPython: gc_init, pairheap.c's NEXT_GET_RIGHTMOST_PARENT, and bound_meth_unary_op.

; CHECK-LABEL: align_down:
; CHECK:      mv a0, a0
; CHECK-NEXT: andi a0, a0, -32
define ptr addrspace(200) @align_down(ptr addrspace(200) %p) addrspace(200) {
  %i = ptrtoint ptr addrspace(200) %p to i64
  %and = and i64 %i, -32
  %conv = zext i64 %and to i128
  %r = inttoptr i128 %conv to ptr addrspace(200)
  ret ptr addrspace(200) %r
}

; CHECK-LABEL: clear_flag_bit:
; CHECK:      mv a0, a0
; CHECK-NEXT: andi a0, a0, -2
define ptr addrspace(200) @clear_flag_bit(ptr addrspace(200) %p) addrspace(200) {
  %i = ptrtoint ptr addrspace(200) %p to i64
  %and = and i64 %i, -2
  %conv = zext i64 %and to i128
  %r = inttoptr i128 %conv to ptr addrspace(200)
  ret ptr addrspace(200) %r
}

; Two capabilities, no constant: both cursors are read.
; CHECK-LABEL: hash_two:
; CHECK-DAG:  mv a0, a0
; CHECK-DAG:  mv a1, a1
; CHECK:      xor a0, a0, a1
define i64 @hash_two(ptr addrspace(200) %a, ptr addrspace(200) %b) addrspace(200) {
  %x = ptrtoint ptr addrspace(200) %a to i64
  %y = ptrtoint ptr addrspace(200) %b to i64
  %h = xor i64 %x, %y
  %w = zext i64 %h to i128
  %p = inttoptr i128 %w to ptr addrspace(200)
  %r = ptrtoint ptr addrspace(200) %p to i64
  ret i64 %r
}
