; Bitwise arithmetic on a CAPABILITY, which is what C that goes through uintptr_t
; and back turns into: align a pointer down, steal a low bit as a flag, hash two
; pointers. The address is read, the operation happens at XLen, and the result is
; UNTAGGED -- which is what the source asked for, since a value built out of
; uintptr_t bits cannot carry a tag.
;
; The file keeps its i128 name for the history that references it. There is no
; i128 left in it: a capability is c128 and its address is i64, so both the read
; (EXTRACT_SUBREG on sub_cap_addr) and the write back (INSERT_SUBREG, an ADDI on
; the address half that clears the tag) are subregister references. Each of these
; three functions is now a SINGLE arithmetic instruction -- the reads and writes
; coalesce away entirely. They used to cost a `mv` on the way in, and before the
; inttoptr lowering existed the way back out was a stack round-trip: two `sd` and
; an `ldc` to assemble a 128-bit value whose metadata half is not data.
;
; The result of the mask is UNTAGGED, and that is the point: nothing here may
; produce a tagged capability out of integer bits. The implicit-check-nots cover
; the WHOLE output, not just the tail after the last CHECK -- each names an
; instruction that would appear if inttoptr tried to preserve provenance instead
; of writing the address half.
; RUN: llc -mtriple=capstone64 -filetype=asm -verify-machineinstrs < %s \
; RUN:   | FileCheck %s --implicit-check-not=init --implicit-check-not=scc \
; RUN:       --implicit-check-not=movc --implicit-check-not=cincoffset

target datalayout = "e-m:e-pf200:128:128:128:64-p:64:64-i64:64-i128:128-n32:64-S128-A200-P200-G200"

; CHECK-LABEL: align_down:
; CHECK: andi a0, a0, -32
; CHECK-NEXT: cjalr zero, 0(ra)
define ptr addrspace(200) @align_down(ptr addrspace(200) %p) addrspace(200) {
  %i = ptrtoint ptr addrspace(200) %p to i64
  %and = and i64 %i, -32
  %r = inttoptr i64 %and to ptr addrspace(200)
  ret ptr addrspace(200) %r
}

; CHECK-LABEL: clear_flag_bit:
; CHECK: andi a0, a0, -2
; CHECK-NEXT: cjalr zero, 0(ra)
define ptr addrspace(200) @clear_flag_bit(ptr addrspace(200) %p) addrspace(200) {
  %i = ptrtoint ptr addrspace(200) %p to i64
  %and = and i64 %i, -2
  %r = inttoptr i64 %and to ptr addrspace(200)
  ret ptr addrspace(200) %r
}

; CHECK-LABEL: hash_two:
; CHECK: xor a0, a0, a1
; CHECK-NEXT: cjalr zero, 0(ra)
define i64 @hash_two(ptr addrspace(200) %a, ptr addrspace(200) %b) addrspace(200) {
  %x = ptrtoint ptr addrspace(200) %a to i64
  %y = ptrtoint ptr addrspace(200) %b to i64
  %h = xor i64 %x, %y
  %p = inttoptr i64 %h to ptr addrspace(200)
  %r = ptrtoint ptr addrspace(200) %p to i64
  ret i64 %r
}
