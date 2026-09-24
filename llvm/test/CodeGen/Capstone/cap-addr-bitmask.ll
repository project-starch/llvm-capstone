; Bitwise arithmetic on a CAPABILITY, which is what C that goes through uintptr_t
; and back turns into: align a pointer down, steal a low bit as a flag, hash two
; pointers. The address is read and the operation happens at XLen.
;
; WHAT COMES BACK CHANGED (CapstoneRecoverProvenance). This test used to assert
; that every result is UNTAGGED -- "nothing here may produce a tagged capability
; out of integer bits". The pass does not do that either: it produces no
; capability out of integer bits. When the integer was computed from exactly one
; pointer (align_down, clear_flag_bit), it rebuilds the result as THAT pointer's
; capability moved to the computed address -- same tag, same bounds, same
; permissions, cursor wherever the arithmetic put it -- which is the rule
; CHERI C's capability-carrying uintptr_t gives the same source. An address
; computed from two pointers (hash_two) has no single owner and stays untagged,
; as before. The old behaviour is kept, and pinned below with its original
; implicit-check-nots, as -capstone-recover-provenance=false.
;
; Each untagged result is a SINGLE arithmetic instruction on the address half
; (EXTRACT_SUBREG on sub_cap_addr to read, INSERT_SUBREG to write back); a
; recovered one adds the delta from the source's address and one cincoffset.
; The file keeps its i128 name for the history that references it.
; MUTATION (ON): make @clear_flag_bit's mask come from a second pointer's
; address -> it has two sources, stays untagged, and its cincoffset line fails.
;
; RUN: llc -mtriple=capstone64 -filetype=asm -verify-machineinstrs < %s \
; RUN:   | FileCheck %s --check-prefix=ON
; RUN: llc -mtriple=capstone64 -filetype=asm -verify-machineinstrs \
; RUN:     -capstone-recover-provenance=false < %s \
; RUN:   | FileCheck %s --implicit-check-not=init --implicit-check-not=scc \
; RUN:       --implicit-check-not=movc --implicit-check-not=cincoffset
; RUN: %llc_cap -O0 < %s -o /dev/null
; RUN: %llc_cap -O1 < %s -o /dev/null

target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"

; ON-LABEL: align_down:
; ON: andi [[M:[a-z0-9]+]], [[A:[a-z0-9]+]], -32
; ON-NEXT: sub [[D:[a-z0-9]+]], [[M]], [[A]]
; ON-NEXT: cincoffset a0, a0, [[D]]
; ON-NEXT: cjalr zero, 0(ra)
; CHECK-LABEL: align_down:
; CHECK: andi a0, a0, -32
; CHECK-NEXT: cjalr zero, 0(ra)
define ptr addrspace(200) @align_down(ptr addrspace(200) %p) addrspace(200) {
  %i = ptrtoint ptr addrspace(200) %p to i64
  %and = and i64 %i, -32
  %r = inttoptr i64 %and to ptr addrspace(200)
  ret ptr addrspace(200) %r
}

; ON-LABEL: clear_flag_bit:
; ON: andi [[M:[a-z0-9]+]], [[A:[a-z0-9]+]], -2
; ON-NEXT: sub [[D:[a-z0-9]+]], [[M]], [[A]]
; ON-NEXT: cincoffset a0, a0, [[D]]
; CHECK-LABEL: clear_flag_bit:
; CHECK: andi a0, a0, -2
; CHECK-NEXT: cjalr zero, 0(ra)
define ptr addrspace(200) @clear_flag_bit(ptr addrspace(200) %p) addrspace(200) {
  %i = ptrtoint ptr addrspace(200) %p to i64
  %and = and i64 %i, -2
  %r = inttoptr i64 %and to ptr addrspace(200)
  ret ptr addrspace(200) %r
}

; Each address read is one integer write (C-31, `mv`), then the xor.  A second
; `mv a0, a0` follows the xor: the combiner re-forms the xor of two truncates as
; a truncate of the xor, and that truncate is read the same way -- harmless, one
; redundant self-move, noted rather than pinned.
; ON-LABEL: hash_two:
; ON-NOT: cincoffset
; ON: xor a0, a0, a1
; ON-NOT: cincoffset
; ON: cjalr zero, 0(ra)
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
