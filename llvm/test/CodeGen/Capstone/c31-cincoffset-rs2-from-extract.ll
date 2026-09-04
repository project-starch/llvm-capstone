; C-31: a pointer's address read with ptrtoint is an EXTRACT_SUBREG of the
; capability register -- no instruction -- so when that integer is then used as
; the rs2 of a cincoffset, the register the hardware reads for rs2 is the live
; capability's own register, whose metadata shadow is still TAGGED.  The RTL
; raises UNEXPECTED_OPERAND on a tagged rs2 (capstone_flu_unit.anvil:30, rtl-oracle
; 2026-09-04); QEMU reads the cursor and carries on (op_helper.c:736-737).  So
; `q + (long)p` is a silicon-only fault in ordinary C.  Measured 2026-09-04 on
; the branch compiler: at -O2 @use_int_of_ptr is `cincoffset a0, a1, a0` with a0
; the untouched capability argument.  -O0 spills and reloads the integer through
; memory (`ld a1, 0(a1)`), which is an integer write and clears the shadow.
;
; Fixed 2026-09-05: every c128 -> i64 truncate selects to PseudoTRUNC_CAP
; (addi rd, rs, 0, printed `mv`), an integer write between the capability and
; any use of its address, as cap_get_cursor already did.  Here rd and rs
; coincide (`mv a0, a0`): the write is what clears the shadow, not the move.
;
; MUTATION: the pre-fix compiler IS the failing case -- `cincoffset a0, a1, a0`
; with no integer write, which the CHECK-NOT below rejected (measured
; 2026-09-05 before the rebuild; this file was XFAIL on it).
;
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=capstone64 -mattr=+m -O1 -verify-machineinstrs < %s | FileCheck %s
; RUN: %llc_cap -O0 < %s -o /dev/null

; CHECK-LABEL: use_int_of_ptr:
; CHECK-NOT: cincoffset a0, a1, a0
; CHECK: {{(mv|addi)}} [[R:a[0-9]+]], a0
; CHECK: cincoffset a0, a1, [[R]]
; CHECK: lb a0, 0(a0)
define i64 @use_int_of_ptr(ptr addrspace(200) %p, ptr addrspace(200) %q) {
  %i = ptrtoint ptr addrspace(200) %p to i64
  %a = getelementptr i8, ptr addrspace(200) %q, i64 %i
  %v = load i8, ptr addrspace(200) %a
  %r = sext i8 %v to i64
  ret i64 %r
}

; CHECK-LABEL: offset_by_ptr:
; CHECK-NOT: cincoffset a0, a0, a1
; CHECK: {{(mv|addi)}} [[R:a[0-9]+]], a1
; CHECK: cincoffset a0, a0, [[R]]
define ptr addrspace(200) @offset_by_ptr(ptr addrspace(200) %base, ptr addrspace(200) %p) {
  %i = ptrtoint ptr addrspace(200) %p to i64
  %a = getelementptr i8, ptr addrspace(200) %base, i64 %i
  ret ptr addrspace(200) %a
}

; CONTROL: an integer instruction (srai) sits between the capability and the
; cincoffset; the read itself is still the integer write (`mv a0, a0` before
; the srai), and the srai/slli pair follows unchanged.
; CHECK-LABEL: hash_ptr:
; CHECK: srai a0, a0, 4
; CHECK-NEXT: slli a0, a0, 3
; CHECK-NEXT: cincoffset a0, a1, a0
define i64 @hash_ptr(ptr addrspace(200) %p, ptr addrspace(200) %tab) {
  %i = ptrtoint ptr addrspace(200) %p to i64
  %s = ashr i64 %i, 4
  %a = getelementptr i64, ptr addrspace(200) %tab, i64 %s
  %v = load i64, ptr addrspace(200) %a
  ret i64 %v
}
