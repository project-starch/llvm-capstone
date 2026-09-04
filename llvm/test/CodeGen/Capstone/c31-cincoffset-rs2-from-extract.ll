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
; The fix has to put an integer write between the capability and its use as an
; integer operand of cincoffset/scc/shrink: PseudoTRUNC_CAP (addi rd, rs, 0), as
; cap_get_cursor already does, instead of a bare sub-register read.  This file
; pins that shape and is XFAIL until the fix lands.
;
; MUTATION: n/a until the fix lands (the negative checks below fail today, which
; is the XFAIL); once green, the control @hash_ptr is the mutation target: its
; `srai` is the integer write, and replacing `ashr` by nothing turns @hash_ptr
; into the @offset_by_ptr shape, which the CHECK-NOT must then reject.
;
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=capstone64 -mattr=+m -O1 -verify-machineinstrs < %s | FileCheck %s
; RUN: %llc_cap -O0 < %s -o /dev/null
; XFAIL: *

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

; CONTROL: an integer instruction (srai) already sits between the capability and
; the cincoffset, so this shape is safe today and must stay a single srai/slli
; pair -- no extra move is wanted where the shadow is already clear.
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
