; A select between two CAPABILITIES under every integer condition code.
;
; lowerSELECT's capability path maps only SETEQ/SETNE/SETLT/SETGE/SETULT/SETUGE
; to a branch and ends in llvm_unreachable for the other four (SGT, SLE, UGT,
; ULE).  That is sound only if those four are always canonicalised away before
; they reach it, by swapping the operands: sgt(a,b) becomes lt(b,a), and so on.
; This file pins that for all ten codes at both optimisation levels the
; production builds use, so a change in canonicalisation lands on this test
; and not on the unreachable.  It also pins the copy form: a capability select
; must move its arms with `movc`, never with `addi rd, rs, 0`, which would
; strip the tag.  Measured 2026-09-04: all ten select to a branch plus movc;
; the four "unsupported" codes appear with swapped operands.
;
; The last function compares two capabilities rather than two integers: the
; addresses are compared as integers (sltu) and the select branches on the
; result -- no lcc, no capability comparison instruction.
;
; MUTATION: change `icmp sgt` in @sel_sgt to `icmp slt` -> the pinned
; `blt a1, a0` (swapped operands) fails against `blt a0, a1` (performed
; 2026-09-04 on a scratch copy).
;
; RUN: llc -mtriple=capstone64 -mattr=+m -O1 -verify-machineinstrs < %s | FileCheck %s --implicit-check-not='addi a0, a1, 0' --implicit-check-not='addi a0, a2, 0' --implicit-check-not='addi a0, a3, 0'
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 -verify-machineinstrs < %s | FileCheck %s --implicit-check-not='addi a0, a1, 0' --implicit-check-not='addi a0, a2, 0' --implicit-check-not='addi a0, a3, 0'
; RUN: %llc_cap -O0 < %s -o /dev/null

; CHECK-LABEL: sel_eq:
; CHECK: beq a0, a1, .LBB
; CHECK: movc
define ptr addrspace(200) @sel_eq(i64 %a, i64 %b, ptr addrspace(200) %x, ptr addrspace(200) %y) {
  %c = icmp eq i64 %a, %b
  %r = select i1 %c, ptr addrspace(200) %x, ptr addrspace(200) %y
  ret ptr addrspace(200) %r
}

; CHECK-LABEL: sel_ne:
; CHECK: bne a0, a1, .LBB
; CHECK: movc
define ptr addrspace(200) @sel_ne(i64 %a, i64 %b, ptr addrspace(200) %x, ptr addrspace(200) %y) {
  %c = icmp ne i64 %a, %b
  %r = select i1 %c, ptr addrspace(200) %x, ptr addrspace(200) %y
  ret ptr addrspace(200) %r
}

; CHECK-LABEL: sel_slt:
; CHECK: blt a0, a1, .LBB
; CHECK: movc
define ptr addrspace(200) @sel_slt(i64 %a, i64 %b, ptr addrspace(200) %x, ptr addrspace(200) %y) {
  %c = icmp slt i64 %a, %b
  %r = select i1 %c, ptr addrspace(200) %x, ptr addrspace(200) %y
  ret ptr addrspace(200) %r
}

; CHECK-LABEL: sel_sge:
; CHECK: bge a0, a1, .LBB
; CHECK: movc
define ptr addrspace(200) @sel_sge(i64 %a, i64 %b, ptr addrspace(200) %x, ptr addrspace(200) %y) {
  %c = icmp sge i64 %a, %b
  %r = select i1 %c, ptr addrspace(200) %x, ptr addrspace(200) %y
  ret ptr addrspace(200) %r
}

; SGT is canonicalised to LT with swapped operands -- this is the code the
; lowering has no case for, reached only in its swapped form.
; CHECK-LABEL: sel_sgt:
; CHECK: blt a1, a0, .LBB
; CHECK: movc
define ptr addrspace(200) @sel_sgt(i64 %a, i64 %b, ptr addrspace(200) %x, ptr addrspace(200) %y) {
  %c = icmp sgt i64 %a, %b
  %r = select i1 %c, ptr addrspace(200) %x, ptr addrspace(200) %y
  ret ptr addrspace(200) %r
}

; CHECK-LABEL: sel_sle:
; CHECK: bge a1, a0, .LBB
; CHECK: movc
define ptr addrspace(200) @sel_sle(i64 %a, i64 %b, ptr addrspace(200) %x, ptr addrspace(200) %y) {
  %c = icmp sle i64 %a, %b
  %r = select i1 %c, ptr addrspace(200) %x, ptr addrspace(200) %y
  ret ptr addrspace(200) %r
}

; CHECK-LABEL: sel_ult:
; CHECK: bltu a0, a1, .LBB
; CHECK: movc
define ptr addrspace(200) @sel_ult(i64 %a, i64 %b, ptr addrspace(200) %x, ptr addrspace(200) %y) {
  %c = icmp ult i64 %a, %b
  %r = select i1 %c, ptr addrspace(200) %x, ptr addrspace(200) %y
  ret ptr addrspace(200) %r
}

; CHECK-LABEL: sel_uge:
; CHECK: bgeu a0, a1, .LBB
; CHECK: movc
define ptr addrspace(200) @sel_uge(i64 %a, i64 %b, ptr addrspace(200) %x, ptr addrspace(200) %y) {
  %c = icmp uge i64 %a, %b
  %r = select i1 %c, ptr addrspace(200) %x, ptr addrspace(200) %y
  ret ptr addrspace(200) %r
}

; CHECK-LABEL: sel_ugt:
; CHECK: bltu a1, a0, .LBB
; CHECK: movc
define ptr addrspace(200) @sel_ugt(i64 %a, i64 %b, ptr addrspace(200) %x, ptr addrspace(200) %y) {
  %c = icmp ugt i64 %a, %b
  %r = select i1 %c, ptr addrspace(200) %x, ptr addrspace(200) %y
  ret ptr addrspace(200) %r
}

; CHECK-LABEL: sel_ule:
; CHECK: bgeu a1, a0, .LBB
; CHECK: movc
define ptr addrspace(200) @sel_ule(i64 %a, i64 %b, ptr addrspace(200) %x, ptr addrspace(200) %y) {
  %c = icmp ule i64 %a, %b
  %r = select i1 %c, ptr addrspace(200) %x, ptr addrspace(200) %y
  ret ptr addrspace(200) %r
}

; Two capabilities compared: their addresses are read with the integer write
; (C-31, `mv`), go through sltu, the select branches on that, and the arm is
; moved with movc.  No lcc anywhere.
; CHECK-LABEL: sel_capcmp_ugt:
; CHECK-NOT: lcc
; CHECK: sltu {{a[0-9]+}}, {{a[0-9]+}}, {{a[0-9]+}}
; CHECK: bnez
; CHECK: movc a0, a1
; CHECK-NOT: lcc
define ptr addrspace(200) @sel_capcmp_ugt(ptr addrspace(200) %x, ptr addrspace(200) %y) {
  %c = icmp ugt ptr addrspace(200) %x, %y
  %r = select i1 %c, ptr addrspace(200) %x, ptr addrspace(200) %y
  ret ptr addrspace(200) %r
}
