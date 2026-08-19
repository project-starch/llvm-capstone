; A pointer difference must not require its operands to be tagged capabilities.
;
; `p - q` used to lower through cap_get_cursor on both sides, which selects to
; `lcc rd, rs, 2`. The spec makes that instruction raise "Unexpected operand
; type (24)" when x[rs1] is not a capability
; (capstone-spec/parts/cap-man-insn.adoc:164-168), so the sequence faulted
; whenever either operand was null -- and `NULL - NULL` is ordinary C, defined
; as 0 by every real implementation.
;
; It is not a hypothetical. mruby's VM entry computes
;
;     ptrdiff_t cioff = c->ci - c->cibase;   // vm.c:1120
;     if (!c->stbase) stack_init(mrb);       // both fields null until here
;
; and faulted before a single Ruby instruction ran.
;
; A plain truncate reads the same address without the tag requirement: it
; selects to PseudoTRUNC_CAP, i.e. `addi rd, rs, 0`, and the spec's opening
; paragraph for existing instructions says an ordinary RISC-V instruction
; reading a capability register uses its cursor. That is also what this backend
; emits for `(uintptr_t)p`, and the point of this test is that the two agree.
;
; RUN: llc -mtriple=capstone64 -mattr=+m -verify-machineinstrs < %s | FileCheck %s

target datalayout = "e-m:e-pf200:128:128:128:64-p:64:64-i64:64-i128:128-n32:64-S128-A200-P200-G200"

; The address of a single pointer: the shape that was already tag-agnostic.
; CHECK-LABEL: addr:
; CHECK-NOT: lcc
; CHECK: cjalr
define i64 @addr(ptr addrspace(200) %p) addrspace(200) {
  %a = ptrtoint ptr addrspace(200) %p to i64
  ret i64 %a
}

; The difference of two pointers: the shape that used to emit lcc twice.
; CHECK-LABEL: ptrdiff:
; CHECK-NOT: lcc
; CHECK: sub
; CHECK: cjalr
define i64 @ptrdiff(ptr addrspace(200) %a, ptr addrspace(200) %b) addrspace(200) {
  %pa = ptrtoint ptr addrspace(200) %a to i64
  %pb = ptrtoint ptr addrspace(200) %b to i64
  %d = sub i64 %pa, %pb
  ret i64 %d
}

; The same, scaled by an element size -- what C emits for `struct S *a - b`.
; The division is incidental; what matters is that neither address read is lcc.
; CHECK-LABEL: ptrdiff_scaled:
; CHECK-NOT: lcc
; CHECK: cjalr
define i64 @ptrdiff_scaled(ptr addrspace(200) %a, ptr addrspace(200) %b) addrspace(200) {
  %pa = ptrtoint ptr addrspace(200) %a to i64
  %pb = ptrtoint ptr addrspace(200) %b to i64
  %d = sub i64 %pa, %pb
  %n = sdiv i64 %d, 80
  ret i64 %n
}

; A capability MINUS AN INTEGER is a different operation and must keep using
; cincoffset: it produces a capability, and narrowing it through an address
; would drop the tag. This is here so the fix above cannot be "simplified" into
; covering both cases.
; CHECK-LABEL: cap_minus_int:
; CHECK: cincoffset
; CHECK: cjalr
define ptr addrspace(200) @cap_minus_int(ptr addrspace(200) %p, i64 %n) addrspace(200) {
  %neg = sub i64 0, %n
  %q = getelementptr i8, ptr addrspace(200) %p, i64 %neg
  ret ptr addrspace(200) %q
}
