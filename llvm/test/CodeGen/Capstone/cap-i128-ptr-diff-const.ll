; RUN: llc -mtriple=capstone64 -mattr=+m -verify-machineinstrs < %s | FileCheck %s
;
; Regression: a pointer DIFFERENCE with a constant element offset on one side,
; e.g. `p - (q + 1)`, was mis-lowered. DAGCombine correctly reassociates it to
; `add(sub(p, q), -elemsize)`, where `sub(p, q)` is a pointer difference (a
; SCALAR byte count). But lowerADD treated every i128 add as capability+offset and
; emitted `CIncOffset(<scalar>, -elemsize)` -- a cincoffsetimm on an untagged
; scalar -- which faults at runtime (cscincoffsetimm rs1->tag).
;
; This is exactly Lua 5.4's lua_gettop: `L->top.p - (L->ci->func.p + 1)`, where
; both pointers are LOADED from the lua_State. Every C-API / base-library call runs
; through it, so the interpreter trapped on the first `print()` (and any other base
; function) even though pure-core chunks ran fine.
;
; Fix: when neither operand of the i128 add is a capability base -- the offset is
; an integer offset and the base is a ptr-ptr SUB of two capability values (or an
; already-lowered sign-extended difference) -- lower the add in the XLen domain
; (scalar `addi`) and sign-extend back. The element adjustment must be a scalar
; add, never a cincoffset on the difference.

target triple = "capstone64-unknown-elf"

%struct.SV = type { [4 x i64] }   ; 32-byte element, like Lua's StackValue

; p - (q + 1): loaded capabilities, difference is a scalar; the -32 element
; adjustment must be a scalar `addi`, never a cincoffset on the difference.
; CHECK-LABEL: ptr_diff_q1:
; CHECK:      sub [[D:a[0-9]+]], {{a[0-9]+}}, {{a[0-9]+}}
; CHECK-NEXT: addi [[D]], [[D]], -32
; CHECK-NOT:  cincoffset
define i64 @ptr_diff_q1(ptr addrspace(200) %slots) addrspace(200) {
entry:
  %qp = getelementptr ptr addrspace(200), ptr addrspace(200) %slots, i128 1
  %p = load ptr addrspace(200), ptr addrspace(200) %slots
  %q = load ptr addrspace(200), ptr addrspace(200) %qp
  %add.ptr = getelementptr %struct.SV, ptr addrspace(200) %q, i128 1
  %lhs = ptrtoint ptr addrspace(200) %p to i128
  %rhs = ptrtoint ptr addrspace(200) %add.ptr to i128
  %sub = sub i128 %lhs, %rhs
  %div = sdiv exact i128 %sub, 32
  %r = trunc i128 %div to i64
  ret i64 %r
}

; (p + 1) - q: same, with a +32 adjustment.
; CHECK-LABEL: ptr_diff_p1:
; CHECK:      sub [[E:a[0-9]+]], {{a[0-9]+}}, {{a[0-9]+}}
; CHECK-NEXT: addi [[E]], [[E]], 32
; CHECK-NOT:  cincoffset
define i64 @ptr_diff_p1(ptr addrspace(200) %slots) addrspace(200) {
entry:
  %qp = getelementptr ptr addrspace(200), ptr addrspace(200) %slots, i128 1
  %p = load ptr addrspace(200), ptr addrspace(200) %slots
  %q = load ptr addrspace(200), ptr addrspace(200) %qp
  %add.ptr = getelementptr %struct.SV, ptr addrspace(200) %p, i128 1
  %lhs = ptrtoint ptr addrspace(200) %add.ptr to i128
  %rhs = ptrtoint ptr addrspace(200) %q to i128
  %sub = sub i128 %lhs, %rhs
  %div = sdiv exact i128 %sub, 32
  %r = trunc i128 %div to i64
  ret i64 %r
}

; Plain p - q stays a clean cursor difference + arithmetic shift (control).
; CHECK-LABEL: ptr_diff_plain:
; CHECK:      sub [[F:a[0-9]+]], {{a[0-9]+}}, {{a[0-9]+}}
; CHECK-NOT:  cincoffset
; CHECK:      srai
define i64 @ptr_diff_plain(ptr addrspace(200) %slots) addrspace(200) {
entry:
  %qp = getelementptr ptr addrspace(200), ptr addrspace(200) %slots, i128 1
  %p = load ptr addrspace(200), ptr addrspace(200) %slots
  %q = load ptr addrspace(200), ptr addrspace(200) %qp
  %lhs = ptrtoint ptr addrspace(200) %p to i128
  %rhs = ptrtoint ptr addrspace(200) %q to i128
  %sub = sub i128 %lhs, %rhs
  %div = sdiv exact i128 %sub, 32
  %r = trunc i128 %div to i64
  ret i64 %r
}
