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
; That question no longer exists. Since a capability became c128, an i128 add is
; integer arithmetic by TYPE, and the element adjustment stays in the pointer
; domain where the source put it: `q + 1` is a GEP, so it is a cincoffsetimm on
; the tagged capability BEFORE the difference is taken. What must never happen is
; a cincoffset on the DIFFERENCE, which is untagged -- so each check below pins
; "no cincoffset between the sub and the return" rather than "no cincoffset".

target triple = "capstone64-unknown-elf"

%struct.SV = type { [4 x i64] }   ; 32-byte element, like Lua's StackValue

; p - (q + 1): loaded capabilities, difference is a scalar. The +1 element is a
; cincoffsetimm on the tagged capability; nothing may cincoffset the difference.
; CHECK-LABEL: ptr_diff_q1:
; CHECK:      cincoffsetimm [[Q:a[0-9]+]], [[Q]], 32
; CHECK:      sub [[D:a[0-9]+]], {{a[0-9]+}}, {{a[0-9]+}}
; CHECK-NEXT: srai [[D]], [[D]], 5
; CHECK-NOT:  cincoffset
; CHECK:      cjalr zero, 0(ra)
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

; (p + 1) - q: same, with the adjustment on the other side.
; CHECK-LABEL: ptr_diff_p1:
; CHECK:      cincoffsetimm [[P:a[0-9]+]], [[P]], 32
; CHECK:      sub [[E:a[0-9]+]], {{a[0-9]+}}, {{a[0-9]+}}
; CHECK-NEXT: srai [[E]], [[E]], 5
; CHECK-NOT:  cincoffset
; CHECK:      cjalr zero, 0(ra)
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
; CHECK-NOT:  cincoffset
; CHECK:      sub [[F:a[0-9]+]], {{a[0-9]+}}, {{a[0-9]+}}
; CHECK-NEXT: srai [[F]], [[F]], 5
; CHECK-NOT:  cincoffset
; CHECK:      cjalr zero, 0(ra)
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
