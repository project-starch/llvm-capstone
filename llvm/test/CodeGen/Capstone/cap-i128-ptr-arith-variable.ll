; RUN: llc -mtriple=capstone64 -mattr=+m -verify-machineinstrs < %s | FileCheck %s
;
; Regression: pointer arithmetic with a VARIABLE offset on a capability, e.g.
; `p - (n + 1)`, was mis-lowered. lowerSUB treated the (scaled, sign-extended)
; integer offset as a capability and emitted a ptrtoint/scalar/inttoptr
; round-trip:
;
;   lcc  cursor(p)            ; ptrtoint the base capability
;   lcc  cursor(offset)       ; LCC ON A SCALAR -- treats the offset as a pointer
;   sub                       ; scalar address arithmetic
;   cincoffsetimm <scalar>    ; cincoffset a SCALAR -- no capability base survives
;
; The result is an UNTAGGED pointer. Surfaced by real Lua: `lua_pcall`'s
; `c.func = L->top.p - (nargs+1)` produced an untagged StkId and the interpreter
; trapped on the first cslcc/cscincoffset that touched it.
;
; Two fixes: isCapstoneIntegerOffset now treats any `shl i128` as an integer
; offset (a capability is never the operand of a shift), so lowerSUB does not take
; the ptr-ptr cursor-difference path; and lowerSUB narrows any remaining
; integer-offset shape via TRUNCATE. Result: a single register CIncOffset on the
; base capability, preserving its bounds and tag.
;
; The constant-offset case (`p - 1`) always worked (plain cincoffsetimm); this
; test guards the variable-offset case.

target triple = "capstone64-unknown-elf"

%struct.SV = type { [4 x i64] }   ; 32-byte element, like Lua's StackValue

; CHECK-LABEL: ptr_sub_var:
; The offset is scaled and negated in the integer (XLen) domain, then applied to
; the base capability with a REGISTER cincoffset -- never lcc'd as if it were a
; pointer, and the base must remain the cincoffset operand (not a scalar).
; CHECK:      slli [[OFF:a[0-9]+]], a1, 5
; CHECK:      neg [[OFF]], [[OFF]]
; CHECK:      cincoffset a0, a0, [[OFF]]
; CHECK-NOT:  lcc
define ptr addrspace(200) @ptr_sub_var(ptr addrspace(200) %p, i32 %n) addrspace(200) {
entry:
  %add = add nsw i32 %n, 1
  %e = sext i32 %add to i128
  %neg = sub i128 0, %e
  %r = getelementptr inbounds %struct.SV, ptr addrspace(200) %p, i128 %neg
  ret ptr addrspace(200) %r
}

; Sanity: the constant-offset case stays a clean cincoffsetimm on the capability.
; CHECK-LABEL: ptr_sub_const:
; CHECK:      cincoffsetimm a0, a0, -32
; CHECK-NOT:  lcc
define ptr addrspace(200) @ptr_sub_const(ptr addrspace(200) %p) addrspace(200) {
entry:
  %r = getelementptr inbounds %struct.SV, ptr addrspace(200) %p, i128 -1
  ret ptr addrspace(200) %r
}
