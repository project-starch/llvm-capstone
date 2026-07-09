; RUN: llc -mtriple=capstone64 < %s | FileCheck %s

; CC_Capstone_FastCC used to have no MVT::i128 case, so a fastcc function taking
; or returning a capability fell through to "CC didn't match" and hit
; llvm_unreachable in analyzeInputArgs/analyzeOutputArgs. GlobalOpt gives
; internal-linkage functions fastcc at -O1 and above, so this crashed on any
; translation unit with a non-inlined static function taking a pointer.

; A capability argument occupies one integer argument register.
define internal fastcc i64 @cap_arg(ptr addrspace(200) %p) {
; CHECK-LABEL: cap_arg:
; CHECK: ld a0, 0(a0)
  %v = load i64, ptr addrspace(200) %p
  ret i64 %v
}

define i64 @call_cap_arg(ptr addrspace(200) %p) {
; CHECK-LABEL: call_cap_arg:
; CHECK: cjalr ra
  %r = call fastcc i64 @cap_arg(ptr addrspace(200) %p)
  ret i64 %r
}

; A capability return value also lands in one integer register.
define internal fastcc ptr addrspace(200) @cap_ret(ptr addrspace(200) %p) {
; CHECK-LABEL: cap_ret:
; CHECK-NOT: movc
; CHECK: cjalr zero, 0(ra)
  ret ptr addrspace(200) %p
}

define ptr addrspace(200) @call_cap_ret(ptr addrspace(200) %p) {
; CHECK-LABEL: call_cap_ret:
; CHECK: cjalr ra
  %r = call fastcc ptr addrspace(200) @cap_ret(ptr addrspace(200) %p)
  ret ptr addrspace(200) %r
}

; FastCC passes arguments in a0-a7 and t3-t6 (12 GPRs); the standard CC stops at
; a7. Exhausting all 12 must spill the 13th capability to a 16-byte stack slot
; rather than falling off the end of the convention.
define internal fastcc i64 @many_caps(ptr addrspace(200) %a0, ptr addrspace(200) %a1,
    ptr addrspace(200) %a2, ptr addrspace(200) %a3, ptr addrspace(200) %a4,
    ptr addrspace(200) %a5, ptr addrspace(200) %a6, ptr addrspace(200) %a7,
    ptr addrspace(200) %a8, ptr addrspace(200) %a9, ptr addrspace(200) %a10,
    ptr addrspace(200) %a11, ptr addrspace(200) %a12) {
; The 13th capability arrives on the stack and is reloaded with a capability load.
; CHECK-LABEL: many_caps:
; CHECK: ldc a0, 0(sp)
; CHECK: ld a0, 0(a0)
  %v = load i64, ptr addrspace(200) %a12
  ret i64 %v
}

define i64 @call_many_caps(ptr addrspace(200) %p) {
; The caller stores the 13th capability to the outgoing slot and fills the
; FastCC-only registers t3-t6, which the standard CC never uses for arguments.
; CHECK-LABEL: call_many_caps:
; CHECK: stc a0, 0(sp)
; CHECK: movc t3, a0
; CHECK: movc t4, a0
; CHECK: movc t5, a0
; CHECK: movc t6, a0
; CHECK: cjalr ra
  %r = call fastcc i64 @many_caps(ptr addrspace(200) %p, ptr addrspace(200) %p,
    ptr addrspace(200) %p, ptr addrspace(200) %p, ptr addrspace(200) %p,
    ptr addrspace(200) %p, ptr addrspace(200) %p, ptr addrspace(200) %p,
    ptr addrspace(200) %p, ptr addrspace(200) %p, ptr addrspace(200) %p,
    ptr addrspace(200) %p, ptr addrspace(200) %p)
  ret i64 %r
}
