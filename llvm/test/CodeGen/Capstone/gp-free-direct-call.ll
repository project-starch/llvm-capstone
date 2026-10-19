; Under the gp-free domain ABI a direct call is the plain PC-relative `call`
; (PseudoCALL: auipc ra + jalr ra under one R_Capstone_CALL_PLT relocation), a
; sibling call `tail`, and the arguments stay where the calling convention put
; them. Before 2026-10-19 the callee address was materialised with PseudoLLA
; (auipc; addi) and called through that register, which cost one instruction
; per call and, at most call sites, a shuffle of the temporary around the
; argument registers (`mv a2, a0; movc a0, s0; jalr a2`).
;
; MUTATION: route the gp-free direct call back through PseudoLLA + jalr (the
; register form) -> `call` is absent and `jalr` present in @calls_direct, and
; the object carries PCREL_HI20/LO12 instead of CALL_PLT.
;
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 -capstone-gp-free -verify-machineinstrs < %s \
; RUN:   | FileCheck %s
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 -capstone-gp-free -filetype=obj < %s -o %t.o
; RUN: llvm-readelf -r %t.o | FileCheck %s --check-prefix=RELOC
; RUN: %llc_cap -O0 -capstone-gp-free < %s -o /dev/null
; RUN: %llc_cap -O1 -capstone-gp-free < %s -o /dev/null

target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"

declare void @ext_fn()
declare i64 @ext_add(i64, i64)

; CHECK-LABEL: calls_direct:
; CHECK-NOT: auipc
; CHECK-NOT: jalr
; CHECK:     call ext_fn
; CHECK-NOT: jalr
; CHECK:     ret
define void @calls_direct() {
  call void @ext_fn()
  ret void
}

; The arguments are already in a0 and a1: nothing moves before the call.
; CHECK-LABEL: args_stay:
; CHECK-NOT: mv
; CHECK-NOT: movc
; CHECK:     call ext_add
define i64 @args_stay(i64 %x, i64 %y) {
  %r = call i64 @ext_add(i64 %x, i64 %y)
  ret i64 %r
}

; A sibling call is `tail`: no ra, nothing after it.
; CHECK-LABEL: tail_direct:
; CHECK-NOT: jalr
; CHECK:     tail ext_add
; CHECK-NOT: ret
define i64 @tail_direct(i64 %x, i64 %y) {
  %r = tail call i64 @ext_add(i64 %x, i64 %y)
  ret i64 %r
}

; An indirect call is still the integer jalr through the capability's address
; (the function is named without "call" in it so that CHECK-NOT can be exact).
; CHECK-LABEL: via_pointer:
; CHECK-NOT: call
; CHECK:     jalr {{[a-z][a-z0-9]+}}
; CHECK:     ret
define void @via_pointer(ptr addrspace(200) %fp) {
  call void %fp()
  ret void
}

; RELOC-NOT: R_Capstone_PCREL_HI20 ext_fn
; RELOC:     R_Capstone_CALL_PLT {{.*}} ext_fn
; RELOC:     R_Capstone_CALL_PLT {{.*}} ext_add
; RELOC-NOT: R_Capstone_PCREL_HI20 ext_add
