; llvm.capstone.cap.set.address: replace the address inside a capability without
; trapping on an operand that holds a plain integer. It expands before register
; allocation into a dispatch on the operand's type: `lcc t, cap, 1` (selector 1 is
; total: an untagged operand answers 7), a branch, then `scc` for a NONLIN
; capability or the integer bridge for anything else. The scc is only ever reached
; behind the branch, which is what makes it safe on an untagged value.
;
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=capstone64 -mattr=+m -O0 -verify-machineinstrs < %s | FileCheck %s --check-prefix=O0

declare ptr addrspace(200) @llvm.capstone.cap.set.address.p200(ptr addrspace(200), i64)

; CHECK-LABEL: set:
; CHECK:      lcc [[T:[a-z0-9]+]], a0, 1
; CHECK-NEXT: addi [[D:[a-z0-9]+]], [[T]], -1
; CHECK-NEXT: bnez [[D]], [[BRIDGE:.LBB[0-9_]+]]
; CHECK:      scc a0, a0, a1
; CHECK:      [[BRIDGE]]:
; CHECK-NOT:  scc
; CHECK:      cjalr zero, 0(ra)
; O0-LABEL: set:
; O0:       lcc {{[a-z0-9]+}}, {{[a-z0-9]+}}, 1
; O0:       bnez
; O0:       scc
define ptr addrspace(200) @set(ptr addrspace(200) %c, i64 %a) addrspace(200) {
  %r = call ptr addrspace(200) @llvm.capstone.cap.set.address.p200(ptr addrspace(200) %c, i64 %a)
  ret ptr addrspace(200) %r
}

; In a loop, the scc stays behind its branch: no `scc` appears before the first
; `bnez` of the dispatch.
; CHECK-LABEL: in_loop:
; CHECK-NOT:  scc
; CHECK:      lcc {{[a-z0-9]+}}, {{[a-z0-9]+}}, 1
; CHECK-NOT:  scc
; CHECK:      bnez
; CHECK:      scc
define void @in_loop(ptr addrspace(200) %c, ptr addrspace(200) %out, i64 %n) addrspace(200) {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %r = call ptr addrspace(200) @llvm.capstone.cap.set.address.p200(ptr addrspace(200) %c, i64 %i)
  %slot = getelementptr ptr addrspace(200), ptr addrspace(200) %out, i64 %i
  store ptr addrspace(200) %r, ptr addrspace(200) %slot, align 16
  %i.next = add i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %loop
exit:
  ret void
}
