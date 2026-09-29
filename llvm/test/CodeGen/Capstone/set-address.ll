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

; The address of null is how clang converts an integer into an __intcap. There
; is no type to dispatch on: it selects to the integer bridge, with no lcc and
; no branch.
; CHECK-LABEL: from_int:
; CHECK-NOT:  lcc
; CHECK-NOT:  scc
; CHECK:      li a0, 41
; CHECK-NEXT: cjalr zero, 0(ra)
define ptr addrspace(200) @from_int() addrspace(200) {
  %r = call ptr addrspace(200) @llvm.capstone.cap.set.address.p200(ptr addrspace(200) null, i64 41)
  ret ptr addrspace(200) %r
}

; `(__intcap)((long)ic + 1)`: the sum is an integer. Through set_address on
; null it stays one: no cincoffset on ic, which would trap on an untagged ic.
; CHECK-LABEL: plus_one:
; CHECK-NOT:  cincoffset
; CHECK-NOT:  lcc
; CHECK:      addi a0, a0, 1
; CHECK-NEXT: cjalr zero, 0(ra)
define ptr addrspace(200) @plus_one(ptr addrspace(200) %ic) addrspace(200) {
  %a = ptrtoint ptr addrspace(200) %ic to i64
  %s = add nsw i64 %a, 1
  %r = call ptr addrspace(200) @llvm.capstone.cap.set.address.p200(ptr addrspace(200) null, i64 %s)
  ret ptr addrspace(200) %r
}

; The control: the same sum through inttoptr is the uintptr_t round-trip shape,
; and the backend rebuilds it as an offset on ic (ptr-arith.ll). This is what
; clang emitted before it used set_address on null.
; CHECK-LABEL: plus_one_inttoptr:
; CHECK:      cincoffsetimm a0, a0, 1
define ptr addrspace(200) @plus_one_inttoptr(ptr addrspace(200) %ic) addrspace(200) {
  %a = ptrtoint ptr addrspace(200) %ic to i64
  %s = add nsw i64 %a, 1
  %r = inttoptr i64 %s to ptr addrspace(200)
  ret ptr addrspace(200) %r
}
