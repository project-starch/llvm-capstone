; RUN: llc -mtriple=capstone64 -mattr=+m -verify-machineinstrs < %s | FileCheck %s
;
; Regression (C-2 family): a select whose result is a *capability* (i128) and
; whose condition compares against a NON-ZERO constant used to ICE with
;   "Cannot select: i128 = CapstoneISD::SELECT_CC ..., Constant:i64<...>, seteq".
;
; On capstone64 TableGen drops the i128 Select_GPRCAP_Using_CC_GPR matcher (it is
; guarded !is64Bit), so an i128 select has no selection pattern. lowerSELECT
; builds the branch-based Select_GPRCAP_Using_CC_GPR pseudo directly in C++ to
; work around that. It previously bailed whenever the SETCC compared against a
; non-zero constant, falling through to a path that formed the unselectable i128
; CapstoneISD::SELECT_CC. The fix materializes a constant compare operand into a
; GPR so the branch-based capability select is always used.
;
; The select MUST pick ONE of the two capabilities via control flow (a branch +
; movc), preserving the tag of whichever arm is chosen. It must never merge, OR,
; or otherwise combine the two capability bit-patterns -- an `or` here would be a
; tag-stripping miscompile.

; The exact lstrlib.c:1190 shape: (n == INT64_MIN) ? a : b, selecting between two
; capability pointers. INT64_MIN is a non-zero constant that must be materialized
; into a GPR for the branch compare.
; CHECK-LABEL: select_cap_eq_min:
; CHECK:      slli [[K:a[0-9]+]], {{a[0-9]+}}, 63
; CHECK:      beq a0, [[K]], [[L0:.LBB[0-9_]+]]
; CHECK-NOT:  or {{[a-z0-9]+}}, {{[a-z0-9]+}}, {{[a-z0-9]+}}
; CHECK:      movc a1, a2
; CHECK:    [[L0]]:
; CHECK-NOT:  or {{[a-z0-9]+}}, {{[a-z0-9]+}}, {{[a-z0-9]+}}
; CHECK:      movc a0, a1
; CHECK:      cjalr zero, 0(ra)
define ptr addrspace(200) @select_cap_eq_min(i64 %n,
                                             ptr addrspace(200) %a,
                                             ptr addrspace(200) %b) {
entry:
  %c = icmp eq i64 %n, -9223372036854775808
  %r = select i1 %c, ptr addrspace(200) %a, ptr addrspace(200) %b
  ret ptr addrspace(200) %r
}

; A smaller non-zero constant on the compare RHS (the -O1 two-line reproducer
; from the handoff notes: n == 10 ? a : b). The `li 10` is the materialized
; compare operand that the old code could not produce.
; CHECK-LABEL: select_cap_eq_ten:
; CHECK:      li [[K1:a[0-9]+]], 10
; CHECK:      beq a0, [[K1]], [[L1:.LBB[0-9_]+]]
; CHECK-NOT:  or {{[a-z0-9]+}}, {{[a-z0-9]+}}, {{[a-z0-9]+}}
; CHECK:      movc a1, a2
; CHECK:    [[L1]]:
; CHECK-NOT:  or {{[a-z0-9]+}}, {{[a-z0-9]+}}, {{[a-z0-9]+}}
; CHECK:      movc a0, a1
; CHECK:      cjalr zero, 0(ra)
define ptr addrspace(200) @select_cap_eq_ten(i64 %n,
                                             ptr addrspace(200) %a,
                                             ptr addrspace(200) %b) {
entry:
  %c = icmp eq i64 %n, 10
  %r = select i1 %c, ptr addrspace(200) %a, ptr addrspace(200) %b
  ret ptr addrspace(200) %r
}

; Signed less-than against 1: translateSetCCForBranch rewrites `X < 1` to
; `0 >= X`, moving a constant (0) onto the LHS of the compare. That constant must
; be materialized to X0 (as `blez`, i.e. a compare against zero) rather than
; bailing -- and still selects a capability with a branch + movc, no merge.
; CHECK-LABEL: select_cap_slt_one:
; CHECK:      blez a0, [[L2:.LBB[0-9_]+]]
; CHECK-NOT:  or {{[a-z0-9]+}}, {{[a-z0-9]+}}, {{[a-z0-9]+}}
; CHECK:      movc a1, a2
; CHECK:    [[L2]]:
; CHECK-NOT:  or {{[a-z0-9]+}}, {{[a-z0-9]+}}, {{[a-z0-9]+}}
; CHECK:      movc a0, a1
; CHECK:      cjalr zero, 0(ra)
define ptr addrspace(200) @select_cap_slt_one(i64 %n,
                                             ptr addrspace(200) %a,
                                             ptr addrspace(200) %b) {
entry:
  %c = icmp slt i64 %n, 1
  %r = select i1 %c, ptr addrspace(200) %a, ptr addrspace(200) %b
  ret ptr addrspace(200) %r
}
