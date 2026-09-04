; CapstoneRegisterInfo's scalable-offset route is reachable: a scalable-vector
; stack object under -mattr=+v needs a VLEN-scaled sp adjustment on a
; capability frame, which Capstone PureCap does not implement, and the fatal
; error must say so rather than emit a wrong frame.  (Without +v the same input
; trips an upstream frame-lowering assertion, and a scalable load/store cannot
; be selected against a c128 pointer at all: RVV is not functional on this
; target and not used by the project -- recorded in the plan's execution log,
; not pinned here.)  Measured 2026-09-04 on the branch tools.
;
; RUN: not llc -mtriple=capstone64 -mattr=+v -o /dev/null %s 2>&1 | FileCheck %s
; CHECK: LLVM ERROR: Scalable stack adjustments are not supported yet in Capstone PureCap

target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"

define ptr addrspace(200) @addr_only() {
  %v = alloca <vscale x 2 x i64>, addrspace(200)
  ret ptr addrspace(200) %v
}
