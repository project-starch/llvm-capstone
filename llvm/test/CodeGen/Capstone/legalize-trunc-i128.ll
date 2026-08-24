; RUN: llc -mtriple=capstone64 -verify-machineinstrs < %s | FileCheck %s
;
; Regression test: DAG legalization can introduce `trunc i128 -> xlen` when
; comparing capability pointers (ptr addrspace(200)). The backend must be able
; to select the truncate without crashing.

; Reading the address costs NO instruction: X is the low half of C, so the
; truncate is EXTRACT_SUBREG on sub_cap_addr.
; CHECK-LABEL: ptr_to_long:
; CHECK-NOT: mv
; CHECK: cjalr zero, 0(ra)
define i64 @ptr_to_long(ptr addrspace(200) %p) {
entry:
  %0 = ptrtoint ptr addrspace(200) %p to i64
  ret i64 %0
}

; CHECK-LABEL: trunc_from_cap_cmp_eq:
; CHECK: xor a0, a0, a1
; CHECK: seqz a0, a0
define i64 @trunc_from_cap_cmp_eq(ptr addrspace(200) %a, ptr addrspace(200) %b) {
entry:
  %cmp = icmp eq ptr addrspace(200) %a, %b
  ; Force the compare result into i128, then truncate back to xlen.
  ; This mirrors the shape seen after legalization in SelectionDAG.
  %z = zext i1 %cmp to i128
  %t = trunc i128 %z to i64
  ret i64 %t
}

; CHECK-LABEL: trunc_from_cap_cmp_null:
; CHECK: seqz a0, a0
define i64 @trunc_from_cap_cmp_null(ptr addrspace(200) %a) {
entry:
  %cmp = icmp eq ptr addrspace(200) %a, null
  %z = zext i1 %cmp to i128
  %t = trunc i128 %z to i64
  ret i64 %t
}

; CHECK-LABEL: trunc_after_phi:
; CHECK: beqz{{[ \t]+}}a2, .LBB{{[0-9]+}}_{{[0-9]+}}
define i64 @trunc_after_phi(ptr addrspace(200) %a, ptr addrspace(200) %b, i1 %c) {
entry:
  ; Ensure there is control flow so the i128 value can't be trivially folded
  ; away before isel.
  br i1 %c, label %t, label %f

t:
  %x1 = icmp eq ptr addrspace(200) %a, %b
  %z1 = zext i1 %x1 to i128
  br label %join

f:
  %x2 = icmp eq ptr addrspace(200) %a, null
  %z2 = zext i1 %x2 to i128
  br label %join

join:
  %phi = phi i128 [ %z1, %t ], [ %z2, %f ]
  %tr = trunc i128 %phi to i64
  ret i64 %tr
}

