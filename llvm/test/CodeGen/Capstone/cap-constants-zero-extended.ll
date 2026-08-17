; RUN: llc -mtriple=capstone64 -verify-machineinstrs -o - < %s | FileCheck %s

; The same 64-bit values as cap-constants.ll, written the way the front end widens a C cast of a
; negative or unsigned integer to a capability-width pointer: ZERO-extended into the i128 carrier
; rather than sign-extended. 0xFFFFFFFFFFFFFFFF and `i128 -1` name one register value, and
; `inttoptr i64 -1` already compiled to `li a0, -1`, so only the widened spelling was rejected.
;
; From MicroPython: MP_OBJ_NEW_SMALL_INT(-1) and the sentinel objects in vm.c and objlist.c.
;
; The boundary in the other direction, 2^64, must STILL be refused and is pinned in
; cap-constants-invalid.ll.

; CHECK-LABEL: small_int_minus_one:
; CHECK: li a0, -1
define ptr addrspace(200) @small_int_minus_one() {
entry:
  ret ptr addrspace(200) inttoptr (i128 18446744073709551615 to ptr addrspace(200))
}

; CHECK-LABEL: sentinel_minus_three:
; CHECK: li a0, -3
define ptr addrspace(200) @sentinel_minus_three() {
entry:
  ret ptr addrspace(200) inttoptr (i128 18446744073709551613 to ptr addrspace(200))
}

; CHECK-LABEL: gep_zext_neg:
; CHECK: cincoffsetimm a0, a0, -3
define ptr addrspace(200) @gep_zext_neg(ptr addrspace(200) %p) {
entry:
  %gep = getelementptr i8, ptr addrspace(200) %p, i128 18446744073709551613
  ret ptr addrspace(200) %gep
}

; CHECK-LABEL: load_zext_neg:
; CHECK: lw a0, -4(a0)
define i32 @load_zext_neg(ptr addrspace(200) %p) {
entry:
  %gep = getelementptr i8, ptr addrspace(200) %p, i128 18446744073709551612
  %v = load i32, ptr addrspace(200) %gep, align 4
  ret i32 %v
}
