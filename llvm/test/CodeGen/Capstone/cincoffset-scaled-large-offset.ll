; RUN: llc -mtriple=capstone64 -mattr=+m,+f,+d -O2 -capstone-gp-free -verify-machineinstrs < %s | FileCheck %s
;
; Large constants must stay in the integer displacement. Splitting the cursor
; at x + k * 8 could lose the tag even when the final address is in bounds.
; Share the scale instead, letting machine LICM hoist integer constants in
; loops. The negative offset exercises the opposite side of the window.

target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"

; CHECK-LABEL: large_offsets:
; CHECK-NOT: slli
; CHECK: slli [[S:[a-z0-9]+]], a1, 3
; CHECK-NOT: slli
; CHECK: cincoffset
; CHECK-NOT: slli
; CHECK: cincoffset
; CHECK-NOT: slli
; CHECK-NOT: cincoffset
; CHECK: ret
define double @large_offsets(ptr addrspace(200) %x, i64 %k) {
  %kp = add i64 %k, 2001
  %km = add i64 %k, -1998
  %pp = getelementptr double, ptr addrspace(200) %x, i64 %kp
  %pm = getelementptr double, ptr addrspace(200) %x, i64 %km
  %vp = load double, ptr addrspace(200) %pp, align 8
  %vm = load double, ptr addrspace(200) %pm, align 8
  %sum = fadd double %vp, %vm
  ret double %sum
}

; An unused path must not derive a capability from x.
; CHECK-LABEL: guarded_large_offset:
; CHECK-NOT: cincoffset
; CHECK: beqz a2,
; CHECK: slli
; CHECK: cincoffset [[P:[a-z0-9]+]], a0,
; CHECK-NEXT: fsd {{.*}}, 0([[P]])
; CHECK-NOT: cincoffset
; CHECK: ret
define void @guarded_large_offset(ptr addrspace(200) %x, i64 %k, i1 %access, double %v) {
entry:
  br i1 %access, label %store, label %exit
store:
  %kp = add i64 %k, 2001
  %p = getelementptr double, ptr addrspace(200) %x, i64 %kp
  store double %v, ptr addrspace(200) %p, align 8
  br label %exit
exit:
  ret void
}

; No rewrite through a truncation: extending (i32 k + C) before scaling is
; different from extending k before adding the byte displacement.
; CHECK-LABEL: narrow_index:
; CHECK: addiw {{[a-z0-9]+}}, a1, 2001
; CHECK: slli
; CHECK: cincoffset
; CHECK: ret
define double @narrow_index(ptr addrspace(200) %x, i32 %k) {
  %kp = add i32 %k, 2001
  %idx = sext i32 %kp to i64
  %p = getelementptr double, ptr addrspace(200) %x, i64 %idx
  %v = load double, ptr addrspace(200) %p, align 8
  ret double %v
}
