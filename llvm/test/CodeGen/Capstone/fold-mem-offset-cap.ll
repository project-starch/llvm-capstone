; CapstoneFoldMemOffset, capability forms (lane C, item 2 of the 2026-11-02
; codegen plan, step 3).
;
; A constant cursor move that feeds a *dynamic* cursor add and then loads/stores
; cannot be folded by SelectAddrRegImm (the outer add is register, not simm12)
; nor by performCIncOffsetCombine (the dynamic offset is not a constant). libm
; exp hits this: `cincoffsetimm a0, base, 0x70 ; cincoffset a, a0, idx ; fld 0(a)`
; where plain RISC-V keeps one `cincoffset base, idx` and folds 0x70/0x78 into
; the loads. CapstoneFoldMemOffset now roots at CIncOffsetImm and walks the
; constant through CIncOffset into the memory immediates:
;   cincoffsetimm t, base, 112 ; cincoffset p, t, idx ; fld 0(p) ; fld 8(p)
;     ->  cincoffset p, base, idx ; fld 112(p) ; fld 120(p)
;
; Legal: the immediate is applied at the access, where bounds and permissions
; are checked; the checked effective address is identical. The only cursor the
; new code forms, base+idx, is <= 2048 bytes below an in-bounds access (the
; combined immediate fits simm12), so it is inside the R-60 representable
; window -- no new cursor farther from the object than before.
;
; MUTATION: drop Capstone::CIncOffsetImm from the root test in
; runOnMachineFunction, or Capstone::CIncOffset from the pass-through switch,
; and @poly keeps `cincoffsetimm {{.*}}, 112` with `fld 0`/`fld 8`.
; RUN: llc -mtriple=capstone64 -mattr=+m,+f,+d -O2 -capstone-gp-free -verify-machineinstrs < %s | FileCheck %s
; RUN: %llc_cap -mattr=+f,+d -O0 < %s -o /dev/null
; RUN: %llc_cap -mattr=+f,+d -O2 < %s -o /dev/null

target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"

@T = external addrspace(200) global [0 x double]

; CHECK-LABEL: poly:
; CHECK:      cincoffset [[P:a[0-9]+]], {{a[0-9]+}}, {{a[0-9]+}}
; CHECK-NOT:  cincoffsetimm {{a[0-9]+}}, {{a[0-9]+}}, 112
; CHECK-DAG:  fld {{f[a-z0-9]+}}, 112([[P]])
; CHECK-DAG:  fld {{f[a-z0-9]+}}, 120([[P]])
; CHECK:      ret
define double @poly(i64 %i) {
entry:
  %b = getelementptr inbounds double, ptr addrspace(200) @T, i64 14
  %p0 = getelementptr inbounds double, ptr addrspace(200) %b, i64 %i
  %i1 = add i64 %i, 1
  %p1 = getelementptr inbounds double, ptr addrspace(200) %b, i64 %i1
  %v0 = load double, ptr addrspace(200) %p0
  %v1 = load double, ptr addrspace(200) %p1
  %s = fadd double %v0, %v1
  ret double %s
}

; A store sinks the same way (STC/SD/FSD users).
; CHECK-LABEL: pstore:
; CHECK:      cincoffset [[P:a[0-9]+]], {{a[0-9]+}}, {{a[0-9]+}}
; CHECK-NOT:  cincoffsetimm {{a[0-9]+}}, {{a[0-9]+}}, 112
; CHECK-DAG:  fsd {{f[a-z0-9]+}}, 112([[P]])
; CHECK-DAG:  fsd {{f[a-z0-9]+}}, 120([[P]])
; CHECK:      ret
define void @pstore(i64 %i, double %a, double %b) {
entry:
  %base = getelementptr inbounds double, ptr addrspace(200) @T, i64 14
  %p0 = getelementptr inbounds double, ptr addrspace(200) %base, i64 %i
  %i1 = add i64 %i, 1
  %p1 = getelementptr inbounds double, ptr addrspace(200) %base, i64 %i1
  store double %a, ptr addrspace(200) %p0
  store double %b, ptr addrspace(200) %p1
  ret void
}

; CONTROL: the combined immediate of the second load would be 255*8 + 8 = 2048,
; which does not fit simm12, so the whole chain bails and the constant cursor
; move stays (foldOffset is all-or-nothing per root).
; CHECK-LABEL: poly_far:
; CHECK:      cincoffsetimm
; CHECK:      ret
define double @poly_far(i64 %i) {
entry:
  %b = getelementptr inbounds double, ptr addrspace(200) @T, i64 255
  %p0 = getelementptr inbounds double, ptr addrspace(200) %b, i64 %i
  %i1 = add i64 %i, 1
  %p1 = getelementptr inbounds double, ptr addrspace(200) %b, i64 %i1
  %v0 = load double, ptr addrspace(200) %p0
  %v1 = load double, ptr addrspace(200) %p1
  %s = fadd double %v0, %v1
  ret double %s
}
