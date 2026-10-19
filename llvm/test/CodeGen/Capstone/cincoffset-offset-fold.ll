; performCIncOffsetCombine: a cursor add whose offset carries a small constant,
; and whose result only addresses loads and stores, is reassociated so the
; constant lands in the memory operand and the group shares one cursor:
;   x[3k], x[3k+1], x[3k+2]  ->  cincoffset once; fld 0 / 8 / 16 off it.
; Before, each element cost its own `addi`/`slli` chain and `cincoffset`
; (544.nab_r mme34, ~120M attempts on the SPEC test input).
;
; MUTATION: delete the CIncOffset case in PerformDAGCombine -> @vec3 keeps
; three cincoffsets and no load at offset 8 or 16.
;
; The gp-free ABI is the application profile; its returns print as `ret`.
; RUN: llc -mtriple=capstone64 -mattr=+m,+f,+d -O2 -capstone-gp-free -verify-machineinstrs < %s | FileCheck %s
; RUN: %llc_cap -mattr=+f,+d -O0 < %s -o /dev/null
; RUN: %llc_cap -mattr=+f,+d -O1 < %s -o /dev/null
; RUN: %llc_cap -mattr=+f,+d -O2 < %s -o /dev/null

target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"

; CHECK-LABEL: vec3:
; CHECK:      cincoffset [[B:a[0-9]+]], a0, {{a[0-9]+}}
; CHECK-NOT:  cincoffset
; CHECK-DAG:  fld {{f[a-z0-9]+}}, 0([[B]])
; CHECK-DAG:  fld {{f[a-z0-9]+}}, 8([[B]])
; CHECK-DAG:  fld {{f[a-z0-9]+}}, 16([[B]])
; CHECK:      ret
define double @vec3(ptr addrspace(200) %x, i64 %k) {
  %k3 = mul i64 %k, 3
  %p0 = getelementptr double, ptr addrspace(200) %x, i64 %k3
  %k31 = add i64 %k3, 1
  %p1 = getelementptr double, ptr addrspace(200) %x, i64 %k31
  %k32 = add i64 %k3, 2
  %p2 = getelementptr double, ptr addrspace(200) %x, i64 %k32
  %v0 = load double, ptr addrspace(200) %p0
  %v1 = load double, ptr addrspace(200) %p1
  %v2 = load double, ptr addrspace(200) %p2
  %s = fadd double %v0, %v1
  %t = fadd double %s, %v2
  ret double %t
}

; Stores fold the same way: the scaled index is the shared offset, the element
; constant the immediate.
; CHECK-LABEL: store3:
; CHECK:      cincoffset [[B:a[0-9]+]], a0, {{a[0-9]+}}
; CHECK-NOT:  cincoffset
; CHECK-DAG:  fsd {{f[a-z0-9]+}}, 0([[B]])
; CHECK-DAG:  fsd {{f[a-z0-9]+}}, 8([[B]])
; CHECK-DAG:  fsd {{f[a-z0-9]+}}, 16([[B]])
; CHECK:      ret
define void @store3(ptr addrspace(200) %x, i64 %k, double %a, double %b, double %c) {
  %k3 = mul i64 %k, 3
  %p0 = getelementptr double, ptr addrspace(200) %x, i64 %k3
  %k31 = add i64 %k3, 1
  %p1 = getelementptr double, ptr addrspace(200) %x, i64 %k31
  %k32 = add i64 %k3, 2
  %p2 = getelementptr double, ptr addrspace(200) %x, i64 %k32
  store double %a, ptr addrspace(200) %p0
  store double %b, ptr addrspace(200) %p1
  store double %c, ptr addrspace(200) %p2
  ret void
}

; CONTROL: the result is not an address (it is returned), so nothing moves:
; the constant is added before the cursor, as the generic combiner leaves it.
; CHECK-LABEL: pointer_out:
; CHECK:     addi {{a[0-9]+}}, {{a[0-9]+}}, 8
; CHECK:     cincoffset a0, a0, {{a[0-9]+}}
; CHECK-NOT: cincoffsetimm
; CHECK:     ret
define ptr addrspace(200) @pointer_out(ptr addrspace(200) %x, i64 %k) {
  %k1 = add i64 %k, 1
  %p = getelementptr double, ptr addrspace(200) %x, i64 %k1
  ret ptr addrspace(200) %p
}

; CONTROL: a constant beyond the simm12 range stays in the offset chain.
; CHECK-LABEL: far:
; CHECK:     lui
; CHECK:     cincoffset {{a[0-9]+}}, a0, {{a[0-9]+}}
; CHECK:     ld a0, 0({{a[0-9]+}})
define i64 @far(ptr addrspace(200) %x, i64 %k) {
  %k1 = add i64 %k, 4096
  %p = getelementptr i64, ptr addrspace(200) %x, i64 %k1
  %v = load i64, ptr addrspace(200) %p
  ret i64 %v
}
