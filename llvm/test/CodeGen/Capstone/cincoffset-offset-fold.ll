; performCIncOffsetCombine: a cursor add whose offset carries a small constant,
; and whose result only addresses loads and stores, is reassociated so the
; constant lands in the memory operand and the group shares one cursor:
;   x[3k], x[3k+1], x[3k+2]  ->  cincoffset once; fld 0 / 8 / 16 off it.
; Before, each element cost its own `addi`/`slli` chain and `cincoffset`
; (544.nab_r mme34, ~120M attempts on the SPEC test input).
;
; The shared cursor X+Y is formed by the rewrite, so it must be one the
; program itself forms AND accesses memory at in the same block (x[3k] here),
; with no call or inline asm in the block; otherwise nothing moves. A cursor
; the program never formed can be refused by the address policy and written
; untagged: for C < 0 it lies above the access (sv39 refuses 2^38 and up), for
; C > 0 below it (phys56 refuses a wrap below 0), and the in-bounds access
; faults. @top_of_user_half, @near_zero, @end_pointer, @call_between,
; @asm_between and @trap_between are those shapes; the earlier combine
; transformed each of them.
;
; MUTATION: delete the CIncOffset case in PerformDAGCombine -> @vec3 keeps
; three cincoffsets and no load at offset 8 or 16. MUTATION: drop the
; isAccessAddress test -> @end_pointer loads at -1 off the stored end pointer.
; MUTATION: drop mayLeaveBlockEarly -> @call_between loads at -8 before the
; call.
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

; A NEGATIVE constant (p[k-1]) folds the same way, into a negative immediate,
; because the shared cursor x+8k is p[k]'s, which the program loads from.
; MUTATION: requiring C > 0 in the combine leaves @prev with an `addi ..., -8`
; and two cincoffsets.
; CHECK-LABEL: prev:
; CHECK:      cincoffset [[B:a[0-9]+]], a0, {{a[0-9]+}}
; CHECK-NOT:  cincoffset
; CHECK-DAG:  fld {{f[a-z0-9]+}}, -8([[B]])
; CHECK-DAG:  fld {{f[a-z0-9]+}}, 0([[B]])
; CHECK:      ret
define double @prev(ptr addrspace(200) %x, i64 %k) {
  %km1 = add i64 %k, -1
  %p0 = getelementptr double, ptr addrspace(200) %x, i64 %k
  %p1 = getelementptr double, ptr addrspace(200) %x, i64 %km1
  %v0 = load double, ptr addrspace(200) %p0
  %v1 = load double, ptr addrspace(200) %p1
  %s = fsub double %v0, %v1
  ret double %s
}

; Both signs around one shared cursor: x[k] = x[k-1] + x[k+1].
; CHECK-LABEL: stencil:
; CHECK:      cincoffset [[B:a[0-9]+]], a0, {{a[0-9]+}}
; CHECK-NOT:  cincoffset
; CHECK-DAG:  fld {{f[a-z0-9]+}}, -8([[B]])
; CHECK-DAG:  fld {{f[a-z0-9]+}}, 8([[B]])
; CHECK-DAG:  fsd {{f[a-z0-9]+}}, 0([[B]])
; CHECK:      ret
define void @stencil(ptr addrspace(200) %x, i64 %k) {
  %km = add i64 %k, -1
  %kp = add i64 %k, 1
  %pm = getelementptr double, ptr addrspace(200) %x, i64 %km
  %p = getelementptr double, ptr addrspace(200) %x, i64 %k
  %pp = getelementptr double, ptr addrspace(200) %x, i64 %kp
  %vm = load double, ptr addrspace(200) %pm
  %vp = load double, ptr addrspace(200) %pp
  %s = fadd double %vm, %vp
  store double %s, ptr addrspace(200) %p
  ret void
}

; The simm12 edge: -256 * 8 = -2048 folds (next to the x[k] access) ...
; CHECK-LABEL: edgeneg:
; CHECK:      cincoffset [[B:a[0-9]+]], a0, {{a[0-9]+}}
; CHECK-NOT:  cincoffset
; CHECK-DAG:  ld {{a[0-9]+}}, -2048([[B]])
; CHECK-DAG:  ld {{a[0-9]+}}, 0([[B]])
; CHECK:      ret
define i64 @edgeneg(ptr addrspace(200) %x, i64 %k) {
  %k1 = add i64 %k, -256
  %p = getelementptr i64, ptr addrspace(200) %x, i64 %k1
  %p0 = getelementptr i64, ptr addrspace(200) %x, i64 %k
  %v = load i64, ptr addrspace(200) %p
  %w = load i64, ptr addrspace(200) %p0
  %s = add i64 %v, %w
  ret i64 %s
}

; ... CONTROL: -257 * 8 = -2056 stays in the integer offset. Sharing the
; scale must not introduce an intermediate cursor outside the window.
; CHECK-LABEL: farneg:
; CHECK-NOT: cincoffset
; CHECK:     slli
; CHECK-NOT: cincoffset
; CHECK:     cincoffset {{a[0-9]+}}, a0, {{a[0-9]+}}
; CHECK-NOT: cincoffset
; CHECK:     ld a0, 0({{a[0-9]+}})
define i64 @farneg(ptr addrspace(200) %x, i64 %k) {
  %k1 = add i64 %k, -257
  %p = getelementptr i64, ptr addrspace(200) %x, i64 %k1
  %v = load i64, ptr addrspace(200) %p
  ret i64 %v
}

; --- No-transform shapes: the shared cursor would be one the program never
; accesses. Each was reassociated by the earlier combine.

; The sv39 boundary (C < 0): with bounds [2^38-4096, 2^38), %b = 2^38-2048 and
; %off = 2048 the program reads 2^38-1 through a tagged cursor; the rewrite
; formed %b+%off = 2^38, refused, and `lbu -1` off it faulted. The constant
; stays in the integer offset and the cursor is the access's own address.
; CHECK-LABEL: top_of_user_half:
; CHECK:       addi [[O:a[0-9]+]], a1, -1
; CHECK-NEXT:  cincoffset [[P:a[0-9]+]], a0, [[O]]
; CHECK-NEXT:  lbu a0, 0([[P]])
; CHECK-NEXT:  ret
define i8 @top_of_user_half(ptr addrspace(200) %b, i64 %off) {
  %o = add i64 %off, -1
  %p = getelementptr i8, ptr addrspace(200) %b, i64 %o
  %v = load i8, ptr addrspace(200) %p
  ret i8 %v
}

; The phys56 boundary (C > 0): with bounds [0, 16), %b = 2 and %off = -2046
; the program reads address 3; the rewrite formed %b+%off = 2^64-2044, a wrap
; below 0 that phys56 refuses, and `lbu 2047` off it faulted.
; CHECK-LABEL: near_zero:
; CHECK:       addi [[O:a[0-9]+]], a1, 2047
; CHECK-NEXT:  cincoffset [[P:a[0-9]+]], a0, [[O]]
; CHECK-NEXT:  lbu a0, 0([[P]])
; CHECK-NEXT:  ret
define i8 @near_zero(ptr addrspace(200) %b, i64 %off) {
  %o = add i64 %off, 2047
  %p = getelementptr i8, ptr addrspace(200) %b, i64 %o
  %v = load i8, ptr addrspace(200) %p
  ret i8 %v
}

; The program forms %b+%n (an end pointer, stored) but never accesses it: at
; the top of the user half it is exactly the refused cursor, so the load of
; %b[%n-1] does not go through it.
; CHECK-LABEL: end_pointer:
; CHECK-DAG:   cincoffset [[E:a[0-9]+]], a0, a1
; CHECK-DAG:   stc [[E]], 0(a2)
; CHECK-DAG:   addi [[O:a[0-9]+]], a1, -1
; CHECK:       cincoffset [[P:a[0-9]+]], a0, [[O]]
; CHECK:       lbu a0, 0([[P]])
; CHECK:       ret
define i8 @end_pointer(ptr addrspace(200) %b, i64 %n, ptr addrspace(200) %out) {
  %end = getelementptr i8, ptr addrspace(200) %b, i64 %n
  store ptr addrspace(200) %end, ptr addrspace(200) %out
  %n1 = add i64 %n, -1
  %p = getelementptr i8, ptr addrspace(200) %b, i64 %n1
  %v = load i8, ptr addrspace(200) %p
  ret i8 %v
}

; x[k] is accessed, but only after a call that may not return (longjmp,
; exit): x[k-1] is read before it and must not go through x+8k, which the
; program may never reach.
declare void @may_not_return()

; CHECK-LABEL: call_between:
; CHECK-NOT:   fld {{.*}}, -8(
; CHECK:       fld {{f[a-z0-9]+}}, 0(
; CHECK-NOT:   fld {{.*}}, -8(
; CHECK:       jalr
; CHECK-NOT:   fld {{.*}}, -8(
; CHECK:       fld {{f[a-z0-9]+}}, 0(
; CHECK:       ret
define double @call_between(ptr addrspace(200) %x, i64 %k) {
  %km1 = add i64 %k, -1
  %p0 = getelementptr double, ptr addrspace(200) %x, i64 %k
  %p1 = getelementptr double, ptr addrspace(200) %x, i64 %km1
  %v1 = load double, ptr addrspace(200) %p1
  call void @may_not_return()
  %v0 = load double, ptr addrspace(200) %p0
  %s = fsub double %v0, %v1
  ret double %s
}

; The same with inline asm between the two reads.
; CHECK-LABEL: asm_between:
; CHECK-NOT:   fld {{.*}}, -8(
; CHECK:       fld {{f[a-z0-9]+}}, 0(
; CHECK-NOT:   fld {{.*}}, -8(
; CHECK:       #APP
; CHECK-NOT:   fld {{.*}}, -8(
; CHECK:       fld {{f[a-z0-9]+}}, 0(
; CHECK:       ret
define double @asm_between(ptr addrspace(200) %x, i64 %k) {
  %km1 = add i64 %k, -1
  %p0 = getelementptr double, ptr addrspace(200) %x, i64 %k
  %p1 = getelementptr double, ptr addrspace(200) %x, i64 %km1
  %v1 = load double, ptr addrspace(200) %p1
  call void asm sideeffect "", ""()
  %v0 = load double, ptr addrspace(200) %p0
  %s = fsub double %v0, %v1
  ret double %s
}

; And with a trap between them: llvm.trap need not end its IR block, and
; after it the x[k] access never runs.
declare void @llvm.trap()

; CHECK-LABEL: trap_between:
; CHECK-NOT:   lbu {{.*}}, -1(
; CHECK:       lbu {{a[0-9]+}}, 0(
; CHECK-NOT:   lbu {{.*}}, -1(
; CHECK:       unimp
define i8 @trap_between(ptr addrspace(200) %b, i64 %n) {
  %n1 = add i64 %n, -1
  %p1 = getelementptr i8, ptr addrspace(200) %b, i64 %n1
  %p0 = getelementptr i8, ptr addrspace(200) %b, i64 %n
  %v1 = load volatile i8, ptr addrspace(200) %p1
  call void @llvm.trap()
  %v0 = load volatile i8, ptr addrspace(200) %p0
  %s = add i8 %v0, %v1
  ret i8 %s
}
