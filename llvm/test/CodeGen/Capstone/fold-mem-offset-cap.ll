; CapstoneFoldMemOffset, capability forms, seen from IR (lane C, item 2 of the
; 2026-11-02 codegen plan; four reviews).
;
; The shapes: libm exp's `cincoffsetimm base, 0x70; cincoffset p, that, idx;
; fld 0(p)` (@poly, @pstore) and 538.imagick_r CloneImageInfo's struct cursor
; `lui 4 + addi 0x220` with fields read off it (@chain). The pass could sink
; the constant into the access immediates and did so for a while, but the
; sound rule it ended with is that no surviving cursor may change value, and
; on this target every memory base is a capability derived from the root
; through at least one cursor op, so the capability forms never fold (see the
; rule in the pass; fold-mem-offset-cap-chain.mir pins the shapes in MIR).
; This file pins the UNFOLDED codegen of those shapes: the constant cursor
; move stays and the accesses keep their own displacements.
;
; MUTATION: replace the rule's `!= 0` by `< 0` in the pass -> @poly reads
; `fld ..., 112(` / `120(`, @pstore `fsd ..., 112(` / `120(`, @chain
; `lw ..., 604(` / `608(`, and no `cincoffsetimm ..., 112` / `addi ..., 544`.
; RUN: llc -mtriple=capstone64 -mattr=+m,+f,+d -O2 -capstone-gp-free -verify-machineinstrs < %s | FileCheck %s
; RUN: %llc_cap -mattr=+f,+d -O0 < %s -o /dev/null
; RUN: %llc_cap -mattr=+f,+d -O2 < %s -o /dev/null

target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"

@T = external addrspace(200) global [0 x double]

; CHECK-LABEL: poly:
; CHECK:      cincoffsetimm [[B:a[0-9]+]], {{a[0-9]+}}, 112
; CHECK-NEXT: cincoffset [[P:a[0-9]+]], [[B]], {{a[0-9]+}}
; CHECK-NEXT: fld {{f[a-z0-9]+}}, 0([[P]])
; CHECK-NEXT: fld {{f[a-z0-9]+}}, 8([[P]])
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

; CHECK-LABEL: pstore:
; CHECK:      cincoffsetimm [[B:a[0-9]+]], {{a[0-9]+}}, 112
; CHECK-NEXT: cincoffset [[P:a[0-9]+]], [[B]], {{a[0-9]+}}
; CHECK-NEXT: fsd {{f[a-z0-9]+}}, 0([[P]])
; CHECK-NEXT: fsd {{f[a-z0-9]+}}, 8([[P]])
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

; The struct cursor base + 0x4220 (lui 4 + addi 0x220) with one field read at
; +0x3c and another through the +0x3c cursor at +4. isel collapses the
; intermediate cursor (rewrites 1b/2 of CapstoneISelLowering / SelectAddrRegImm),
; so both loads sit directly on the cursor at 60 and 64; the pass leaves the
; addi 544 in place.
; CHECK-LABEL: chain:
; CHECK:      lui [[L:a[0-9]+]], 4
; CHECK-NEXT: addi [[L]], [[L]], 544
; CHECK-NEXT: cincoffset [[S:a[0-9]+]], a0, [[L]]
; CHECK-NEXT: lw {{a[0-9]+}}, 60([[S]])
; CHECK-NEXT: lw {{a[0-9]+}}, 64([[S]])
; CHECK:      ret
define i32 @chain(ptr addrspace(200) %src) {
entry:
  %s = getelementptr inbounds i8, ptr addrspace(200) %src, i64 16928
  %sub = getelementptr inbounds i8, ptr addrspace(200) %s, i64 60
  %q0 = load i32, ptr addrspace(200) %sub
  %p1 = getelementptr inbounds i8, ptr addrspace(200) %sub, i64 4
  %q1 = load i32, ptr addrspace(200) %p1
  %r = add i32 %q0, %q1
  ret i32 %r
}

; A constant move at the simm12 edge (255 * 8 = 2040) is the same shape.
; CHECK-LABEL: poly_far:
; CHECK:      cincoffsetimm [[B:a[0-9]+]], {{a[0-9]+}}, 2040
; CHECK-NEXT: cincoffset [[P:a[0-9]+]], [[B]], {{a[0-9]+}}
; CHECK-NEXT: fld {{f[a-z0-9]+}}, 0([[P]])
; CHECK-NEXT: fld {{f[a-z0-9]+}}, 8([[P]])
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
