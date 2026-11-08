; An XLen AND reaches the generated matcher, so a mask that needs no register
; is selected without one: SLLI/SRLI for leading- and trailing-ones masks, and
; with Zbb ZEXT_H for 0xffff and ANDN for an AND with an inverted operand.
; Select() used to end its ISD::AND case by selecting every XLen AND itself, as
; ANDI or as AND with the mask materialised (lui/li + addi/shift), which kept
; all of those patterns from ever matching.
; RUN: llc -mtriple=capstone64 -mattr=+m -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK,BASE
; RUN: llc -mtriple=capstone64 -mattr=+m,+zbb -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK,ZBB

; CHECK-LABEL: mask16:
; BASE: slli a0, a0, 48
; BASE-NEXT: srli a0, a0, 48
; ZBB: zext.h a0, a0
; CHECK-NOT: lui
; CHECK: cjalr zero, 0(ra)
define i64 @mask16(i64 %x) {
  %a = and i64 %x, 65535
  ret i64 %a
}

; CHECK-LABEL: mask20:
; CHECK: slli a0, a0, 44
; CHECK-NEXT: srli a0, a0, 44
; CHECK-NEXT: cjalr zero, 0(ra)
define i64 @mask20(i64 %x) {
  %a = and i64 %x, 1048575
  ret i64 %a
}

; CHECK-LABEL: mask32:
; CHECK: slli a0, a0, 32
; CHECK-NEXT: srli a0, a0, 32
; CHECK-NEXT: cjalr zero, 0(ra)
define i64 @mask32(i64 %x) {
  %a = and i64 %x, 4294967295
  ret i64 %a
}

; CHECK-LABEL: high32:
; CHECK: srli a0, a0, 32
; CHECK-NEXT: slli a0, a0, 32
; CHECK-NEXT: cjalr zero, 0(ra)
define i64 @high32(i64 %x) {
  %a = and i64 %x, -4294967296
  ret i64 %a
}

; CHECK-LABEL: andnot:
; BASE: not a1, a1
; BASE-NEXT: and a0, a0, a1
; ZBB: andn a0, a0, a1
; CHECK-NEXT: cjalr zero, 0(ra)
define i64 @andnot(i64 %x, i64 %y) {
  %n = xor i64 %y, -1
  %a = and i64 %x, %n
  ret i64 %a
}

; Unchanged: a simm12 mask is still ANDI, two registers still AND.
; CHECK-LABEL: andi_small:
; CHECK: andi a0, a0, 2047
; CHECK-NEXT: cjalr zero, 0(ra)
define i64 @andi_small(i64 %x) {
  %a = and i64 %x, 2047
  ret i64 %a
}

; CHECK-LABEL: and_reg:
; CHECK: and a0, a0, a1
; CHECK-NEXT: cjalr zero, 0(ra)
define i64 @and_reg(i64 %x, i64 %y) {
  %a = and i64 %x, %y
  ret i64 %a
}

; Unchanged: masking a capability still moves its cursor with CINCOFFSET by the
; difference, never replaces the capability with the masked integer.
; CHECK-LABEL: cap_mask:
; CHECK: andi a1, a0, -16
; CHECK-NEXT: sub a1, a1, a0
; CHECK-NEXT: cincoffset a0, a0, a1
; CHECK-NEXT: cjalr zero, 0(ra)
define ptr addrspace(200) @cap_mask(ptr addrspace(200) %p) {
  %q = call ptr addrspace(200) @llvm.ptrmask.p200.i64(ptr addrspace(200) %p, i64 -16)
  ret ptr addrspace(200) %q
}

declare ptr addrspace(200) @llvm.ptrmask.p200.i64(ptr addrspace(200), i64)
