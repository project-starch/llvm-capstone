; RUN: llc -mtriple=capstone64 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK,DEFAULT
; RUN: llc -mtriple=capstone64 -verify-machineinstrs -data-sections < %s | FileCheck %s --check-prefixes=CHECK,DATASECTIONS

; C-85: a 16-byte common global (a C tentative definition built with -fcommon,
; here a capability and an i128) has no section. The 16-byte emission path
; asked for one and reached "Unknown section kind". It is a .comm symbol with
; its size and alignment, as every other common global is.

@p = common addrspace(200) global ptr addrspace(200) null, align 16
@w = common addrspace(200) global i128 0, align 16
@small = common addrspace(200) global i32 0, align 4
; Not common: still a 16-byte object in .bss.
@q = addrspace(200) global ptr addrspace(200) null, align 16

; CHECK:      .type p,@object
; CHECK-NEXT: .comm p,16,16
; CHECK:      .type w,@object
; CHECK-NEXT: .comm w,16,16
; CHECK:      .comm small,4,4
; DEFAULT-NEXT:      .bss
; DATASECTIONS-NEXT: .section .bss.q,"aw",@nobits
; CHECK-NEXT: .p2align 4
; CHECK-NEXT: .globl q
; CHECK-NEXT: q:

define ptr addrspace(200) @load_p() {
  %v = load ptr addrspace(200), ptr addrspace(200) @p, align 16
  ret ptr addrspace(200) %v
}
