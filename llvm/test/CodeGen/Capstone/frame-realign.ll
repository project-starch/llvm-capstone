; RUN: llc -mtriple=capstone64 -verify-machineinstrs < %s | FileCheck %s

target triple = "capstone64"

define ptr addrspace(200) @need_realign() {
; CHECK-LABEL: need_realign:
; CHECK: cincoffsetimm sp, sp, -64
; CHECK: cincoffsetimm s0, sp, 64
; CHECK: lcc [[CUR:a[0-9]+]], sp, 2
; CHECK-NEXT: andi [[CUR]], [[CUR]], -64
; CHECK-NEXT: scc sp, sp, [[CUR]]
; CHECK: cincoffsetimm a0, sp, 0
; CHECK: cincoffsetimm sp, s0, -64
; CHECK: cjalr zero, 0(ra)
entry:
  %slot = alloca i8, align 64, addrspace(200)
  ret ptr addrspace(200) %slot
}


