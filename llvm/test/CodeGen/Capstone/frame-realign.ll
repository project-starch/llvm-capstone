; RUN: llc -mtriple=capstone64 -verify-machineinstrs < %s | FileCheck %s
; RUN: %llc_cap -O0 < %s -o /dev/null
; RUN: %llc_cap -O1 < %s -o /dev/null

target triple = "capstone64"

define ptr addrspace(200) @need_realign() {
; CHECK-LABEL: need_realign:
; CHECK: cincoffsetimm sp, sp, -64
; CHECK: movc s0, sp
; CHECK-NEXT: cincoffsetimm s0, s0, 64
; The cursor of sp is read with the plain integer move (C-31 / Tier 4.2: no
; `lcc ..., 2` anywhere), then aligned and written back with scc.
; CHECK: mv [[CUR:a[0-9]+]], sp
; CHECK-NEXT: andi [[CUR]], [[CUR]], -64
; CHECK-NEXT: scc sp, sp, [[CUR]]
; CHECK: cincoffsetimm a0, sp, 0
; CHECK: cincoffsetimm sp, s0, -64
; CHECK: cjalr zero, 0(ra)
entry:
  %slot = alloca i8, align 64, addrspace(200)
  ret ptr addrspace(200) %slot
}


