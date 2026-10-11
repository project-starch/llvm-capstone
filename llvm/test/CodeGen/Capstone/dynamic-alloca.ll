; Pinned to -capstone-shrink-stack=false (default on since 2026-07-03): this test
; checks the base dynamic-alloca lowering, not stack narrowing; the narrowed path
; is covered by cap-shrink-dynalloca.ll.
; The new stack pointer is also the returned pointer, so its source stays live after the copy
; into sp. By default (CapstoneLiveSourceCopy) that copy goes through the live-source copy slot, which
; is fp-relative here because of the VLA; with +movc-keeps-integer-source it is a plain movc.
; RUN: llc -mtriple=capstone64 -capstone-shrink-stack=false -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK,RULE
; RUN: llc -mtriple=capstone64 -mattr=+movc-keeps-integer-source -capstone-shrink-stack=false -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK,KEEP
; RUN: %llc_cap -O0 < %s -o /dev/null
; RUN: %llc_cap -O1 < %s -o /dev/null

target triple = "capstone64"

define ptr addrspace(200) @vla(i64 %n) {
; CHECK-LABEL: vla:
; CHECK: addi a0, a0, 15
; CHECK-NEXT: andi a0, a0, -16
; CHECK-NEXT: neg a0, a0
; CHECK-NEXT: cincoffset a0, sp, a0
; RULE-NEXT: stc a0, [[SLOT:-[0-9]+\(s0\)]]
; RULE-NEXT: ldc sp, [[SLOT]]
; KEEP-NEXT: movc sp, a0
entry:
  %p = alloca i8, i64 %n, align 16, addrspace(200)
  ret ptr addrspace(200) %p
}


