; CapstoneLiveSourceCopy: a MOVC whose source is read again afterwards becomes STC+LDC through a
; 16-byte stack slot, because on today's silicon MOVC writes cnull into an untagged (integer) source
; (C-32). A MOVC whose source is dead, or comes from a register that always holds a tagged
; capability or null (c0, sp, gp, tp, fp, bp), stays a MOVC. With +movc-keeps-integer-source, the
; setting for a bitstream that implements Q-04 (b), every copy is a MOVC again. That run is the
; control: each RULE check fails against it.
;
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK,RULE
; RUN: llc -mtriple=capstone64 -mattr=+m,+movc-keeps-integer-source -O2 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK,KEEP
; RUN: llc -mtriple=capstone64 -mattr=+m -O0 -verify-machineinstrs < %s -o /dev/null
; The slot must exist before the frame is laid out, so a function that needs it is not
; shrink-wrapped. Forcing shrink-wrapping is a hard error, not a silently unallocated slot.
; RUN: not llc -mtriple=capstone64 -mattr=+m -O2 -enable-shrink-wrap=true < %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=SW
; SW: LLVM ERROR: capstone: early_return needs the live-source copy slot but was shrink-wrapped

declare void @g(ptr addrspace(200)) addrspace(200)
declare void @many(ptr addrspace(200), ptr addrspace(200), ptr addrspace(200)) addrspace(200)

; The argument is saved in s0 and passed to two calls. The first copy back to a0 reads s0 again
; afterwards, so it goes through the slot. The second is s0's last use and stays a movc.
; CHECK-LABEL: saved_across_call:
; RULE: delin [[F:s[0-9]]]
; RULE-NEXT: stc s0, [[S:[0-9]+\(sp\)]]
; RULE-NEXT: ldc a0, [[S]]
; RULE-NEXT: cjalr ra, 0([[F]])
; RULE-NEXT: movc a0, s0
; RULE-NEXT: cjalr ra, 0([[F]])
; KEEP: movc a0, s0
; KEEP-NEXT: cjalr ra
; KEEP-NEXT: movc a0, s0
; KEEP-NEXT: cjalr ra
define void @saved_across_call(ptr addrspace(200) %p) addrspace(200) {
  call addrspace(200) void @g(ptr addrspace(200) %p)
  call addrspace(200) void @g(ptr addrspace(200) %p)
  ret void
}

; A copy whose source dies stays a movc and costs no frame: the slot is reserved only when some
; copy can still have a live source when the rewrite runs.
; CHECK-LABEL: dead_source:
; CHECK-NOT: sp
; CHECK: movc a0, a1
; CHECK-NEXT: cjalr zero, 0(ra)
define ptr addrspace(200) @dead_source(ptr addrspace(200) %a, ptr addrspace(200) %b) addrspace(200) {
  ret ptr addrspace(200) %b
}

; A leaf that returns one value in two registers. The copy's source is itself returned, so the
; leaf gets a 16-byte frame for the slot.
; CHECK-LABEL: leaf_dup:
; RULE: cincoffsetimm sp, sp, -16
; RULE: stc a0, [[L:0\(sp\)]]
; RULE-NEXT: ldc a1, [[L]]
; RULE-NEXT: cincoffsetimm sp, sp, 16
; KEEP-NOT: sp
; KEEP: movc a1, a0
define { ptr addrspace(200), ptr addrspace(200) } @leaf_dup(ptr addrspace(200) %a) addrspace(200) {
  %r0 = insertvalue { ptr addrspace(200), ptr addrspace(200) } poison, ptr addrspace(200) %a, 0
  %r1 = insertvalue { ptr addrspace(200), ptr addrspace(200) } %r0, ptr addrspace(200) %a, 1
  ret { ptr addrspace(200), ptr addrspace(200) } %r1
}

; Several copies of one live source share one STC. Two LDCs from the slot behave exactly like two
; MOVCs from the source, for every type. An integer or a NONLIN value reloads unchanged. A LINEAR
; value is taken by the first LDC, which clears the slot, as the first MOVC would null the source.
; CHECK-LABEL: one_store_many_loads:
; RULE: stc a0, [[M:[0-9]+\(sp\)]]
; RULE-NEXT: ldc a1, [[M]]
; RULE-NEXT: ldc a2, [[M]]
; RULE-NOT: stc a0
; RULE: cjalr ra
; KEEP: movc a1, a0
; KEEP-NEXT: movc a2, a0
define void @one_store_many_loads(ptr addrspace(200) %p) addrspace(200) {
  call addrspace(200) void @many(ptr addrspace(200) %p, ptr addrspace(200) %p, ptr addrspace(200) %p)
  ret void
}

; Shrink-wrapping would put the prologue on the %work path only. The copies there need the slot,
; so the prologue stays in the entry block under the rule.
; CHECK-LABEL: early_return:
; RULE: cincoffsetimm sp, sp, -{{[0-9]+}}
; RULE: beqz
; KEEP: beqz
; KEEP: # %work
; KEEP-NEXT: cincoffsetimm sp, sp, -{{[0-9]+}}
define ptr addrspace(200) @early_return(ptr addrspace(200) %p, i1 %c) addrspace(200) {
entry:
  br i1 %c, label %out, label %work
work:
  call addrspace(200) void @g(ptr addrspace(200) %p)
  call addrspace(200) void @g(ptr addrspace(200) %p)
  ret ptr addrspace(200) %p
out:
  ret ptr addrspace(200) null
}

; With a frame pointer the slot is addressed from fp (s0), next to the callee-saved area.
; CHECK-LABEL: with_fp:
; RULE: stc s1, [[P:-[0-9]+\(s0\)]]
; RULE-NEXT: ldc a0, [[P]]
; KEEP-NOT: (s0)
define void @with_fp(ptr addrspace(200) %p) addrspace(200) "frame-pointer"="all" {
  call addrspace(200) void @g(ptr addrspace(200) %p)
  call addrspace(200) void @g(ptr addrspace(200) %p)
  ret void
}

; With a base pointer (an over-aligned object plus a dynamic alloca) the slot is addressed from
; bp (s1): sp moves at the alloca, and fp is on the other side of the realignment.
declare void @h(ptr addrspace(200), ptr addrspace(200)) addrspace(200)
; CHECK-LABEL: with_bp:
; RULE: delin [[G:s[0-9]]]
; RULE-NEXT: stc [[P:s[0-9]]], [[B:[0-9]+\(s1\)]]
; RULE-NEXT: ldc a0, [[B]]
; RULE-NEXT: cjalr ra, 0([[G]])
; RULE-NEXT: movc a0, [[P]]
; KEEP: delin [[G:s[0-9]]]
; KEEP-NEXT: movc a0, [[P:s[0-9]]]
; KEEP-NEXT: cjalr ra, 0([[G]])
; KEEP-NEXT: movc a0, [[P]]
define void @with_bp(ptr addrspace(200) %p, i64 %n) addrspace(200) {
  %big = alloca i8, i64 64, align 64, addrspace(200)
  %dyn = alloca i8, i64 %n, align 16, addrspace(200)
  call addrspace(200) void @h(ptr addrspace(200) %big, ptr addrspace(200) %dyn)
  call addrspace(200) void @g(ptr addrspace(200) %p)
  call addrspace(200) void @g(ptr addrspace(200) %p)
  ret void
}
