; A gp-derived global-address capability must be NON-LINEAR before anything can
; copy it. cincoffset produces a LINEAR capability, and the ISA consumes a
; non-NONLIN source on copy (movc rd, rs1 nulls rs1). If the LINEAR base is left
; as an SSA value with more than one use -- which MachineCSE creates when it
; hoists the pure cincoffset out of several branches while the tied, side-
; effecting delin stays behind in each -- the first copy nulls the shared
; register and every later delin faults on an untagged operand.
;
; selectLGA emits the cincoffset+delin pair as ONE pseudo (PseudoCapGlobalBase),
; so no LINEAR value ever has multiple uses. Each materialised base is delin'd
; exactly once, at its definition; later copies are of the NONLIN result.
;
; See CapstoneISelDAGToDAG.cpp selectLGA, CapstoneInstrInfo.td
; PseudoCapGlobalBase, and its expander in CapstoneExpandPseudoInsts.cpp.

; RUN: llc -mtriple=capstone64 -mattr=+m -capstone-shrink-globals=false < %s \
; RUN:   | FileCheck %s

target datalayout = "e-m:e-p:64:128-p200:128:128:128-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"

@g_count = internal unnamed_addr addrspace(200) global i32 0, align 4
@g_a = internal unnamed_addr addrspace(200) global ptr addrspace(200) null, align 16
@g_b = internal unnamed_addr addrspace(200) global ptr addrspace(200) null, align 16

; The store target is one of two globals on divergent paths, so the shared
; _MergedGlobals base is live across the branches -- the exact multi-use LINEAR
; base that used to be nulled by the first copy.
;
; Every delin here operates on a freshly gp-derived base, never on the result of
; a movc/cincoffset copy: the def-then-delin pair is atomic. There must be no
; delin whose operand is a register a prior movc wrote.

; The gp-relative base is delinearised at its point of definition, immediately
; after the cincoffset that derives it -- so the value that leaves this pair is
; already NONLIN and every later use (here cincoffsetimm) copies it safely.
; CHECK-LABEL: domain_main:
; CHECK: cincoffset [[B:a[0-9]+]], gp, {{a[0-9]+}}
; CHECK-NEXT: delin [[B]]
;
; The old miscompile left the LINEAR base multi-use and copied it with movc
; before a second delin. With the base already NONLIN there is no such second
; delin: exactly one delin is emitted for the merged-globals base.
; CHECK-NOT: delin
define dso_local void @domain_main(ptr addrspace(200) noundef %arg, i32 noundef %func) local_unnamed_addr addrspace(200) {
entry:
  %cmp = icmp eq i32 %func, 1
  %0 = load i32, ptr addrspace(200) @g_count, align 4
  br i1 %cmp, label %if.then, label %if.end6

if.then:
  switch i32 %0, label %if.end5 [
    i32 0, label %if.end5.sink.split
    i32 1, label %if.then4
  ]

if.then4:
  br label %if.end5.sink.split

if.end5.sink.split:
  %g_b.sink = phi ptr addrspace(200) [ @g_b, %if.then4 ], [ @g_a, %if.then ]
  store ptr addrspace(200) %arg, ptr addrspace(200) %g_b.sink, align 16
  br label %if.end5

if.end5:
  %inc = add i32 %0, 1
  store i32 %inc, ptr addrspace(200) @g_count, align 4
  br label %return

if.end6:
  store i32 %0, ptr addrspace(200) %arg, align 4
  br label %return

return:
  ret void
}
