; C-32: `movc rd, rs1` on the RTL NULLS the source register for every source
; that is not NONLIN -- including an UNTAGGED one (capstone_flu_unit.anvil:13-26
; has no NOT_CAP exclusion; rtl-oracle 2026-09-04), while QEMU only nulls a
; tagged, non-copyable source (op_helper.c:580-585).  A value bridged from an
; integer (inttoptr: an integer written into the address half of an undefined
; capability, untagged by construction) that is COPIED while it stays live
; therefore loses its value on silicon and keeps it under QEMU.  Observed live on
; the board in the Sublet port's setupLookaside at -O2.
;
; FIXED 2026-09-15 by the lead's design A: the inttoptr lowering emits the
; rematerializable PseudoBRIDGE_CAP instead of a bare INSERT_SUBREG, so the
; bridged value is RE-BRIDGED at each use and no capability copy of it is formed.
; The XFAIL is gone.
;
; WHY THE `mv` COUNT IS CHECKED AND NOT JUST `CHECK-NOT: movc`.  A bare
; CHECK-NOT cannot tell "remat fired" from "the pseudo exists but remat never
; ran and the copy happened to be elsewhere".  The bridge must appear at EACH
; use, which is what remat doing its job looks like; that is the positive half of
; this test.  What makes it fire is the pseudo's isReMaterializable /
; isAsCheapAsAMove pair: drop both, rebuild, and these `mv` lines become `movc`
; again.  (An earlier draft of this comment credited a Capstone override of
; isReallyTriviallyReMaterializable.  There is no such override -- one was
; written, measured to produce a byte-identical -O2 image, and removed.  See the
; pseudo's definition in CapstoneInstrInfo.td.)
;
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 -verify-machineinstrs < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,O2
; RUN: llc -mtriple=capstone64 -mattr=+m -O0 -verify-machineinstrs < %s \
; RUN:   | FileCheck %s --check-prefix=O0
; RUN: %llc_cap -O1 < %s -o /dev/null

declare void @use(ptr addrspace(200))

; The shape ordinary C produces: `void *p = (void *)x; use(p); return p;`.
; p is live across the call, and the copy that used to keep it there was a movc.
; CHECK-LABEL: bridged_copied_live:
; CHECK-NOT: movc
; O2: mv a0, s0
; CHECK: cjalr ra
; O2: mv a0, s0
; CHECK-NOT: movc
; CHECK: cjalr zero, 0(ra)
define ptr addrspace(200) @bridged_copied_live(i64 %x) {
  %p = inttoptr i64 %x to ptr addrspace(200)
  call void @use(ptr addrspace(200) %p)
  ret ptr addrspace(200) %p
}

; CONTROL, and it is what bounds the CHECK-NOT above: a REAL capability saved
; across a call must still be copied with movc.  If this arm ever stops emitting
; one, the negative check above has stopped meaning anything.
; CHECK-LABEL: real_cap_copy:
; O2: movc
define ptr addrspace(200) @real_cap_copy(ptr addrspace(200) %p) {
  call void @use(ptr addrspace(200) %p)
  ret ptr addrspace(200) %p
}

; KNOWN RESIDUE, pinned deliberately rather than left to be discovered.  Design A
; cannot remove a PHI copy: the two bridged values meet at a join, register
; allocation places a copy in a predecessor, and that copy is a GPCR copy, i.e.
; a movc.  This function MUST still emit one.  It is the residue the lead
; accepted when choosing design A over the register-class route, and it is
; recorded here so that a later change which removes it is noticed as a change
; rather than assumed to have always been true.
; CHECK-LABEL: bridged_phi_residue:
; O2: movc
define ptr addrspace(200) @bridged_phi_residue(i64 %x, i64 %y, i1 %c) {
entry:
  br i1 %c, label %a, label %b
a:
  %pa = inttoptr i64 %x to ptr addrspace(200)
  br label %join
b:
  %pb = inttoptr i64 %y to ptr addrspace(200)
  br label %join
join:
  %p = phi ptr addrspace(200) [ %pa, %a ], [ %pb, %b ]
  call void @use(ptr addrspace(200) %p)
  ret ptr addrspace(200) %p
}

; -O0 has its OWN prefix because its output is not the -O2 output: at -O0 the
; copy is copyPhysReg's GPR->GPCR arm emitting the ADDI directly, and
; real_cap_copy emits no movc at all, so the shared CHECK lines cannot be
; expected to hold here.  What -O0 must show is the same thing in the same
; place: the bridge as an integer write, never a movc of the bridged value.
; O0-LABEL: bridged_copied_live:
; O0-NOT: movc
