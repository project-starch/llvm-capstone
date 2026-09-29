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
; 2026-09-25: CapstoneLiveSourceCopy closes the class design A could not. A MOVC whose source is
; read again after it becomes STC+LDC through a stack slot, which leaves an untagged source intact
; on the RTL as on QEMU. So the residue arms below (bridged_phi_residue, bridged_callarg_plus_phi)
; and real_cap_copy's saved-across-the-call copy now go through the slot by default (RULE). The
; old pins still hold under +movc-keeps-integer-source (KEEP), the setting for a bitstream whose
; MOVC leaves an untagged source alone. They are kept as the control that shows the RULE checks
; can fail.
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 -verify-machineinstrs < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,O2,RULE
; RUN: llc -mtriple=capstone64 -mattr=+m,+movc-keeps-integer-source -O2 -verify-machineinstrs < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,O2,KEEP
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
; KEEP: movc a0, s0
; KEEP-NEXT: cjalr ra
; RULE: delin [[F1:a[0-9]]]
; RULE-NEXT: stc s0, [[S1:[0-9]+\(sp\)]]
; RULE-NEXT: ldc a0, [[S1]]
; RULE-NEXT: cjalr ra, 0([[F1]])
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
; KEEP: movc a0, s0
; KEEP-NEXT: cjalr ra
; RULE: delin [[F2:a[0-9]]]
; RULE-NEXT: stc s0, [[S2:[0-9]+\(sp\)]]
; RULE-NEXT: ldc a0, [[S2]]
; RULE-NEXT: cjalr ra, 0([[F2]])
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

; THE SHAPE THAT IS LIVE ON SILICON, and the one `bridged_phi_residue` above does NOT
; cover.  Added 2026-09-17 after an audit found the gap.
;
; The difference from `bridged_phi_residue` is that there, each bridge's ONLY use is the
; PHI -- so nothing is lost to a declined fold, and its `movc` are copies of the PHI
; RESULT.  Here the bridge has TWO uses: a conforming copy into a GPCR physreg (the call
; argument) AND the PHI.  MachineSinking::PerformSinkAndFold scans every use of a def and
; declines for the WHOLE def on the first non-conforming one, so the PHI costs the call
; argument its fold and that copy is emitted as `movc` too.
;
; This is `setupLookaside` at 0x267a0 in the Sublet cell 6 -O2 image, reduced: the
; INT-ONLY site the scanner flags, and the one the harm story runs through.  Without this
; arm a fix that removed THAT copy would leave the suite green, because
; `bridged_phi_residue` would still emit its own `movc` and nothing else here looks at
; this shape.
;
; SO: when a C-32 fix lands, THIS IS THE ARM THAT SHOULD FAIL FIRST.  It is a pin on
; current behaviour, not a statement that the behaviour is wanted -- update it
; deliberately, with the reason, exactly as `bridged_phi_residue` would be.
; The check is `movc` IMMEDIATELY BEFORE the `cjalr`, not a bare `movc`, and that is
; load-bearing: this function also emits `movc <reg>, zero` on the null arm to materialise
; cnull, which a bare CHECK would match happily FOREVER -- including after a fix had
; removed the copy this arm exists to catch.  Pinning the call-argument copy by its
; position is what makes the arm able to fail.
; CHECK-LABEL: bridged_callarg_plus_phi:
; KEEP: movc
; KEEP-NEXT: cjalr
; RULE: mv s0, a0
; RULE: delin [[F3:a[0-9]]]
; RULE-NEXT: stc s0, [[S3:[0-9]+\(sp\)]]
; RULE-NEXT: ldc a0, [[S3]]
; RULE-NEXT: cjalr ra, 0([[F3]])
define ptr addrspace(200) @bridged_callarg_plus_phi(i64 %x, i1 %c) {
entry:
  br i1 %c, label %then, label %nul
then:
  %p = inttoptr i64 %x to ptr addrspace(200)
  call void @use(ptr addrspace(200) %p)
  br label %join
nul:
  br label %join
join:
  %q = phi ptr addrspace(200) [ %p, %then ], [ null, %nul ]
  ret ptr addrspace(200) %q
}

; -O0 has its OWN prefix because its output is not the -O2 output: at -O0 the
; copy is copyPhysReg's GPR->GPCR arm emitting the ADDI directly, and
; real_cap_copy emits no movc at all, so the shared CHECK lines cannot be
; expected to hold here.  What -O0 must show is the same thing in the same
; place: the bridge as an integer write, never a movc of the bridged value.
; O0-LABEL: bridged_copied_live:
; O0-NOT: movc
;
; THE O0-NOT IS BOUNDED DELIBERATELY, and the bound is the label below.  A trailing
; CHECK-NOT runs to END OF FILE, so before this bound existed the -O0 arm silently policed
; every later function too -- and adding bridged_callarg_plus_phi broke it, because that
; function legitimately materialises cnull for its null PHI input and `movc <rd>, zero` is
; not a copy of a bridged value at all.  Bounding it here keeps exactly the old coverage
; (bridged_copied_live, real_cap_copy and bridged_phi_residue all still emit no movc at
; -O0) while letting the new arm say what IT must show.
; O0-LABEL: bridged_callarg_plus_phi:
; O0: movc {{[a-z0-9]+}}, zero
;
; That is the ONLY movc this function may have at -O0.  At -O0 the bridge goes through
; memory (`stc` then `ldc`), so the bridged value is never copied with movc here -- which
; is why C-32 has never been seen on an -O0 image, and why the -O0 arm is not where a fix
; would be noticed.
