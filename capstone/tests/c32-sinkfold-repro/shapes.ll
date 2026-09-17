; C-32 after design A: the shapes that decide whether the bridged integer is
; copied with `movc`, reduced from the Sublet cell 6 -O2 image
; (sha256 113221f93b0ac994..., function setupLookaside, site 0x267a0).
;
; WHAT DECIDES IT.  Design A lowers inttoptr to the rematerializable
; PseudoBRIDGE_CAP, but RA-side rematerialization never runs for it: the generic
; TargetInstrInfo::isReallyTriviallyReMaterializable refuses any instruction with
; a virtual-register use, and this pseudo has one.  What actually removes the
; movc is MachineSinking::PerformSinkAndFold (pre-RA, MachineSink.cpp), which
; rewrites ISel's `$c10 = COPY %bridged` into `$c10 = PseudoBRIDGE_CAP %int` and
; so duplicates the bridge into each use, leaving no GPCR vreg for RA to copy.
;
; That fold is ALL-OR-NOTHING PER DEF.  It walks every use of the bridge's def
; and the first use that is not a copy chaining to a physreg of the SAME
; register class (GPCR), and not a foldable load/store address, makes it decline
; for the whole def.  One such use therefore leaves the OTHER uses' copies as
; movc as well.  That is why design A holds on the lit-test shape and not on
; setupLookaside.

declare void @use(ptr addrspace(200))

; 1. EVERY USE CONFORMS: the bridged value is only ever copied into a GPCR
;    physreg (the call argument, then the return).  The fold fires; no movc.
; A movc here would be a DEFECT: it copies an untagged bridged integer.
;    EXPECT-MOVC: 0 defect
define ptr addrspace(200) @fold_ok_all_uses_conform(i64 %x) {
  %p = inttoptr i64 %x to ptr addrspace(200)
  call void @use(ptr addrspace(200) %p)
  ret ptr addrspace(200) %p
}

; 2. ONE NON-CONFORMING USE, and it is the only difference from shape 1: the
;    address half is read back as an integer.  That use chains to a copy into a
;    GPR physreg, which is not in GPCR, so the fold declines FOR THE WHOLE DEF
;    and the call-argument copy stays a movc.
;    EXPECT-MOVC: 1 defect
define i64 @fold_declines_int_readback(i64 %x) {
  %p = inttoptr i64 %x to ptr addrspace(200)
  call void @use(ptr addrspace(200) %p)
  %i = ptrtoint ptr addrspace(200) %p to i64
  ret i64 %i
}

; 3. THE OTHER NON-CONFORMING USE: a PHI at a join.  A PHI is neither a copy nor
;    a foldable address, so the fold declines the same way.  This is the residue
;    the lead accepted when choosing design A -- recorded here as the SAME
;    mechanism as shape 2, not a separate family.  TWO, because the joined value
;    is used twice -- the call argument and the return -- and the fold having
;    declined, each use gets its own movc.
;    EXPECT-MOVC: 2 defect
define ptr addrspace(200) @fold_declines_phi(i64 %x, i1 %c) {
entry:
  br i1 %c, label %then, label %nul
then:
  %p = inttoptr i64 %x to ptr addrspace(200)
  br label %join
nul:
  br label %join
join:
  %q = phi ptr addrspace(200) [ %p, %then ], [ null, %nul ]
  call void @use(ptr addrspace(200) %q)
  ret ptr addrspace(200) %q
}

; 4. THE SILICON SHAPE, instruction for instruction.  The Sublet port takes the
;    lookaside base out of a capability as an INTEGER with inline asm, bridges
;    it on one arm of a branch, passes it to an indirect call, and joins it with
;    null.  Compare against the image at 0x26790-0x267f4:
;        beqz a0, .+0x34 / mv s3, a0 / movc a0, s3 / jalr a1 / movc s3, zero /
;        mv a0, s3
;    On the RTL the movc nulls cs3, so the join reads zero and the lookaside is
;    silently off.
;    EXPECT-MOVC: 1 defect
define i64 @silicon_setuplookaside_shape(i64 %in, ptr addrspace(200) %fp) {
entry:
  %v = call i64 asm "lcc $0, $1, 3", "=r,r"(i64 %in)
  %z = icmp eq i64 %v, 0
  br i1 %z, label %nul, label %then
then:
  %p = inttoptr i64 %v to ptr addrspace(200)
  %r = call i32 %fp(ptr addrspace(200) %p)
  br label %join
nul:
  br label %join
join:
  %q = phi ptr addrspace(200) [ %p, %then ], [ null, %nul ]
  %qi = ptrtoint ptr addrspace(200) %q to i64
  ret i64 %qi
}

; 5. CONTROL, and it is what bounds every "0" above: a REAL capability held
;    across a call must still be copied with movc.  If this ever reports 0 the
;    counts above have stopped meaning anything.  These three are CORRECT and
;    must not be "fixed": the incoming argument saved to a callee-saved
;    register, the call argument, and the return.
;    EXPECT-MOVC: 3 correct
define ptr addrspace(200) @real_cap_copy_control(ptr addrspace(200) %p) {
  call void @use(ptr addrspace(200) %p)
  ret ptr addrspace(200) %p
}
