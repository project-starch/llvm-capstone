# C-32: why design A does not fix `setupLookaside` — the mechanism, settled

**2026-09-17, compiler lane (apollo).** The handover
(`docs/plans/2026-09-17-compiler-lane-handover.md`, §2) recorded that design A is on
dev and measured not to fix the site it was chosen for, and left the mechanism
explicitly unproven with two candidate explanations (§10). This note settles it. The
answer is neither candidate.

## The finding

**Design A's protection is not rematerialisation. It is `MachineSinking::PerformSinkAndFold`,
and it is all-or-nothing per def.**

RA-side rematerialisation never runs for `PseudoBRIDGE_CAP` at all.
`TargetInstrInfo::isReallyTriviallyReMaterializable` (`llvm/lib/CodeGen/TargetInstrInfo.cpp`,
the loop ending "Don't allow any virtual-register uses") returns false for any
instruction with a virtual-register use, and the pseudo — `(outs GPCR:$rd), (ins GPR:$rs)`
— has one. Capstone's override at `CapstoneInstrInfo.cpp:244` is the inherited RVV one and
falls through to the generic check for this opcode. This is consistent with, and explains,
the predecessor's result that adding an override produced a byte-identical image: the hook is not
what is running. (The commit message and the `.td` comment call that hook
`isReallyTriviallyReMaterializableImpl`. **No such name exists in this tree** — `grep -rn` over
`llvm/` and `capstone/` returns nothing. It is `isReallyTriviallyReMaterializable`,
`TargetInstrInfo.h:182`, overridden at `CapstoneInstrInfo.h:78`. A reader will otherwise hunt for
it.)

**This is proof, not inference, and the note claims it as such.** Every remat client in
`llvm/lib/CodeGen` routes through `TargetInstrInfo::isTriviallyReMaterializable`
(`TargetInstrInfo.h:151-156`), which is a conjunction of `MI.getDesc().isRematerializable()` **and**
`isReallyTriviallyReMaterializable(MI)`; `grep -rn "isRematerializable()" llvm/lib/CodeGen/
llvm/lib/Target/Capstone/` returns **zero hits**, so there is no consumer of the descriptor flag
alone. With the second conjunct false, the whole predicate is false whatever the flag says — so of
design A's two remat flags, `isReMaterializable` is provably inert in CodeGen and only
`isAsCheapAsAMove` is load-bearing, read by `PerformSinkAndFold` at `MachineSink.cpp:494`
(`CapstoneInstrInfo::isAsCheapAsAMove` falls through to `MI.isAsCheapAsAMove()` for this opcode, so
the `.td` flag is what that call sees). The clients checked individually:
`RegisterCoalescer::reMaterializeTrivialDef` (`RegisterCoalescer.cpp:1330`),
`LiveRangeEdit::checkRematerializable` (`LiveRangeEdit.cpp:75`), `InlineSpiller.cpp:733`,
`SplitKit.cpp:380`, `MachineLICM.cpp:780`, `CalcSpillWeights.cpp:125`.

What actually removes the `movc` is `MachineSinking::PerformSinkAndFold`
(`llvm/lib/CodeGen/MachineSink.cpp:405`), which runs **pre-RA, in SSA**. It rewrites ISel's
`$c10 = COPY %bridged` into `$c10 = PseudoBRIDGE_CAP %int`, duplicating the bridge into
each use so that no GPCR vreg survives for register allocation to copy.

The gate is the use scan in that function ("Scan uses of the destination register. Every
use, except the last, must be a copy, with a chain of copies terminating with either a copy
into a hard register, or a load/store instruction where the use is part of the address"):
the first use that is **not** a copy chaining to a physreg of the same register class
(GPCR), and not a foldable load/store address, returns false **for the whole def**. One
non-conforming use therefore leaves every *other* use's copy as a `movc` as well. The pass says so
itself: the rewrite loop opens with `// Now we know we can fold the instruction in all its users.`,
reached only after the entire scan has succeeded.

**Why "rematerialisation" was a reasonable thing to conclude, and is still the wrong name.** The
rewrite is performed by `TII->reMaterialize(...)` — the mechanical "emit a copy of this instruction
defining DstReg" helper. The word is right there in the pass that does the work. What it is *not* is
the remat legality path: nothing here consults `isTriviallyReMaterializable`, which is why the flag
that matters is `isAsCheapAsAMove` (checked in the use scan: "If we are going to replace a copy, the
original instruction must be as cheap as a copy") and not `isReMaterializable`.

**The fold is also opportunistic in a second way**, which matters for how much weight the design can
carry: for a use in a DIFFERENT block from the def, it additionally declines on register pressure
(`registerPressureSetExceedsLimit`, same function, inside the `UseInst.getParent() != MI.getParent()`
arm — same-block uses are not pressure-checked). So design A's protection is not a property of the
pseudo; it is a peephole that usually fires. This session did not construct a case that declines on
pressure alone, so that path is read from the source and not measured.

`setupLookaside`'s bridged value has such a use: the **PHI** at the join. So the fold declines, a
GPCR vreg reaches RA, and ISel's call-argument COPY is lowered by `copyPhysReg` to `movc` at
`postrapseudos`.

**Two distinct roles, which are easy to conflate and which this note separates deliberately.** The
PHI is what makes the fold DECLINE, and so is what produces the `movc`. The integer read-backs of
the address half (`mv a0, s3` at `0x267f4` and `0x26820` — the `a = (uptr)pStart` this site's
registry entry describes) are uses of the JOINED value, not of the bridge's def, and they are not
what declined the fold; they are what makes the nulling OBSERVABLE, by reading the source after the
`movc` has written `cnull` over it. C-32 needs both: a declining use to produce the copy, and a
later read of the source for the loss to show. An integer read-back can also decline the fold on its
own when it is a use of the bridge's own def — shape 2 of the reproducer is exactly that, and it is
a second, independent route into the same defect.

## Evidence

Toolchain: `llvm/cmake-build-debug` on apollo, freshness gate exit 0
(`libLLVMCapstoneCodeGen.so 82f944546c332b6d`, built 2026-09-16 17:30), carrying design A
(`46c53b7b6ae2` is an ancestor of the checkout).

1. **Pass-level trace.** `llc -O2 -print-after-all -filter-print-funcs=bridged_copied_live`
   over the committed lit test: the bridge count goes 1 → 2 at *Machine code sinking*, and
   the MIR after it reads `$c10 = PseudoBRIDGE_CAP %0:gpr` — the def written straight into
   the physreg. No later pass changes it.
2. **The lit test's residue and the live site share a ROOT but are DIFFERENT manifestations —
   corrected after audit.** Same trace over `bridged_phi_residue`: machine-sink changes nothing and
   the two `MOVC` first appear at *Post-RA pseudo instruction expansion*, i.e. `copyPhysReg` on
   COPYs that survived RA. But its ISel MIR shows **each bridge's only use is the PHI**:

       bb.1.a:    %0:gpcr = PseudoBRIDGE_CAP %3:gpr
       bb.2.b:    %1:gpcr = PseudoBRIDGE_CAP %4:gpr
       bb.3.join: %2:gpcr = PHI %1:gpcr, %bb.2, %0:gpcr, %bb.1
                  $c10 = COPY %2:gpcr        <- the two MOVC come from THESE

   So nothing was lost to the all-or-nothing decline there: there was no conforming copy among the
   bridge's uses to lose, and its `movc` are copies of the **PHI result** `%2`, whose def is a PHI
   and is never a `PerformSinkAndFold` candidate at all. An earlier draft of this note called the
   two "the same failure"; that was wrong and is retracted here.
3. **One-variable control.** Starting from the lit test's passing shape and adding a single
   `ptrtoint` read-back — nothing else — `mv a0, s0` becomes `movc a0, s0`.
4. **Positive control.** `-capstone-enable-sink-fold=false` turns the *passing* shape into
   two `movc`, reproducing the pre-design-A defect on demand. Design A's protection rides
   entirely on that switch.
5. **The silicon site reproduced as the same sequence** (not identical: the image interleaves
   `ldc a1, 0x2e0(gp)` / `ldc a1, 0x50(a1)` between the `beqz` and the `mv s3, a0`, and lays the
   arms out the other way round — the image falls through from the then-arm, the shape emits a
   `j`), from the transferred image
   (`xfer/c32-f1-2026-09-16`, `cell6-O2-c32fixed.dom`, sha256 verified
   `113221f93b0ac994…` against the branch's own `SHA256SUMS` before reading it):

       image    26790 beqz a0 · 2679c mv s3,a0 · 267a0 movc a0,s3 · 267a4 jalr a1
                267c4 movc s3,zero · 267f4 mv a0,s3
       shape 4  beqz a0 · mv s0,a0 · movc a0,s0 · cjalr ra,0(a1)
                movc s0,zero · mv a0,s0

   The `movc s3, zero` on the other arm is what identifies `cs3` as a **PHI register**: it is
   written on both arms of the branch at `0x26790` and read after the join at `0x267f4`.
6. **The opt-level signature matches the board's.** The five shapes at `-O0`/`-O1`/`-O2` give
   0 / 4 / 4 defect `movc`: `-O1` and `-O2` identical, `-O0` emitting none at all. That is what the
   C-32 registry entry records from silicon — "the -O1 image diverges identically; the -O0 image
   does not (its int-to-pointer cast goes through memory)" — and `real_cap_copy_control` dropping to
   0 at `-O0` matches the lit test's own note that `real_cap_copy` has no `movc` there. The
   reproducer was not tuned to these; they are a fidelity check it passed.
7. **The four-site table is re-verified at primary source, not taken from the handover.**
   `python3 capstone/tests/movc-cfg-scan.py <image> postfix-113221f9` over the transferred image:
   17285 `movc` with `rd != rs`, 10273 with the source read again on some path, of those **2
   INT-ONLY and 2 MIXED** — `setupLookaside+0x267a0 -> read @0x267f4`,
   `setupLookaside+0x26a28`, `main+0x3aabc`, `renameResolveTrigger+0x10b5b8` — plus 1780 whose
   source is only a call return or argument and so are not statically classifiable. Same four
   functions and classifications the handover records, and the first one's source/read pair is
   exactly the pair analysed above.

   **Re-scanned after the scanner was repaired, and the table does not move.** That first scan
   predated the board lane's fix to `movc-cfg-scan.py` (the one-operand `jalr rs` had been read as a
   definition of `rs`, injecting a reaching definition that does not exist — reported from this
   investigation, fixed at `fe2866d71437`). Re-running the repaired scanner on the same
   hash-verified image gives a **byte-identical** report: same four sites, same classifications,
   same 2/2/1780 split. The negative is a loaded one rather than a void one — the construct the fix
   acts on is present (seven one-operand `jalr` in `setupLookaside` alone) and the two scanner
   versions differ as files, so the instrument changed and the answer did not. `0x26a28` in
   particular stays MIXED, and for the reason already given: its `cnull` arm, not the `jalr`
   defect. For `main+0x3aabc` the fix removes the spurious `jalr s5` reaching definition, but three
   genuine `ldc` capability loads remain among its twelve, so its classification and the conclusion
   drawn from it are unaffected.
8. **The other three sites do NOT share one cause — an earlier draft of this note said "four
   sites, one cause" and that is RETRACTED.** It was written from a few lines of disassembly
   context around each; replicating the scanner's flow-sensitive backward walk and printing the
   actual reaching definitions splits them three ways:

   * **`setupLookaside+0x26a28` — same cause.** Reaching defs are exactly `0x2679c [int]` and
     `0x267c4 [cnull]`: the same register, same value, same merge as the live site. Its MIXED tag
     is an artefact of the null arm being `movc s3, zero`, not a genuine capability.
   * **`renameResolveTrigger+0x10b5b8` — same CLASS, cause not shown.** Single reaching def
     `0x10b42c mv s11, a2`, INT-ONLY: a bridged integer copied around a back-edge. No merge is
     visible, so "a fold declined by a PHI" is not established for it.
   * **`main+0x3aabc` — NOT SHOWN to be a bridged value at all.** Twelve reaching definitions,
     three of them capability loads (`0x38434 ldc s5, 0x0(s3)`, `0x3a3b0` and `0x3a474
     ldc s5, 0x2e0(gp)`). That is a callee-saved register recycled across a ~2965-instruction
     function, not one bridged value.

   **This also failed to reconcile with prior art, which is the rule this note broke.**
   `docs/history/15-09-2026_02-40-00_c32-movc-scan-one-site-or-a-class.md` already characterises
   the analogous `main` site as "**Not PHI-shaped**". Searching the per-bug history before
   generalising from three lines of context would have caught it; it was an audit that did.

Reproducer and its six negative-tested exit paths: `capstone/tests/c32-sinkfold-repro/`.

**Three instrument notes for anyone repeating this:**

* **`-filter-print-funcs` does NOT gate `LLVM_DEBUG` output**, only `-print-after-all`. A
  `-debug-only=machine-sink` run over a multi-function module attributes every function's folds to
  whichever one you were looking at. Split the module one function per file.
* **`movc-cfg-scan.py` treats `jalr rs` as a DEF of `rs`** (`defs_reads` returns `r[0]`), so an
  indirect call terminates the backward walk and injects a spurious `cap` reaching definition. It
  makes the reaching-def union a lower bound and can flip a site to MIXED. The tool is the board
  lane's; this is reported to them, not fixed here.
* **Keying a per-pass counter by pass NAME loses data** — `Greedy Register Allocator` and
  `Slot index numbering` each run more than once, and later dumps overwrite earlier ones. Key on
  dump index.

## What this corrects in the record

* The handover's §10 dichotomy — "remat under-applied" vs "a post-RA `copyPhysReg` copy" —
  is not the right split. The copy is created by **ISel, pre-RA**, and merely *lowered* by
  `copyPhysReg` post-RA; what fails is a pre-RA fold; and remat was never reachable.
* `CapstoneInstrInfo.td`'s `PseudoBRIDGE_CAP` comment and `46c53b7b6ae2`'s message both
  explain the fix as rematerialisation ("the value is RE-BRIDGED at each use"). The
  *effect* is right and the *mechanism named is not*. Of the two remat flags, only
  `isAsCheapAsAMove` is load-bearing, and it is consumed by `PerformSinkAndFold`
  ("If we are going to replace a copy, the original instruction must be as cheap as a
  copy"), not by any remat client. That explains the predecessor's mutation result —
  dropping both flags did bring the `movc` back — without supporting the mechanism it was
  read as confirming. **Not corrected in the source yet:** that comment rides with whatever
  design the lead picks, and is noted here so it is not repeated as fact meanwhile.
* The live site is **not** a shape design A was expected to cover and missed — it is a bridged
  value reaching a merge, which is the root design A was accepted as not covering. But it is **not
  the same thing `bridged_phi_residue` pins**, and an earlier draft of this note said it was.
  `setupLookaside` contains both kinds, and they must be told apart:

  * **`0x267a0`, the live site** — a bridge def with *two* uses: the call-argument physreg COPY
    (conforming) **and** the PHI (not). All-or-nothing bites here, and turns the conforming copy
    into a `movc` as well. This is the shape this note is about, the one the scanner flags
    INT-ONLY, and the one the harm story runs through.
  * **`0x26a28` / `0x26a54`** — copies of the *merged* value, reaching defs `0x2679c [int]` and
    `0x267c4 [cnull]`. **These** are the `bridged_phi_residue` kind.

* **Consequence worth stating on its own: until this change the lit test could not detect a change
  to the live site's shape.** A fix that removed the `0x267a0` copy would have left
  `bridged_phi_residue` still emitting its `movc` and the lit suite still green;
  `bridged_phi_residue` does not exercise the fold-declined shape and cannot. A real gap in the
  regression net, found by audit rather than by the net.

  **It is CLOSED here, not merely recorded.** `c32-movc-untagged-live.ll` gains
  `bridged_callarg_plus_phi` — the live shape reduced — which pins the call-argument copy the way
  `bridged_phi_residue` pins its own residue, and it is the arm that should fail first when a C-32
  fix lands. Negative-tested two-sided: it passes on current codegen and FAILS when the PHI use is
  removed so the fold fires, so it is a proven guard rather than an unproven one.
  `c32-sinkfold-repro` is a development instrument and runs in **no** suite; the lit arm is what
  actually runs.

  Two defects in the existing test surfaced while adding it, both fixed here. Its `O0-NOT: movc`
  was **unbounded** — a trailing `CHECK-NOT` runs to end of file, so it had been silently policing
  every function added after it, and the new arm's legitimate `movc <rd>, zero` cnull
  materialisation tripped it; it is now bounded by a label, preserving exactly the old coverage.
  And a bare `O2: movc` on the new arm would have been satisfied forever by that same cnull
  `movc` — including after a fix removed the copy the arm exists to catch — so the check is pinned
  to the call-argument copy by position (`O2: movc` / `O2-NEXT: cjalr`) instead.
* **The handover's statement that the register-class alternative "was rejected partly on design A
  being sufficient here" is not supported by the record, and this note does not repeat it.** The
  decision table the lead ruled from (`docs/plans/DECISIONS-WAITING-2026-09-10.md:559-561`) says
  something different and more useful: design A "leaves **PHI copies**, which remat cannot reach",
  and the register class "would cover PHI copies too" but is "probably unworkable: *untagged* is a
  property of the **value**, a `RegisterClass` is a set of **physical registers**, and
  `copyPhysReg` sees only physical numbers". So the register-class route was set aside on a
  structural objection of its own, independent of design A. Repeating the handover's version would
  have told the lead that a rejection premise had been falsified, when the recorded premise is
  untouched by anything measured here.

## What this does not establish

* That the PHI is the declining use in `setupLookaside` rests on the register census, not on that
  function's MIR, which cannot be produced on this host. Every write of `s3` in the function is at
  `0x26550` (an earlier, unrelated value, read at `0x266d0`), `0x2679c` (the bridge, then-arm),
  `0x267c4` (`movc s3, zero`, else-arm) and `0x26aac` (`ldc s3, 0x70(sp)`, the epilogue restore),
  with post-join reads at `0x267f4`, `0x26820`, `0x26870` (`stc s3, 0x2e0(s1)` — where the
  possibly-nulled capability is stored INTO the sqlite3 struct), `0x26888`, `0x26a28` and `0x26a54`.

  The merge itself is **machine-checked, not a hand census**: replicating the scanner's
  flow-sensitive backward walk over the image's CFG prints, at each of `0x267f4`, `0x26820` and
  `0x26870`, exactly two reaching definitions — `0x2679c [int]` and `0x267c4 [cap]`, one per arm of
  the branch at `0x26790` — while `0x267a0` has the single def `0x2679c`. Both arms reaching one
  read is the definition of a merge, so "cs3 holds two unrelated values" does not fit. The one
  remaining alternative, that the merge is a `select` expanded post-RA rather than an IR PHI, is
  also closed: `Select_GPRCAP_Using_CC_GPR` goes through `emitSelectPseudo`
  (`CapstoneISelLowering.cpp:23269`), which builds a real `Capstone::PHI` at ISel time, long before
  machine-sink. What is still NOT available is `setupLookaside`'s own MIR, which this host cannot
  produce.
* Anything about the workload. A candidate fix still has to be scanned with
  `movc-cfg-scan.py` over a real Sublet cell ⑥ build (`run-speedtest1-measure.sh` with
  `SPEEDTEST1_SUBLET=1` — handover §8), on a host that can build it. Apollo cannot.
* Anything about the RTL half. That `movc` nulls a non-NONLIN source is unchanged.

## What follows, for the lead's decision

**Design A did exactly what the decision table said it would do, including leaving exactly what the
table said it would leave.** The table's own entry for it reads "leaves **PHI copies**, which remat
cannot reach"; the live site is a PHI copy. Nothing misbehaved. What was not anticipated is that the
documented gap would turn out to be the live case — so the design question is not "why did design A
miss this", it is: **the residue design A was accepted WITH is the live silicon site, so accepting it
is no longer separable from leaving C-32 open.** Design A remains a real improvement — it removes the copies on defs all of whose uses conform.
Whether it can be *extended* is deliberately left open here, because an earlier draft asserted it
could not be and that does not follow from anything measured. All-or-nothing is a property of
upstream's current `PerformSinkAndFold`, not a law: "fold the conforming uses and leave the
non-conforming ones" is a conceivable tuning, and it would remove `0x267a0` — the site the harm
story actually runs through — while leaving the merged-value copies at `0x26a28`/`0x26a54`. This
note does not propose it, has not costed it, and flags that it would change an upstream CodeGen
pass's contract rather than Capstone-local code. It is raised only so the option is not dismissed
by a sentence.

Three routes, for the lead, not for this lane to pick:

1. **The register-class route** — the only recorded option whose entry says it "would cover PHI
   copies too", which is now the property that matters. Its recorded objection is structural and is
   NOT disturbed by anything here: untagged is a property of the value, a `RegisterClass` is a set of
   physical registers, and `copyPhysReg` sees only physical numbers. So this is a request to
   re-examine that objection on its own merits, not a claim that its premise has fallen.
2. **Make the bridged value not a GPCR value at all** until it is genuinely used as a
   capability, so a GPCR copy of it cannot be formed. Larger, and touches the ABI boundary.
3. **Attack the copy rather than the value** — i.e. a lowering where a GPCR copy of an
   untagged value is not `movc`. This is C-46's box (`MOVC`'s definition is untouched and
   must stay so), so it is a hardware-side conversation, not a codegen one.

**Two facts that narrow this, both read from the source today rather than assumed:**

* **The register-class objection is real as stated.** `CapstoneInstrInfo::copyPhysReg`
  (`CapstoneInstrInfo.cpp:513`) takes `Register DstReg, Register SrcReg` — physical registers — and
  decides by class membership, `if (Capstone::GPCRRegClass.contains(DstReg, SrcReg))`. There is no
  value-level information at that point, so "a register class marking a bridged integer" would need
  a disjoint set of *physical* registers for bridged integers, which the X/C aliasing makes a much
  larger change than it sounds. Route 1 is not cheap; it is the one that covers the shape.
* **The obvious instruction swap is already ruled out, in that function's own comment.**
  `cincoffsetimm rd, rs, 0` "would be the non-destructive alternative and it faults outright on an
  untagged source, which is worse." So route 3 cannot be done by picking a different existing
  opcode. The same comment already anticipates this exact case — "An untagged value can sit in a
  capability register after an inttoptr, so that case is not impossible here" — which is C-32,
  written down before it was observed.

No fix is proposed here on purpose: the handover's §10 closes with "do not skip to a fix",
and the three wrong answers it records were all plausible ones.
