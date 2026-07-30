# C-14 fix proposal — the compiler uses `movc` (a MOVE) for scalar copies

**Status:** proposal, awaiting a decision. Nothing implemented.
**Root cause:** see `ref/ISSUES.md` C-14.

> **REVISED TWICE ON 2026-07-30 -- see `ref/ISSUES.md` C-14 for the full trail.** The
> current position is: the spec is UNDER-SPECIFIED for MOVC with a scalar source, and the
> weight of evidence favours scalars being EXEMPT (`mem-access-insn.adoc:45` glosses
> `type != 1` as "scalar or non-linear"; `:105` writes an explicit "is a capability" guard
> for STC; the MOVC clause predates scalars being allowed at all, spec commit `a1db3c2`;
> QEMU's guard is deliberate, commit `b9c53f0d09` "movc allows scalars"; and the RTL's own
> STC exempts scalars while its MOVC does not). So the RTL's MOVC is probably an oversight
> -- but that goes to the board owner as a QUESTION about which behaviour is normative,
> never as an accusation.
>
> Unchanged throughout: the mechanism, the numeric proof, and that LLVM emits the wrong
> instruction. Option C is the plan regardless of how the spec question resolves.

## The constraint

RTL `movc rd, rs` nulls `rs` unless `rs` is a `NONLIN` capability
(`capstone_flu_unit.anvil:13-21`). QEMU only nulls it when `rs` is a *tagged*
non-copyable capability (`op_helper.c:580-584`), so scalars survive under QEMU and die on
silicon. `copyPhysReg` emits `MOVC` for every GPR-to-GPR copy
(`CapstoneInstrInfo.cpp:520-523`), so this reaches ordinary compiled code everywhere.

No single instruction is a safe universal copy:

| candidate | scalar source | capability source |
|---|---|---|
| `addi rd, rs, 0` | correct, preserves source | drops metadata -- not a capability copy |
| `movc rd, rs` | **destroys source** | correct for NONLIN; destroys others |
| `cincoffset rd, rs, x0` | RTL preserves rs1, QEMU nulls it (C-4b) | same divergence |
| `cincoffsetimm rd, rs, 0` | traps UNEXPECTED_OPERAND on NOT_CAP | works, but see above |

`copyPhysReg` runs after register allocation and sees only `GPR`, which holds both
scalars and capabilities. It therefore cannot choose correctly today. **That is the whole
problem** -- everything below is a way of getting the type to the copy.

## Options

### A. Separate register classes for capabilities and scalars
The principled fix. Give capabilities their own class so `copyPhysReg` dispatches on
`RegClass.contains(...)` exactly as it already does for `GPRF16`/`GPRF32`
(`CapstoneInstrInfo.cpp:526-536`). Scalar copies become `ADDI`, capability copies stay
`MOVC`.

- Correct by construction, and fixes every future copy site, not just loops.
- Largest change: affects ISel patterns, calling convention, spill/reload, and the
  gp-captable lowering. Not a same-week change, and it touches the ABI the board
  firmware already agrees with.

### B. A copy pseudo selected by type at ISel, expanded late
Emit `PseudoCopyScalar` / `PseudoCopyCap` where the DAG still knows i64 vs i128, and
expand them after RA. `copyPhysReg` keeps MOVC only for the cases it genuinely cannot
attribute.

- Much smaller than A and reversible.
- Does not cover copies the register allocator itself invents (spills, coalescing
  artefacts, two-address fixups) -- which is exactly where the failing `movc a4, a6`
  came from. **This is the main risk and needs checking before committing to it.**

### C. Targeted late peephole: rewrite `MOVC rd, rs` where `rs` is live-after
A MachineFunction pass over the post-RA output: for each `MOVC rd, rs` whose `rs` is live
after the instruction, substitute a safe form.

- Smallest, most auditable, and directly aimed at the measured failures.
- Still needs the scalar/capability distinction to pick the substitute. If `rs` is live
  after the copy, it cannot be a LINEAR capability (duplicating one is illegal anyway), so
  it is either a scalar or NONLIN -- and MOVC is already safe for NONLIN. So the pass only
  has to be right about "is this a scalar", which liveness plus the defining instruction
  can usually answer.
- Verification is cheap: `tests/runtime-qemu/silicon-ladder/check-movc-reuse.py` already
  classifies all 13 measured rungs correctly and reports 444 sites in SQLite.

### D. Fix QEMU to match the spec (NOT the RTL)
QEMU's `helper_csmovc` adds an `rs1_v->tag &&` conjunct that the spec does not have, so it
silently preserves scalar sources that a conforming core destroys. Removing that conjunct
makes QEMU spec-accurate and turns this entire bug class into something the model can
catch before it ever reaches the board -- which is the real long-term win, because the
same blind spot hid DELIN and the bounds-compression divergence. Worth raising regardless of what we do in the compiler, because the
divergence will bite any future toolchain.

- Cheap, entirely in our tree, and no board time. It will make some currently-green QEMU
  tests fail -- that is the point, those are the latent silicon bugs.
- Still worth reporting to the board owner separately: `check_fwd_rs1`
  (`ariane_pkg.sv:925-931`) lists `{SPLIT, MOVC, CJALR, CCSRRW, STC}` and is **dead code**,
  while the rs1 write-back is gated on the much broader `check_cap_op`
  (`commit_stage.sv:278-281`). That widening is unrelated to C-14's attribution but looks
  unintended.

## Is `movc` consuming its source a bug at all?

**No. It is the specified behaviour, and the RTL implements it correctly.**

`capstone-spec/parts/cap-man-insn.adoc:33-37`:

    * If `rs1 = rd`, the instruction is a no-op.
    * Otherwise
    . Write `x[rs1]` to `x[rd]`.
    . If `x[rs1]` is not a non-linear capability (i.e., `type != 1`),
      write `cnull` to `x[rs1]`.

Types are `0` linear, `1` non-linear, `3` uninitialised, `5` sealed-return
(`parts/existing-insn.adoc:60-65`). A plain scalar is not a non-linear capability, so
`type != 1` holds and the source MUST be zeroed. `parts/intro.adoc:59-61` gives the
intent: instructions "can only **move**, but not copy, linear capabilities between
general-purpose registers." MOVC is a MOVE.

Our own C3 note (`capstone-qemu/tests/capstone-mrev-codegen/README.md:117-124`) already
called this "correct linear-capability semantics"; what nobody checked was that it applies
to scalars too.

So the attribution is:
* **RTL: conforming.** Do not patch it.
* **QEMU: deviates.** `helper_csmovc` adds an `rs1_v->tag &&` conjunct the spec does not
  have, exempting scalars. That deviation is why every one of these bugs is QEMU-green.
* **LLVM: the bug.** `copyPhysReg` emits a MOVE where a COPY was meant. Wrong on any
  conforming implementation, independent of this board.

Option C is therefore **not a workaround for a hardware defect** -- it is the correctness
fix the compiler needs regardless of which core runs the code.

## If we CAN patch the RTL and reflash, is C still best?

**Yes -- and stronger than before: patching the RTL is now the wrong thing to do at all,
not merely the expensive thing.** The spec mandates the current behaviour, so a patch
would take the board out of conformance and make our measurements describe a core nobody
else has. The reasons below were written when this looked like a hardware bug; they remain
true as secondary costs, but conformance is now the decisive argument.

* **It invalidates the measured perf numbers.** Every silicon cycle count we have was taken
  on the current bitstream. A new bitstream means either re-measuring the whole ladder or
  publishing a table that mixes two hardware revisions. With the deadline on 2026-08-02
  there is not enough board time to re-run the set.
* **Reflashing is a hard stop** requiring explicit human approval, and needs the
  Anvil -> SystemVerilog -> Vivado flow, which is hours and is not currently set up here.
* **It changes what we are measuring.** A patched core is no longer the Capstone hardware
  the paper describes. Defensible if documented, but that is the project lead's framing
  call, not a lane's.
* **C costs nothing in performance.** `addi rd, rs, 0` replaces `movc rd, rs` one-for-one,
  same instruction count, so the existing measurements stay comparable. This is the
  decisive practical point: the workaround does not distort the numbers the paper reports.
* **Both fixes address exactly the same defect.** Neither changes the C3 behaviour for
  genuine capability copies, which is intended. So the RTL fix buys correctness for other
  toolchains, not extra coverage for us.

The one real argument for the RTL fix is **completeness**: a compiler fix can miss copies
the register allocator invents, whereas the hardware fix is total. That is measurable
rather than speculative -- `check-movc-reuse.py` currently reports 444 sites in SQLite, so
if the compiler fix drives that to 0 the coverage question is answered without a reflash.

**Recommended order:** C now; D (report to the board owner) in parallel, since it costs one
short message and fixes it properly for everyone; the RTL patch + reflash only after the
deadline, or sooner if C turns out to be incomplete. If a reflash does happen, budget board
time to re-run the full ladder, not just the failing rungs.

## Recommendation

**C now, D in parallel, A or B after the deadline.** C is the only one that can be
written, QEMU-gated and board-validated in the time available, and it has a ready-made
verification gate. D costs one short message and may fix it properly for everyone.

## Validation plan for whichever is chosen

1. `check-movc-reuse.py` reports 0 on gpw16, gpn2, gpn4, gpw2 (currently 1 each).
2. Rebuild and re-run those four plus the `beebs_primer1` control in ONE board session.
   Expect all to return their exact oracles; `gpw2`'s true oracle is 3983810698, not the
   `%u` placeholder its old file contained.
3. Only then rebuild SQLite and re-run it.
4. Do not touch the paper's SQLite claim until 3 passes -- it currently says SQLite has
   not run on the board, which remains accurate.
