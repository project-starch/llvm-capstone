# Four open questions closed by simulation, and one false alarm I raised about myself

**Date:** 2026-08-21
**Method:** each run in a detached worktree at the revision under test, with the test source
copied verbatim from `f231b5af0` so both sides run identical binaries, and **each run carrying
its own model-identity control** — without that, identical-looking numbers are indistinguishable
from a worktree that silently reused another model.

## 1. S-10 alone closes the write-buffer residual — PROVEN, was PLAUSIBLE-UNPROVEN

Worktree at `4fee13b2d` (S-10 only, no S-10b):

```
s07-wbuf-forward-residual        SUCCESS  9371 cyc   17 exceptions    (shipped tree: 9)
s07-wbuf-forward-residual-ctl    SUCCESS 26361 cyc   17 exceptions    (control, pinned)
s07-wbuf-tag-reorder             SUCCESS  9138 cyc    1 exception     <- identity control
```

17 is the ceiling — sixteen legs plus the baseline — so every leg traps and the window is shut.
The control sits at 17 exactly as on the shipped tree, so the single variable is S-10. The
identity control returns the **post-fix** signature (1 exception at 9138) rather than the pre-fix
one (4 at 9150), so the model really was built at this revision.

**Consequence: S-10b was never needed for this.** S-10b cannot be built at all (`DRC LUTLP-1`),
while S-10 is the arm that reached `write_bitstream` with exit 0 at `80843404c`. The 8-of-16 leak
on the currently flashed bitstream has a synthesis-proven closer one commit away.

## 2. S-07 does not worsen the residual — found already measured, not re-run

The test's own header at `4fee13b2d` records identical binaries against pre-fix RTL at
`a3dbae618`:

```
                                pre-fix a3dbae618      fixed
s07-wbuf-forward-residual       9 exc /  9234 cyc     9 exc /  9234 cyc
s07-wbuf-forward-residual-ctl  17 exc / 26361 cyc    17 exc / 26361 cyc
```

Identical in both arms, with its own identity control (`s07-wbuf-tag-reorder` 4 exc pre-fix
against 1 fixed). **"Strict improvement" survives.** Re-running this would have been a wasted
build; the answer was already in the tree.

## 3. Pre-fix controls for the three security tests — one gained, one is not evidence, one was aimed at the wrong revision

Worktree at `013e162fd` (pre-S-06), tests copied verbatim from `f231b5af0`
(`asm_insn.h` is identical between the two, so they are portable):

```
s06sec-raw-alias-no-launder   FAILED tohost=4   697 cyc     -> SUCCESS 723 after   PROVEN CONTROL
s06sec-csr-raw-no-forge       SUCCESS           511 cyc     -> SUCCESS 511 after   NOT EVIDENCE
s06sec-ctx-scalar-roundtrip   SUCCESS           589 cyc     -> see below           WRONG REFERENCE
s06-lowhalf-zero              FAILED            729 cyc                            IDENTITY CONTROL
```

The identity control matches the committed baseline row `s06-lowhalf-zero FAIL 729` exactly, so
the worktree was genuinely the pre-fix model.

**`s06sec-csr-raw-no-forge` passes on both sides at the identical cycle count.** It does not
demonstrate the S-06 fix. Either it is a non-regression check or its triggering condition was
never created. That should be said wherever it is cited.

**`s06sec-ctx-scalar-roundtrip` was aimed at the wrong revision by me.** It is the S-08 test, and
S-08 is a regression *introduced* by S-06 P4 — so pre-S-06 RTL passes it and proves nothing. The
correct reference is `25035c4c0`, the commit immediately before the S-08 fix:

```
25035c4c0  post-S-06, PRE-S-08   FAILED tohost=3   612 cyc
f231b5af0  post-S-08, shipped    SUCCESS           592 cyc   (committed sweep row)
s06-lowhalf-zero @25035c4c0      SUCCESS           731 cyc   IDENTITY CONTROL (FAILED 729 pre-S-06)
```

A proper matched pair, both halves sourced.

## 4. The capability-mint test was NOT written, and that is the result

See `21-08-2026_02-10-00_...md`. Domains run at M privilege with capmode on, and the four minting
ops are register-only, so neither privilege nor CPMP constrains them — and a bare-metal M-mode
test has no authority boundary to violate, so it would have returned clean having created
nothing. **Not spending a run on a test that cannot fail.**

## The false alarm, recorded because the shape matters

While checking the S-08 message I could not find its "612 cycles FAIL / 592 PASS" figures in the
history note or in the original commit body, and concluded I had fabricated them in a commit
bound for a shared branch. **I had not.** 612 was measured minutes later at `25035c4c0`, and 592
is a committed sweep row in `s07-strip.txt`. Both were real; I had grepped two files, found
nothing, and treated absence as proof.

That is the project's own standing lesson — *a zero from a matcher means nothing until the
matcher is shown able to fire on the set being counted* — applied to myself and failed. The
correct move when a citation cannot be found is to widen the search or measure it, not to assume
invention. Raising a false alarm about fabricated evidence is not free: it would have discredited
a correct commit message.

## Net effect on the branch

`fpga-testing-dev` messages for S-06 and S-08 were rewritten to carry the measured controls and,
just as importantly, to name what is **not** evidence. Trees unchanged — `core/` and `corev_apu/`
remain byte-identical to `f231b5af0`, verified after the rebuild.
