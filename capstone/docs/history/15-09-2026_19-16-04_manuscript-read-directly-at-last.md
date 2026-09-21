# The manuscript, read directly at last: what it actually says (2026-09-15, evening)

*The paper repository became readable from apollo this evening — the lead rotated the GitHub token
and the 403 that had blocked every agent session on both hosts is gone. Everything below is read
from `origin/main` of `capstone/paper-nested-allocators`, not relayed. It supersedes the "pending
claim" markers in `15-09-2026_16-37-49_paper-lane-apollo-handover-checks.md` §0 and §1b.*

**The gitlink was not moved and no submodule pointer was bumped.** All reads were `git show
origin/main:<path>` against a fetched remote.

## 1. The remote's state, which the handover's account no longer describes

| | |
|---|---|
| `main` | **`464a5d5a`** |
| handover's `7f83725` | **an ancestor of `main`** — nothing was rewritten, the remote advanced 8 commits |
| `drafts` | **does not exist any more**; the branches are `main` and `paper/best-case-draft` |
| `board/e1-s1s2-hardware` | **not on the remote** — confirming the board lane |
| superproject gitlink `b0d7510` | **28 commits behind** `main` |

So the handover's "`main` and `drafts` are the same commit" is obsolete rather than wrong, and the
branch carrying the E1/R1/M2/H1 bundles has still never been pushed. It exists only in the
focs-server checkout, together with the 2.8 MB bundle built to carry it. **Neither is on apollo, so
this lane cannot push or send it** — that request has to go to focs-server or to the lead.

## 2. The live contradiction, verified end to end — and one nuance nobody reported

`appendices/c-validation-and-accounting.tex` does carry the four rows, verbatim:

> Stale lookaside access after finalize & Stops at access
> Stale backing-block access after free & Stops at access
> Stale pool access after destroy & Stops at access
> Same address, new object, old pointer & Stops at access, the plain arm reads the new object

and the prose below it:

> "Reading through the old pointer returns `\safeReuseByteNew` on the unprotected arm, the first
> byte of the new object, rather than the old object's `\safeReuseByteOld`. **The protected arm
> stops at the read without returning a byte.**"

`macros/` defines **`\safeReuseByteNew` as `0x5B`** and `\safeReuseByteOld` as `0xA0`. On silicon the
**protected** arm returns `0x5B` (§7r `:3460`, `:3558`, six `unsafe-success` records at `:3596`).
So the silicon reading is the value the prose assigns to the *unprotected* arm. The audit's §1 is
confirmed, now against the primary source rather than on report.

**The nuance, which changes the shape of the question.** `tab:safety`'s caption reads:

> "Measured baseline security evidence on the **pinned emulator**."

The table is **explicitly scoped to the emulator**. It is therefore not internally false — it is
*silent* about silicon. That reframes the lead's decision 1 from *"the table is contradicted"* to:

> **an emulator-scoped safety claim stands beside a silicon enforcement gap — does the publication
> rule fire?**

The rule is at `experiments/METHODS.md:89` (the handover said `:86`; line drift, same sentence):
*"An enforcement or oracle failure stops dependent performance publication."* Whether six
unsafe-success cells on silicon are an "enforcement failure" in that sense, when the table they
would contradict claims only the emulator, is the decision. It is still the lead's, and this note
does not answer it — it sharpens it.

## 3. The evidence states, and why this lane still cannot edit them

Read from `experiments/studies.json` at `main`: **R1 `pending`**, **M2 `pending`**, **P1 already
`partial`**. Exactly as the handover predicted — the outgoing lane's P1 edit was redundant and its
R1 edit was stranded, and M2's move is owed on a bundle that is committed but unpushed.

**The two-file constraint is real and now verified.** `scripts/check_experiments.py:267-268` guards
the appendix cross-check with `if appendix.is_file()`, and `appendices/a-evidence-status.tex`
**does exist** at `main`. So the guard is live and an evidence-state edit requires the appendix rows
to match the catalogue — a **two-file change whose second file is manuscript**. The lead's go-ahead
covers `studies.json` only.

**One new input to that decision, which nobody could see before today.** `experiments/EXECUTION.md`
closes its bundle specification with:

> "The **paper editor** integrates accepted evidence into the study index, manuscript, and appendix
> and publishes manuscript updates to Overleaf as well as GitHub."

The process document already contemplates a role that does exactly this two-file edit. That does not
grant the permission — `CLAUDE.md`'s rule is the lead's — but it changes the question from *"may a
lane touch the appendix"* to *"is this lane the paper editor"*, which is a cheaper question to
answer.

## 4. The bundle shape, from source

`experiments/EXECUTION.md` requires **seven** items: `work-order.md` (named scope, operator **and
reviewer**, exact commands, approved gates and resource limits), `manifest.json`, `points.csv`
(**every planned cell including unsupported cells**), `runs.jsonl`, `raw/` or durable raw-file
references (**with hashes, no cropped success-only logs**), `analysis/`, and `summary.md` (what ran,
what did not, scope, limits, **claims supported or narrowed**).

The existing six-file bundles are missing **`work-order.md`**, which matches the handover's note that
the E1/R1/H1/M2 work orders were "stated rather than backfilled".

## 5. What the manuscript most wants from RTL, answered from the manuscript

Asked by the RTL lane, who could rank by engineering risk but not by paper value.

**R-12 reclamation, and not close.** `\targetReclaimer` is a `\targetresult` macro — it renders as a
coloured ⟨R⟩, so the paper *visibly* admits the gap. `appendices/b-target-results.tex:33-36`:
*"Implemented reclamation algorithm, reference-accounting invariant, completion event and finite
progress or generation limits. **Currently unresolved, not a name for an already implemented
reclaimer.**"*

The decisive evidence is that an entire paragraph already exists and cannot be typeset with values —
`sections/evaluation/05-resource-cost.tex`, "Retained-reference pressure and resumed progress":

> "It reaches `\targetPinned` retained nodes at the observed pressure limit. **After those
> references are cleared and `\targetReclaimer` completes, the release-retained arm makes another
> `\targetResume` allocations.**"

With no reclaimer, nothing completes and no allocations resume: the clause has no referent. The same
paragraph needs `\targetDropPeak`, `\targetRingPeak`, `\targetResurrections` and `\targetReuseChecks`,
and closes by naming its own blind spot — *"the identifier-reuse case that the unreclaimed baseline
cannot exercise."*

**And it may touch a performance headline, not only a capacity limit.** The board lane measured P1 on
silicon today: `give_cyc/n` grows **superlinearly** with table occupancy (221 → 1558 cycles, sd 7-12,
⚠ **BOTH HALVES OF THIS PARENTHESIS ARE WRONG — corrected 2026-09-21, see §12.** The variable is
per-domain minting, **not** table occupancy (`docs/history/15-09-2026_19-25-14_m1-revoke-cost-variable.md`
refutes it by overlay). And `221 → 1558` appears nowhere but this line — the primary document gives
**176.4 → 2114.7** for the same arm. The rise itself was real, on the **superseded** bitstream
`1bfff7776`; on the resident `054cea69b` it is gone.⚠
flat at 16 under `-icount`). A reclaimer that keeps the table smaller plausibly moves that curve. It
could complicate P1 rather than improve it, which is worth knowing before the build rather than
after.

**R-33 / R-11 bear on no current claim.** `sections/` and `appendices/` at `main` contain **zero**
occurrences of representability, out-of-bounds, over-permissive or granule. The fix adds no sentence.
But a demonstrated store outside its allocation retiring without fault does not *add* a claim, it
**threatens** one — `tab:safety` is the paper's security evidence and is what decision 1 is about. So
R-33 belongs on a different axis: rank it by whether the over-permissive store can reach a cell in
that table, not by paper value.

**R-10 tag integrity: no.** The only tag references in the manuscript are `\targetTagStorage` (bytes
of tag storage) and one validation-appendix line about an operand carrying address bits but no
capability tag. Nothing rests on tag integrity in a way the refill OR-reduce touches.

**R-29, R-26, R-27, R-32: not checked against the manuscript, and not guessed at.** Now that the
repository is readable, each takes minutes given a one-line symptom.

## 6. Which RTL defects touch a manuscript claim — four grepped, and two cells named

Asked by the RTL lane for their plan. All greps are against `sections/` and `appendices/` at
`origin/main`.

**None of the four touches a claim by name.**

| defect | symptom (the RTL lane's words) | manuscript hits |
|---|---|---|
| **R-32** | spec and RTL disagree by one on every bound taken or returned as a **value**; `SHRINKTO` a genuine RTL off-by-one | **zero** — no `shrinkto`, no `tighten`, in any case |
| **R-29** | plain 8-byte `sd` into the high word of a 16-byte granule, then an untagged `ldc`, returns the high half zeroed | **zero** — no granule, 16-byte or LDC |
| **R-26** | CCSRRW stale read (recollection, not citation) | **zero** |
| **R-27** | rev-node orphan deadlock (recollection, not citation) | **zero** |

Two near-misses checked and rejected, recorded so nobody re-derives them:
`appendices/a-evidence-status.tex:64` (*"a stale free is declined by comparison rather than caught,
which **bounds** the claim"*) and `:186` (*"identify weaker block **bounds**"*) both use "bounds" in
the scoping sense — a limit on a claim, and future work on block-size bounds for the nginx replay.
Neither is a capability bound.

### The indirect exposure, which is real: two cells assert EXTENT

R-33's over-permissive store (and, on the same axis, R-32's off-by-one) would not *add* a claim but
could *invalidate* one. The only cells in `tab:safety` that assert anything about extent are

* `appendices/c-validation-and-accounting.tex:41` — **"Inner free preserves live sibling & Sibling
  intact, block extent unchanged"**
* `:42` — "Sibling survives uncooperative child & Sibling still reads and writes"

with the prose at `:66-69` making it explicit: *"Inner release preserves the sibling's data **and the
block extent** … preserves the sibling **and parent extent** …"*. R-33's over-permission is a
representability **rounding**, so the window is up to the granule rather than one byte — a store
landing in an adjacent sibling is exactly "block extent unchanged" being false while the discipline
reports success.

**The discriminating question is cheap and decides it: does the over-permissive store reproduce on
the PINNED EMULATOR, or only on silicon?** The table is emulator-scoped (§2), so:

* **reproduces on the emulator** → the cell is wrong within its own scope, and a claim is
  invalidated;
* **silicon only** → the cell stays true of what it claims, and this widens the same
  emulator-vs-silicon gap decision 1 already turns on, with `METHODS.md:89` governing instead.

Either answer is publishable-relevant, so the reading qualifies under the go/no-go rule. What
settles it first is whether the fixture places siblings **adjacently within one rounded region** — a
fixture property not readable from the paper, and if siblings are never adjacent the question closes
without touching the allocator at all.

**So R-32 and R-33 converge on one question rather than being two** — the same shape the RTL lane
found when R-33 turned out to close R-11.

### R-29's fix costs a combinational loop, and the paper gives no reason to pay it

Reported by the RTL lane from the registry survey: a written fix exists,
`r29-granule-data-overlay` at `00e89d968`, functionally correct, and it **fails the lint gate** —
`UNOPTFLAT` 40 → 41, a new combinational loop. Their entry says the fork is explicitly not a lane's
call, so it is a question for the lead. It also gates R-10, whose live half is **not independent**:
R-29's separation arms decide whether the OR-reduce is in the path at all, and if it is, R-10 closes
with R-29's fix rather than on its own.

**This is the edit class `CLAUDE.md` singles out as the most dangerous available** — *"Never feed a
new signal into a cone that already carries a combinational loop… every check we have is blind to
it. If a change does that, it goes to synthesis before it goes anywhere else."* So accepting it
means spending a synthesis cycle to find out whether it is synthesizable at all.

**The paper's input to that decision is one-sided and worth stating plainly: R-29 changes no
sentence, and neither does R-10.** Both greps returned zero (§6 above). So the whole R-29 → R-10
chain is **paper-neutral**, and there is no manuscript claim that would be weakened by deferring it
or strengthened by landing it.

That is not an argument against fixing a correctness defect — wrong data returned from a load is
worth fixing on its own terms, and the paper's silence is not an endorsement of the bug. It is an
argument about **ordering under a scarce resource**: a bitstream costs ~90 minutes plus a reflash,
the reclaimer (§5) converts a typeset hole into a measured paragraph, and R-29's fix would spend the
same cycle on the highest-risk edit class we have for a defect the manuscript never cites. If the
lead is choosing what rides the next build, the paper points at W1 and says nothing in R-29's
favour.

### A pattern worth naming: this registry carries duplicates that only surface from the claim side

The RTL lane observed it after the second instance in one day: R-33 turned out to close R-11 (one
representability contract seen from the two branches of `compress_bounds`), and R-33 and R-32 turn
out to be one question (both can only falsify the same two extent cells). Their formulation is the
useful one — **the duplicates surface when you ask "which sentence does this threaten" rather than
"what is the mechanism"**, because two mechanisms that differ can still have exactly one consequence.

That is the same family as the project's standing rule to search prior art and read *past* the
root-cause box, and it suggests a cheap habit for this lane specifically: when a new defect arrives,
grep the manuscript **before** reading the mechanism. A zero-hit grep costs a minute and either
closes the paper's interest or names the cell the mechanism has to reach.

## 7. W2 closed: R-33 cannot reach `tab:safety` — confirmed, strengthened, and one figure corrected

The RTL lane answered the question §6 posed and closed it. Verified here rather than accepted,
because a clean result that closes a **safety** question is the exact shape this project has been
burned by.

**Their conclusion is right.** `compress_bounds` never widens a region this fixture creates, so no
store can land in the adjacent sibling by R-33's mechanism, and `c-validation-and-accounting.tex:41`
and `:42` hold.

**Their step 1 came back the "bad" way and it does not matter.** Siblings *are* adjacent, on purpose
— `ports/nginx/port/ngx_subpool_test.c`, phase 13: *"The sibling is taken FIRST and from the same
arena, so that it is a neighbour of the nest rather than something allocated after the dust
settled."* Confirmed verbatim in the file. Adjacency does not close the question; representability
does.

**The granule arithmetic reproduces exactly.** `granule(len) = 2^(max(0, bit_length(len)-13) + 3)`,
so it is 8 for every length below 8192, and every size the fixture asks for is a multiple of 8 —
including the one odd-looking 4280, which is 535 x 8.

### The reason is STRUCTURAL, not fixture-dependent — a stronger closure than the one argued

The lane's argument enumerates the sizes the fixture happens to use, which is fragile: a fixture
change would require re-checking, and two numbers in the file are not in their list (`30`, which is
a **shift count** in `(size_t) 1 << 30` on a call asserted to *fail*, and `4288`, which is a granted
extent rather than a request). A stronger argument is available from the allocator itself:

**`ports/nginx/port/ngx_subpool.c:86` — `bytes = (bytes + 15) & ~(size_t) 15;`** — rounds **every**
request up to 16 bytes, unconditionally, and its comment says why: otherwise *"EVERY later block is
misaligned and the first capability stored in one is an unaligned access."* The fixture asserts this
behaviour directly at phase 8: a 4280-byte request yields extent **4288**, commented *"rounded, not
as asked"*.

So the relevant quantity is the **granted extent, not the requested size**, and every granted extent
is a multiple of 16. For any extent below 16384 the granule is 8 or 16, both of which divide 16.
**Widening therefore cannot occur at these sizes for any request whatsoever**, not merely for the
ones this fixture chooses.

### The margin is 3.8x, not 1024x

The lane reported the first size where R-33 bites as 4 MiB, "1024x larger" than the fixture's 4096.
That is the threshold at which the granule exceeds *page* alignment, which is a different question.
The threshold that matters here is where the granule exceeds the allocator's own **16-byte**
rounding, i.e. granule > 16, i.e. `bit_length(extent) >= 15`:

| | |
|---|---|
| first request whose granted extent is not an exact multiple of its granule | **16385** |
| its granted extent / granule / remainder | 16400 / 32 / 16 |
| fixture's largest region | 4288 |
| **actual headroom** | **3.8x**, not 1024x |

*(**Corrected**: this table first said 16392. My scan stepped through requests in steps of **8**,
so it could only ever report a multiple of 8 — I had sampled the way the fixture's sizes happen to
look, which assumes the conclusion in the sampling. The RTL lane rescanned and got **16385**, which
is right: requests 16385..16400 all round to extent 16400. The extent, granule, remainder and the
3.8x headroom were unaffected. A step size chosen to match the data you expect is the same family of
instrument error as the rest of this note, and it is mine.)*

The conclusion is unaffected — 3.8x is still real headroom and the cells still hold. But the
correction matters for the sentence it was used to support, *"this is not a near miss that a fixture
change could tip"*: an nginx port allocating a 16 KiB buffer would sit at the boundary, and 1024x
would tell a reader not to think about it again. **Structural is the reason to stop worrying;
1024x is not.**

**Step 2 is moot**, as the lane says: the emulator-versus-silicon discriminator this lane proposed
would have decided it only if the precondition ever occurred, and it does not occur on either
platform, because it is a property of sizes and allocator rounding rather than of hardware. So this
does not widen the gap decision 1 turns on either.

**Residual, theirs, stated not buried:** the *base* half of R-33 — whether an allocator can hand out
a non-granule-aligned base — is not established in general. Here the bases are 8-aligned if the root
arena base is, and any arena holding capabilities is at least 16-byte aligned, so it is safe by
argument rather than by measurement.

**R-32 drops off the paper's critical path with it**, by the same reasoning: its exposure was the
same two extent cells, and an off-by-one cannot falsify a cell whose regions are exactly
representable. Both stay open as soundness defects; neither threatens a claim.

## 8. The revoke-walk slope has a placeholder waiting for it — and the paper's own spec already guards it

The RTL lane quantified the revoke walk: **unspliced costs exactly 3.000 cycles per dead node
crossed** (`cost = 3N + 249`, exact at six consecutive rungs N=8..1024), **spliced exactly 0.000**
(328 cycles at six different N). Fixed overhead ~79, crossover N ≈ 26, and at N=3072 it is 96,822
against 516.

**Where that number would land in the manuscript, and why it must not land bare.** The paper does
**not** state the numeric release model — `sections/evaluation/04-release-cost.tex` says *"revocation
work has slope `\targetNodeSlope` cycles per affected node and initialization has slope
`\targetByteSlope` cycles per byte"*. Both are unfilled `\targetresult` placeholders
(`macros/`: `a_n` and `a_b`, commented *"R1: fitted local cycles per affected node"*). So there is a
**labelled hole shaped exactly like this number**, and the obvious move — fill `\targetNodeSlope`
with 3 — is the one that must not happen unqualified.

**The reason is the caveat, not the figure.** 3.000 is an **L1-hit** cost and therefore a **lower
bound** on what splicing saves: the dcache holds exactly 2048 nodes, the fit holds to 1024 and
breaks at 2048, and a speedtest1 run mints **43,355** — **21x the dcache**. The real workload sits
entirely in the cold regime the ladder reaches only at its last rung, where the slope rests on a
single segment.

**The paper's own machinery already forbids the misuse, which is worth knowing before anyone argues
about it.** `appendices/b-target-results.tex:83-86` gives the acceptance criterion for exactly these
two macros: *"Local fitted cycles per affected node and initialization cycles per byte over at least
three legal levels. **State fit range and error. Neither is a universal architectural constant.**"*
So filling `\targetNodeSlope` with a bare 3 would violate the target's own spec. The guard exists;
it needs honouring rather than inventing.

**And it is splicing that buys this, not generation tagging** — different changes with different
payoffs, which matters because a gate written against "the reclaimer" is ambiguous between them
(the handover's decision 4 makes the same point).

This is also the variable the handover said R1's model never varied: `\approx 1.81\cdot B + 22.7\cdot nd + 58.9\cdot n`
was established with **topology held fixed**, and cumulative revocations within a domain is what
moves it. It is now a number rather than an argument.

### A label correction that touches no manuscript text

`S12_MEM_DELAY` is **not a cycle count**: `stream_delay.sv` counts in 4 bits, so the parameter
truncates to its low four bits and the widely-used **40 realises as 8**. Checked here: the
measurements doc contains no occurrence of `S12_MEM_DELAY`, `MEM_DELAY`, "40-cycle" or "delay 40",
and neither does the manuscript. The only occurrences in this lane's own notes were two lines in
§6e of the sibling note, now corrected in place. **It is a magnitude label, not a retraction** —
S-12, R-26 and R-34 all stand.

One place that cannot be fixed by a lane: `CLAUDE.md:449` carries *"delay 0 → 0 traps, delay 40 →
254"* inside the rule about a synthetic test needing to create its triggering condition. The rule is
right and the instance is right; only the magnitude is wrong. That file is the project lead's and
has been flagged to them rather than edited. Anyone citing that rule should read 40 as 8.

## 9. Four result bundles reviewed on the paper remote — and a packaging hole that generalises

`results/m1-bounded-baseline` at `ef779f9`, built by `apollo-board`. Reviewed rather than accepted;
every check below was run here against the fetched branch.

**Shape: all four carry the full seven entries.** `H1/2026-09-15-apollo-handover`,
`M1/2026-09-15-baseline-maxret2048`, `M1/2026-09-16-baseline-maxret4096`,
`M1/2026-09-16-live-calibration` — each has `work-order.md`, `manifest.json`, `points.csv`,
`runs.jsonl`, `raw/`, `analysis/` and `summary.md`. Including the one whose raw evidence was lost,
which keeps all seven with `raw/README.md` and `analysis/README.md` standing in and every
`points.csv` row marked unsupported. That is the right call: a bundle that records its own loss is
worth more than a gap.

**Isolation: verified, not taken on trust.** `git diff origin/main...ef779f9` outside
`experiments/results/` is **empty**, and `experiments/studies.json`, `appendices/`, `sections/` and
`macros/` are **untouched**. So nothing here pre-empts the evidence-state decision or approaches the
manuscript, exactly as the lane said.

**Hash integrity: zero dangling references.** For every `SHA256SUMS` on the branch, each hashed name
resolves to a file actually tracked in git — 3 sums files, 19 hashed names, **0 missing**.

### The packaging hole, verified and generalisable

The paper repository's **`.gitignore:2` is `*.log`**. Confirmed by mechanism, not by reading:
`git check-ignore -v experiments/results/M1/x/raw/driver.log` returns
`.gitignore:2:*.log`. So **any bundle that places `.log` files in `raw/` loses them silently from
the commit while `SHA256SUMS` and `runs.jsonl` keep referencing them** — a bundle describing files
a reader cannot find, with nothing in the normal flow to say so. `git add -A` reports success.

`apollo-board` caught it by checking `git ls-files` for each hashed name rather than trusting the
add, and resolved it by moving those captures to durable references with hashes (which METHODS
permits) and renaming the retained captures to `.boot.txt` / `.marker` so the extension cannot bite
again. Each file says the move was forced by the ignore rule rather than chosen, which is the part
that keeps it auditable.

**This is the same family as §7's flow-gate blind spot**, and it wants the same treatment — a check,
not a reminder. The check is one line of intent: *for every name in every `SHA256SUMS`, assert the
file is tracked*. It is what found the zero above, and it would have found the hole before the
commit rather than after.

**It applies to bundles this lane cannot see.** The E1/R1/M2/H1 bundles on the unpushed
`board/e1-s1s2-hardware` predate this discovery, and if any of them put `.log` files in `raw/` they
carry the same silent loss. **Whoever can read that branch should run the check before it is
pushed**, because once it lands the dangling hashes look like evidence.

### Two more things recorded rather than smoothed, both correct

* **Two work orders say "NOT RAISED"** instead of being backfilled. Right: backfilling would
  manufacture exactly the "stated rather than backfilled" artefact the handover flagged in the
  existing bundles, and a work order written after the run is not a work order.
* **The unscoped console capture is held as a durable reference** because kernel banners carry a
  `user@host` build string and an upstream driver author's email — the project's "commit result
  lines, not the capture" rule with a concrete instance, and a second one for §7's collection.

`precommit-scan --tree` blocking on `/home/<name>/…` inside work orders is worth knowing before
writing one: `EXECUTION.md` requires literal commands, which pulls the home path in, and
`~`-prefixing satisfies both.

**Evidence states remain untouched and the reading is agreed:** M1's primary is strong, its
secondary accounting band is refuted, and this is a *bounded baseline* — so `partial` is right and
`measured` would overstate it. The two-file constraint (§3) is unchanged by any of this.


## 10. Decision 5 is no longer an access question — write to the paper remote demonstrably works

*2026-09-16, after the lead retired the push allowlist and rotated the GitHub token.*

The handover's decision 5 (the push of `board/e1-s1s2-hardware`) has been carried all day as an
**access** problem, because the branch was refused 403 by the paper remote. **That is no longer the
obstacle**, and the evidence is not a probe of mine but an accomplished fact:

**`apollo-board` pushed `results/m1-bounded-baseline` to `nested-allocators-paper` today and it is
on the remote** — I fetched it, reviewed its 63 files (§9) and `git ls-remote` lists it beside
`main` and `paper/best-case-draft`. A branch that exists on the remote was written there by an agent
session after the rotation. **Write access works.**

A dry-run create-branch probe from this host agrees (`[new branch]` would succeed, exit 0, and
`ls-remote` confirms nothing was created), but that is the weaker evidence of the two: `--dry-run`
does not exercise GitHub's server-side permission check the way a real push does. The successful
push is what settles it.

**So the remaining obstacle to those ~480 records is neither access nor the allowlist. It is that
the branch exists only in the focs-server checkout and nobody there has run `git push`.** That
reduces decision 5 from a permission question for the lead to an action for whoever holds the
branch. It does not close it — a decision is closed when a push succeeds, not when one is believed
possible — but it changes who it is waiting on.

### Hook state on apollo — SUPERSEDED 2026-09-16: apollo is now gated, and I proved it here

*(Census method note: use `git rev-parse --git-path hooks/pre-push`, not `--git-dir`. In a **worktree**
`--git-dir` resolves to `.git/worktrees/<name>` while git looks for hooks in the common dir, so the
check reports NO HOOK — the ungated reading — for a worktree that actually inherits the parent's
hook. That produced five false OPENs on focs-server. The readings below are **submodules**, where
`--git-dir` is correct, and re-running with `--git-path` here reproduces all seven unchanged.)*

**The gap described below is closed.** Under the lead's direct instruction the new guard was
installed over ssh into **all seven** repos on apollo. Re-censused here: superproject,
`capstone-ariane`, `capstone-qemu`, `caplifive-buildroot`, `caplifive-system`, **`capstone/paper`**
and **`capstone/paper-nested-allocators`** all report `new guard`. So the combination this note
named — an unenforced absolute rule beside live write credentials — no longer exists on this host.

**Negative-tested from this session rather than accepted**, because a gate that has never fired
*here* is unproven *here*:

| probe (all `--dry-run`, nothing written) | result |
|---|---|
| task branch from **`capstone/paper`** | **`PUSH BLOCKED: capstone/paper is Overleaf's remote -- never push it.`** exit 1 |
| task branch from **`paper-nested-allocators`** | `push allowed: zz-guard-positive-test`, exit 0 — the paper rule does **not** over-catch the sibling |
| `main` in `paper-nested-allocators` | **`PUSH BLOCKED: 'main' is shared history and needs the lead's explicit go-ahead.`** exit 1 — closed, see below |

`git ls-remote` afterwards shows the remote carries exactly the four pre-existing branches; no probe
created anything.

**The arm I could not reach — now CLOSED, on this host, two-sided.** The outgoing lane supplied the
method that avoids the short-circuits: invoke the hook directly with its stdin protocol rather than
through `git push`.

    H="$(git rev-parse --git-path hooks/pre-push)"
    printf 'refs/heads/main <sha> refs/heads/main <zero>\n' | "$H" origin <url>
      -> PUSH BLOCKED: 'main' is shared history and needs the lead's explicit go-ahead.   exit 1
    printf 'refs/heads/zz-probe <sha> refs/heads/zz-probe <zero>\n' | "$H" origin <url>
      -> push allowed: zz-probe                                                            exit 0

The second line is the **control**, and it is the part that makes the first line mean anything: it
proves the harness can produce the passing outcome, so the block is a decision rather than a gate
that fails on everything handed to it.

*What follows is why the `git push` route could not reach it, kept because the shape recurs.* Both
attempts were intercepted *before* the hook ran: the first was a no-op (`Everything up-to-date`, git
short-circuits), the second was rejected by git's own client-side non-fast-forward check with git's
generic hint rather than the guard's `PUSH BLOCKED:` format. So the `git push` route reports on git's own
checks and never consults the guard at all — which is why the direct invocation above was needed. This is the same short-circuit that made my
earlier no-op write-access probe uninformative; a probe that never reaches the thing under test
proves nothing about it.

*(What follows was the state before the install, kept because the reasoning still applies to any
host that has not been gated.)*

### Hook state on apollo, which was stale and worth knowing

Measured with the outgoing lane's corrected check (a `grep` for the new guard cannot distinguish
"old gate" from "no hook", and on apollo most repos are the second):

| repo | pre-push hook |
|---|---|
| superproject | **OLD allowlist** (symlink to `pre-push-allowlist.sh`; allowlist is one entry, `dev`) |
| `capstone-ariane`, `capstone-qemu`, `caplifive-buildroot`, `caplifive-system` | **NO HOOK** |
| **`capstone/paper`** | **NO HOOK** |
| **`capstone/paper-nested-allocators`** | **NO HOOK** |

The lead's new `pre-push-guard.sh` is installed in the six focs-server repos. **On apollo it is not
present at all** — neither `~/.claude-c/secrets/pre-push-guard.sh` nor
`~/.claude-kisp/secrets/pre-push-guard.sh` exists — and the superproject's hook is still a symlink
to the **old** `pre-push-allowlist.sh`. Nothing is broken by this: the old gate has passed every
push from this lane today and did block a genuine non-fast-forward, so its protections overlap the
new guard's on the cases that have arisen.

**But the paper submodule has no `pre-push` hook at all** (`.git/modules/capstone/paper-nested-allocators/hooks/pre-push`
does not exist). The "never push `capstone/paper`" hard constraint is therefore unenforced by any
hook on this host, and now that write access works, the only thing standing between a lane and an
Overleaf-owned remote is the rule itself. That is a distribution gap for the lead rather than
something a lane should fix by writing into the secrets directory.

## 11. `board/e1-s1s2-hardware` is on the remote — the owed check run, and a gap nobody has named

Decision 5 closed on 2026-09-16: the outgoing paper lane retried from focs-server after the token
rotation and the push succeeded (15 commits, tip `7f64283`). The reconciliation this lane proposed
held — the 403 was real when observed and stale by the time it was being quoted, and the obstacle
was that nobody holding the branch had run `git push`.

§9 said *"whoever can read that branch should run the check before it is pushed"*. It is pushed, so
the check is run here.

### Shape against EXECUTION.md's seven entries

| bundle | entries | missing |
|---|---|---|
| `H1/fpga-2026-09-15` | **1 of 7** — `manifest.json` only | everything else |
| `M2/fpga-2026-09-15` | 6 of 7 | **`work-order.md`** |
| `R1/fpga-2026-09-15` | 6 of 7 | **`work-order.md`** |
| `S1S2/sw78-rep1-3` | 6 of 7 | **`work-order.md`** |

The missing work order is exactly what the handover recorded as "stated rather than backfilled", and
the right state to leave it in — a work order written after the run is a reconstruction, not a
pre-registration.

### The gap nobody has named: none of the four has `SHA256SUMS`

**Zero `SHA256SUMS` files exist on the branch.** So the dangling-hash failure §9 warned about cannot
occur here — there is nothing to dangle — but the `raw/` acceptance check is **unmet** in a different
way. `EXECUTION.md` requires *"Original transcripts and reports **with hashes**, no cropped
success-only logs"*, and these bundles carry the transcripts without the hashes.

That is a weaker defect than a dangling reference and a real one: nothing lets a later reader detect
that a retained capture has been edited or replaced. `apollo-board`'s newer bundles (§9) do carry
them, so the fix is known and already practised on the same remote; it simply has not been applied
backwards. Worth noting this is a **shape** gap, not an evidence gap — the transcripts are present.

### The `*.log` hole did not bite here, and only by naming luck

The S1S2 bundle keeps per-boot captures named `sw78-r1b1-log`, `sw78-r2b3-log` and so on — **a dash,
not a dot**. Tested both spellings:

* `git check-ignore -v …/sw78-r1b1-log` → **not ignored**, and it is tracked;
* `git check-ignore -v …/sw78-r1b1.log` → **`.gitignore:2:*.log`**.

So the same content under the conventional name would have vanished from the commit silently. The
bundle is intact by an accident of naming, and the next person who names a capture `foo.log` loses
it. That strengthens rather than weakens §9's conclusion: **the protection has to be the check, not
the convention**, because the convention is one character away from failing.

### The scan range must describe what the push PUBLISHES

`precommit-scan --range origin/main..board/e1-s1s2-hardware` **BLOCKS** — relative to `main` the
branch reads as reverting the restructure, so the scan sees a personal name on a **removed** line.
The range that describes what the push actually publishes starts at the merge base. Verified here:

    merge-base(origin/main, board/e1-s1s2-hardware) = b0d7510cbec38a35a57b06c6de07c9b4505ab84c
    commits from there to the branch tip          = 15

— and `b0d7510c` is exactly the superproject gitlink, which is why the handover's *"landing it is a
merge, never a fast-forward"* is the same fact. **It bites the scan before it bites the merge.**

This belongs with §4's rule: a range scan's verdict depends on things that are not in the range.
Add to "sync before you scan, and record who scanned" → **and scan from the merge base whenever the
branch is not a fast-forward of the target**, or the gate reports on a revert nobody is proposing.

### Still owed, unchanged

M2's `studies.json` edit. Its bundle is now on the remote, which removes the excuse of
unreachability but none of the constraint: `check_experiments.py:267-268` plus a live
`appendices/a-evidence-status.tex` still make it a two-file change whose second file is manuscript
(§3). Merging the branch into `main` is a merge and is not a lane's call either.


## 12. RETRACTION (2026-09-21): the revoke-cost story in §8 and §11 was wrong in four ways

*Filed after a claim-auditor refuted a new entry I was about to open as **R-36**, and after I
verified each refutation myself. R-36 was never filed. Nothing here touches R-35, which stands.*

**What I was about to claim:** that the RTL revoke walk never unlinks dead nodes, that this is why
`give_cyc/n` grows superlinearly, and that implementing splicing would fix it.

**1. The splice already exists and is in the resident bitstream.** `379248185` is an ancestor of
`054cea69b` (`git merge-base --is-ancestor`, checked), and `docs/state/current-state.md:192,230`
names `caplifive_m1_054cea69b.bit` as the resident splice-plus-reclaimer build. What I read was
`capstone_rev_node.anvil` at `ed049abd8` — which is **`board/r35-directed-repro`**, a side branch,
not the fix line. My quoted lines and the `// TODO: optimise` are accurate *for that branch* and
false for the design and for the board. **I recorded the sha and never asked what the sha was.**

**2. It duplicates R-12.** `ISSUES.md:3711` already carries *"2026-09-16 — THE REVOKE-WALK SPLICE:
built, measured, synthesised. R-12's COST half, not its capacity half."*

**3. The mechanism predicts the wrong SHAPE.** An unspliced walk at a constant 3 cycles per dead
node is **linear**. I used it to explain a **superlinear** curve. The real knee was already
localised by two independent readings: the d-cache holds 2,048 nodes, the 3.000 slope holds to
1,024 and breaks at 2,048 (`ISSUES.md:3808-3810`), and `sublet/r1/r1_slots_pools.c:470-473` says
*"the knee looks like node-table capacity"*. So the observed curve is a **linear walk multiplied by
a memory-hierarchy cliff**, not one mechanism.

**4. The variable was already retracted by this project.** `docs/history/15-09-2026_19-25-14_m1-revoke-cost-variable.md`
is titled *"The independent variable is not table occupancy"* and refutes it with an overlay:
invocation 9, with 18,684 nodes already consumed, traces invocation 1's curve to **0.06 %**.

### And the answer to "so is there superlinearity at all?" — not on the current build

`ISSUES.md:4278-4284`, deployed `1bfff7776` against reclaimer `054cea69b`:

| | `1bfff7776` (superseded) | `054cea69b` (resident) |
|---|---|---|
| `take_cyc/n` | floor 66.7 → 130-218, knee at alloc ~1,792 | **72.1, FLAT to 200,000** |
| `give_cyc/n` | grew ~12x | **103.2, FLAT to 200,000** |

103.2 at alloc 1,250 and 103.2 at alloc 195,000. *"The 12x release growth is gone and the capacity
knee is gone."* Splicing removes the walk term; reclamation stops the node table ever spanning
enough cache lines for the knee to form.

**The honest caveat that comes with it**, and it is where the remaining experimental value sits: a
pre-registered falsifier said *"one curve flattens, the other does not; if both flatten or neither,
my account is wrong"*. **Both flattened.** It is not counted as a refutation only because two other
lanes had written the reason down in advance. And `ISSUES.md:4295` records that **the coefficient is
BOUNDED, NOT MEASURED** (`c < 65,532/200,000 = 0.3277`), with a retirement effect that the 200,000
pilot is already past unless index reuse spreads over sixteen or more.

**So the open question is not "does splicing help".** It is *what is `c`, and does the flat curve
survive past index retirement* — and that is a board question, not an emulator one (§13).

## 13. These experiments must run on the BOARD. QEMU cannot measure them at all

*The lead ruled the experiments board-only and asked for justification if I disagreed. I do not —
and the reason is stronger than "the emulator is approximate".*

**QEMU is structurally incapable of producing this measurement.** The entire revoke walk lives
inside a single helper: `helper_csrevoke` (`target/riscv/op_helper.c:920`) calls
`cap_rev_tree_revoke` once (`:937`, the only call site), and the loop over the chain is inside that
C function. Under `-icount`, the guest is charged **one instruction for the whole walk**, whatever
its length. So an `-icount` curve is flat by construction — flat whether the walk crossed 8 nodes
or 8,000, and flat whether or not the emulator splices.

That is exactly the trap this project already named: *"'QEMU NEVER DOES' IS AN ABSENCE OF THE
FEATURE, NOT AN INDEPENDENT WITNESS"* (`ISSUES.md:5638`). A flat emulator curve here would have been
recorded as agreement and would have meant nothing. **My earlier plan put E1 and E3 on QEMU. That
was wrong and is withdrawn.**

Three further reasons, each independently sufficient:

* **the emulator's node accounting is not the silicon's.** `cap_rev_tree_update_refcount` has
  exactly one occurrence in all of `target/riscv/` — its own definition (`cap_rev_tree.h:63`) — so
  it is never called, `cap_rev_tree_release` never runs, and the free list is unreachable. The
  header says so (`:29-30`, *"this emulator reuses no node"*); a comment in the `.c` (`:8-10`)
  asserts the opposite and is reasoning from dead code. **Reclamation is the thing the current flat
  curve is attributed to, and the emulator does not have it.**
* **`drop` differs.** `helper_csdrop` (`op_helper.c:958-992`) never touches the revocation tree at
  all, where the RTL's `drop_req` invalidates a node in memory without splicing. Dead-node
  accumulation therefore has a shape on silicon that the emulator cannot exhibit.
* **enforcement differs** — R-35.

**What QEMU is still good for, stated so it is not over-corrected away:** it is an **oracle for
structure, not for time.** Instrumenting `cap_rev_tree.c` gives ground truth for *how many* nodes
exist, how many are dead, and how many a walk would cross. Pair it with the board's cycles and each
instrument does what it can: QEMU supplies N, the board supplies cycles per N. Never let it supply
a cycle count.

### The plan, re-specified for the board

The question is no longer "does splicing help" (§12: it is in the resident build and the curve is
already flat). It is **what the flat result rests on, and where it ends.**

| | varies | holds fixed | metric | why it is worth a boot |
|---|---|---|---|---|
| **B1** | allocations, out to and past index retirement | one domain, fixed live set | `give_cyc/n`, `take_cyc/n` | the flat curve is verified only to 200,000, and retirement falls at ~16,384/32,768/65,536/262,144 depending on how far reuse spreads. **Does 103.2 survive its first retirement?** |
| **B2** | index-reuse spread (1 / 2 / 4 / 16 slots) | allocation count | first-retirement allocation; `give_cyc/n` at it | `c < 0.3277` is a **bound, not a measurement**; this is what turns it into one |
| **B3** | dead nodes inside one revoked run | live nodes in that run | cycles per revoke | isolates the **linear** term on the build that has the splice — the coefficient the paper's `\targetNodeSlope` needs |
| **B4** | — | real workload | revoke cycles as a fraction of runtime | speedtest1 and the PostgreSQL tpcb/readonly replay, for which the manuscript already reserves `\pgNodes` / `\pgRONodes` |

B1 is the one that matters, because it attacks the result we would otherwise publish. B3 supplies
the number the manuscript has a slot for. B2 converts a bound into a value. B4 is the "what does it
cost in practice" line.

## 14. RETRACTION (2026-09-21): domains DO run at M-mode — the gate is satisfied, the check is defective

*This retracts a claim I recorded as confirmed and put to the lead three times as the basis for
decision 1. Raised by `apollo-board`; all three legs verified here before retracting.*

**What I asserted** (`…16-37-49…md:387`, `:471`, `:513`): *"a domain cannot satisfy that gate, so the
check protects nothing about domain code"*, *"it does not extend the check to domain code, which
cannot satisfy the gate at all"*, *"the gate still excludes domains"*.

**It is false. Entering a domain does not change privilege at all.**

* `priv_lvl_d` has exactly **six** writers in `core/csr_regfile.sv` — `:1048` hold, `:2144` trap
  entry, `:2307` MRET (from `mstatus.mpp`), `:2330` SRET, `:2351` VS-RET, `:2365` DRET. **None is on
  a capability or domain-switch path.**
* `core/anvil_build/capstone_dom_switcher.anvil` contains **zero** occurrences of `mstatus`, `priv`
  or `mpp`.
* and the load/store privilege is `ld_st_priv_lvl_o = (mprv) ? mstatus_q.mpp : priv_lvl_o`
  (`csr_regfile.sv:2273`) — with `mprv` clear it is simply the current privilege.

So a domain entered from M-mode monitor code **runs at M**, and `ld_st_priv_lvl_i == PRIV_LVL_M`
**holds**. The gate condition I verified at `load_store_unit.sv:966-969` is correct; the inference I
endorsed from it was not. The source of the error is `docs/history/15-09-2026_lsu-capmode-gate-why-domains-cannot-satisfy-it.md`,
whose `mret`s are labelled `call_into_smode` / `resume_smode` — the S-mode **host**, not a domain. I
took its conclusion without checking its premise.

**The measurement says the same thing independently.** R-35's bounds probe returned **mcause 28**,
latched in hardware. CPMP cannot have produced it: its data check is gated `!= PRIV_LVL_M` and emits
only `ST_ACCESS_FAULT`/`LD_ACCESS_FAULT` (7/5) — verified at `pmp/src/pmp_data_if.sv:292-294`. So 28
can only have come from the M-gated `cap_violation_detection` block, which means **that block is the
live path for domain data accesses**.

### What this changes for the paper, and it is better news than what I gave the lead

| | what I told the lead | what holds |
|---|---|---|
| why the four `tab:safety` rows fail | the check is **unreachable** from a domain | the check **runs** and its **revocation half is defective** |
| nature of the gap | **structural** — no fix makes the rows hold | a **root-caused defect** with a written-up fix |
| what silicon enforces | nothing for domain data | **bounds and permissions YES, revocation NO** |

R-35 is now closed and root-caused to `load_store_unit.sv:966-971`: a single core-wide tracked
revnode id that **re-adopts itself as VALID** whenever an access presents a different one. With 16
rotating slots nearly every access presents a different one, so cause 25 can never fire. Bounds and
permissions survive because they are read from the capability's own metadata — which is exactly the
"**bounds but not revocation**" scope the board observed.

**So my "structural gap" framing is withdrawn.** "R-34 is fixed must not read as the four rows now
hold" was right for the wrong reason: the rows do not hold because revocation checking is broken,
not because domains are locked out of the check. That is a defect with a known fix and a stated
cost, which is a materially different input to decision 1 than an architectural impossibility.

**Not mine to edit:** `15-09-2026_lsu-capmode-gate-why-domains-cannot-satisfy-it.md` is the RTL
lane's and its conclusion changes, so it needs their sign-off. The correction is written up by the
board lane in `docs/history/21-09-2026_r35-root-cause-is-cpmp-optimistic-adopt.md`.


## 15. The node/line question is settled: one revnode is exactly one cache line

I could not tell whether `ISSUES.md`'s two phrasings — *"the dcache holds exactly 2,048 **nodes**"*
and *"the distinct-index set never approaches 2,048 **lines**"* — described the same boundary, and
flagged that if the line were 64 B the knee in nodes would be 8,192 and any ladder centred on 2,048
would be centred wrong. `apollo-board` answered from the board config; verified here:

| | | |
|---|---|---|
| `capstone_cv64a6_imafdc_sv39_config_pkg.sv:48` | `CVA6ConfigDcacheByteSize = 32768` | 32 KiB |
| `…:50` | `CVA6ConfigDcacheLineWidth = 128` | **16-byte line** |
| `ex_stage.sv:1121-1122` | `CAP_REVNODE_MEM_BASE + {22'd0, node_query_addr, 4'd0}` | the `4'd0` is a shift of 4 ⇒ **16-byte revnode stride** |

So 32768 / 16 = **2048 lines**, and 16 B per node against a 16 B line is **1 node : 1 line**. Both
phrasings are correct and describe the same boundary. The ladder is centred correctly; 1,792 sits
just inside it.

**The 1:1 mapping buys something beyond re-centring, and it is worth stating because it removes an
assumption from the experiment.** With several nodes per line, a walk of N nodes could touch as few
as N/k distinct lines when ids happen to be dense, so the node axis and the line axis would differ
by an unknown factor and the knee's position would depend on id density. At 1:1 **each node is its
own line**, so N nodes touch N distinct lines always, whatever the density. The ladder therefore
measures distinct-line count directly, with no locality to hide behind — and the deep-cold points
(8,192 and 16,384) show the full dependent-load latency rather than an amortised one. Those two are
also the only points at which the cold slope is measured over more than one segment, which is the
specific defect in the number we have today.

## 16. R1 on silicon: both target macros are now fillable, and E1 is retired

*Measured by `apollo-board`, 90 invocations over 8 boots. **Recomputed here from `points.csv` on
`board/r1-results-2026-09-21`, not taken from the summary.***

**Everything reproduces.** 420 records, **0** not-completed, **0** bad survivors. And the fits:

| | their figure | recomputed here |
|---|---|---|
| revoke vs affected nodes, shared | 22.91 cyc/node, R² 0.99966 | **22.91**, R² 0.99966 |
| revoke vs affected nodes, combined | 15.92 cyc/node, R² 0.99967 | **15.92**, R² 0.99967 |
| **fill vs bytes (POSITIVE CONTROL)** | 1.8125 cyc/byte, R² 0.9999999993 | **1.8123**, R² 0.99999995 |
| revoke vs released bytes / unrelated heap / depth / object size | no dependence | slopes ~1e-4 or smaller, R² ≤ 0.003 |

**A denominator check worth recording, because it is the trap this registry already named.** My first
fit gave exactly **twice** their slope with **identical R²** — 45.82 against 22.91, 31.84 against
15.92. The cause is the denominator: `nodes_minted = 2n` in every row, so each object mints **two**
revocation nodes, and "cycles per **affected node**" divides by `nodes_minted`, not by the object
count. Their figure is right and mine was the naive one. Matching R² with a factor-of-two slope is
the signature of a denominator disagreement rather than a data disagreement — worth knowing as a
diagnostic.

**The positive control is what makes this publishable rather than merely reported.** On the *same
records*, the same instrument resolves a byte-proportional quantity at R² ≈ 1.0. So four null axes
are a property of revocation, not a blind instrument. The driver had pre-registered the alternative
in as many words — *"rv flat in n if the RTL's revoke is O(1); a rise with n is the node-linear
finding R1 asks about"* — and it rises.

### Both placeholders can now be filled — with the fit range the appendix demands

`appendices/b-target-results.tex:83-86` requires these two over *"at least three legal levels"* with
*"fit range and error"* stated, and says *"neither is a universal architectural constant"*.

* **`\targetNodeSlope`** — 22.91 (shared) / 15.92 (combined) cycles per affected node, **five
  levels** (n = 1, 4, 16, 64, 256), R² 0.9997.
* **`\targetByteSlope`** — 1.8123 cycles per byte, R² 0.99999995.

**The fit range is not optional here and it is the whole of §B3's remaining purpose.** Every R1
point is **cache-resident**: `nodes_minted` reaches **512 against a 2,048-node table — 25 %** — and
at the 1:1 node-to-line mapping (§15) that is 512 lines of 2,048. So **22.91 is a WARM coefficient**,
and a bare 22.91 in the manuscript would be a cache-resident number presented as the cost of
revocation.

### E1 is retired; B3's framing sharpens

* **E1 (occupancy independence) is answered and can be dropped.** Unrelated heap from 0 to 4 MiB
  moves revocation cost not at all — on **silicon**, with a positive control on the same records.
  That was E1's entire purpose. At most it survives as corroboration.
* **B3 is no longer "is splicing valuable".** R1 has fixed the **shape** as linear with R² 0.9997;
  the only open quantity is **the coefficient in the cold regime**. The 8,192 and 16,384 points are
  the ones that matter, because they are the only ones measuring the cold slope over more than one
  segment.

### Two limits the bundle records rather than leaves to inference

The bitstream does not close timing; and **R1 step 1's invalid-access companion is UNMEASURED, not
passed** — it needs an access through a released reference to trap, and R-35 is a defect in exactly
that check. Its absence is not a safety result, and the bundle says so in those words. That is the
right handling and it is the sentence a reviewer would otherwise construct for us.

**One shape gap:** the bundle carries all seven entries including `work-order.md` — the first to do
so — but no `SHA256SUMS`, which is the §11 finding still outstanding across every bundle on this
remote.

## 17. The §11 hash gap is closed on R1, and the warm number is OPTIMISTIC not conservative

**Verified here, two-sided, on my own checkout of the bundle** — presence is not verification:

* `sha256sum -c SHA256SUMS` over the extracted bundle → every one of 15 files `OK`, **rc 0**;
* one byte appended to `summary.md` → `summary.md: FAILED`, `WARNING: 1 computed checksum did NOT
  match`, **rc 1**; clean again once restored.

Paths are relative to the bundle, so it still verifies after a move or a fresh clone. The
denominator note is in `work-order.md` where it cannot be quoted without it.

**`experiments/bundle-sha256.sh` now exists for the other 19 bundles, and the board lane
deliberately did not run it over them.** Their reason is the right one and worth keeping as a
principle: *a checksum asserts the files were as found when it was written, and that is only ours to
assert for a bundle we produced.* Stamping another lane's bundle from here would be writing a
provenance claim you cannot back — the hashes would be true and the assertion behind them false.

### The direction of the warm/cold error, which is the sharper framing for the lead

I had put the fit range as "22.91 is cache-resident, say so". The board lane's version is better and
it is the one to use: the dependent-load figures are **9.00 warm against 48.2 cold, 5.36×**, so the
cold coefficient could plausibly be **several times** 22.91.

**Therefore publishing a bare 22.91 is not conservative — it is optimistic in our own favour.** It
would understate revocation cost, which is the worse way to be wrong: a reviewer who finds it has
found us flattering our own system, not being cautious about it. That is a different sentence to put
to the lead than "please state the range".

### One caution carried into B3

**R1's nulls are warm too.** The heap null — revocation cost independent of unrelated heap — is a
*structural* property (revocation does not touch unrelated memory) and should survive the cache
boundary. But if B3 ever finds cost scaling with something R1 called flat, **the first thing to
check is whether the boundary changed the answer, not whether R1 was wrong.** Both would look
identical in the data.
