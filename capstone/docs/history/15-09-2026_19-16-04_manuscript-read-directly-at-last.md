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
