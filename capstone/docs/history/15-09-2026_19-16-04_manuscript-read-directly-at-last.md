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
