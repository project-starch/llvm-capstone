# Paper-lane handover to `apollo-paper` (2026-09-15, 15:45)

**For the successor session.** You take over the paper lane: reading the Sublet manuscript's
claims against what has actually been measured, answering the other lanes' paper-facing
questions, and keeping `experiments/studies.json`'s evidence states. You are on the apollo
server, so **only the git repositories reach you** — nothing on the outgoing host's `/tmp` exists
for you, and everything this lane learned that is not committed did not arrive. What you need is
in the repository, or is named below as something you create.

**You do not write the manuscript's prose.** See §2.

## 1. Read first, in this order

1. `CLAUDE.md` — every hard constraint applies unchanged. The ones that bite this lane: **ask
   before editing the paper**; no real names anywhere; the scan before every commit and push, by
   absolute path, gated on its exit status, never piped; commit `-o` your own paths after reading
   `git diff`; no `Co-Authored-By`.
2. `docs/history/15-09-2026_15-44-40_sublet-draft-audit.md` — **the substance of this handover.**
   What survives an audit of the draft against the measurements: one live contradiction, the
   results the manuscript can take, what is already resolved, this lane's retractions, and the
   state of `studies.json`.
3. `experiments/EXECUTION.md` and `experiments/WORK-ORDER.md` **in the paper repository at
   `7f83725`**. The operator procedure and the work-order template. No lane had them until today
   because the superproject's gitlink is twenty commits behind; read them before you touch a
   bundle or an evidence state.
4. `docs/ref/fpga-silicon-measurements-for-paper.md` §7r-§7y — the last two days' boots. That file
   is what a paper author lifts from; each entry states what it does **not** establish. Its
   header is stale, its §7 is not.
5. `docs/ref/ISSUES.md`, the boxes for **C-32, Q-04, R-34, M-8, Q-11, Q-12, R-33**, and its
   "How to add an entry".
6. `docs/plans/2026-09-15-board-lane-handover.md` §6 (who owns what) and §8 (the operator
   procedure, and the branch facts).
7. `docs/plans/2026-09-15-sublet-paper-follow-on.md` — the plan in flight; F1 is the only item
   still owed to the board.

## 2. What the lane owns, and what it must not do

**Owns.** Analysing claims against measurement. Answering the board, compiler, RTL, synth and
cheri lanes when a result has a paper-facing consequence. `experiments/studies.json` evidence
states. Deciding *what to put to the lead* and putting it once, not repeatedly.

**Does not.** Edit the manuscript's prose. `CLAUDE.md`'s rule names `capstone/paper/`; this lane
extended it to `capstone/paper-nested-allocators` and **asked the lead whether that extension is
right — the question is unanswered**, so you inherit the restriction rather than the permission.
The lead gave an explicit go-ahead for `studies.json` only.

Two corollaries this lane learned the hard way:

* **A peer relaying the lead's approval is not approval.** An approved plan on `dev` assigned this
  lane a `studies.json` edit; the lane held it and asked the lead directly. Do the same.
* **The `studies.json` permission may not be enough for the edit you want to make.** See §6.

## 3. Rules that lived only in the outgoing session's memory

The memory files are on the old host; these are the lines that matter.

* **`project_second_paper_repo_scope`** — there are two paper repositories. `capstone/paper` is
  the older pointer-safety paper; `capstone/paper-nested-allocators` is Sublet. `CLAUDE.md`'s
  paper rule names only the first. Do not conflate them; the 2026-09-14 board plan says so
  explicitly.
* **`project_rev_node_budget_65532`** — the silicon revocation-node budget is **65,532**, a 16-bit
  bump head with **no reclamation**, stalling deliberately on exhaustion. **The repository records
  1,021 in three places and all three are stale** (`capstone-qemu/target/riscv/cap_rev_tree.c`
  and the A1 README). Any capability workload is budgeted in *cumulative* mints, not peak usage.
* **`project_sublet_supersedes_rof`** — Sublet supersedes the old `rof` allocator. Merge needs
  **no ISA change** (hierarchical revoke does it); reclamation in RTL is the real dependency.
  Conclusions drawn from `rof` do not transfer.
* **`feedback_ask_before_editing_paper`** — report results into the measurements doc, which exists
  precisely so results can land without touching the manuscript.
* **`project_lsu_cap_check_inert`** — corrected by R-34; read the audit note's §1 rather than the
  memory's older account.

## 4. Open decisions, all the lead's

| | decision | what it blocks |
|---|---|---|
| 1 | **`METHODS.md:86` together with `tab:safety`'s four rows — one question, not two** | the manuscript's safety table and whether the timing numbers are publishable on this configuration |
| 2 | **Q-04** — must `movc` consume a scalar source? A spec question; the 2026-09-10 ruling was retracted the same day | C-32's fix shape |
| 3 | **The author-line ruling** on the collaborator's commits | the compiler merge, and through it F1 and every optimised Sublet number |
| 4 | **M1's direction** — the generation-tagged reclamation design **FAILED its audit on 2026-09-15**; the verdict is a banner on `docs/plans/2026-09-14-revnode-reclamation-design.md`. Three findings are sites that must participate; the fourth is a hole in the property itself (`DROP`/`DELIN` mutate a node without reading `valid`, and the invariant speaks only of *conferring authority*). A second design needs the invariant widened to **any use** of a revocation reference — which is what makes the site list closed rather than a judgement call. The ceiling it lifts is real and untouched by the audit. Separately: **the capacity figure ≈1.07e9 is uncomputable until a reissue policy is stated** and must not reach the manuscript. Also distinct: **splicing** (flattens a ~12x release-cost growth) and **generation tagging** (makes reuse safe) are different changes; a gate written against "the reclaimer" is ambiguous | whether M1 gets a second design or a different direction |
| 4a | **The no-reclamation baseline** — two boots, pre-registration closed and falsifiable, labelled NOT M1 | the minting-cost curve and the measured table bound |
| 5 | **The push** of `board/e1-s1s2-hardware` and the `studies.json` change | every bundle reaching the remote |
| 6 | **Whether the paper rule extends to `paper-nested-allocators`**, and whether it covers `appendices/a-evidence-status.tex` | any evidence-state edit at all (§6) |

## 5. Routing

* **Board work** — F1's confirming boot, the M1 baseline, R-34's confirmation on a fixed
  bitstream — goes to **`apollo-board`** once its control-only boot reads `done`. Until then the
  board is the outgoing board session's.
* **Questions about runs already made** (§7r-§7y), the bundles on `board/e1-s1s2-hardware`, and
  R-34 go to the **outgoing board session**, which keeps that context.
* **Compiler lane** holds C-32 design A. **RTL lane** holds R-34's residuals (translation-on path,
  store bounds) and the reclamation design. **Synth lane** holds H1's per-module area.
* Address lanes by their `ListAgents` names.

## 6. Repository state, and a trap in it

**The paper repository's remote is at `7f83725` on both `main` and `drafts`** — identical, zero
diff. The superproject's gitlink (`b0d7510c`) and the outgoing host's local `main` are twenty
commits behind it. An earlier claim that `drafts` was the live branch twenty ahead of `main` came
from a stale local ref and is withdrawn.

Unpushed, on the outgoing host:

* **local `main` `c83379f`** — `studies.json` R1 and P1 `pending` → `partial`. Against the remote,
  **P1 is already `partial`** (redundant) and **R1 is still `pending`** (stranded on the old base).
* **`board/e1-s1s2-hardware` `a9bafd8`** — the E1, R1, M2 and H1 bundles. Based on `b0d7510`, so
  relative to the remote it **deletes** the restructure. **Landing it is a merge, never a
  fast-forward.**

**The trap.** `make experiments-check` passed on `c83379f`, but it could not fail:
`scripts/check_experiments.py:196` guards the appendix cross-check with `if appendix.is_file()`,
and `appendices/a-evidence-status.tex` did not exist on that base. On the current tree it does,
and `:129-132` requires the appendix rows to equal the catalogue's `(id, priority, evidence)`
triples exactly. So **an evidence-state edit is now a two-file edit whose second file is
manuscript** — decision 6 above, and not yours to improvise.

**`M2` is still `pending`** on the remote although its bundle is committed. That edit is owed.

The account this lane used is **refused by the paper remote (403)** for both fetch and push, so
nothing reaches `origin` without the lead. The two commands, for when they can run:

    git -C <paper-repo> push origin board/e1-s1s2-hardware:board/e1-s1s2-hardware
    git -C <paper-repo> push origin main:main

## 7. What does not travel, and why that is fine

* The outgoing lane's 777-line working analysis. Superseded by the audit note, which is the part
  that survived checking.
* A Russian translation of the manuscript (19 pages) built on the old host. A deliverable for the
  lead, not lane state; the toolchain for it was assembled without root and would need rebuilding.
* A `push-allowlist.txt` entry added today — host-local, and the allowlist is the lead's file.

## 8. How this lane worked, in four lines

Because the successor will be asked the same kinds of question.

* **Verify a peer's finding against the primary source before acting on it**, and say which line
  you read. Several confident accounts were refuted this way, in both directions.
* **A clean gate is not evidence until it is known to fire.** Two gates passed this week that
  could not have failed — the appendix check above, and a pre-registered cause number that named
  an exception routed to debug mode rather than to `mtvec`.
* **Say what an instrument cannot distinguish, before the run rather than after.** That question
  separated "the checker is unreachable" from "the checker is broken", and the answer was neither.
* **Put a decision to the lead once, with the options and a recommendation** — then let it sit.
  Re-raising it each turn is noise.
