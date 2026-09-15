# Sublet draft: what an audit against the measurements leaves standing (2026-09-15)

*Written as the paper lane hands over to `apollo-paper`. The lane's working analysis lived in a
scratch file on the outgoing host and does not travel; this is what survived checking it against
the manuscript as it actually stands. Paths and line numbers are against the paper repository's
remote, `7f83725` (identical on `main` and `drafts`).*

**Read `docs/plans/2026-09-15-paper-lane-handover.md` beside this.**

## 0. One branch fact first, because it misled this lane

`origin/main` and `origin/drafts` are **the same commit**, `7f83725`, with a zero-line diff. What
is twenty commits behind is the superproject's gitlink (`b0d7510c`) and this checkout's **local**
`main`. An earlier reading of "drafts is the live branch, twenty ahead of main" came from
comparing against the stale local branch and is withdrawn — it also reached
`2026-09-15-board-lane-handover.md` §8 and has been corrected there.

The consequence that matters: a checkout at the gitlink has the old flat `parts/` layout and
lacks `experiments/EXECUTION.md`, `experiments/WORK-ORDER.md`, `sections/`, `appendices/`,
`macros/` and the rest of the restructure.

## 1. The one live contradiction

`tab:safety` in `appendices/c-validation-and-accounting.tex` (`:27`) claims **"Stops at access"**
on four rows — `:34` stale lookaside access after finalize, `:35` stale backing-block access after
free, `:36` stale pool access after destroy, `:37` same address / new object / old pointer. The
prose at `:55-58` says it again in words: the old pointer returns `\safeReuseByteNew` on the
unprotected arm, and *"The protected arm stops at the read without returning a byte."*

**On silicon the protected arm returns `0x5B`** — the new object's first byte, the value the prose
attributes to the *unprotected* arm. Three repetitions, every cell identical (E1, §7r; s3 and s5,
six unsafe-success records over ten boots with a control and a banner in each).

The cause is fully known and **is not Capstone's**, which is the part that should shape the
wording:

* the LSU's capability block is gated on `capmode_i && ld_st_priv_lvl_i == PRIV_LVL_M`
  (`load_store_unit.sv:966-969`), and a domain cannot satisfy it — `mret` clears `mprv` on entry
  to a lower privilege and `mprv` is machine-mode only, so the only route left would be running
  the domain at machine mode, which discards the isolation the check exists to enforce;
* **and it would not matter if it could**: R-34, the LSU raises its exceptions and the load unit
  drops them. Demonstrated by the **stock** `rv64mi-p-ma_addr` compliance test failing with
  capmode never set. A base CVA6 defect that also swallows the capability causes.

So the four rows are unsupported for two independent reasons, neither of them about the
discipline. **The capability-access rows stand** — the DYN unit's node-validity query carries no
privilege or capmode gate at all.

**This is one decision, not two.** It is the same question as `METHODS.md:86` (*"An enforcement or
oracle failure stops dependent performance publication"*): whether the six unsafe-success cells
bind the P1/R1 timing numbers, or are a labelled configuration fact. Answering the table and the
publication rule differently would be incoherent. Both are the lead's.

## 2. Measured results the manuscript can take

Each is recorded in `docs/ref/fpga-silicon-measurements-for-paper.md` at the section named.

| result | value | where |
|---|---|---|
| Compound stack cost vs native | **1.607x at `-O2`**, 1.334x at `-O0` | §7s |
| — its two factors | ABI 1.3624, discipline 1.1797 (`-O2`); 1.2103 / 1.1024 (`-O0`) | §7s |
| Release cost, separable | **≈ 1.81·B + 22.7·nd + 58.9·n** | §7t |
| — the byte term's share | **95-99 %** of block release | §7t |
| — per-object release | 593 / 612 / 608 cycles at 64 / 256 / 4096 B, **no fill** | §7t |
| Memory ledger | metadata **1.74x peak payload** (tables 1,344,064 + nodes 693,680 + tags 215,076 against 1,291,712) | §7u |
| Node-validity query | **flat over 1,366x occupancy** (6 to 8,196 live nodes), arms indistinguishable | §7y |
| Dependent load | 9.00 cycles in the 32 KiB D-cache, 48.2 from DRAM | §7v |

Three of these carry a caveat that must travel with them:

* the `-O2` figures rest on the **C-32 workaround** (`setupLookaside` alone at `-O0` inside an
  otherwise `-O2` image) and are labelled a bounded-prototype diagnostic until the compiler's
  bridge fix lands;
* every `-O1`/`-O2` Sublet run before that workaround was a **lookaside-OFF** run, and the `-O2`
  ⑥/⑤ ratio from those is not citable;
* the whole §7 series was measured on a bitstream that does not meet timing (WNS −12.425 ns).
  Its bound is measured rather than argued: cross-boot reproducibility is 0.02-0.05 % on this
  workload family, far below the differences the ratios rest on.

**Two results are positive findings about the design and should not read as controls**: release
cost is flat across unrelated heap from 64 KiB to 4 MiB (spread 0.0034 %), so revocation is
bounded by the subtree rather than the pool; and flat across nesting depth 1-8 at fixed node
count (0.0132 %). Both answer objections a reviewer raises against any revocation scheme, with a
flat line over a wide range.

**Linearity, scoped honestly** (§7w): measured conformant on the deployed bitstream for
`cincoffset`, `scc`, `ldc` and `stc`, six readings each with an instrument control and NONLIN
conformance controls; `init` covered by R-25's silicon confirmation; **`tighten` and `shrinkto`
untested on silicon**.

## 3. Already resolved — do not re-raise

Two findings this lane carried for days are **spent**, and the manuscript resolved them without
prompting:

* the contribution claiming *"the first RTL implementation of Capstone that enforces linearity
  and reclaims revocation-tree nodes"* is **gone**; reclamation is now correctly pending (M1), and
  `sections/evaluation/05-resource-cost.tex:34` says *"The unreclaimed baseline bounds a run"*;
* **"constant release" survives only in negated form** — `appendices/a-evidence-status.tex:99`
  (*"One instruction does not establish constant-time release"*) and
  `sections/evaluation/04-release-cost.tex:22` (*"ranges, not universal constants or constant-time
  release"*).

Worth knowing why the first one resolved itself: the archived planning branch
`archive/drafts-before-lifetime-20260914` holds the previous generation of experiment plans
(`experiments/A2-…` through `A19-…`), and `A6-rtl.md` carries that exact contribution sentence
with its own pre-registered refutation condition — a run that allocates past the pool boundary
must keep the head bounded, *refuted while the head grows monotonically* — and the instruction
that **the paper does not write the claim while that hypothesis is open**. The head does grow
monotonically. The claim came out because the plan's own gate fired.

`A3-freigabe.md` did the same for release: it pre-registered the two slopes R1 later measured
(cycles per free over heap size, and over object size, each below 0.1 in a log-log fit) and both
are satisfied. So the **per-object half of the release claim is confirmed**, not refuted — an
earlier blanket "constant release is refuted" from this lane was too broad and is withdrawn.

## 4. Retractions from this lane, so they are not re-run

Each was stated, then refuted by measurement or by reading the source. They are listed because a
successor with the same instincts will reach for them again.

1. **"Enforcement is live for the trusted monitor and dead for untrusted domain code."** Endorsed
   from the RTL lane's framing, refuted hours later by a directed RTL test: with both gate inputs
   satisfied and witnessed, none of the three clauses fires. Dead at every privilege.
2. **"Expect a trap, cause 24."** Pre-registered for the M-mode arm from an August note. On this
   core `riscv_pkg.sv:348` makes **24 = DEBUG_REQUEST**, routed to debug mode and never to
   `mtvec`; the capability causes are 25-28. A live clause could not have produced a trap at all,
   so that arm would have read "no trap" either way.
3. **A lookaside-OFF stall risk for cell ⑥.** Wrong: Sublet ports *both* allocators, so ⑥ is ⑤
   with the discipline applied and lookaside is on by construction, not an independent knob.
4. **R-29 as the `-O2` counter divergence.** Refuted by a CFG scan with the `-O0` image as its
   positive control: zero qualifying sites in the optimised images.
5. **`sublet_type` branching as that divergence.** Refuted at the call site — the only call is
   lazy, off the allocation path, and the grant is never revoked during a run.
6. **"The retention pattern is M1's independent variable."** Void on the deployed table: nothing
   is reclaimed, so every allocation mints one node whatever is retained.
7. **"`drafts` has nothing `main` lacks", then "`drafts` is twenty ahead of `main`."** Both from a
   stale local ref; see §0.

## 5. `studies.json` is not where this lane left it

The lane's own commit `c83379f` (local `main`, **unpushed**) moved R1 and P1 from `pending` to
`partial`. Against the remote:

* **P1 is already `partial`** upstream — that half is redundant;
* **R1 is still `pending`** — that half is stranded on a base twenty commits old and needs
  re-applying;
* **M2 is still `pending`**, although its bundle (four boots, 225 records) is committed on
  `board/e1-s1s2-hardware`. That edit is owed and was never made.

**And the gate that passed could not fire.** `make experiments-check` returned clean on `c83379f`,
but `scripts/check_experiments.py:196` guards the appendix cross-check with `if appendix.is_file()`,
and `appendices/a-evidence-status.tex` did not exist on that base. On the current tree it does, and
`:129-132` requires the appendix rows to equal the catalogue's `(id, priority, evidence)` triples
exactly. So re-applying the R1 change — and making the M2 one — is a **two-file edit whose second
file is manuscript**, and therefore needs the lead's go-ahead separately from the `studies.json`
permission already given.

`board/e1-s1s2-hardware` is based on `b0d7510` and so *deletes* the restructure relative to the
remote. **Landing it is a merge, not a fast-forward.**
