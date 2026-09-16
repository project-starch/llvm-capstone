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

## 6. Two rules this investigation produced, for citing rather than restating

Both came out of the reclamation-design audits on 2026-09-15 and generalise well past them.

### A quantity constant in every variable the experiment varied, and linear in one it did not

This is the shape that survives its own repetition. Two instances the same evening:

* **R1's release model.** `≈ 1.81·B + 22.7·nd + 58.9·n` was established as *separable* — each term
  depending only on its own variable — by holding topology fixed and varying unrelated heap
  (64 KiB-4 MiB, flat to 0.0034 %) and nesting depth (1-8, flat to 0.0132 %). It never varied
  **cumulative revocations within a domain**, and that is the variable that moves: `REVOKE_NODE`
  re-enters its FSM per visited node with revoked nodes never spliced out, so round *r* costs
  *r+2* dependent reads. Measured growth ~12x within a domain. The model is not wrong; it is a
  **snapshot at low cumulative-revocation state rather than a law**, and §7t should say so.
* **The reclamation design's capacity figure.** "≈1.07 x 10⁹ lifetimes before the first index
  retires" holds only under perfectly uniform round-robin reissue. Under lowest-free-first or LIFO
  a mint/revoke loop returns the same index every time and retires it after 16,384 reclaims **of
  that index** — a figure optimistic by up to 65,532x, and **not computable at all until a reissue
  policy exists**, which the design never states.

**Quantified 2026-09-15 by the RTL lane, with a caveat that must travel with the number.** The
unspliced walk costs exactly **3.000 cycles per dead node crossed** — `cost = 3N + 249`, fitting
exactly at **ten** points from N=8 to N=1024 (8, 9, 10, 12, 16, 64, 128, 160, 512, 1024), reproduced
value for value across separate runs and separate testlist files; spliced is exactly **0.000**, 328
cycles at six different N, identical to the cycle. Crossover at N ≈ 26, and at N=3072 it is 96,822 against 516.

**The intercept is scoped, the slope is not — and the bound now has a mechanism rather than an
empirical edge.** State it as *"the intercept of a fit valid for N >= 8"*. An earlier reading that
revoke's fixed cost "steps by roughly 4x somewhere in 1..8" is **withdrawn**: there is no single
step. Exactly two rungs sit exactly **+91 cycles** above the fit, and they are **N=6 and N=7**.

The cause was measured rather than inferred, by a matched-pair manipulation. The write-through
dcache write buffer is **8 deep** (`CVA6ConfigWtDcacheWbufDepth`); halved to 4 and re-run, the +91
pair moved from {6,7} to {2,3} — a shift of **exactly 4** — while every rung from N=8 upward stayed
**byte-identical** between the two configurations. So the pair sits at depth−2 and depth−1, tracks
the buffer depth exactly, and the perturbation provably does not reach the fitted range. Tree
restored and re-verified afterwards.

**A caution about the lower bound that is worth more than the bound.** At the default depth **N=5
lands exactly on the fit — and is not support for it.** The sub-depth curve rises at ~60
cycles/node and merely *crosses* the slope-3 line there. **A point that agrees for the wrong reason
is how a fit range gets overstated**, and it would have been counted as evidence by anyone
extending the range downward until agreement stopped. Do not count N=5.

None of this touches the 3.000 slope, the spliced 0.000, the crossover at N ≈ 26 (inside the range),
or the conclusion that splicing removes the term entirely.

**Do not quote 3 cycles/node as the correction to R1's model.** It is an **L1-hit** cost and therefore
a **lower bound** on what splicing saves. The dcache holds exactly 2048 nodes, the fit holds to 1024
and breaks at 2048, and a speedtest1 run mints 43,355 nodes — **21x the dcache** — so the real
workload sits entirely in the cold regime the ladder reaches only at its last rung. The cold per-node
slope rests on **a single segment** and is not a constant.

Neither could have been caught by repeating its own measurement, because repetition varies the
variables the design already chose. The question that catches both is asked before the run:
**which variable does this quantity depend on that I am holding fixed?**

Earlier instances of the same shape, for the record: S-12's zero-latency testbench (the store
buffer could never fill, so the reproducer read zero for a day), and a delay-40 near-miss the same
afternoon.

### An emulator result is evidence only about mechanisms the emulator implements

*"The emulator does not show X"* is weak evidence about X until someone establishes that the
emulator **models the encoding or mechanism X lives in**. The named instance: R-11 reads *"the RTL
truncates a capability TOP past a 2 MiB window; QEMU never does"* — but `grep -rn cursorless` over
`capstone-qemu/target/riscv/` returns **zero hits**, while the RTL carries the whole branch
(`ariane_pkg.sv:625`, `:633`, `:666-670`). QEMU never does it **because it cannot**. That is not a
reference model disagreeing with silicon; it is a reference model that lacks the feature, and it
should not be counted as a divergence.

This sharpens rather than replaces the earlier rule that **neither machine can be assumed
conservative relative to the other** — established when the divergences turned out to run in both
directions (Q-04/C-32: the RTL nulls where the emulator keeps; Q-12: the emulator omits clears the
RTL performs). A third case now exists: one side may not implement the thing at all.

**Scope, so this is not over-read.** It bears on **bounds**-behaviour claims, where the cursorless
encoding lives. It does not touch the temporal-safety rows of `tab:safety`, whose mechanism is
node validity rather than bounds encoding. Present harm is **UNRESOLVED**: no artifact has been
found in which the two sides exchange a compressed metadata word. The check that would settle it
is whether any differential test compares a compressed word or a `CAPNODE` result across QEMU and
silicon.

### An intermediate representation you built yourself is not evidence

Contributed by the RTL lane from two of its own errors the same day, and placed here at its request
because the other two rules live here.

Both errors were the same move: take a **representation** of the thing, read it carefully, and treat
the reading as the measurement — without returning to the source that would settle it.

* A generated slice printed as `[93:0]` was read as a field **order**. It is a **width**. The order
  came from the declaration sequence, assumed low-bits-first; Anvil packs MSB-first. Settled in one
  line by the consumer that reads the field — `ex_stage.sv:1207` broadcasts on `!node_wr_req[31]`,
  and the consumer's comment says that fires when validity is written to zero, so that bit is `valid`,
  which holds only under MSB-first.
* A `sed -n '1147,1150p' | cat -n` range was labelled by hand — *"printed line 1 = file line 1147"* —
  and the third printed line read as 1148. It is 1149. Settled by `grep -n`, which carries the line
  numbers with it.

**What makes it actionable rather than a reminder to be careful: in both cases the authoritative form
was cheaper than the one used.** `grep -n` costs the same keystrokes as a `sed` range. Reading the
consumer costs one line against reconstructing a packing. The rule is not "check your work" but
**prefer the form that carries its own provenance**, because it removes the step where your own label
can be wrong.

**It is adjacent to the narrowed-view rule and not the same.** Narrowed view: the query shows you what
you went looking for. This one: the artifact is faithful and your reading of it is not.

**And the uncomfortable half, which the contributing lane volunteered and which no review protects
against.** In the packing case its own auditor had derived the answer correctly, hours earlier, in the
same session. The two were never put side by side — not because either was hidden, but because the
question had stopped being treated as open. Adversarial review runs on questions someone is still
asking; this failure is upstream of it.
