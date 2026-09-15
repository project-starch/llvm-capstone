# Consolidated board plan for `apollo-board` (2026-09-15, evening)

## Context

The lead asked for one plan carrying the whole project's current state, to be handed to `apollo-board`
to launch. Two lanes moved to the apollo server this afternoon: the board lane
(`docs/plans/2026-09-15-board-lane-handover.md`) and the paper lane
(`docs/plans/2026-09-15-paper-lane-handover.md`, with its draft audit
`docs/history/15-09-2026_15-44-40_sublet-draft-audit.md`). Those documents say **how** to work. This
plan says **what to run, in what order, and on which gate**, and it is built from each lane's own
statement of its state rather than from this session's memory of it. Repository state: dev
`49a0a9bdd745`; the board branch `board/e1-s1s2-hardware` at `a9bafd8` (E1, R1, H1, M2 bundles),
unpushed, travelling to apollo as a git bundle. The board is one serialized resource and passes to
`apollo-board` when P0 below reads `done`.

**Environment prerequisites are assumed complete** (the lead's call): the driver's Python dependencies
on the apollo host — `python-socketio[client]`, whose `[client]` extra is `websocket-client`, plus
`aiohttp` for the dry-run — and the `docker` group re-login for RTL simulation. If a boot cannot open
the console, that is the cause and it is not a board fault.

**Delivery.** This plan lands in the repository as `docs/plans/2026-09-15-consolidated-board-queue.md`
(one commit, scanned by absolute path before commit and `--range` before push), and is then handed to
`apollo-board` by message with its path and commit. It supersedes the queue in §5 of the board-lane
handover, which stays authoritative for *how* to work.

## Where everything stands (one table, so the successor needs no archaeology)

| thing | state |
|---|---|
| dev tip | `49a0a9bdd745`; the drivers, chains, lists, the pinned `lpc` and both handovers are on it |
| the follow-on plan's items | F2 done (M-11 GATED; both bundle generators read through the transcript module), F3 hygiene done / **push is the lead's**, F4 done (§7w, Q-12 filed), F5 done with its pre-read correction (§7y), F6 prepared and not launched, **F1 not done** (the fix is not on dev) |
| board branch | `board/e1-s1s2-hardware` at `a9bafd8`, **14 commits ahead of the paper remote's `7f83725`**, carrying `H1/`, `M2/`, `R1/`, `S1S2/` (and `a11-nginx/`); it lands as a **merge, never a fast-forward** — a fast-forward would revert the paper's restructure |
| open registry items the board touches | C-32 open (design pending on dev), Q-04 open (masks C-32 on every emulator pass), R-24 refuted and needing a ruling, R-34 demonstrated in simulation and handed to the RTL lane, R-12 characterised (no reclamation; ~65,532 then a visible hang), R-33 open at both ends now, M-1 open and ours, M-11 gated, Q-11/Q-12 with the collaborator |
| H1's own manifest | already records `tighten`/`shrinkto` as **not probed** — which is what the P1 ride-along closes |

## Three corrections that would otherwise cost a run

1. **Q0 cannot be a control-only boot.** `board-r1e4.sh:112` refuses a boot with no invocations
   (`[ "$NINV" -ge 1 ] || fail`), so the handover's "an empty invocation list" aborts before the board
   is touched. Q0 is a **one-invocation** boot: control → one trivial harness run → control.
2. **F1's scan gate is enumerated by FUNCTION NAME, never by offset** (the compiler lane: the same
   `renameResolveTrigger` site landed at three different offsets across three builds of one source).
   An offset-pinned gate reports a failure that is not one.
3. **R-34's confirmation pre-registration is "the untagged arm TRAPS with cause 24"**, not "enters
   debug mode" (the RTL lane: the `DEBUG_REQUEST` renumber ships in the same batch as the delivery
   fix). And the test is expected to **reach its final report and then time out**, because completing
   requires a store to the host interface through an integer base, which the fix correctly traps —
   the timeout after the last print is the healthy shape, not a lost run.

## The board queue, in gate order

### P0 — the hand-over boot (runnable now)

One invocation between the two controls, using the M1 image already validated on the apollo host
(`9a01b12a4db639b6`) and a one-line list, e.g. `1 drop m1 shared 2097152 --cap 16 --budget 4000`:

```
R1_BOOT=1 R1_OUT_TAG=q0 R1_BOOT_TAG=q0 R1_BOOT_DESC="apollo hand-over: control, one trivial harness run, control" \
R1_PREREG="RESULT k800 retval=4 twice; 1 x speedtest1-ran=0x4EB1xxxx; one 'R1 m1 end ... stop=target'; marker done" \
R1_IMG=<build11 dom> R1_HASH=9a01b12a4db639b6 R1_LIST=<the one-line list> R1_HOST=<sqlite_host_rr.user> \
R1_QEMU_LOG=<flow check drop boot.log> R1_QEMU_GATE='R1 m1 end' R1_QEMU_GATE_MIN=1 \
bash capstone/tests/rtl-smoke/drivers/board-r1e4.sh
```

Pass = both controls at `retval=4`, one `ran` code, marker `done`, and no runner left behind (check the
task list). Report it; the state doc header then names `apollo-board` the board lane, and this session
becomes backup, hands-off. No work order needed: P0 is an instrument proof, not a measurement.

### P1 — the no-reclamation baseline, two boots (runnable now, on the lead's word)

`drivers/chain-m1.sh` with the apollo build11 image and the flow check's log. Everything is
pre-registered in the driver header: **primary** = `take_cyc/n + give_cyc/n` flat in cumulative
allocations (least-squares slope × the run's allocation range below 1 % of the mean; last quarter's
mean within 1 % of the first's); **secondary, scored apart** = the sum in the band 384–391 raw cycles
per allocation. Labelled **"no-reclamation baseline, NOT M1"** everywhere; boot 1's four patterns at
the 256-entry capacity are **diagnostic by capacity, not by platform**; the retention comparison is
**void** until a reclaimer exists (no reclamation ⇒ one node per allocation whatever is retained).
A flat curve at a different magnitude confirms the primary and refutes the accounting model, which is a
result, not a failure.

**Ride-along worth taking in boot 1** (the paper lane names it as one of only three board runs that
move an evidence state): H1's linear-copy family is complete except **`tighten` and `shrinkto`**, the
last two instructions in it untested on silicon. `--series linear` already runs eight arms through
`lcc_type_after_op` (`sublet/r1/r1_slots_pools.c:392-410`) with `expect_spec`/`expect_r21r22` per arm;
adding the two is a harness edit of the same shape, an emulator pass, and two extra list lines in a
boot that uses 4 of its 12 invocations. Keep it a **separate study**: its own work order, its own
records (`study_id` H1, the same `boot_id`), its own bundle addendum — never mixed into the baseline's.

### P2 — F1's C-32 confirming boot (gated on the lead's ruling, then the compiler lane's push)

The paper lane ranks this **first in value**: it is the only queued run that removes a caveat from a
headline number (the 1.607× compound currently carries "bounded-prototype diagnostic, one setup
function at `-O0`"). It is blocked, so P1 runs first — but **if the merge lands while P1 is in
flight, let P2 preempt it**; the baseline's boots will still be there.

Sequence when the compiler lane gives the dev hash: rebuild the toolchain from it (never during a
suite), rebuild cell ⑥ at pure `-O2` (`ports/sqlite/build-sqlite-silicon.sh`, `SQLITE_OPT_LEVEL=-O2`,
no `SQLITE_OPTNONE_FUNCS`), run `tests/movc-cfg-scan.py` — gate: **0 integer-only sites other than
`renameResolveTrigger`'s block-entry copy live around its back-edge, named by function**, mixed and
opaque buckets reported beside — then SQLLogicTest on the `-O2` image as the emulator pass with the
`movc` density read back (about 17.3k, not about 6.7k), then one `drivers/board-c6var.sh` boot.
Pre-registered: counters 5568/37966/32565/37966/5401 and cycles within 0.1 % of arm C's 1,376,190,813.

Two constraints from the compiler lane, both of which would otherwise waste the boot:
* **Do not stage the before/after pair in one boot.** Two Sublet SQLite workloads exhaust the
  65,532-node table; the before-image number comes from its own boot or from arm C, as it does today.
* **Spend the marginal arms on repetitions of the pre-registered arm** — three draws give the 0.1 %
  band a spread; a single draw on a machine with known nondeterminism is thin for a paper number.
  Not the allocation census, not the `minstret` bracket: both perturb, and the second is a separate
  image by design.

Write into the §7 entry what this boot does **not** establish: it does not exercise the PHI residue
(`renameResolveTrigger` is reached per trigger and the main testset defines none), so a pass says the
fix works and says nothing about whether the residue bites.

### P3 — R-34's confirmation boot (gated, and its worth is a question for the lead)

Blocked several deep: the RTL lane's batch is at `c77c65324` on `r34-r24-exception-delivery` with the
renumber audited and the delivery fix written, but **the sweep is not clean and no hash goes to synth
today**; then synthesis; then a reflash, which is ask-first. Pre-registration when it comes: the
write-only arm 27, bounds 28, misaligned 4 and 6, the store side 27/6/28, and the untagged arm
**cause 24** (correction 3 above), with the expected post-report timeout.

**Put to the lead before a reflash is spent on it:** the paper lane's position is that this boot
changes no sentence in the manuscript — the plain-data rows are already unsupported three independent
ways (the directed simulation with both gate inputs witnessed, the stock `rv64mi-p-ma_addr` failing
with capmode never set, and the monitor's `rdtime` path as uncontrived silicon on every clock read) —
so if the reflash is spent, it should be for the RTL lane's reasons, not the paper's.

## Desk work (no board time), in parallel with the queue

* **D1 — finish H1's bundle.** It is one file (`manifest.json`) where E1, R1 and M2 carry manifest,
  `points.csv`, `runs.jsonl`, `summary.md`, `raw/` and `analysis/`. E4's instruction tests and
  calibration are real readings, already in §7v, with the transcript captured: completing the bundle
  from data in hand costs no boot and is the paper lane's explicit request. (The **missing work
  orders** on E1/R1/H1/M2 stay stated, not backfilled — a template exists to be filled before
  measurement, and a retrospective copy would be theatre.)
* **D2 — `sublet/r1/m1-bundle.py`**, modelled on `m2-bundle.py` (same record shape, transcript module,
  `analysis/` with a rerunnable script), parsing `R1 m1 start` / `R1 m1 snap` / `R1 m1 end` and
  emitting the two cost curves per snapshot interval. Write it before P1's boots so the bundle is not
  improvised afterwards.
* **D3 — the monitor's readiness for R-34's fix, which is now a prerequisite for that bitstream
  rather than a follow-up.** The RTL lane's early sweep has `capsbi-init`, `ccsrrw` and `cbnz` timing
  out under the delivery fix: with delivery working, the `NOT_CAP` clause fires on every plain access
  through an integer-derived base in machine mode with capmode set, and the monitor does exactly that
  — `sbi_capstone.S:113` computes `add t5, sp, t5` and then stores `sd a0, 16(t5)` through the
  integer result, and the `rdtime` emulation does the same at every clock read. The monitor is this
  lane's file. The shape of the fix is to keep the capability base: `CINCOFFSET`/`CINCOFFSETIMM` on
  the capability register instead of integer arithmetic that untags it, audited and validated on the
  emulator against the current bitstream first (it must be a no-op there). Wait for the RTL lane's
  full sweep numbers before scoping it — they promised a count rather than an impression.
* **D4 — a work order per measurement**, now that `experiments/EXECUTION.md` and
  `experiments/WORK-ORDER.md` are visible (paper remote `7f83725`, absent from the pinned submodule,
  carried in the transfer bundle). P1, P2 and P3 each get one filled **before** the run, with no
  REQUIRED field unresolved: scope, question, exclusions, operator and **reviewer** (the lead
  assigns), the full paper commit, every submodule commit, inputs and hashes, positive controls and
  oracles, expected faults and survivor checks, parser validation against a known-good log **and a
  deliberately wrong oracle**, peak and cumulative node capacity, stop conditions, exact commands with
  cwd and outputs, timer brackets stated as exclusive and nested, repetitions with seed and boot order.

## Tracked dependencies (other lanes' work; do not do it, track it)

| lane | state today | what unblocks the board |
|---|---|---|
| compiler | design A at `46c53b7b6ae2`, `19bc05cf21b1` (pushed, their branch); the dev merge `d257f1abad4f` is **performed, verified, lit 106/106, unpushed** — the scan's range mode blocks on the collaborator's author lines, and the scan is working as designed (author identity is scanned deliberately) | the lead's ruling, then their push, then they give the dev hash → P2 |
| RTL | `c77c65324` on `r34-r24-exception-delivery` (renumber audited, delivery fix restored from upstream #2528's diff, lint at baseline); sweep early and already showing monitor-side timeouts; **no hash to synth today** | their clean sweep + D3's monitor work → synthesis → the lead's reflash → P3 |
| paper → `apollo-paper` | handover sent; `studies.json` still has M2 `pending` though its bundle is committed; most states are blocked on non-board work (M1's reclaimer, M-8's port fix, S3's corpus), and P1 addresses only the resource half | nothing; address `apollo-paper` from now on |
| synth | per-module area for H1 done; next is a bitstream when a hash exists | nothing |

## Decisions to put to the lead (with the queue running, not before it)

1. **The author-line ruling** — it holds P2 and every optimised Sublet number. The compiler lane
   withdrew its own first suggestion and frames the choice as: rewrite those 11 commits' authorship and
   force-push (all lanes re-sync), or keep a narrow identity allowlist outside the repo beside the
   denylist. Not a lane's call, and no lane touches the gate.
2. **P1's go/no-go** (the paper lane recommends taking it, headline the cost curves).
3. **Whether P3's reflash is worth spending** on the paper's account (the paper lane says no).
4. **A reviewer's name** for D4's work orders.

## Verification

* **P0:** both controls `retval=4`, one `ran` code, marker `done`, task list clean; `make
  experiments-check` passes in the apollo paper worktree once the branch bundle is fetched.
* **P1:** every arm reaches `stop=target` (boot 1) or `stop=budget` (boot 2); `minted` = fixture + one
  per allocation; retained 0/16/640/0 as the emulator flow check read; the flatness criterion scored
  and reported before the magnitude band; the bundle regenerates its summary from the raw records.
* **P2:** the scan reads zero integer-only sites outside the named function; board counters equal the
  emulator's to the unit; each of the three draws within 0.1 % of 1,376,190,813; §7s updated and C-32
  closed with the scan as its gate.
* **P3:** the R-34 folder's test inverts — the arms that retire today trap with 27/28/4/6 and the
  untagged arm with 24 — and the post-report timeout is recorded as the expected shape.
* **Every boot:** control first and last, the pre-registration in the driver header, results cited by
  image hash from the run-scoped transcript, the node budget inside 80 %, one Sublet workload per boot.

## Documents to update (with each item)

* Per boot: a §7 entry in `docs/ref/fpga-silicon-measurements-for-paper.md`, the state doc header, an
  `ISSUES.md` box when an issue moves, and the bundle on `board/e1-s1s2-hardware` with
  `experiments-check` and the scan run from inside the worktree.
* P2 closes C-32 (FIXED, scan as gate) in `ISSUES.md` and §7s; P1 lands `M1-baseline` under
  `experiments/results/`; D1 completes `experiments/results/H1/`; D3 lands a history note plus the
  monitor change on its branch.
* This plan and the two handover documents live in `docs/plans/`; `apollo-board` owns this one from the
  moment P0 passes.
