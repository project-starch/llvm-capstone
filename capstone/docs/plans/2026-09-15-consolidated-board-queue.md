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
the console, that is the cause and it is not a board fault. **One more, found 2026-09-15 and not
optional: set `git config user.name` and `user.email` on that host.** Without them `git commit` refuses
outright, and `precommit-scan.sh --range` bypasses its own committer-identity filter by design (`ME`
collapses to `" <>"`), so every author line in a range is fed to the denylist — which is what
`apollo-paper` measured there as "399 of 400 commits". With an identity configured, a range is clean
until it reaches commits by a *different* identity; on the reference host `dev~40..dev` passes and
`dev~70..dev` blocks with 22 hits, all identity lines, no message or diff text. A board result's push
range is `origin/dev..HEAD` and stays clean.

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

## The paper's experiment catalogue, read 2026-09-15 (and what it changed here)

`experiments/` in the paper repository is the consolidated experiment specification and it has been
**restructured** since our board branch's base: nineteen studies under
`experiments/protocols/{host,emulator,hardware}/`, `studies.json` driving a generated execution map,
beside `METHODS.md`, `EXECUTION.md` and `WORK-ORDER.md`. It is unreachable over the network for these
accounts (403); read it locally with
`git -C <paper worktree> show origin/main:experiments/<file>` at `7f83725`, which the transfer bundle
carries. Five things in it bear on this queue:

* **The sanctioned order** (`protocols/hardware/README.md`): *"The central order is H1, S1, S2, M1, then
  sustained timing. M1 needs a reclamation implementation. Available board time alone does not unblock
  it. Capacity-bounded R1/M2 diagnostics and a bounded P1/M3 comparison may begin earlier within their
  protocols. Record those limits in the result bundle."* Our bounded work is sanctioned in the
  catalogue's own words, and the baseline is explicitly not M1.
* **The repetition rule** (`METHODS.md`, *Repetitions and Timing*): *"For primary FPGA timing, use five
  measured runs per arm or point, spanning at least three boots. A repetition completes the whole
  declared workload. The default is a fresh process or domain, with no workload warm-up."* M2 (15 runs
  per point over 4 boots) and R1 (5 over 8) comply; **the baseline as first scheduled did not**, which
  is why P1 below is now four boots.
* **The capacity rule** (`METHODS.md` rule 5): at most 80 % of usable nodes unless exhaustion is being
  tested explicitly, a no-reclamation point must fit cumulatively including warm-up, *"a shortened point
  has its own configuration label"* — which is what `stop=budget` is.
* **The publication rule**, which settles half of a question this lane had been holding for the lead:
  *"An enforcement or oracle failure stops dependent performance publication. It does not erase the run.
  Small bounded diagnostics may proceed without M1 completion, but cannot establish sustainable
  full-workload performance."* Bounded diagnostics may proceed. The open half is narrower and should be
  asked that way: are S1/S2's `unsafe-success` cells an *enforcement failure* in that sentence's sense?
* **A cross-cutting blocker that is a ruling, not board time.** `EXECUTION.md` requires fault reporting to
  have an independently tested positive control; the `trapctl` rung is that control and `H1-platform.md`
  records that it *"has not returned on the bitstream of record. Its deliberate out-of-bounds load ends
  QEMU without a trap, so it cannot earn the recorded QEMU pass that the preflight's check 13 demands for
  every baked rung, and the same holds for any rung whose intended outcome is a fault. Until the lead
  rules how a fault rung earns that record, the fault half of the FPGA matrix is blocked and the control
  half is not."* So every invalid-operation cell in S1, S2 and H1's fault half waits on that ruling, and
  no amount of board time produces it. It is compounded by M-1 (a domain runs with `mtvec` 0, so a
  capability fault looks like a hang) — which is exactly why M-1 is chronic rather than cosmetic.
* **Three hardware studies this queue never named**, all `fpga: required` in `studies.json`: **P3**
  (attribution of a measured P1 cost — Optional, needs P1's frozen pair and validated H1 counters,
  *"creates no new performance headline"*), **P4** (cache effects of release — Optional, needs H1 and
  R1's fixture plus a survivor-window adapter that does not exist, and must not be fed P2's reuse
  histograms), and **R2** (PostgreSQL context replay — needs H1/S1/S2/**M1** and the real tpcb and
  readonly traces from the artifact owner, which are not committed; do not launch a replay with a
  stand-in). P3 and P4 are board-runnable on the current bitstream **only on an explicit Optional
  assignment**, which `EXECUTION.md` requires and both protocols gate on. R2 is blocked.

Two more rules that bind every boot in this queue: **a run may not be stitched** — *"Full runs cannot be
assembled by rebooting and concatenating prefixes"* (`METHODS.md`), and *"Fresh boots between independent
repetitions are valid. Reboots inside one claimed complete workload, or concatenated prefixes, are not"*
(`EXECUTION.md`) — and **one run is one sample**: *"Do not relabel one run as five samples."* The
arithmetic consequence is worth stating plainly, because it sets expectations for every request for board
time: **one more boot adds at most one repetition toward a point and cannot close any primary timing cell
in any study.**

Also from `EXECUTION.md`: a result bundle is **seven** files — `work-order.md`, `manifest.json`,
`points.csv` (every planned cell *including unsupported ones*), `runs.jsonl`, `raw/` with hashes,
`analysis/` with its reproduction command, `summary.md` — and *"the reviewer regenerates the summary
from raw records"*. E1, R1 and M2 have six of the seven; H1 has one. And `studies.json` on the remote
still reads **R1 `pending`** and **M2 `pending`** although both bundles are committed on the board
branch: owed edits, and they are the paper lane's.

## The board queue, in gate order

### P0 — the hand-over boot (runnable now)

One invocation between the two controls, using the M1 image **rebuilt at `DOMAIN_BASE_VA=0x410000`** and
re-validated so it carries its own emulator-pass record (the prepared `9a01b12a4db639b6` was built at the
script's default `0x10000`, which is the `k800` control's own entry VA: preflight C15 refuses that boot
before the board is touched, as it did on the first attempt) and a one-line list, e.g. `1 drop m1 shared 2097152 --cap 16 --budget 4000`:

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

### P1 — the no-reclamation baseline, FOUR boots (runnable now, on the lead's word)

**Rescheduled 2026-09-15 against `METHODS.md`'s repetition rule**, which the original two-boot schedule
missed: boots 1–3 each run **three repetitions of each of the four retention patterns** (12 invocations,
the M-9 cap exactly) for **nine runs per pattern over three boots**, and boot 4 is the single
deployed-table run to the 80 % budget — a *shortened point* under rule 5, carrying its own configuration
label (`stop=budget`), reported as the bound and a corroborating curve rather than primary timing (five
runs of that point would cost five more boots; that is the lead's call, not a default). `lists/m1-diag.txt`
is now 36 lines and `chain-m1.sh` loops over four boots, giving the capacity boot its own output tag so
it cannot write into boot 1's directory.

`drivers/chain-m1.sh` with the harness rebuilt at entry `0x410000` (not the prepared default-base build11) and that image's own flow-check log. Everything is
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

### P1's OUTCOME (2026-09-15, apollo) and what it changed

Four boots, all `done`. **The pre-registered primary is REFUTED in the informative direction**: scoring the
two brackets apart (the combined metric hid it) `take` is near-flat at about 70 cycles while `give` grows
221 → 1,558 and superlinearly, window slopes 202 / 121 / 341 / 791 / 2,570 cycles per 1,000 nodes — the rise
against cumulative allocations that the driver header named as "the alternative that would matter". The
secondary band (384–391) is missed at 465–707, so the accounting model is refuted and the primary stands, per
its own terms. Range-matched over the common span the four patterns agree (308/327/305/315, sd 7–12), which
confirms empirically that the retention comparison is void without a reclaimer. The discriminator is the
valuable half: the same harness under the emulator has `give/n` flat at 16 with slope exactly 0.00, and
`-icount` counts instructions, so the instruction sequence is constant and the growth is the cost of running
a fixed sequence against a fuller table — which rules out an algorithmic walk in the monitor and does not name
a hardware structure.

**Read with M2 this is one sentence:** the node table is free to READ as it fills (the validity query is flat
across 1,366× of occupancy) and not free to WRITE.

**Three instrument defects the run exposed, fixed 2026-09-15 (harness `cf4a25172cbc521e`, built at
`DOMAIN_BASE_VA=0x410000`; emulator-checked at the capacity that exposed them):**
* `M1_MAXRET` was 2,048 while the target is 10·C = 2,560 at C = 256, so the retaining patterns stopped at
  `stop=buffer` and the release arm's phase 2 — gated on reaching the target — never ran: three of the four
  patterns were really two. **The buffer is an instrument limit, not a property of the system, and a
  measurement must not be bounded by its own instrument**, so it is raised to 4,096 rather than C being
  lowered (lowering C would change what the point is about). Emulator check at C = 256: `drop` reaches
  `stop=target` at 2,560, `release` reaches it at 3,072 with `released=1`.
* Boot 4 could never have measured the bound: a retaining pattern fills its buffer long before the table.
  **It is now the `drop` pattern** at C = 65,532 to the 80 % budget — which, after the finding above, is also
  the high-occupancy end of the release-cost curve and the most interesting stretch in the queue.
* `stale_alias_type` read 1 on silicon against a pre-registered 2. **Not a Q-11 amendment** (Q-11 is a
  subordinate handle after an ANCESTOR's revoke; this is a delin'd alias after its own object's release), and
  the boring explanation must be excluded first: the RTL's type numbering and the numbering `sublet.h`'s
  constants use differ by one, so 1 and 2 may be one capability in two conventions. The check is off-board —
  the same probe under the emulator and in simulation at the flashed revision — and only if all three still
  disagree is it a new entry.

### NEW — the occupancy-versus-object pair (ahead of the ride-along, behind P2)

R1 measured release as O(1) per OBJECT (593/612/608 cycles over a 64× size range) and REVOKE at 23.0 cycles
per node, flat in heap and depth; P1's baseline has `give` growing with CUMULATIVE ALLOCATIONS. Those are
consistent only if the growth is driven by table OCCUPANCY — an independent variable neither study varied.
The pair: release the same object shape at about 500, 1,500 and 2,500 cumulative allocations, three
repetitions each, everything else identical. Pre-register both readings before the boot: tracking occupancy
with the object fixed makes "release cost depends on how full the table is" a new and publishable finding;
not tracking it narrows the superlinear claim to something the churn varies incidentally.

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
Pre-registered: counters 5568/37966/32565/37966/5401 and cycles within 0.1 % of arm C's 1,376,190,813. One caution the protocol states and this boot must respect
(`P1-application-cost.md`): *"The existing 911104-byte effective Sublet heap versus 2097152-byte static
heap is not an isolated protection-cost comparison."* The 2 MiB arena staged here is the matched-backing
configuration E2 established for exactly that reason; say so in the §7 entry rather than leaving the
reader to assume it.

Two constraints from the compiler lane, both of which would otherwise waste the boot:
* **Do not stage the before/after pair in one boot.** Two Sublet SQLite workloads exhaust the
  65,532-node table; the before-image number comes from its own boot or from arm C, as it does today.
* **Spend the marginal arms on repetitions of the pre-registered arm** — three draws give the 0.1 %
  band a spread; a single draw on a machine with known nondeterminism is thin for a paper number.
  Not the allocation census, not the `minstret` bracket: both perturb, and the second is a separate
  image by design.

Write into the §7 entry what this boot does **not** establish: it **verifies a pre-registered value** and is
not new primary P1 timing (that would need five runs over three boots per `METHODS.md`); and it does not
exercise the PHI residue
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

### P5 — the two Optional studies: a proposal, not a launch

`P3-cost-attribution` is runnable on the current bitstream the day the lead names the P1 cost to explain
and H1's counters are validated; it explains a measured cost and creates no headline of its own.
`P4-cache-effects` needs a survivor-window adapter written on top of R1's fixture first — desk work
before any boot. Neither may start without an explicit Optional assignment. Put them to the lead as
available capacity once P1 is running; do not queue them unasked.

## Desk work (no board time), in parallel with the queue

* **D1 — finish H1's bundle to `EXECUTION.md`'s seven-item shape**: it is one file (`manifest.json`)
  today, and needs `points.csv` listing every planned cell *including the unsupported ones*,
  `runs.jsonl` linking each attempt to its point, repetition, boot and oracle, `raw/` with hashes,
  `analysis/` with a reproduction command, and a `summary.md` that states what did not run. E4's instruction tests and
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
  the capability register instead of integer arithmetic that untags it. **Where it can be validated is not
  obvious and matters: on the DEPLOYED bitstream the change is untestable, because with delivery broken the
  before and the after both run clean — only simulation of the RTL lane's fix branch can show the monitor
  surviving live enforcement** (RTL lane, 2026-09-15). Validate it there, and separately confirm it is a
  no-op on the current bitstream so the change can land ahead of the reflash. Wait for the RTL lane's
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
| RTL | `9a7bd598c` on `r34-r24-exception-delivery` (**pushed**, the lead allowlisted it; boundary and miss-path sufficiency both measured, so no companion RTL change is needed)| their clean sweep + D3's monitor work → synthesis → the lead's reflash → P3 |
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
5. **Optional assignments for `P3-cost-attribution` and `P4-cache-effects`**, or explicitly not yet.
6. **How a fault rung earns its recorded QEMU pass** (`H1-platform.md`, quoted above). Until it is ruled,
   the fault half of the FPGA matrix is blocked — S1's and S2's invalid-operation cells and H1's fault
   family — while the control half runs. This is the ruling with the widest reach in the catalogue, and
   it is not a board-time question.
7. The METHODS question, now narrowed to one sentence: **are S1/S2's `unsafe-success` cells an
   *enforcement failure* that stops dependent performance publication?** The rest of that rule is
   answered in the catalogue — bounded diagnostics may proceed.

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
