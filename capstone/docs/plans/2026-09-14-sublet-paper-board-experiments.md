# Plan: board experiments for the Sublet paper (2026-09-14, evening)

## Context

**The article.** `capstone/paper-nested-allocators` (submodule added 2026-09-14 by the peer lane) is
*Sublet: Delegating Temporal Safety to Custom Allocators*, at `b0d7510` = `origin/main`, tree clean,
no Overleaf remote configured (GitHub `main` is the source of truth). It is not the pointer-safety
paper the 2026-09-14 proposal addressed. Its evaluation is a catalogue of 18 studies
(`experiments/studies.json`, protocols `experiments/*.md`, measurement standard `METHODS.md`), each
with an evidence state; the manuscript keeps a placeholder until a study delivers. **The manuscript
cites none of this lane's silicon numbers** (paper lane, grep of every `*macros*.tex`: no 911104, no
1.0951, no 1.18/1.19, no cycle counts; the only board-derived macros are the 65,536-entry table and
the 2.8 % nginx node share). Today's re-basing therefore invalidates nothing, and the numbers enter
clean when a study takes them.

**What the paper wants from the board** (Core studies, evidence state, the quantity):

| study | state | wants from silicon |
|---|---|---|
| P1 application cost | pending | complete speedtest1 cycles, five arms, **O2 in every capability arm**, 128 MiB arena, 5 runs over ≥3 boots, M3 alongside |
| R1 release cost | pending | release-to-reuse cycles on a slots-and-pools harness: affected-nodes 1/4/16/64/256, released-bytes 4–1024 KiB, unrelated-heap 0–4096 KiB, depth 2/3/5/9, object-free 16–4096 B |
| M1 node reclamation | partial | drop / retain-ring / retain-pressure / release-retained arms to 10 C cumulative node allocations on a *reclaiming* configuration |
| S1, S2 lifetime and authority | partial | the case-by-arm matrix repeated 3× on hardware (emulator rows exist since today; "emulator results alone do not fill FPGA cells") |
| S3 real-bug corpus | pending | detection on FPGA, gated behind H1+S1+S2 |
| M2 access path | pending | working-set and active-nodes pointer-chase series on the board; memory-latency series in the simulator |
| M3 memory ledger | partial | nine disjoint categories, occupied and reserved apart, on P1's configurations |
| H1 platform | partial | manifest, instruction tests 3× on hardware, counter calibration, one bounded exhaustion diagnostic, synthesis reports for three seeds |

**Three blockers, found while reading, that shape the order:**

1. **M1's reclaiming configuration does not exist.** `capstone_rev_node.anvil` mints at two sites, both
   `head++`, no free list anywhere in the RTL, exhaustion is a deliberate stall (R-12: one size-1 Sublet
   run mints 43,355 of 65,532 nodes; a second wedges). P1 and R1 list M1 as a hard dependency; both
   protocols allow **capacity-bounded diagnostics, labelled as such**, and forbid publishing complete
   long-running FPGA performance without it. The reclaimer is an RTL-lane design question, not a board
   experiment.
2. **P1's O2 requirement is unbuildable today.** The SQLite domain builds only at `-O0`
   (`build-sqlite-silicon.sh:42`); at `-O1` the backend stops with `Cannot materialize arbitrary
   >64-bit constants as capabilities` (C-17, lowering of an i128 `select_cc` via two halves queued as
   B6). Every §7 row is `-O0` SQLite over `-O1` support code. P1 says the O0 replays "are not timing
   baselines for this study". Compiler-lane dependency.
3. **P1's complete workload vs the node budget.** P1 wants the largest of sizes 100/30/10/3/1 that
   completes in all capability arms; the Sublet arm is pinned to size 1 by the budget. The paper lane
   names three honest outs and calls the choice the lead's: stitched multi-boot phases under a stated
   protocol, M1 first, or scope P1 to size 1 explicitly.

**What already exists and is reused:** the §4g cells ②/④/⑤/⑥ (custom-plain, custom-spatial without and
with the pool, custom-sublet) on silicon and QEMU; the `sublet.h` primitives and the SQLite/PostgreSQL/
nginx Sublet patches; the S1/S2 probe images `ngx_uaf.c` (7 stages), `ngx_subpool_test.c` (13 phases),
`pg_subpool_test` (8 levels) with their QEMU gates (`run-nginx-uaf.sh`, `run-subpool-test.sh`); the
ladder timing harness and `DOMAIN_BASE_VA` linking (`build-ladder-fpga.sh`, `build-sqlite-silicon.sh`);
the July per-primitive costs and the fill probes (`fillcost`/`fillsd`/`fillwarm`: the byte-linear INIT
fill on silicon, §5); the revoke-reshare probe and `rtpc`; the R-12 exhaustion reading (head `0xFFFF`,
trap latch, bounded by the watchdog); the staged-probe driver with pre-registration, hash-cited
readings and the entry watchdog.

**Where each result lands in the manuscript** (read from `parts/`): E1 → `tab:safety`'s FPGA cells and
the S2 hierarchy matrix (`eval-safety.tex`); E2 → `fig:performance`'s left panel, cycles relative to
custom/spatial, and the memory panel with E5; E3 → the three panels of `fig:release-cost` (affected
nodes, released bytes, unrelated heap, plus the depth sweep); E4 → `tab:hardware`'s five `\pending`
rows (board/core/bitstream revision; clock, caches, memory latency; compiler/ABI/runtime revisions;
node format and reserved bytes; LUTs/registers/BRAM/tag storage) and `tab:eval-setup`'s FPGA row;
E6 → `fig:resource-scaling`'s right panel; E7 → its left panel and `tab:safety`'s "retained stale
pointer after node reuse" row, which the current prototype cannot cross because it reuses no node.

**Decisions taken with the lead (2026-09-14 evening).** Scheduled now: E1, E2, E3, E4 (E6 deferred).
P1 is scoped to **size 1 at -O0, labelled a bounded-prototype diagnostic** in the arm description,
upgraded when O2 (B6) and M1 land; `\sublAdvantageKept` stays unfilled because system-sublet does not
exist, and P1's own rule is to report `protection_cost` alone in that case. Three hand-offs go out now
through this lane's cross-session messages: compiler lane (B6), RTL lane (M1 reclaiming table), synth
lane (H1's three-seed reports and matched baseline).

**Boot order.** E1's three boots first (existing programs, fills Core cells), E2's one boot, E4's one
boot, then E3's three or four once its harness passes QEMU — the harness is built while the earlier
boots run, since only the board is serialized.

## Experiments, in evidence-per-boot order

Every board arm keeps today's discipline: control first, one unknown last, pre-registered readings in
the driver header, results cited by image hash from the run-scoped log, `BUDGET` per arm, ONE Sublet
workload and ONE 128 MiB region-arena workload per boot. Every study result lands in the paper's bundle
format (`results/<STUDY>/<RUN>/manifest.json, points.csv, runs.jsonl, raw/, summary.md`) with the
seeded arm order (seed 20260914), and the measurements doc keeps the narrative row.

### E1 — S1/S2 on hardware: the safety matrix repeated three times (2–3 short boots)

The cheapest Core evidence on the list, and the only one that needs no new program: the three probe
images already encode the S1 cases and the S2 series (inner-life, combined, hostile-child, foreign-
authority, stale-authority, positive) and pass on QEMU as of today's paper commits.

* Build the three images and their hosts from the pinned commits; same-program gate on each host
  (`pg_host.c` and `ngx-guest.c` include `libcapstone.h`, the shared struct, so the #3 module accepts
  them — verify by content, as sw70 taught); QEMU pass record per image and selector.
* Add `DOMAIN_BASE_VA` to `build-nginx-domain.sh` / `build-subpool-test.sh` (the linker flag the SQLite
  build already takes) so the three images share a boot at distinct entry VAs; else one image per boot.
* Boot: control → `ngx_uaf` all seven stages → `ngx_subpool_test` all thirteen phases →
  `pg_subpool_test` all eight levels → control. Three repetitions = three boots (METHODS: 3× on each
  claimed target). Pre-registered per cell: the emulator's status (`enforced-fault` with the same
  mcause, or `rejected`) and the survivor markers; any `unsafe-success` stops dependent timing.
* Deliverable: the FPGA column of the paper's `tab:safety` and the hierarchy matrix; S1/S2 move from
  partial toward measured on hardware; S3 unblocks.

### E2 — P1's matched-geometry pair at size 1 (QEMU first, then one boot)

The only P1 quantity the board can settle now is the one the configuration caveat has been carrying
since sw60: ⑥ (Sublet, arena 1,419,584 rounded to 1,421,312) against ⑤ (memsys5 + pool, 2 MiB static
heap) is a two-geometry comparison. P1 asks for one arena limit held constant across arms.

* No new image: the Sublet cell's arena is a host argument (`run-speedtest1-measure.sh:166`,
  `--arena $SUBLET_ARENA --tables $SUBLET_TABLES`; the domain reads its size off the grant), so ⑥ is the
  same program `ceeded2533a74bce` run with `--arena 2097152` (P1's power-of-two rule; representable, so
  the arena gate passes and R-33's rounding is out of the picture). QEMU icount at size 1 first: oracle
  `112006 38bb59fd`, `HEAP 2097152`, the `sublet:` counters (5,481 splits, 37,874 mrevs) unchanged —
  the arena size is not on the allocator's hot path, so a different count means a different run, stop.
* One boot: control → ⑥ `--arena 2097152 --size 1` (the boot's one Sublet workload) → control.
  Pre-registered: cycles within ±1 % of 2,794,183,730, and ⑥/⑤ on matched geometry against sw75's
  2,551,483,818 (today's two-geometry value 1.0951 is the comparison).
* Add the **system-spatial** arm if the port takes it cheaply: SQLite on the domain's own malloc with
  memsys5 and lookaside disabled (`SQLITE_SYSTEM_MALLOC`, one build knob; QEMU pilot decides). It is
  the third of P1's five arms; system-sublet does not exist and P1 says to report `protection_cost`
  alone in that case.
* Deliverable: P1's `protection_cost` at size 1 on matched geometry, labelled *bounded-prototype
  diagnostic, -O0, size 1*, with the arm table P1 prescribes; the state doc's "configuration vs
  discipline" caveat closes.

### E3 — R1 as a capacity-bounded release-cost series (new harness; 3–4 boots)

The paper lane's highest-value board item: the byte-linear INIT fill is why Sublet costs 9.6 % on
silicon and 1.8 % on QEMU, and R1's released-bytes series turns that single ratio into a curve.
R1 permits "capacity-bounded diagnostics … labelled separately" without M1.

* New domain program `slots_pools.c` (rtl-smoke, built by `build-ladder-fpga.sh`'s recipe with
  `DOMAIN_BASE_VA`), using `sublet.h` for the fixture: one root region, 16 leaves (n varies), depth
  2 by default; three lifetime patterns (individual free, shared death, inner frees then ancestor
  withdrawal); custom-spatial twin without the temporal primitives. Timing per R1 step 3: `rdcycle`/
  `rdinstret` brackets around bookkeeping, revoke, initialization and reissue, plus the outer bracket
  ending after a checked load+store in the returned region. Series and points as the protocol table;
  every point preflighted for node demand (≤ 80 % of C cumulatively per boot — the affected-nodes
  series at n = 256 with five repetitions is the budget's binding point) and for arena bytes (the
  unrelated-heap 4 MiB point needs a CMA region: `cma=` on the command line, the #3 module).
* QEMU first for oracles (survivor checks, stale-access rejection in the untimed companion probes),
  then boots: one per series family, five repetitions spanning three boots as METHODS asks — the
  driver pre-registers the slope sign (cycles linear in released bytes at the fill probes' rate,
  ~2 cycles/byte; linear in affected nodes at the July mrev/revoke costs) and the control arm's
  constant.
* Deliverable: R1's figure (total release-to-reuse cost against nodes, bytes, unrelated heap, depth,
  with exclusive components), the first measured explanation of the silicon-vs-QEMU Sublet gap;
  labelled capacity-bounded, node identifiers not recycled.

### E4 — H1's board half: manifest, instruction tests ×3, calibration, the exhaustion diagnostic (1 boot + desk)

* Manifest from records: bitstream `caplifive_r30r31_1bfff7776` (hash, WNS −12.425 ns, 169,207 LUTs
  placed), core 25 MHz / timebase 12.5 MHz, cache geometry from the CVA6 config, node table 65,536
  entries / 65,532 usable / reset head 3, bytes per node and tag storage read from the RTL, monitor
  `4274268`, module `d04bd83`, toolchain commits, counters (`mcycle`, `minstret`, the `sublet:`
  counters, RCLM/RCPR/RCSH/RCRE, the debug mux).
* One boot: the directed rungs that already exist for linear-move clearing, forbidden linear copy,
  split bounds/alignment, mrev/delin/revoke, subordinate-handle invalidation, INIT-before-use — three
  passes each (they return in seconds), a timer-overhead rung and a dependent-load chase over a
  buffer larger than the caches (the "memory latency" figure `tab:hardware` asks for, and M2's
  calibration), and the bounded exhaustion diagnostic **last** (a Sublet cell's second run: allocation count, occupancy
  `0xFFFF`, outcome = stall, released by the watchdog).
* Synthesis reports for seeds 1/2/3 and the matched baseline core are the synth lane's (never run
  from here); the plan hands the request over with the manifest's field list.

### E5 — M3's ledger on the P1 arms (desk work, no boot)

Nine categories per arm from the images and the run logs: payload and allocator overhead from
`--stats`/`HEAP`, free arena from the arena size, side metadata from the Sublet patch's tables
(`tables` region, `ALEN`), node metadata as 65,536 × bytes-per-node (from E4), tags from the RTL,
runtime (domain stack, glue) from the loader's `Domain requirement`, code+rodata via `llvm-size`.
Occupied and reserved kept apart; the sum reconciled to the created regions.

### E6 — M2's board points (DEFERRED by the lead; new microbenchmark, after E3)

Dependent pointer chase over 64-byte records, custom-spatial vs custom-sublet, working-set
16…4096 records and active-nodes 1…256 identities at a fixed 256-record chase, seeds 1/2/3, 100,000
accesses after two warm-up traversals; five board runs per point. The memory-latency series
(delay 0/10/50/100) is Verilator work under the `rtl-sim` skill, not a boot.

### E7 — M1 and complete P1: what waits, and what is prepared now

* M1 needs a reclaiming configuration (free list or generation-tagged node reuse) in the RTL: an
  RTL-lane design, ask-first bitstream. Prepared here: the four-arm driver (drop / retain-ring /
  retain-pressure / release-retained, snapshots every C/16, isolated stale probes at each
  checkpoint) so the first reclaiming bitstream runs it the day it lands; today only retain-pressure
  runs (= E4's exhaustion diagnostic).
* Complete P1 needs O2 (B6, compiler lane) and M1, plus the lead's scoping decision (size 1 labelled /
  stitched multi-boot / M1 first). Prepared here: the five-arm driver with the seeded order and
  three-boot schedule; E2 is its size-1 pilot at -O0.

## Hand-offs (sent now by cross-session message, as the lead decided; no names, roles only)

* Compiler lane: B6 — lower the i128 `select_cc` (two i64 halves) so the SQLite domain builds at
  `-O1`/`-O2` (C-17); P1's O2 arms wait on it; the proof is the size-1 QEMU pair at the oracle.
* RTL lane: M1's reclaiming node table — a free list or generation-tagged reuse with a stated safety
  invariant (the event that makes a node reclaimable; retained stale aliases must not regain
  authority), then synthesis; also the bytes-per-node and tag-storage figures for H1/M3 read from the
  RTL. Ask-first before any reflash.
* Synth lane: three-seed (1/2/3) utilization and timing reports for the deployed configuration and an
  otherwise matched baseline core (same memory interface, caches, peripherals, part, tool version,
  clock constraint) for `tab:hardware`'s area row; achieved frequency kept apart from estimated.
* Paper lane (reply to today's message): the per-study columns the bundles need for E1–E4, and whether
  `tab:safety`'s FPGA column wants the three repetitions' statuses or one consolidated row.

## Verification

* E1: every cell's status equals the emulator's with the same cause on all three repetitions; survivor
  markers intact; no `unsafe-success`; hosts proven the same program before staging.
* E2: QEMU hash at the oracle and the `sublet:` counters identical to ⑥'s before any boot; the board
  reading inside the pre-registered band; the ratio labelled bounded-prototype, -O0, size 1.
* E3: node-demand preflight per point recorded; slopes fitted only where ≥3 legal levels exist; the
  spatial twin's curve flat where Sublet's rises; the fill rate agrees with `fillcost`.
* E4: each instruction test passes 3/3 with expected faults isolated; the exhaustion arm classified
  `exhausted` with allocation count and occupancy; the manifest's every field sourced.
* All: bundles pass `make experiments-check` in the paper repo; results cited by image hash.

## Documents to update (with each item)

* `docs/ref/fpga-silicon-measurements-for-paper.md`: a §7r per boot; the CONFIGURATION block's
  geometry caveat closed by E2.
* `docs/ref/ISSUES.md`: R-12 gains the H1 exhaustion record; C-17/B6 the P1 dependency line.
* `docs/state/current-next-step.md` / `current-state.md`: the paper's study table as the open list.
* The paper repo: result bundles under `experiments/results/<STUDY>/`; no manuscript edit from this
  lane (the paper lane owns the prose).
* This plan lands as `docs/plans/2026-09-14-sublet-paper-board-experiments.md`.
