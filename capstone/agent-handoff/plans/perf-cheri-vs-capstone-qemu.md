# Plan: temporal-safety overhead — our system vs CHERI, QEMU-to-QEMU (then RTL)

*Proposal for review before implementation. Direction set on 2026-07-14
(meeting + Slack). Feeds `paper/evaluation.tex` §\ref{sec:eval-perf-compare}
(stub already added). Companion to the security table (`tab:cheri`) and the
borrow-cost microbenchmark (`sec:eval-perf`).*

## Why (the argument)

The security table is settled and well-received, but the direction reframed the headline:

- CHERI's **deployable** config (async revocation) does **not** catch the corpus's
  temporal defects. Good — that is the security point.
- CHERI's **eager** config (revoke-on-every-free) **does** catch them — so **on
  security, eager CHERI matches us** (modulo row~3r). The direction was explicit: the
  double-free `abort` is discounted (software, not a capability check — already
  fixed in the table), and eager is "a very slow version of what Capstone is
  doing… like a garbage collector."
- Capstone realizes the *same* revoke-at-free semantics as a **fast O(1)**
  capability op. So **the axis that separates the two systems is performance**, not
  security. Guidance: *"My only argument now is: is this performantly better than the
  hardware?"* and *"show how much faster we are compared to the Cornucopia design."*

Slack (verbatim): *"Cheri-sync is not the default, due to perf overheads. But its
security matches capstone. So to show the improvement, we should test the perf.
distinction on QEMU-Capstone and Qemu-cheri. Hopefully we win there. Please add
that performance data separately in a table."*

## The methodology constraint (explicit)

- **Do not** compare CHERI-QEMU against Capstone-RTL — *"that's incomparable."*
- **QEMU-to-QEMU first:** measure the temporal-safety overhead of **CHERI-QEMU**
  vs **Capstone-QEMU** on the *same vehicle*, so the perf difference is established
  on equal footing. *"measure the number of instructions… some way to measure the
  slowdown."*
- **Then RTL on top:** report the Capstone cycle-accurate number and say "on real
  hardware we are obviously much faster." (The RTL borrow-cost port is staged in
  `tests/rtl-smoke/`; the RTL temporal-overhead run is the follow-on.)
- **Ask the collaborator** how to set up the QEMU-to-QEMU CHERI-vs-Capstone overhead
  measurement (methodology question — see below).

## The experiment

**Question:** what does *temporal safety* cost, and is contract-point revocation
(ours) cheaper than sweep-based revocation (CHERI)?

**Metric:** runtime overhead of the temporal-safety mechanism = instruction-count
(and, on RTL, cycle) slowdown over an **unprotected baseline**, same workload,
same vehicle class.

**Two systems, one vehicle class (QEMU):**

| System | Vehicle | Temporal mechanism | Configs to measure |
|--------|---------|--------------------|--------------------|
| **Ours (Capstone)** | `capstone-qemu` (icount / `csrdicount`) | revoke-at-free, O(1) capability op (umm_malloc revoke-on-free, [[project_umm_malloc_heap_allocator]] / #78) | baseline (no revoke) vs revoke-on-free |
| **CHERI** | `qemu-system-riscv64cheri` (CheriBSD, `CHERI_CAPREVOKE`) — the stack task-015 already built via cheribuild | pointer revocation (CHERIvoke/Cornucopia sweep) | baseline (revocation off) vs **async** vs **eager** |

Overhead = (protected instrs − baseline instrs) / baseline, per system. The
claim we hope to show: Capstone's revoke-on-free overhead ≪ CHERI's async, and
≪≪ CHERI's eager (the config that actually matches our security).

**Workload.** Candidates, in priority order:
1. **A malloc/free-heavy microbenchmark** — the cleanest isolation of revocation
   cost (many short-lived allocations = many revocations/sweeps). Deterministic,
   directly comparable across both QEMUs.
2. **SQLite** — the case-study workload; but recall upstream SQLite does not run
   CHERI purecap standalone in our env (see `sec:eval-method`), so the CHERI side
   may need the minimal SQLite-lifecycle harness rather than the full engine.
3. **RV8/BEEBS** — already green on Capstone; less allocation-bound, so a weaker
   probe of revocation cost, but a useful cross-check.

Recommend **(1)** for the headline number (isolates the mechanism the comparison cares
about) plus **(2)** as the applied case if the CHERI-side SQLite harness is cheap.

**Instrumentation.** Capstone: the existing `csrdicount` icount readout
(`tests/runtime-qemu/borrow-cost-probe/`). CHERI: an equivalent QEMU
instruction-count readout (qemu `-d`/plugin, or `rdinstret` under a comparable
`-icount` setup) — the exact mechanism is the main open question for the collaborator.

## Methodology — resolved ourselves (no the collaborator gate)

The collaborator is the FPGA/RTL collaborator, not a CHERI expert, so the CHERI-QEMU
methodology is ours to settle. It is not hard: `qemu-system-riscv64cheri` is a
standard QEMU fork, so it has the usual instruction-count readouts —
`-plugin .../libinsn.so`, `-d nochain` counting, or `rdinstret`/`rdcycle` under
`-icount`. We standardise on **`rdcycle`/`rdinstret` under `-icount`** on both
sides (the Capstone side already does this — see below), so the two are directly
comparable with no bespoke op.

Remaining self-answerable items (no external dependency):
1. Canonical malloc/free-heavy workload — start with the microbench used on the
   Capstone side (`revoke-cost-probe`), port it verbatim to the CHERI side.
2. Confirm the CHERI revocation configs (spatial-off baseline / async default /
   eager) map to the `malloc_revoke`/`CHERI_CAPREVOKE` knobs task-015 exercised —
   already documented in `tests/cheri-baseline/RESULTS.md`.

## Deliverable

- A **separate performance table** (Slack directive): temporal-safety overhead,
  our system vs CHERI async/eager, on the shared workload, QEMU-to-QEMU
  instruction-count slowdown — fills `paper/evaluation.tex`
  §\ref{sec:eval-perf-compare} (stub in place).
- Then the **RTL cycle-accurate** Capstone number layered on top.

## Lane / sequencing

- Likely a **B-lane** task (it reuses task-015's CHERI stack + the Capstone
  revoke-on-free allocator; heavy QEMU runs, serialize the rootfs lock —
  [[project_matrix_runs_serialize_rootfs_lock]]).
- **Gated on the collaborator's methodology answer** for the CHERI-QEMU instruction-count
  readout (question 1). Draft the B task once that lands; do not sink time into a
  CHERI-side counter that won't compare cleanly.
- Independent of the RTL borrow-cost run (`tests/rtl-smoke/`), which proceeds in
  parallel.

## Status

**Capstone half DONE (2026-07-14).** `tests/runtime-qemu/revoke-cost-probe/`
(build/run scripts `build-`/`run-revoke-cost-probe.sh`) measures the malloc/
touch/free microbench under three allocator configs on `capstone-qemu`
(`-icount`, `rdcycle` readout). Result (`revoke-cost-probe/RESULTS.md`):

| config | per-op | |
|---|---|---|
| bump (unprotected baseline) | 7.0 instr | |
| revoke-on-free, revoke suppressed | 60.0 instr | alloc-side: +53 |
| **full revoke-on-free** | **65.0 instr** | **+58 over baseline (9.28x)** |

Breakdown: **revoke-at-free itself is +5 instr/op (O(1), cheap)**; the temporal
cost is dominated by making each allocation independently revocable (+53), a
property of the naive Phase-0 allocator, not the revoke primitive. This is the
number to put opposite CHERI's async/eager sweep cost.

**CHERI half DONE (2026-07-14).** `tests/cheri-perf/` (build/run `run.sh`) runs the
IDENTICAL malloc/touch/free loop on CheriBSD purecap under
`qemu-system-riscv64cheri`, `CHERI_CAPREVOKE` off/async/eager via the
cheri-baseline sysctl knobs, `rdinstret` bracket (counts user+kernel, so the
kernel revocation sweep is included). Result (`cheri-perf/RESULTS.md`):

| config | per-op | overhead vs spatial |
|---|---|---|
| spatial (rev off) | 3,760 instr | — |
| async (deployed default) | 23,977 instr | 6.4x |
| **eager (revoke-every-free)** | **14.03 M instr** | **3,731x** |

**Eager — the only CHERI config that matches our security — costs ~14M instr per
free** (address-space sweep), vs our revoke-at-free **+5 instr/op**: ~6 orders of
magnitude at equal temporal security. Async is cheaper (6.4x) but catches 0/11
UAF at the contract point. Method notes: `-icount` infeasible for eager
(~2.8e12 instr/trial); eager is n=2 (3rd trial exceeded the pexpect window, 2
trials agree to 3%); the two QEMU baselines are different vehicles (bare-metal
domain vs full-OS jemalloc) so only within-vehicle overhead + O(1)-vs-sweep
asymptotics are load-bearing.

**Real-workload arm DONE (2026-07-14).** A shared binary-search-tree
build/lookup/destroy workload (`tests/shared/tree_workload.h`, 2000-node live-set,
20k node lifecycles) run on both sides: CHERI `tests/cheri-perf/tree_cheri.c`
(3 configs), Capstone `tests/runtime-qemu/revoke-cost-probe/revoke_cost_tree.c`
(bump/norevoke/revoke). Result:

| | spatial/bump | async/norevoke | eager/revoke |
|---|---|---|---|
| CHERI | 10,095 | 19,281 (**1.9x**) | **16.77M (1,661x)** |
| Capstone (-O2) | 230 | 24,300 | **+5** revoke-at-free (**O(1)**) |

Real-workload confirms + sharpens the microbench: CHERI async is **1.9x** (real
per-op work dilutes the amortized sweep -> the representative deployed number),
eager stays ~16.8M/op, our revoke-at-free stays **O(1) +5 instr/op**
(workload-independent). **Now measured at -O2** (2026-07-15): the capability-value
select ICE that had pinned this to -O0 was fixed
(`history/15-07-2026_03-43-21_cap-select-o2-ice-fixed.md`), so the tree number now
**matches the microbench's +5 exactly** (the earlier -O0 build measured +10; the
delta is O-level-robust either way). Alloc-side +24k is the Phase-0 allocator's
O(n) slot scan, not the mechanism. (Tree domain build now enables `+m` at -O2 —
the workload's key-gen multiply needs it; identical across configs so it cancels
in the delta.)

**Paper DONE:** `paper/evaluation.tex` §`sec:eval-perf-compare` = `tab:perfcompare`
(microbench) + `tab:perftree` (real workload, -O2 +5) + analysis paragraphs.

**Remaining:** the Capstone RTL cycle-accurate follow-on (`tests/rtl-smoke/`,
human-in-the-loop). ~~The -O2 backend cap-select ICE~~ FIXED 2026-07-15.

Original proposal above retained for context; the the collaborator gate is dropped.
