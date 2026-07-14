# Plan: temporal-safety overhead — our system vs CHERI, QEMU-to-QEMU (then RTL)

*Proposal for review before implementation. Direction set by the PI on 2026-07-14
(meeting + Slack). Feeds `paper/evaluation.tex` §\ref{sec:eval-perf-compare}
(stub already added). Companion to the security table (`tab:cheri`) and the
borrow-cost microbenchmark (`sec:eval-perf`).*

## Why (the PI's argument)

The security table is settled and well-received, but the PI reframed the headline:

- CHERI's **deployable** config (async revocation) does **not** catch the corpus's
  temporal defects. Good — that is the security point.
- CHERI's **eager** config (revoke-on-every-free) **does** catch them — so **on
  security, eager CHERI matches us** (modulo row~3r). The PI was explicit: the
  double-free `abort` is discounted (software, not a capability check — already
  fixed in the table), and eager is "a very slow version of what Capstone is
  doing… like a garbage collector."
- Capstone realizes the *same* revoke-at-free semantics as a **fast O(1)**
  capability op. So **the axis that separates the two systems is performance**, not
  security. PI: *"My only argument now is: is this performantly better than the
  hardware?"* and *"show how much faster we are compared to the Cornucopia design."*

Slack (verbatim): *"Cheri-sync is not the default, due to perf overheads. But its
security matches capstone. So to show the improvement, we should test the perf.
distinction on QEMU-Capstone and Qemu-cheri. Hopefully we win there. Please add
that performance data separately in a table."*

## The methodology constraint (PI, explicit)

- **Do not** compare CHERI-QEMU against Capstone-RTL — *"that's incomparable."*
- **QEMU-to-QEMU first:** measure the temporal-safety overhead of **CHERI-QEMU**
  vs **Capstone-QEMU** on the *same vehicle*, so the perf difference is established
  on equal footing. *"measure the number of instructions… some way to measure the
  slowdown."*
- **Then RTL on top:** report the Capstone cycle-accurate number and say "on real
  hardware we are obviously much faster." (The RTL borrow-cost port is staged in
  `tests/rtl-smoke/`; the RTL temporal-overhead run is the follow-on.)
- **Ask Jason** how to set up the QEMU-to-QEMU CHERI-vs-Capstone overhead
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

Recommend **(1)** for the headline number (isolates the mechanism the PI cares
about) plus **(2)** as the applied case if the CHERI-side SQLite harness is cheap.

**Instrumentation.** Capstone: the existing `csrdicount` icount readout
(`tests/runtime-qemu/borrow-cost-probe/`). CHERI: an equivalent QEMU
instruction-count readout (qemu `-d`/plugin, or `rdinstret` under a comparable
`-icount` setup) — the exact mechanism is the main open question for Jason.

## Open questions (→ Jason; methodology)

1. How should we measure **QEMU-to-QEMU runtime overhead for CHERI vs Capstone** on
   equal footing? Specifically, the CHERI-QEMU instruction-count readout analogous
   to our `-icount`/`csrdicount` (a qemu plugin? `-d instr`? `rdinstret`?).
2. Is there a canonical malloc/free-heavy benchmark we should standardize on so the
   two systems run *identical* workloads?
3. For CHERI, confirm the revocation configs to report (spatial-off baseline,
   async default, eager) map to the `malloc_revoke`/`CHERI_CAPREVOKE` knobs
   task-015 already exercised.

## Deliverable

- A **separate performance table** (Slack directive): temporal-safety overhead,
  our system vs CHERI async/eager, on the shared workload, QEMU-to-QEMU
  instruction-count slowdown — fills `paper/evaluation.tex`
  §\ref{sec:eval-perf-compare} (stub in place).
- Then the **RTL cycle-accurate** Capstone number layered on top.

## Lane / sequencing

- Likely an **Agent-B** task (it reuses task-015's CHERI stack + the Capstone
  revoke-on-free allocator; heavy QEMU runs, serialize the rootfs lock —
  [[project_matrix_runs_serialize_rootfs_lock]]).
- **Gated on Jason's methodology answer** for the CHERI-QEMU instruction-count
  readout (question 1). Draft the B task once that lands; do not sink time into a
  CHERI-side counter that won't compare cleanly.
- Independent of the RTL borrow-cost run (`tests/rtl-smoke/`), which proceeds in
  parallel.

## Status

Proposal only — awaiting review + Jason's methodology answer. No runs started.
