# Sublet and PoisonCap application memory study

The primary question is the cost of protecting an application's **internal
allocator boundary**. Default CheriBSD malloc revocation does not by itself
observe those lifetimes. PoisonCap is the implemented nested-allocator
comparator. Keep the existing default-CheriBSD experiment as a separate
reference, and retain its results and failures.
The [full-application campaign contract](application-memory-campaign.md)
fixes the four reusable experiment families, metrics, benchmark units, and
paper figure/claim gates for every admitted application case. The measurements
below record discovery and readiness for that campaign.

## Two matched pairs

| Primary arm | Internal allocator | Platform control |
|---|---|---|
| `capstone` | Original allocator | Same SDK outer heap and resource grants |
| `capstone-sublet` | Sublet at the selected boundary | Same Capstone stack |
| `poisoncap-spatial` | Spatial control of the nested adapter | Published PoisonCap stack, fixed outer policy |
| `poisoncap-temporal` | Same adapter with nested poisoning and reclamation | Same kernel, libc, compiler and outer policy |

Retain `cheribsd-revocation-on/off` as two secondary reference arms. PoisonCap
has a different patched software stack; substituting the latest CheriBSD off
binary for its spatial control confounds protection with platform changes.
Freeze the nested scope, application version, useful work, input and output
oracle across the primary four. Record the allocator layout and adaptations.

Report **absolute accounting and within-platform increments** together:
`Sublet − Capstone baseline` and `PoisonCap − PoisonCap spatial control`.
The spatial control must use the same layout as the PoisonCap adapter. Also
account separately for structural adaptation from the original allocator:
metadata moved out of payload, alignment, backing size and static tables.
Otherwise subtracting the spatial control hides storage needed by both
PoisonCap modes. An optional original-layout control isolates this adaptation;
it is not a replacement for the matched control.

For the requested **protected/original ratio**, the
[metric specification](../../experiments/study/memory-metrics.md) requires
matching baseline definitions. The current Capstone original-layout spatial
control and PoisonCap adapter-spatial control cover different adaptation
costs. An original-layout PoisonCap spatial reference is required before
ranking full adaptation-overhead factors; preserve the existing layout-matched
PoisonCap pair for the separate incremental policy question.

For the original SQLite and FFmpeg inner-boundary experiments, hold automatic
outer libc revocation off in both PoisonCap modes, as in their nested adapters;
explicit inner revocation remains active. The mruby GC-slot campaign instead
holds outer revocation **on** in both PoisonCap modes and changes only the
nested GC policy. Each comparison is within its own matched pair. Neither
setting establishes whole-process coverage equivalence across platforms.
Preserved-default outer revocation has known historical guest failures.
The locally patched P-prime libc is a different platform, never a silent repair
of the published platform. Hash the running libc as well as the kernel and SDK.

## Anchor on the published SQLite workload

The [PoisonCap paper](https://arxiv.org/html/2605.13210v1) uses SQLite 3.22.0
`speedtest1` and a memsys5 integration with quarantine. Its reported nested
revocation run completed 15 of 32 phases; 17 hung or panicked. Treat that as a
reproduction/coverage issue, not an advantage for Sublet. SPEC results in that
paper exercise a different, outer allocation path. They do not establish the
nested allocator's memory cost.

The [published artifact](https://github.com/Yuecheng-CAM/Ajs38RO4Vz-p/tree/731c5720d1a6321d436a8d7b85bf3aa2e4971fe0/sqlite)
contains the full SQLite fork. The archive was already pinned by the
[existing platform recipe](../../ports/ffmpeg/buffer-pool/host/cheribsd/poisoncap/platform.json).
Extract only the needed sources under `$CAPSTONE_TMP_ROOT`; do not vendor them.
Our source audit found:

- SQLite 3.22.0; `src/mem5.c` hardcodes `NESTED_TEMPORAL`.
- The driver reserves 256 MiB for memsys5 before parsing arguments.
- Free-list links move to a separate mapping. At a 16-byte atom, an 8-byte
  link per managed atom is substantial storage; measure the actual mapping,
  including its residency, instead of inferring RSS from the requested size.
- A 4,096-entry quarantine; ordinary threshold checking starts at 16 MiB of
  allocated-or-quarantined bytes and triggers when quarantine reaches one
  quarter of that amount. These are source conditions, not a generic policy
  description inferred from a graph.
- The full-queue path calls `quarantine_flush()` directly. The threshold path
  calls the revoker and then flushes; its return value is not checked.
  Count full-queue drains and failed revocation calls. Such paths cannot be
  credited as low-cost protected reuse merely because SQL returns correctly.
- The driver initializes SQLite before some configuration arguments are
  processed. Check effective configuration; passing `--lookaside 0 0` alone
  is not evidence that lookaside was disabled.
- Twelve of the 32 main phases are commented out in the artifact. The
  remaining phase 160 still says indexed although index creation is disabled.
  `--verify` checks other testsets, not main SQL results. Restore the full
  declared work and add a result/error oracle before claiming an upstream
  benchmark result; successful process exit alone is insufficient.

Preserve an unchanged-source build attempt first. A workload-only driver may
remove the artifact's retained-alias startup demonstration; record that patch.
No security suite is part of this memory study. A passing SQL run is only a
functional result until the selected policy and accounting are observed.

The earlier [readiness pilot](../../experiments/study/results/20260927-poisoncap-sqlite.json)
builds both modes and completes the artifact's 20 active main phases at size 1
on the published libc, with automatic outer revocation off. This is execution
evidence only: no inner-policy counters or independent main-result oracle were
added, and no comparison against Sublet was measured then. The subsequent
[full 32-phase memory pilot](../../experiments/study/results/sqlite-322-memory-20260927/README.md)
matches an independent native oracle in four nested arms, with effective
lookaside off, a memsys5-only Sublet backport, per-phase allocator accounting,
and explicit corrected-policy identity. It reports selected passing memory
reservations and failure attempts; corrected PoisonCap trials at 4.5 and 7 MiB
panic in the published kernel, so a minimum capacity is not established. The
subsequent [budget and selective-adapter follow-up](../../experiments/study/results/memory-followup-20260927/README.md)
finds another protected PoisonCap panic at 7.5 MiB and an incomplete size-2
matrix; its FFmpeg control removes the earlier full-copy adapter's apparent
memory disadvantage. Neither failure establishes a protected minimum. The separate
[mruby lists pilot](../../experiments/study/results/20260927-mruby-lists.json)
passes the full upstream work count on all four original arms, with explicit
remaining budget/counter qualification gaps.

Use 3.22.0 for the reproduction anchor: the matched memsys5-only Sublet
speedtest build is now qualified for the single size-1 pilot. The existing
newer Sublet SQLite integration is 3.53.3 and covers memsys5 plus lookaside.
The first comparison disables lookaside **effectively in all arms** and uses
the 3.22.0 Sublet backport. For later work, keep versions matched; never
compare 3.22.0 against 3.53.3 as a protection delta.
A corrected full-queue path must have its own policy identity and evidence;
keep the published attempt visible. Do not sweep per free as the sole baseline.

## Measurements and plots

1. **Memory versus reclamation work.** For each fixed SQL or application work
   count, plot peak committed backing plus attributable metadata against
   completed revocation passes and adapter bytes poisoned/cleared/copied.
   Show the published batching policy first, then explicitly identified
   policy variants. Include Sublet operation counts and authority metadata.
   Plot different operation kinds separately; one Sublet operation is not
   assumed to cost one PoisonCap sweep.
2. **Reuse delay in allocation events.** At the internal boundary, retain only
   integer address/size observations. Count events from retirement to reuse;
   include allocations never reused as censored observations and include
   failed requests. Plot a CDF and distinct starts per useful work unit.
   Existing outer-malloc hooks do not observe GC slots or memsys5 objects.
3. **Release and refill curves.** Across actual statement finalization, request
   completion, frame-pool return or context reset, report live payload,
   rounded live capacity, quarantined bytes, reusable free capacity, metadata
   and returned backing. Keep process reservations and resident pages distinct.
4. **Capacity under a fixed budget.** Preselect workloads and budgets, then
   report completed work and all failures. Distinguish rounding slack, free
   space unable to satisfy a request, quarantine retention and allocator
   caches. `resident − requested` alone is not fragmentation.
5. **State-preserving pool cost.** FFmpeg RefStruct state can persist across
   returns. Measure actual snapshot retention and copy bytes. Our current
   PoisonCap extraction snapshots both pool types for convenience; selective
   preservation and batching are valid competitor policies. Those snapshots
   are not an architectural lower bound.

A useful hypothesis is that Sublet permits prompt reuse with less quarantine
retention as allocation churn increases. Another is lower payload rewriting
for state-preserving pools. Test both over the complete declared workload
range; retained graphs or buddy occupancy may favor the competitor. Include
negative and crossover results. No result is selected solely because it wins.

QEMU supplies functional evidence and operation counts. It does not supply
comparable native latency, bandwidth, cache traffic or energy. The current
Capstone node collector itself sweeps tags: repeat phase-equality checks with
collection-free, adequately provisioned controls for each new workload before
calling byte/reuse metrics invariant. The twelve old controls do not establish
invariance for a new benchmark. Exclude collector timing and metadata-cost
rankings until the intended reclamation implementation is available.

## Application readiness

| Application and known workload | Current reuse | Missing for primary nested comparison |
|---|---|---|
| SQLite `speedtest1` | Full 32-phase size-1 pilot passes all four nested arms with native SQL oracle | Shared runner, multiple sizes/repeats, corrected PoisonCap kernel panic diagnosis, matched complete platform metadata |
| mruby upstream lists / Mandelbrot / AO | [AO width-8/16 GC-slot campaign](../../experiments/study/results/mruby-gc-memory-20260927/README.md) passes 24/24 full-interpreter processes with native PPM oracles and 32-bin reuse counts in all four arms; generic planner binding now emits the same 24 qualified cells for future runs | Larger AO sizes toward the upstream default and selected physical backing/node cost; lists and Mandelbrot still need four-arm nested measurements |
| CPython pyperformance bodies | Real Capstone interpreter; PoisonCap pymalloc component exists | Link PoisonCap into full interpreter; fixed-work controller; same freelist/GC settings and dependencies |
| PostgreSQL pgbench | Single-user Capstone backend; PoisonCap context component exists | Full backend integration; matching database lifecycle; pgbench server/client support or explicitly labeled SQL-body subset |
| FFmpeg known decode corpus | Full 9.0.1 decoder pool lease-gap study passes 36/36 four-arm 1/4/16-stream runs with three repetitions and exact frame oracle | Wider FATE input coverage and complete platform metadata accounting; encoding remains outside current scope |
| Perl upstream performance cases | Real interpreter port | Identify and implement a common nested allocator boundary on both systems; outer-malloc results remain useful but answer another question |

Component examples and allocator recordings do not qualify as these application
workloads. Do not count six PoisonCap application ports merely because extracted
allocators build. Full benchmark-suite completion is also separate from one
passing benchmark body.

## Campaign organization

The existing [study planner](../../experiments/study/README.md) now accepts
`--comparison nested-poisoncap --profile nested`. The existing comparison
remains the default. Each plan retains four arms, all failed/unavailable cells,
fixed work parameters and seed. The separate
[published SQLite catalog](../../experiments/study/catalog-poisoncap.json)
fixes the version and source without changing the newer application catalog.

PoisonCap planning is available; execution qualification is deliberately closed.
The current application runner's `malloc_revoke_enabled()` check observes outer
malloc, not nested poisoning. Reusing that check would silently mislabel an arm.
Before opening the gate, extend the shared runner and binding contract with:

- Observed internal mode and boundary, policy parameters, all quarantine drain
  paths, revocation result/count, and allocator state at every phase.
- Kernel, running libc, compiler, firmware and emulator hashes; full original
  versus modified platform/adapter identities.
- Same useful-work oracle and input bytes, common inner observer identity,
  measured observer reservation/capacity and no table-overflow verdicts.
- Matched backing budgets, effective configuration, adaptation cost accounting,
  and within-platform controls that differ only in the intended policy.

Keep readiness pilots separate from confirmatory plans. Freeze the qualified
plan before collecting a publishable campaign; use paired repetition blocks,
retain faults/timeouts and do not silently rerun them. Reuse the existing guest
lifecycle and runner rather than introducing per-application VM scripts.
