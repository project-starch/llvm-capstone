# Full-application nested-memory campaign

This is the common experiment contract for a paper comparison of Capstone +
Sublet and PoisonCap at an application's **internal allocator**. It replaces
one-off allocator traces or favorable application-specific plots with the same
questions, counters, axes, failure rules, and figure slots for every admitted
benchmark. The [benchmark catalog](../../experiments/study/README.md) fixes
recognized workload sources; the [platform design](sublet-poisoncap-memory-study.md)
defines the four matched arms. This document defines experiments, not new
measurements or a claim that every listed port already qualifies.

The [metric definitions](../../experiments/study/memory-metrics.md) fix byte
ledger identities, simultaneous peaks, paired denominators, reuse cohorts,
recovery thresholds and claim limits. They are the normative definitions for
the figure slots below; a metric name alone does not qualify an existing log.

## Paper question and comparisons

The thesis to test is: *for repeated useful work, can protection of inner
allocations preserve prompt reuse and a stable memory footprint without
requiring growing withheld data storage?* The hypothesis can fail. Sublet's
tables, node state, rounding, or retained graphs may cost more; a well-tuned
PoisonCap adapter may reclaim promptly. Publish both outcomes.

Run four configurations of the **same application version and input**:
Capstone spatial, Capstone + Sublet, PoisonCap spatial, and PoisonCap temporal.
The mechanism effect is measured as protected minus spatial **within each
platform**. Show the absolute byte ledgers as well. Different OS, ABI,
allocator, and metadata implementations prevent a raw cross-platform
subtraction from being called a hardware effect. The published PoisonCap
policy and a corrected policy have distinct identities; neither a drain
without revocation nor a failed kernel run is a successful protected result.
Default CheriBSD revocation on/off remains a separately labeled reference.

## Build and execution denominator

Before admitting a four-arm memory figure, record the **actual compiler argv**
for the application translation unit in every arm, rather than reconstructing
flags from a build script later. Pin the same upstream archive digest, benchmark
driver digest, input digest and logical run arguments. Split source changes into
an upstream base, a platform/ABI/VFS patch, a protection patch, and observer
instrumentation; retain a digest for each stage. The platform patch may differ
between Linux/Capstone and CheriBSD, but must be the same for spatial and
protected builds *within* a platform. A published fork with a new source ID is
a ported derivative, not an identical upstream translation unit merely because
its version string matches.

Compile the application at the same declared optimization level and with the
same effective application feature switches in all four arms. For SQLite,
compare every explicit `SQLITE_*` definition or undefinition, including
`TEMP_STORE`, `DEFAULT_LOOKASIDE`, `ZERO_MALLOC`, and every `OMIT_*` switch;
only `SQLITE_OS_OTHER` is a necessary platform exception. Record target ABI,
compiler binary digest, sysroot/libc, link mode, and VFS separately. These
cannot be made byte-identical across the two platforms. Within each platform,
the spatial and protected arms must share the compiler, target, VFS, base port
patch, and non-protection flags. The protection patch and its metadata are the
intended difference.

Record effective runtime configuration as well: pool size, memsys5 minimum
request/atom, lookaside hit count, page size, journal/temp mode, cache settings,
and any benchmark-specific options. Run the same native output oracle and
verify per-unit useful-work and allocation-request counts; identical SQL
answers alone do not imply identical allocator demand. If a requested common
option fails on one platform, keep the failed build/run and identify the
smallest necessary exception before changing the denominator. Do not quietly
choose different options for the two systems.

The generic [build-comparability gate](../../experiments/study/check-build-comparability.py)
checks the recorded four-arm manifest for common source, driver, input,
runtime configuration, optimization and SQLite flags, plus within-platform
compiler/target/VFS/base-patch identity. It rejects absent compile argv and
unknown optimization. The existing SQLite pilot predates that manifest and
fails this publication gate; its plots remain explicitly exploratory. A
successful compile of the published PoisonCap SQLite fork with Capstone's
deployed SQLite define set (except `SQLITE_OS_OTHER`) demonstrates that this
normalization is technically plausible; the corresponding CheriBSD probe
linked too. Its first guest attempt panicked in the published CheriBSD kernel
(`vm_map.c:6103`, `share->excl`) during `scp`, before SQLite ran. Keep this as
an infrastructure failure, not as evidence about the normalized workload.

## One application adapter, four experiments

An admitted benchmark supplies a pinned upstream workload body and a native
output oracle. A thin host-controlled adapter exposes `prepare`, `unit`,
`release`, and `finish` boundaries. A **unit** completes an independently
checkable amount of useful work: a full benchmark body, decoded stream,
transaction, or query batch. `release` is the benchmark's real object/context
release, not an inserted allocator-wide reset. The same logical unit count,
inputs, mode, and oracle apply to all four arms. A repeated-unit variant is
identified separately from the unmodified upstream reference run.

Predeclare at least one memory-active case per application and a contrasting
case when the upstream suite offers one. Apply the same contract to **every
admitted case**, including cases whose result disfavors the thesis. A benchmark
is admitted only if the chosen unit exercises the nominated nested allocator, its release is
observable, the output oracle passes, and its input/retained-set scale is
reproducible. Workload adaptations and missing axes are published as cells,
not silently replaced by a different benchmark. Use the existing planner's
case × variant × arm × repetition denominator; add adapter fields to the
binding rather than a per-application VM script.

All axes below are fixed by *completed work*, not QEMU elapsed time. A fresh
process begins each cell. Perform one declared warm-up unit, retaining its
state in the ledger, then run the measured sequence in that same process.
Sample before/after every `unit` and `release`, with intra-unit peaks. Three
independent process repetitions are the initial confirmatory minimum, with
platform order alternated by block. Keep raw failures, timeouts, panics,
observer overflow, and unavailable ports in the planned denominator.

| Experiment | Fixed and varied quantities | Primary figure and falsifiable question |
|---|---|---|
| **A. Churn at fixed live demand** | Keep the reference input and retained state fixed. Run 16 consecutive units in one process; report prefixes 1, 4, 16. A 64-unit extension is predeclared for cases whose observer and runtime qualify. Each post-warm-up unit's peak live requested bytes and post-release live bytes must be within 5% of the corresponding 16-unit median; otherwise report this as stateful growth, not fixed-demand churn. | Allocator bytes unavailable for new requests, withheld bytes, reusable capacity and backing versus useful units. Supporting panels show distinct starts, interval union and retirement-side reuse fractions on a fixed follow-up cohort. Does storage demand stay stable over the observed horizon? A changing live set disqualifies the “fixed-demand” interpretation; address reuse alone does not establish memory savings. |
| **B. Live-set scaling** | Run four units each at three predeclared, valid application input/retention sizes (`small`, `reference`, `large`). Choose them in native discovery so observed peak inner live requests increase approximately 1×, 2×, 4×; use the observed bytes, not nominal input size, on the x-axis. Do not alter one arm's work to fit memory. | Peak *simultaneous* allocator-held data, reusable capacity, protection metadata, and total allocator reservation versus peak live requested bytes. Does the within-platform protection tax scale with the live set, object count, or rounding? |
| **C. Burst and recovery** | Four reference units, one unit with 4× the reference live demand, its normal release, then eight reference units in the same process. Preserve any intentional retained graph. | Aligned phase curves for live requests, withheld/quarantined bytes, reusable free bytes, backing grants, and metadata. Report the first post-burst unit returning to the pre-burst envelope, or right-censor at eight. Does freed capacity become usable for subsequent work, and does storage return to the allocator or OS? |
| **D. Fixed-budget progress** | Repeat A at the reference input under an **identical absolute allocator budget** across the four arms, charged for payload, application metadata, quarantine/snapshots, and protection metadata. Let `B0` be the smallest common successful *spatial-control* allocator budget from a separate discovery grid. Freeze page-rounded `B0 × {1, 1.25, 1.5, 2, 3, 4, 6, 8, 12}` before protected runs; run every point in a fresh process and right-censor above-grid requirements. | Completed units and exact output per budget, with allocation OOM, capability fault, kernel panic, timeout, and observer failure shown separately. What capacity permits the declared work? A panic is not an out-of-memory bound. If a platform component cannot yet be charged or capped, label this experiment provisional rather than reporting a minimum. |

A–C are the common application figures. D is required for a *minimum viable
memory* claim, but is not turned into such a claim while kernel metadata or
the published PoisonCap kernel prevents a valid budget test. The existing
SQLite size-1/size-2 trials are discovery data, not the frozen D grid.

## What “working set” and “reuse” mean here

Record these layers at **the same inner boundary** for all four arms:

| Layer | Required observations | Meaning |
|---|---|---|
| Useful demand | Live and peak requested bytes/objects, allocation and release counts, completed units, output oracle | Logical live allocation set, not cache working set. The four arms must execute comparable requests; report any differences. |
| Address reuse | Integer `(start, usable size, allocation index, release index)` observations, in-place realloc and unreused/insufficient-follow-up counts | Distinct starts, union of allocated address intervals, allocation-side reuse fraction and retirement-side reuse within 1/8/64/512/4096 attempts on a fixed eligible cohort. These are virtual address histories, not resident pages. |
| Nested allocator state | Rounded live capacity, reusable free capacity, capacity withheld from reuse, grants/returns to parent, high-water capacity | Separate useful live storage, rounding, reusable cache, and quarantine. `held = live + withheld` only where those categories really partition the same backing. |
| Protection and process storage | Sublet tables and live/retired/reclaimed node counts; PoisonCap link and queue tables, snapshot backing and revoker/page-table state; committed/mapped pages and process RSS where measured | Attribute metadata to the process or platform once, never sum independent peaks. Distinguish fixed reservations, touched pages, allocator-owned bytes, and OS-returned pages. |

Every sample carries application/case/variant, arm, repetition, unit/phase,
source/binary/kernel/libc/SDK identities, effective allocator and revocation
policy, observer capacity, and raw-log digest. The observer stores integer
events only; it must not keep guest heap capabilities alive. Check its own
storage and overflow. Requested bytes and address events at the system allocator (`malloc`)
alone do **not** qualify `memsys5`, GC slots, pymalloc, PostgreSQL contexts,
or FFmpeg pools as measured nested allocators.

Working-set growth has three different meanings and receives three different
axes: current *live requested bytes*, current *allocator-held/committed bytes*,
and cumulative *distinct virtual addresses*. A flat live set with a growing
historical address set indicates delayed or absent reuse; a growing held set with a flat live
set is retention. Neither alone proves physical fragmentation or RSS growth.
Page residency needs measured committed/resident pages. Failure despite free
capacity needs request-size and free-block geometry before being called
fragmentation.

For A, keep the allocation-side reuse fraction and retirement-side reuse
curve separate, using the denominators in the metric definitions. Never
drop unreused eligible retirements from the latter curve. For B/C, report both
absolute bytes and protected/spatial differences within each platform at
matched unit boundaries. Any paired normalization uses the same spatial
reference denominator for both arms and only when that denominator is nonzero.
Do not compare an application's high-water mark in one arm with another arm's
final value. Record drops as well as growth; a late sweep may flatten a curve.

## Benchmark mapping and current gates

The same four experiments apply to each *admitted* case. This table names the
work unit and a real scaling knob; a component port does not by itself qualify
the whole application. Exact input values, source hashes, oracle, and effective
configuration belong in frozen bindings.

| Application and recognized case | Unit, release boundary, live-set knob | Current four-arm gate |
|---|---|---|
| SQLite 3.22.0 `speedtest1 main` | One complete 32-phase pass; finalize statements and reset a private database between passes without resetting `memsys5`; upstream `--size` for B/C | One size-1 pass works in four arms. Repeated-pass adapter and inner address observer missing; size 2 Sublet bounds fault and PoisonCap temporal kernel panic block scaling. |
| FFmpeg 9.0.1 configured decoder, then pinned FATE decode inputs | One independently decoded stream; destroy decoder and return pool leases; input dimensions or simultaneously live decoder contexts for B/C | Whole decoder runs in four arms for sequential streams. Need inner pool address observer, a reviewed concurrent-live knob, broader corpus, and full metadata ledger. The selective PoisonCap adapter is the fairer control. FATE is an input/correctness corpus, not an official performance score here. |
| mruby upstream `bm_so_lists.rb` | One complete upstream work body; drop transient lists and let its normal GC run; list size/retained graph for B/C | Original Capstone/CheriBSD arms pass. Full PoisonCap GC-slot integration and inner observer are missing. |
| CPython pinned `pyperformance` body | One fixed-work JSON/pickle or similar body; release temporary Python objects under the benchmark's ordinary GC policy; input or retained-object size for B/C | Full PoisonCap interpreter, dependencies, output oracle, and inner pymalloc observer missing. A pymalloc component alone is insufficient. |
| PostgreSQL `pgbench` select-only/simple-update | One transaction with its normal context reset; database/input scale or transaction result size for B/C, after checking that this changes context demand | Current single-user backend is not a `pgbench` server. Full backend/PoisonCap context integration and transaction-level oracle required. A SQL-body subset is a separately named derived workload. |
| Perl pinned core workload | One complete case, dropping its temporary values; benchmark-defined input size for B/C | No matched protected nested allocator on both platforms yet. Keep outer-malloc/reference results separate; do not count this as a four-arm nested case. |

If the native benchmark offers no repeatable unit or monotone live-set knob,
admit a different *predeclared* case from the same suite or mark B/C unavailable.
Do not change allocator policy, force GC/sweeps, or introduce artificial retained
aliases just to manufacture a favorable curve. Retained-graph or policy sweeps
are separately labeled sensitivity experiments after the common campaign.

## Figures, aggregation, and publication gate

### Cross-application presentation and paper precedents

The primary overview uses **one metric across every admitted application**,
with benchmark cases grouped under application names. A second layer of
small multiples explains the per-application trajectory. Neither layer may
substitute a different byte quantity when instrumentation is missing.

Relevant precedents, inspected in the papers rather than inferred from abstracts:

* [Cornucopia Reloaded, ASPLOS 2024, Figure 3, PDF page 8](https://www.cl.cam.ac.uk/research/security/ctsrd/pdfs/202404asplos-cornucopia-reloaded.pdf#page=8)
  places normalized peak RSS for several workloads in one grouped figure,
  with the absolute baseline MiB above each group. Its caption also explains
  how a minimum quarantine affects small heaps. Adopt the common metric,
  visible denominator and explanatory context. Our current allocator ledger
  is not RSS, and must retain its own label. That paper uses ratios of averaged
  peaks; this campaign instead computes paired run ratios before aggregation.
* [mimalloc, 2019 technical report, Figure 5, PDF page 15](https://www.microsoft.com/en-us/research/wp-content/uploads/2019/06/mimalloc-tr-v1.pdf#page=15)
  groups allocators by benchmark using normalized peak RSS. This makes
  workload-specific differences visible. Adopt the grouped comparison;
  with only two protected factors, paired dots are less crowded than bars.
  All cases admitted to our campaign remain visible, rather than selecting
  cases after observing a favorable result.
* [Mesh, PLDI 2019, Figures 6–7, PDF page 11](https://people.cs.umass.edu/~mcgregor/papers/19-pldi.pdf#page=11)
  shows Firefox and Redis memory trajectories. The Firefox discussion
  distinguishes similar peaks from lower memory through much of execution;
  Redis illustrates reclamation over a run. Adopt the trajectory companion.
  Our x-axis is completed useful work, not emulator seconds, and our
  selected allocator quantities cannot inherit Mesh's physical-memory claim.
* [PoisonCap, arXiv v1, Figures 6–7, PDF pages 10–11](https://arxiv.org/pdf/2605.13210v1#page=10)
  shows SQLite phase runtime factors and SPEC runtime/DRAM-traffic factors.
  Those figures do not supply the common nested-allocator memory/recovery
  matrix specified here. The opportunity is to measure that missing behavior
  across allocator types, not to reinterpret DRAM traffic as storage capacity
  or claim that the paper evaluated no memory-related effects.

These references motivate the following presentation; they do not establish
our experimental results:

| Sheet | One shared endpoint across all cases | Layout and supported conclusion |
|---|---|---|
| Memory cost | `peak H(protected) / peak H(own spatial)` in A; the separate `B` panel uses the same rule | Two dots per benchmark, baseline at 1×, absolute four-arm bytes alongside. Lower relative cost at the selected allocator boundary, subject to equal ledger and baseline scope. Never replace `H` with pool extent for FFmpeg. |
| Reuse change | Control minus protected retirement-side `R(h)`, in percentage points, on the same eligible cohort and horizon | One row per benchmark, columns for fixed `h`; zero means preserved reuse fraction. Full per-case CDFs accompany the summary. This explains placement/delay, not physical memory savings. |
| Recovery | First sustained return of `H` to the declared pre-burst envelope in C | One row per benchmark, four arm markers; right-censor when recovery is not observed. Separate `B` and `F` explain retained backing versus usable capacity. No numerical averaging of censored observations. |
| Mechanism | A's phase ledger, B's live-demand scaling, C's burst/recovery ledger | Rows are applications/cases; columns are the same experiments with identical definitions. Common work checkpoints; show live demand, withheld storage, reusable bytes and metadata. Explain where overview differences arise. |

Use application order consistently across sheets. Predeclare reference cases,
input levels, warm-up, release boundaries and observation horizon. Normalized
phase curves must use a fixed, declared denominator from the same platform's
spatial reference run, not a moving per-phase denominator that masks growth.
Retain absolute curves and baselines. Never concatenate application events
into a single pooled CDF: a high-allocation workload would dominate it.

There is no need to collapse three applications into one headline average.
Report each application, effect range and the number of admitted cases with
lower/equal/higher values. If a multiplicative summary is later needed after
all gates pass, use geometric means of positive paired factors within cases,
then equal-weight application summaries; report the formula and all individual
effects. This is an optional descriptive summary, not a significance test or
a replacement for the pre-existing median/range case reports. Do not average
absolute bytes across differently sized applications or compute geometric
means of percentage-point changes. Three identical process repeats do not
establish coverage of other inputs.

The current [cross-application reuse overview](../../experiments/study/results/published-policy-20260928/cross-application-reuse.pdf)
is an **exploratory allocation-side** CDF summary from existing measured runs,
at three display cuts (15, 1,023 and 65,535 issues). It preserves the existing
all-issues denominator and has a [per-run CSV](../../experiments/study/results/published-policy-20260928/cross-application-reuse.csv).
It is not the future retirement-side endpoint, a preregistered result, or
evidence that the A–D execution schedules are already matched. Keep the full
CDFs visible; no cross-application score is calculated.

Implementation order for the current three-application core:

1. Complete and validate one disjoint `L/C/Q/F/M/S/Z/B` observer schema at the
   inner boundary. SQLite, GC pages and decoder pools must emit the same
   fields with explicit unavailable values. Record grants/returns and exact
   intra-unit peaks; expose address union separately. Add retirement cohorts
   for the future reuse endpoint without keeping guest capabilities alive.
2. Fix the baseline-kind mismatch and remaining mruby build mismatch before
   claiming equal-scope protection cost. Keep adapter-spatial controls as
   separate policy-isolation measurements; add missing original-layout
   controls rather than relabeling existing data.
3. Qualify `prepare/unit/release/finish` adapters on native output oracles and
   each of the four arms, then execute A identically for SQLite, mruby and
   FFmpeg (one warm-up plus 16 units). Validate stable live demand and record
   natural GC/destructor behavior; no forced sweeps or replay.
4. Freeze the native-discovered live-set sizes and run B/C on those same cases.
   Generate the three common endpoint sheets and their explanatory panels.
   Extend exactly this contract to the remaining applications. Keep D pending
   until the same full budget can be accounted and enforced in all arms.

Use the same panel template for each admitted case: A availability and reuse,
B footprint versus live demand, C phase ledger and recovery, D budget/status.
Show the four absolute arms and the two within-platform protection deltas.
The main cross-platform summary uses **protected / spatial on each platform**:
both baselines equal `1.0x`, with Sublet and PoisonCap factors plotted per
application for the same declared byte metric. Calculate factors within paired
repetition blocks, then summarize them. Show absolute bytes alongside these
factors; the PoisonCap spatial adapter already includes structural adaptation
cost whereas Capstone's current spatial control uses the original allocator.
Before comparing these ratios as full protection overhead, qualify an
original-layout PoisonCap control and record matching `baseline_kind` values
on both platforms. Preserve the existing adapter-spatial arm for policy
isolation. The metric specification fixes the exact formula, baseline-scope
gate and zero-denominator rules.
The paper's main figure can select one representative app of each allocator
type, but supplementary figures and a table retain **every predeclared case**
and failure. Aggregate only after per-app plots: equal weight per application,
median across its cases/repetitions, then a distribution of paired effects
across applications. Do not pool millions of allocation events from one app
to drown out another or infer significance from three deterministic repeats.

The claim gate is: exact useful-output oracles; complete inner-boundary event
accounting; stable work and effective policy; complete four-arm denominators;
matched within-platform controls; metadata and observer ledger; collection
invariance controls for the temporary Capstone QEMU node sweep; and successful
completion under the stated configuration. For a physical-footprint claim,
add attributable committed/resident pages and kernel metadata. A missed gate
becomes a visible limitation, not an excluded row. Existing SQLite temporal
panics, SQLite size-2 Sublet faults, and the selective FFmpeg counter-result
remain visible when deciding whether the thesis survived.

Implement the contract with host-side Python descriptors and the shared
persistent-guest runners. Keep benchmark sources, inputs, build trees, raw
transcripts, and exploratory grids under `$CAPSTONE_TMP_ROOT`; commit only
pinned descriptors, compact checked records, plots, and methods. Freeze the
confirmatory plan after qualification and before observing its protected
results. No new benchmark submodule, application-specific VM manager, or
allocator-trace replay is needed.
