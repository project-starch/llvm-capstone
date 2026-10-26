# T1 (Sublet spatial) against T2 (spatial + temporal) on silicon: the sampling plan, and why it is this one

Revision 2, 2026-10-26. It follows an adversarial audit of revision 1 (/tmp/capstone/perf-sampling-audit-2026-10-26.md).
That audit refuted four sentences and found five defects; every change is listed at the end. Numbers marked TBD come
from the pilot. Nothing has been measured on the board yet.

## What is measured, and what it means

**Question.** For each program of CoreMark, RV8 and BEEBS, built as a virtual-profile Linux program on bitstream 2
(capstone-ariane dcc6daf6e), how much slower is it with per-object lifetimes (T2) than without them (T1)?
- T2 is the default heap: a CDERIVE child per allocation, a CREVOKE and a scrub on free.
- T1 is the same heap with `CAPSTONE_HEAP_LIFETIMES=0` and `CAPSTONE_HEAP_SCRUB_FREE=0`: a SHRINK copy with exact
  bounds, no revoke, no scrub.
- The two SDKs come from one dev tree and differ only in heap.c's object. Programs are built at -O2, without
  `+movc-keeps-integer-source`.

**Framing.** This is the marginal cost of per-object lifetimes plus scrub-on-free. It is NOT the cost of temporal
safety as a whole:
- T1 still has every access checked against a lifetime node in hardware, the mapping's node;
- T1 still has the per-mapping lifetimes (heap.c `mapping_group`).

The ratio isolates what per-object temporal safety adds on top of that.

**Prerequisite (fixed before any board run).** Until 2026-10-26, every malloc, realloc and aligned_alloc made a full
delegated service call (heap.c `ensure_ids(256)`, unconditional since cdd3c15a3554). Both arms paid it, and it
compressed T2/T1 toward 1.
- The fix, branch runtime/heap-ids-budget, counts the records the heap takes and asks once per about 255.
- It is landed, and gated on QEMU against dev check by check, before the measured SDKs are built.

**Metrics, per timed iteration.** Each iteration is bracketed by two marks. A mark reads three sums that the perf
build of the module keeps: cycle, instret and CSRUNV entries, accumulated only inside CSRUNV windows, with interrupts
off (`CAPSTONE_VM_BENCH_COUNTERS`; sysfs `bench_{cycles,instret,entries}`). From each iteration we keep:
1. **cycles inside the context.** This is the primary metric. The slowdown is T2/T1 of it.
2. **instret inside the context.** This says how much of the slowdown is extra work.
3. **entries.** Every entry after the first is an escape. The escape itself (the switcher walk back in, the satp
   write's flush of every TLB, the refill) lands inside the next window. The count is reported, so the in-window cost
   of escapes is visible rather than hidden.
4. **end-to-end ns** (clock_gettime). This is wall time, Linux's work included, and it is reported beside (1).

**Read on silicon (probe, board 1, 2026-10-26, lane 7ac87738ef71):**
- The counters work from S-mode, and the Tier 1 contract of this module revision passes.
- An empty region costs 18 entries and about 20,400 instructions; that is the mark overhead.
- A loop of 2·10⁷ instructions reads 1,765–1,989 instructions above the prediction, with 15 more entries than the
  empty region. That is about 125 instructions counted inside the window per extra CSRUNV entry, consistent with
  M-mode work on a timer escape.
- So instret is exact only up to entries × ~125. At an iteration of ~10⁷ instructions with tens of entries, that is
  ~10⁻⁴, two orders below h.
- No correction is applied. Entries are recorded beside every reading, so it can be applied.

**What each mark costs.** A mark is itself an escape: three preads into static buffers, no stdio, no malloc, parsed
only after the end mark. The `ctl/empty` control measures that overhead, which is the same in both arms.

## Why N is derived and not chosen

Revision 1 picked N by feel; the lead rejected that. N depends on three things that are not yet measured for this
setup: where the variance sits, what each repetition costs, and the precision wanted.

**Levels of variance.**
- **Board.** The two boards are identical, and they split the programs. No program runs on both: the lead's rule, to
  halve wall time.
- **Boot.** Both arms of a program run in the same boot, interleaved, so boot effects cancel in the ratio. A boot ×
  arm interaction is possible and is not measured; this is stated as a limitation.
- **Process (one launch).** This is the unit of analysis. Every launch has a new heap history and a new phase against
  the timer tick.
- **Iteration within a process.** These share the process's state, so they are not independent of the process level
  (Kalibera and Jones, ISMM 2013).

**Cost of each repetition,** from the bitstream-2 Juliet batch005 of part 01: 16 min for 71 steps, with bake and
boot. That gives:
- boot ≈ 6 min;
- launch ≈ 5–8 s, so `c_p ≈ 7 s`;
- timed iteration ≈ 0.3–0.5 s, so `c_i ≈ 0.4 s`.

So iterations are cheap and launches are expensive, and the best split depends on where the variance sits.

## Layout control, by construction

On silicon, moving code alone changed cycle ratios by up to ±4 % at identical instret (ladder K0–K3). The README gives
the order: janne 7.7 %, bs 5.7 %, insertsort 4.7 %. Repetition does not remove this, and process launches randomise
nothing that matters, because the L1 ways are 4 KiB (config_pkg) and ASLR moves only page numbers. So the plan
controls layout in the build:
- **Same addresses for everything but the allocator.** The allocator's object is linked LAST, and heap.c is padded to
  the same section sizes in both arms. Then every other input section has the same address in both arms: the
  benchmark, libc and all of `.rodata`.
- **The gate:** `llvm-readelf -S` plus the linker map, compared per image pair. Any difference outside heap.c blocks
  that program.
- **The allocator itself necessarily differs between the arms.** For the programs that allocate inside the timed
  region, the processes of each arm are spread evenly over 4 allocator alignment pads, K = 0..3 (the ladder's
  method). This puts the allocator's layout sensitivity inside the interval instead of biasing it. It costs builds,
  not launches.

## The pilot (one boot per board, programs split)

**Programs.** Five, chosen to span the expected variance:
- CoreMark: compute, no allocation;
- BEEBS janne: the most layout-sensitive kernel on the ladder;
- RV8 aes: malloc/free per operation, run at the 4 allocator pads;
- RV8 miniz: large allocations;
- BEEBS sglib-rbtree-libc: many small allocations through Sublet's path.

**Controls, in the same boot:**
- `ctl/empty`: the mark overhead;
- `ctl/loop`: a known instret, which must read its predicted value;
- the module's modinfo shows the three bench parameters. The pre-boot gate refuses a kit whose module lacks them.

**Design.**
- Per program and arm: 6 processes, each with 1 warm-up iteration and 10 timed iterations.
- Within a program, processes run in ABBA order. Half the programs start with T1 and half with T2.
- Programs run in a fixed random order, seed 20261026.
- The quantum CSR (0x7C3) is read before and after, and recorded (S-26: it depends on boot history).

**What the pilot decides:**
- σ_p and σ_i, the between-process and within-process standard deviations of log cycles, from a one-way
  random-effects decomposition;
- whether instret is constant;
- a trend test over iteration index, to decide whether one warm-up iteration is enough;
- entries per iteration.

## Dimensioning the main run (pre-registered rule; scripted in perf-analyze.py `pilot`)

- **Iterations.** `I = round(sqrt(c_p · σ_i² / (c_i · σ_p²)))`, clamped to [3, 20].
- **Processes.** P per arm is the smallest P ≥ 3 with
  `t(0.975, 2(P−1)) · sqrt(2 (σ_p²/P + σ_i²/(P·I))) ≤ log(1 + h)`.
  - h = 1 % is the design target (the lead, 2026-10-26: minimise board time; the per-suite geomean is several
    times tighter). P scales as 1/h², so 1 % needs about a quarter of the processes 0.5 % would.
  - σ_p and σ_i are the largest across the pilot programs, each at its upper 80 % confidence bound.
  - It is Student's t, not 1.96: the report's interval is Welch's on P process means.
- **Cap.** At most 3 h of runs per board. If the rule needs more, P is cut to fit and the achievable width is stated.
- **No top-up.** P is fixed for the whole run once the pilot is read. A rule that adds runs to programs whose interval
  came out wide biases coverage: an audit simulation put the early-stopping programs at about 85 % coverage. Every
  interval is reported as measured, whatever its width.

## Running the main experiment

- **Boards.** Programs are split between the two boards, balanced by expected time.
- **Boot.** All of a program's processes, both arms, run in one boot, in ABBA order with alternating start.
- **Order.** Programs run in a fixed random order.
- **Warm-up.** The warm-up iterations decided by the pilot are dropped.
- **Outliers.** None are removed.
- **Failures.** A failed process (rc ≠ 0, a missing BENCH line, a failed verify) is recorded. It is re-run once and
  both records are kept.

## Estimation and figures (perf-analyze.py `report`)

- **Per program.**
  - Each process gives the mean of log cycles over its timed iterations.
  - An arm's estimate is the mean of its processes.
  - T2/T1 = exp(difference), with Welch's 95 % interval on the process means.
  - A two-level bootstrap (processes, then iterations) is the sensitivity check.
  - The same is done for instret. When instret is constant it is exact.
- **Verdicts.**
  - If the interval contains 1, the verdict reads "no measurable difference", together with its bounds.
  - "Within ±x %" is claimed only where the whole interval lies inside 1 ± x % (an equivalence statement).
- **Per suite.** The geometric mean of the ratios, with a two-level bootstrap interval over every program together.
- **Box plots.** Every timed iteration, normalised to the program's T1 median, T1 and T2 side by side: whiskers at
  1.5 IQR, outliers drawn.
- **Records.** The raw per-iteration TSV is committed, and every table and figure is regenerated from it by the
  committed script. Negative tests of the script run on synthetic data with planted effects. A +0.6 % and a +5 % are
  detected; two nulls read "no measurable difference"; missing data refuses.

## Predictions, before the pilot

- **CoreMark, and every program with no allocation inside its timed region:** T2/T1 contains 1, and the instret
  difference is 0.
  - This holds only if the heap fix holds. With it, an allocation-free region makes no heap calls in either arm.
- **RV8 aes and miniz, and the libc-routed BEEBS programs:** T2/T1 > 1.
  - Predicted instret difference: the program's malloc and free counts (recorded on QEMU) × heap.c's per-call
    instruction delta between the arms.
  - Predicted cycle difference: that, plus the measured silicon costs of CDERIVE and CREVOKE from the measurements
    doc.

## Changes from revision 1, from the audit

1. The per-allocation service call is fixed first; it is a runtime defect.
2. The P rule uses Student's t, and the adaptive top-up is removed.
3. Entries are counted, marks do no stdio or malloc, `ctl/empty` is added, and modinfo is gated.
4. Layout: everything but the allocator is equal by construction, and the allocator is spread over K = 0..3 pads.
5. The analysis implements this plan (perf-analyze.py replaces perf-stats.py).
6. The framing is the marginal cost of per-object lifetimes.
7. The pilot uses janne, not insertsort, as the layout-sensitive kernel.
8. "No measurable difference" is not stated as "< h".

## Pilot read and the main run's dimensions (2026-10-26, before any main-run boot)

**Pilot run.** Pilot 2 ran on board 2, lane 59093fd4d3ca, transcript `perfpilot2/guest/board.txt`. The images were
built on dev 4f054964a7d8.
- All 68 processes read rc=0, with all 748 iteration rows verified (parse-board.py: 0 problems).
- **Per pilot program, σ_p | σ_i of log cycles,** each at its upper 80 % bound:
  - coremark 0.11 % | 0.30 %;
  - janne 0.04 % | 0.13 %;
  - aes 0.15 % | 0.38 %;
  - **miniz 0.35 % | 1.01 % (the worst);**
  - rbtree-libc 0.20 % | 0.84 %.
- **Instret** is not constant across iterations: 4–50 distinct values per arm. That fits the per-entry overhead read
  by the probe, plus data-dependent paths in the allocating programs.
- **Controls.** ctl-loop's instret is as the probe read. ctl-empty is excluded from dimensioning, by the plan's
  wording: its region is ~2e4 instructions, and its relative noise does not describe a benchmark.
- **Costs measured here.** A launch plus its warm-up and overhead is c_p ≈ 10.7 s; c_i ≈ 1.6 s per iteration
  (median; rbtree 3.3 s).

**Choice of I and P.** At those values the formula gives I* = 7.5. The rule as written (I = round(I*) = 8, then the
least P) gives P = 4 at 2.8 h per board. Enumerating every I from 3 to 20 with its least P finds a cheaper pair that
also meets h = 1 %: **I = 5, P = 4, at 2.3 h per board.** The formula only approximates the cost optimum and ignores
that P is an integer. So the main run takes the least-cost pair, on the same σ and the same h. This refinement is
recorded here, before the main run.

**Programs run one iteration per process** (crc32, huffbench, huffbench-libc, wikisort: not re-entrant). The same
rule with I = 1 gives P = 11 per arm.

**Allocating programs.** The 4 processes per arm take K0, K1, K2 and K3, one each.

## T3: lifetimes without scrub (2026-10-27, before any T3 boot)

**Why.** T2/T1 bundles two allocator costs: per-object lifetimes (a CDERIVE child per malloc, a CREVOKE per free) and
the scrub on free. T3 has lifetimes on and scrub off (`CAPSTONE_HEAP_LIFETIMES=1`, `CAPSTONE_HEAP_SCRUB_FREE=0`).
So T3/T1 is the lifetimes' cost and T2/T3 the scrub's. The lead approved the arm on 2026-10-27, to run after the
T1/T2 main run, which is complete.

**Programs.** The 15 that allocate inside the timed region, and nothing else: miniz-libc, huffbench-libc, the five sglib-*-libc (dllist, hashtable, listinsertsort, listsort,
rbtree), the three ctl-*-libc, and rv8 aes, miniz, norx, primes and qsort.
- A program with no allocation in its timed region reads T2/T1 at 1.003 or below with equal instret. T3 cannot
  differ there, so it is not run.

**Images.**
- T3's SDK is built from the same dev tree as T1's and T2's.
- check-sdks.py, T2 against T3, lists the expected differences: heap.c.obj (archive member and object),
  sdk.json `heap_scrub_free`, the heap-switch lines in build.ninja and CMakeCache, and output paths.
  - It lists one unexpected file, `sdk/__pycache__/capstone-cccpython-312.pyc`, present only in sdk-T3.
  - That file is the bytecode cache of the capstone-cc wrapper, whose source is byte-identical in both SDKs. It is
    not linked into any image.
- The three-SDK heap-layout record gives every arm the same padded heap region.
- check-arms.py passes 60/60 for T3 against T2 (`perf/build3/gate-T2-T3.tsv`, its PASS rows are exactly the
  60 T3 images): every input section outside the allocator sits at the same address.
- T1 and T2 images are byte-identical to the main run's.

**Design.**
- Three kits, `perf/main3b/t3k{1,2,3}`: one boot each, each program's three arms in that same boot.
- Arms alternate in ABCCBA order, T1 T3 T2 T2 T3 T1, starting with C for odd positions.
- The dimensions are the main run's:
  - P = 4 processes × (W = 1, I = 5) per arm;
  - huffbench-libc, which is not re-entrant, gets P = 11 × (0, 1) per arm;
  - the four processes of an allocating program take K0 to K3.
- Run-script hashes, sha256 first 16:
  - t3k1 0ecbbc7c4c9f5ec8 (81 processes);
  - t3k2 fc24bc09d77351fc (60);
  - t3k3 37861b84fca1ac04 (60).
- A first build of these kits ran huffbench-libc at (1, 5) because `--single` was left out (`perf/main3`, hashes
  6570bfd8…, 8d80b9a8…, a13d53d3…). It was withdrawn before any boot.

**Analysis.** perf-analyze.py `report` on the T3 boots' rows, with `--arms T1,T3`, `--arms T3,T2` and
`--arms T1,T2`. Each is the same estimator as the main run, on a pair of arms from the same boot.

**Predictions.**
1. **Instructions.** For every program, the median instret per iteration orders T1 ≤ T3 ≤ T2. T3 does the
   lifetime work, and T2 adds the scrub.
2. **miniz-libc.** Most of its slowdown is the scrub: T2/T3 carries more than 80 % of log(T2/T1), so T3/T1 is
   below 1.17.
   - The basis is an inference from the main run, not a measurement: escapes of 1,714 against 589, and instret
     1.265.
   - If T3/T1 is instead close to T2/T1, the cost is the lifetimes and the inference is wrong.
   - The escape counts move with it: T3's entries are near T1's if the escapes are the scrub's first-touch faults.
3. **T2/T1 replicate.** These boots re-measure T2/T1 for the 15 programs on a second boot. This is the first
   boot-to-boot reading; the main run measured each program in one boot only.
   - Expected: each program's Welch 95 % interval from these boots overlaps its interval from the main run.
   - A program whose intervals are disjoint is reported as a boot effect, with no top-up.
   - Amendment, 2026-10-27, after launch and before any T3 transcript was read: the first wording, "lies inside
     the union of the two Welch intervals", holds by construction. Each estimate lies inside its own interval, so
     it could never fail. The overlap test replaces it.
