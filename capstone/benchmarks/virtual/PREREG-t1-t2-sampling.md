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
1. **Instructions.** For every program, the median instret per iteration orders T1 ≤ T3 ≤ T2 within a tolerance
   of 0.1 % of T1's: T3 ≥ T1 × 0.999 and T2 ≥ T3 × 0.999. T3 does the lifetime work, and T2 adds the scrub.
   - Amendment, 2026-10-27, before any T3 transcript was read: the first wording had no tolerance.
   - A dry run on the main run's rows showed instret is not constant: rv8-qsort's T1 median exceeds its T2 median
     by 35 of 11.4 million. So the strict order fails on noise alone.
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

## Bitstream 3: rerun of ludcmp, st and minver (2026-10-27, before any bitstream-3 boot)

Bitstream 3 is capstone-ariane e7af48457, which carries the R-62/R-63 divsqrt fix. Its memory map is unchanged from
dcc6daf6e: CAP_TAG_MEM_BASE 0xBC2D_2D2D and CAP_REVNODE_MEM_BASE 0xBFF0_0000 at both commits. The board lane flashes
it, runs a Tier 1 control boot and the rtl lane's bare FP kit, then this kit.

**Kit** `perf/main4/b3k1`, one boot, run-b3k1.sh sha256 cbf60f75e5eb79ae.
- First, the identification step, native with no context: `ludcmp 1 3` and `st 1 3`.
  - These are the same binaries (sha256 3b9f085a…, b486bd0a…) that read VERIFY-FAILED on every iteration on
    bitstream 2, in the FP discriminator boot.
- Then T1 and T2 for ludcmp, st and minver, with the main run's images and dimensions: P = 4 × (W = 1, I = 5) per
  arm, ABBA.

**Predictions.**
1. **Identification:** both native programs read verify_failed=0 and BENCH-DONE failed=0.
   - Bitstream 2 read VERIFY-FAILED on every iteration.
   - If either still fails, the board is not running the fixed RTL, or the fix is incomplete on silicon. Nothing
     after it is read until that is settled.
2. **Correctness:** every process passes in both arms.
   - check= equals the twin's for minver: 0xd6cc5fbda167dc7b.
   - Bitstream 2 read 0x74ceeb9e3de2c383, the RTL-rounding model's value.
   - ludcmp and st carry no hash; for them the reading is VERIFY pass.
   - The rtl lane's dynamic census found no division in R-64's, R-65's or R-66's reach in these three programs. So
     a remaining difference from the twin would point elsewhere.
3. **Timing:** T2/T1 for all three is 1.000 within the main run's precision. None allocates in its timed region.
   - These are reported as bitstream-3 rows, never pooled with the bitstream-2 suite geomean.

## T3 on the other 88 programs, bitstream 3 (2026-10-27, before any of these boots)

**Why.** The lead asked for T3 on every program, not only on the 15 that allocate inside the timed region. This
section covers the remaining 88, so every one of the 103 programs gets all three arms.

**Platform.** Bitstream 3 (caplifive_vp2c_e7af48457) on both boards, with the board lane's bitstream-3 perf platform
(the one b3k1 ran on). The 15 keep their bitstream-2 T3 rows; the two sets are reported side by side with the
bitstream named on every row, and never pooled into one suite geomean across bitstreams.

**Images.**
- T1 and T2: `perf/build2`, the images the main run booted. The 85 programs `build` and `build2` share are
  byte-identical there; coremark, matmult-int and rv8-dhrystone are the resized b2k1 images.
- T3: `perf/build4`, built for these 88 by `build-t3all.sh` from the same tree (wt-perf 86f638759ecf), compiler
  (cc-prmerged), SDK (`sdk-T3`, runtime 3d74dc21d4ec4f4b) and controlled heap layout (`heap-layout-3arm.json`) as
  the first T3 images.
- Layout gate `check-arms.py build2/T2 build4/T3`: 88 of 88 pairs PASS (the other 60 pairs are the 15 programs'
  images, which have no build4 counterpart).

**Kits** `perf/main5/t3a1..t3a4`, 22 programs each, three arms in one boot, ABC..CBA, P = 4 × (W = 1, I = 5) per
arm; crc32, huffbench and wikisort (not re-entrant) at P = 11 × (0, 1). Launcher 978c09d61f774315. Run scripts:
t3a1 95581eee5424dd8d, t3a2 bd4ef4ee77c21fc6, t3a3 8d7f4b3cec3b260c, t3a4 c2f7cc1b012f7c57. Board 1 runs t3a1
then t3a3, board 2 t3a2 then t3a4; make-kits estimates 1.5 h per kit.

**Predictions.** None of the 88 calls the runtime's malloc or free inside its timed region; on bitstream 2 every one
read T2/T1 at 1.003 or below with equal instret.
1. **Correctness.** Every process exits 0 with no VERIFY-FAILED, and within each program check= is the same in all
   three arms. Where the QEMU twin has a hash (`twin-checks.tsv`), check= equals it. A program that differs from its
   twin is reported with its value; R-64/R-65/R-66 remain in bitstream 3 and are the first suspects.
2. **instret.** Per program, the three arms' instret agree within 0.1 %.
3. **Cycles.** Per program, T3/T1 and T2/T3 lie within 1 % of 1.000 (point estimate). The geomeans of T3/T1 and of
   T2/T3 over the 88 lie in [0.995, 1.005].
4. **Replicate.** This run's T2/T1 overlaps the main run's interval for each program. A program that does not is
   reported with both intervals; a cross-bitstream difference is expected only where the divider runs.

**Analysis.** parse-board.py, then perf-analyze.py `report --arms` for T1,T3 / T3,T2 / T1,T2, as for the first T3
run; the checks above by a script committed with the results.

## The 15 allocating programs on bitstream 3: every program on one bitstream (2026-10-28, before any of these boots)

**Why.** The lead asked for one figure with three groups, BEEBS, RV8 and CoreMark, each showing T1, T3 and T2.
A suite geomean must come from one bitstream. The 88 programs have all three arms on bitstream 3; the 15 that
allocate inside the timed region have them only on bitstream 2. RV8 and BEEBS-libc are split across the two. This
run puts the 15 on bitstream 3, so all 103 programs have T1, T3 and T2 there.

**Kits.** The T3 kits `perf/main3b/t3k{1,2,3}`, unchanged: the same images, launcher and run scripts as the
bitstream-2 T3 run. Run-script sha256/16: t3k1 0ecbbc7c4c9f5ec8, t3k2 fc24bc09d77351fc, t3k3 37861b84fca1ac04;
launcher 978c09d61f774315. Each kit's SHA256SUMS verifies against its stage (62 files each) on 2026-10-28.
Platform as for the T3-all run: bitstream 3 (caplifive_vp2c_e7af48457), the board lane's bitstream-3 perf platform.
Board 1 runs t3k1 then t3k3, board 2 runs t3k2.

**Predictions.** The 15 are integer programs; bitstream 3 differs from bitstream 2 by the divsqrt fix.
1. **Correctness.** Every process exits 0 with no VERIFY-FAILED. Within each program check= is one value across the
   three arms, and it equals the bitstream-2 T3 run's value.
2. **instret.** Per program and arm, the median instret per iteration equals the bitstream-2 T3 run's within 0.1 %.
3. **Cycles, replicate across bitstreams.** Per program, each of T3/T1, T2/T3 and T2/T1 has a Welch interval that
   overlaps the bitstream-2 T3 run's interval for that pair. A pair whose intervals are disjoint is reported with
   both. miniz-libc is the one program already seen to move between boots: its T2/T1 read 2.1653 [2.1563, 2.1743] in
   the main run and 2.1819 [2.1751, 2.1886] in the T3 run.
4. **Suite geomeans on bitstream 3, all 103 programs** (this run's 15 with the T3-all run's 88). Composed from the
   current per-program point estimates (15 from bitstream 2, 88 from bitstream 3), each suite geomean lies within
   0.005 of:

   | suite | n | T3/T1 | T2/T3 | T2/T1 |
   |---|---|---|---|---|
   | beebs (upstream) | 80 | 1.0001 | 0.9999 | 1.0000 |
   | beebs-libc | 13 | 1.0265 | 1.0775 | 1.1061 |
   | rv8 | 7 | 0.9997 | 1.0247 | 1.0244 |
   | coremark | 1 | 1.0004 | 0.9998 | 1.0003 |
   | ctl | 2 | 0.9980 | 1.0156 | 1.0136 |

   CoreMark's three arms come from the T3-all run alone; it is listed for completeness, not as a prediction.

**Analysis.** parse-board.py on each transcript; perf-analyze.py `report --arms` for T1,T3 / T3,T2 / T1,T2 on this
run's rows (per-program replicate); then on this run's rows plus the T3-all run's rows (suite geomeans over all
103 programs). The checks above by a script committed with the results. Nothing from bitstream 2 enters a
bitstream-3 geomean.

## Caps: the allocator without per-object bounds, bitstream 3 (2026-10-30, before any Caps boot)

**Why.** MapLife's malloc returns a capability over the object's own bytes. It is a SHRINK of the slot's authority,
checked to have come out exact. Caps returns the group's authority instead, cursor at the object. That authority is
the whole mapping, or the whole nested group. So MapLife/Caps is what per-object spatial bounds cost inside the
allocator. With the existing arms it completes the chain Caps → MapLife (per-object bounds) → ObjLife (plus
per-object lifetimes).

Arm names: Caps is new. MapLife is this document's T1, and ObjLife its T3 (lifetimes, no scrub). The campaign's
directories keep the T1 and T3 labels.

**Images.**
- **The switch.** `CAPSTONE_HEAP_OBJECT_BOUNDS=0`, together with LIFETIMES=0 and SCRUB_FREE=0. It lives in
  heap.c, CMake, Application.cmake and sdk.json `heap_object_bounds`; the worktree is `perf-t0a` at 4f054964a7d8,
  the campaign SDKs' source commit.
  - At the default (1) the edited tree rebuilds the campaign SDKs byte for byte. Whole `libapplication-runtime.a`:
    sdk-T1 ab0c38405c8b740c, sdk-T3 3d74dc21d4ec4f4b, sdk-T2 5a79a20fdf806bfb. Every libc and builtins object is
    equal too.
  - The strict comparison fires on the Caps SDK (heap.c.obj only).
- **The Caps SDK.** `perf/sdk-T0a`: runtime 49b443f93d8328ad, heap.c.obj 9a8b75f2675b4129 (text 12548 B).
  - The 4-arm layout record `perf/heap-layout-4arm.json` keeps the regions at 13376 and 65792.
  - Caps images: `perf/T0a/build/Caps`, 103 programs and 148 images, from the campaign's build.py recipe.
- **Layout gate.**
  - `check-arms.py` against `build2/T1`: 148 of 148 pairs PASS, 0 objects differ.
  - Against `build3/T3`: 60 of 60; against `build4/T3`: 88 of 88.
  - The default-link.ld negative control fails 148 of 148.
  - Every input section outside heap.c sits at the same address in the three arms.
- **QEMU twin.**
  - Caps runs every program checked, with rc 0 and check= equal to MapLife and `twin-checks.tsv` (18 of 18 images).
  - The must-fire control holds: one byte past a 48-byte malloc faults with cause 28 in MapLife and ObjLife. In
    Caps it does not fault, and the same capability reaches the next object.
  - Under `--icount`, MapLife − Caps is ~10.3 instructions per malloc+free pair (sglib).

**Kits.**
- Seven kits, `perf/T0a/kits/heap15/capsh{1,2,3}` and `perf/T0a/kits/other88/capsa{1,2,3,4}`, from `make-kits.py`.
- Arms are `Caps,MapLife,ObjLife`, alternating ABC CBA (Caps MapLife ObjLife ObjLife MapLife Caps), starting with C
  at odd positions.
- Dimensions are the T3 runs':
  - P = 4 × (W = 1, I = 5) per arm;
  - huffbench-libc, crc32, huffbench and wikisort at P = 11 × (0, 1);
  - an allocating program's four processes take K0 to K3;
  - seed 20261026.
- Each kit's plan equals the campaign kit's (t3k1..3 and t3a1..4) position for position, with the arms renamed
  T1 → Caps, T3 → MapLife and T2 → ObjLife.
- Every MapLife and ObjLife image is byte-identical to the campaign kits' T1 and T3 images (299 files, 0 differ).
- Launcher 978c09d61f774315.

| kit | programs | processes | run script | SHA256SUMS | stage wrapper run.sh |
|---|---|---|---|---|---|
| capsh1 | 5 | 81 | a9cf7a545a94fabd | 3144ed7b4aa66277 | 7f294b3a8ddbadf3 |
| capsh2 | 5 | 60 | 924c389068df6cf4 | 73b2dcb9ff7bb6f7 | b811333d3465bfdb |
| capsh3 | 5 | 60 | d397aeef95eec45a | bb61c5c122b12e81 | 9bd1538bc9a9df89 |
| capsa1 | 22 | 285 | 6c70804efc06228a | b0034c8472d4f462 | 2427a8891d5d1a2c |
| capsa2 | 22 | 285 | 8ce971358f3285f4 | 388fcc005c31fdf8 | 0e4e4d25517dc3b9 |
| capsa3 | 22 | 285 | b47b07cc743c83af | 16c52874f44e8e51 | 52ebedb323fe2e35 |
| capsa4 | 22 | 264 | 7dc69f214d0f301e | 6456ab2224df7fa1 | 0d0cda42c40599dc |

**Boots.**
- **Board and route.** Board 1, bitstream 3 (caplifive_vp2c_e7af48457), through the SPEC lane's chain:
  `chain-board.sh 1 caps-b1.list`, which runs `spec-board.sh`, which runs `run-stage.py run --backend board`.
- **Board kit.** `bkit12-perf-b3-b1` (launcher 978c09d61f774315, module 9c60fb53f80ce67e). spec-board.sh checks
  both by hash.
- **Control.** Each boot first runs the known-good control: crc.K0 T1, bd909cc6eba4bf2d, W 1 and I 5.
- **The stage.** `perf/T0a/board/stages/perf-caps-<kit>/` is a thin wrapper (`make-caps-stages.py`).
  - Its run.sh hashes every kit file ON THE BOARD and compares the hashes with the kit's SHA256SUMS; a mismatch ends
    the step before the first process.
  - It then runs the kit's own run-<kit>.sh unchanged, so the @@PERF and BENCH lines reach the transcript as in the
    campaign.
- **List lines** (stage, step_s, booking_min, twin):
  - capsh1..3 at 3600 s and 90 min.
  - capsa1..4 at 10800 s and 210 min.
  - The step bounds are the campaign's own (its t3k gate bound 3600 s, its t3a bound 10800 s).
  - The measured spans were 24 to 29 min of step for t3k1..3 (33 to 38 min wall), and 78 to 84 min for t3a1..4 (87
    to 94 min wall). Caps replaces the scrub arm (T2), so no Caps kit should run longer than its campaign kit.
- **Firmware size.** The stages are 15.6 to 17.2 MB, against 16.95 MB for the largest campaign t3a stage.
  - The payload pads in steps of 2 MiB, and a 16.7 to 16.95 MB stage gave 23,759,368 B.
  - The next step, 25,856,520 B, is still under BOARD_FW_CAP 26,000,000.
- **Dry run.** A stubbed dry run (`perf/T0a/board/dryrun/`) took all seven lines to END rc 0, with `RUN control
  PASSED` and the real run-stage.py.
  - Stubs: busy checks, hub, and BOARDVM_STUB_TRANSCRIPT built from the campaign's own bitstream-3 transcripts
    relabelled by position.
  - The runner was invoked with this kit and each line's stage. Each staged kit verified against its SHA256SUMS.
  - parse-board.py read 81/60/60/285/285/285/264 processes, each set equal to the kit's steps.tsv.
  - capsh1..3 parsed to the campaign's committed rows value for value.
  - A failed control stopped the chain (CB STOP, VOID).

**Predictions.** Ratios are MapLife/Caps (perf-analyze `--arms Caps,MapLife`); above 1 means Caps is cheaper.
1. **Correctness.**
   - Every process in every arm exits 0, with W + I BENCH lines, no VERIFY-FAILED and BENCH-DONE failed=0.
   - Within each program check= is one value across the three arms, and it equals the bitstream-3 T3 runs' value.
2. **instret, the 15 allocating programs.**
   - MapLife − Caps per timed iteration is the QEMU twin's `--icount` difference, within 0.05 % of MapLife's instret.
     The difference comes from ~10 instructions per malloc+free pair: 9 on malloc's path (a SHRINK, two `lcc`, two
     compare-and-branches, three ALU ops, one capability store) and one store in free.
   - Predicted MapLife/Caps instret:

   | program | heap calls / it | Δ instr / it | MapLife/Caps instret | MapLife/Caps cycles band | (T3/T1 cycles, b3) |
   |---|---|---|---|---|---|
   | sglib-dllist-libc | 4,200 | 43,772 | 1.0047 | 1.0009 – 1.0052 | 1.0604 |
   | sglib-hashtable-libc | 4,600 | 47,480 | 1.0053 | 1.0010 – 1.0061 | 1.0594 |
   | sglib-listinsertsort-libc | 4,455 | 45,998 | 1.0050 | 1.0009 – 1.0056 | 1.0630 |
   | sglib-listsort-libc | 4,500 | 46,448 | 1.0051 | 1.0010 – 1.0059 | 1.0640 |
   | sglib-rbtree-libc | 3,600 | 37,520 | 1.0039 | 1.0007 – 1.0043 | 1.0467 |
   | ctl-stack-libc | 1,400 | 18,059 | 1.0016 | 1.0003 – 1.0018 | 1.0198 |
   | ctl-string-libc | 1,400 | 16,808 | 1.0015 | 1.0003 – 1.0019 | 1.0193 |
   | ctl-vector-libc | 1,480 | 19,248 | 1.0017 | 1.0003 – 1.0018 | 1.0193 |
   | huffbench-libc | 76 | 844 | 1.0001 | 1.0000 – 1.0002 | 1.0015 |
   | miniz-libc | 36 | 541 | 1.0001 | 1.0000 – 1.0001 | 1.0000 |
   | rv8 aes, miniz, norx, primes, qsort | ≤ 6 | ≤ 52 | 1.0000 | 1.0000 | ≤ 1.0012 |

   ctl-stack-libc's and ctl-vector-libc's per-iteration instret varies by up to ~25 k within an arm on the twin,
   so their rows are ±0.0003.
3. **Cycles, the 15.**
   - **Direction: Caps ≤ MapLife.** MapLife/Caps ≥ 0.999 (point estimate) on every program.
   - **Size.** The band in the table assumes each removed instruction costs 1 to 6 cycles. The bounds are:
     - 1: CVA6 issues one instruction per cycle at best.
     - 6: the programs' own average CPI on bitstream 3 is 4.5 to 5.5 for sglib and ctl. The removed instructions are
       register operations and two stores to lines the allocator has just touched, so they should cost no more than
       the average.
   - **sglib.** The geomean of MapLife/Caps over the five sglib-*-libc lies in [1.000, 1.008], with point estimate
     1.001 to 1.006.
     - A single program's Welch interval is NOT predicted to exclude 1. The T3 runs' per-program intervals on these
       programs are ±0.5 to 0.6 %, the same size as the effect.
   - **Far below ObjLife/MapLife.** On each sglib program, MapLife/Caps is about an order of magnitude smaller in
     log terms than the same boot's ObjLife/MapLife (the band's top, 1.006, against 1.047 to 1.064). A CDERIVE plus a CREVOKE costs ~600 cycles per allocation; this
     difference is ~10 to 60.
   - **Falsifier.** A MapLife/Caps above 1.02 on any program means a SHRINK or an `lcc` is not cheap on silicon. It
     is reported as a finding, not explained away. A MapLife/Caps below 0.99 on an sglib or ctl program means Caps
     is slower. With byte-identical layout outside heap.c, that would point at heap.c's own placement (K pads) or
     the group-sized capability's effect on the hardware, and is reported with the K split.
4. **The 88 that do not allocate in the timed region.**
   - Per program, MapLife and Caps instret agree within 0.01 %.
   - The geomean of MapLife/Caps over the 88 lies in [0.995, 1.005], and each per-program point estimate lies within
     1 % of 1.000.
5. **Replicate of the existing arms on board 1.**
   - Per program, the median instret per iteration of MapLife and of ObjLife equals the bitstream-3 T1 and T3
     runs' within 0.1 %. Those are `results/2026-10-28-t3heap-bitstream3/` (the 15) and
     `results/2026-10-27-t3all-bitstream3/` (the 88).
   - ObjLife/MapLife has a Welch interval that overlaps that run's T3/T1 interval.
   - capsh2, capsa2 and capsa4 move programs that ran on board 2 in the campaign to board 1. A cross-board
     difference is reported, and the within-boot ratios stand.

**VOID.** A boot is VOID, and none of its rows enter any ratio, unless `check-caps-boot.py` passes. It requires:
- C1 `RUN control PASSED`;
- C2 `@@CAPS-SUMS OK`, and the board-side hash block equal to the kit's SHA256SUMS line for line;
- C3 `@@PERF-<KIT>-DONE` and `@@SPEC-DONE`;
- C4 parse-board.py exit 0;
- C5 the parsed processes exactly the kit's steps.tsv;
- C6 no `@@STEP-BOUND`, no runner TIMEOUT, no kernel oops or OOM.

C5 is needed because parse-board.py alone passes a boot cut short: the dry run cut a transcript after process 40 of
60, parse-board exited 0, and the gate returned VOID.

Also VOID:
- a boot in which `spec-board.sh` printed `SB STOP` (wrong board kit, bitstream or control hash);
- a boot whose MapLife instret misses prediction 5 by more than 0.1 % on more than one program. That means the
  platform is not the T3 runs', and the Caps ratios from it are not used until it is explained.

Each check was shown to fire on tampered inputs before any boot:
- a failed control;
- a cut transcript;
- a missing out dir (NO DATA, exit 2);
- a kit with one image byte changed: the wrapper, run on the host, exited 3 with nothing run.

**Analysis.**
```
cd <work root holding perf/> || exit 1; G=perf/T0a/board; K=perf/T0a/kits; O=<chain out root>
V=<tree with the Caps parse-board.py / perf-analyze.py edits>/capstone/benchmarks/virtual
for k in capsh1 capsh2 capsh3; do python3 $G/check-caps-boot.py $O/perf-caps-$k $K/heap15/$k --runs $G/runs-$k.tsv; done
for k in capsa1 capsa2 capsa3 capsa4; do python3 $G/check-caps-boot.py $O/perf-caps-$k $K/other88/$k --runs $G/runs-$k.tsv; done
# only VALID boots: concatenate (one header) per group
python3 $V/perf-analyze.py report heap15.tsv --arms Caps,MapLife  --out caps-heap15/Caps-MapLife
python3 $V/perf-analyze.py report heap15.tsv --arms Caps,ObjLife  --out caps-heap15/Caps-ObjLife
python3 $V/perf-analyze.py report heap15.tsv --arms MapLife,ObjLife --out caps-heap15/MapLife-ObjLife
(the same three for other88.tsv)
```
- The instret and replicate checks (predictions 2, 4 and 5) are made by a script committed with the results. It is
  written after the boots and before any transcript is read, and each check is shown to fire on tampered inputs.
- perf-analyze.py reports TEST/BASE, so `--arms Caps,MapLife` gives MapLife/Caps.
- The Caps arm names need the parse-board.py regex (`T[123]|Caps|MapLife|ObjLife`) and perf-analyze.py KNOWN edits
  from `perf-t0a`. dev's parse-board.py reads a Caps transcript as "no @@PERF steps".
