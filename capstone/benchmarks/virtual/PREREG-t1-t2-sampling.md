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

## ObjBounds: per-object bounds without the per-access lifetime check (2026-10-30, before the flash and any ObjBounds boot)

**Why.** The ladder Plain → Caps → ObjBounds → MapLife → ObjLife needs the rung where a program has per-object bounds
and the hardware does no lifetime check. "The cost of temporal safety" is ObjLife / ObjBounds; MapLife / ObjBounds is
what the per-access lifetime check alone costs. Arm names: ObjBounds is new. It is not a new image: it is MapLife's
images (this document's T1) run on a bitstream without the check.

**Bitstream.** capstone-ariane objbounds-b 603528b8b; control objbounds-off-b 1782d20bd, which is bitstream 3
(e7af48457) by preprocessed tokens and by its five generated Anvil units' hashes. The record and the synthesis
pre-registration are rtl's, land/rtl-objb c8be354d28ef (`capstone/docs/plans/2026-10-30-objbounds-variant.md`).
`git diff 1782d20bd 603528b8b -- core` is the config package's `CAPSTONE_NO_LIFETIME_CHECK` define and two lines of
the DYN unit's Anvil source. What the variant removes, by consumer (rtl's audit by call site):
- the LSU verdict on plain loads, stores and AMOs: a hit in the 256-node cache cost 0 cycles; a miss cost a replay and a
  record probe through D-cache port 2, and the cache is emptied at every context entry and escape;
- the PC tracker's cause 25 on fetch (cost 0);
- the DYN unit's uncached node query on EVERY LDC and STC: 3.96 cycles per LDC and 4.67 per STC on a resident node
  (simulation, memory delay 12: a pattern, not a silicon value), more when the record read misses.

What it keeps: the DYN unit's 25s on CDERIVE, CREVOKE, DROP, REVOKE, MREV, SPLIT, DELIN, TIGHTEN and LCC operands (the
ISA semantics, so the same SDKs and images run), the CPMP trackers, and the node query on CALL, RETURN and CSSUPERVISE:
ObjBounds still pays one query per domain entry. A stale access on this bitstream is NOT refused. It is a measurement
platform, not a safe one.

The sealed .bit's sha256 and name are synth's; this section is pushed before they exist. objb-first.sh takes both as
arguments and stops on a mismatch. rtl's board pre-registration (its kit vectors for the variant) is
<RTL-BOARD-PREREG-HASH, pending>; its first-boot checks, if any, run before this run's pilot.

**Images and kits.** No new image. Every T1 and T3 image is byte-identical to the bitstream-3 campaign's
(`perf/build2/T1`; `perf/build3/T3` for the 15 allocating programs, `perf/build4/T3` for the 88): 296 images, 0 differ.
Launcher 978c09d61f774315 (`perf/main3b/t3k1/stage/capstone-vexec`).
- Seven kits, `perf/objb/kits/heap15/objbh{1,2,3}` and `perf/objb/kits/other88/objba{1,2,3,4}`, from dev's
  `make-kits.py`.
- Arms `T1,T3`, alternating ABBA; P = 4 × (W = 1, I = 5) per arm; huffbench-libc, crc32, huffbench and wikisort at
  P = 11 × (0, 1); an allocating program's four processes take K0 to K3; seed 20261026.
- Each kit holds exactly the programs of the campaign kit and the Caps kit at its position (t3k1..3 / capsh1..3,
  t3a1..4 / capsa1..4). The arm labels stay T1 and T3, so dev's parse-board.py reads them unchanged. The bitstream is
  what makes T1 ObjBounds.

| kit | programs | processes | run script | SHA256SUMS | stage wrapper run.sh |
|---|---|---|---|---|---|
| objbh1 (pilot) | 5 | 54 | 201e1ee4b2cf920b | 00b9a5a3f89b593a | 8fc116b3ebcdcd61 |
| objbh2 | 5 | 40 | d978f4f7c674b382 | 028c6c13477f5f98 | 5c08babdf34c1f04 |
| objbh3 | 5 | 40 | dae6215c7b3c78cc | 66148da740684d52 | 98466d960267338f |
| objba1 | 22 | 190 | 298561a522889952 | 9c92f9c66ef1534b | 884d5b22ecb1d636 |
| objba2 | 22 | 190 | 2292eccab4798f61 | 8f12b118961ffb2d | 5c3cd4d7ef80da8c |
| objba3 | 22 | 190 | 007b7ee3dc4afa95 | 6eecb237da40c589 | 1eae273ad57f7bb2 |
| objba4 | 22 | 176 | c4432903c97046f7 | cabf8c78de4c086a | 5a83af83f5fa078b |

objbh1 holds huffbench-libc, rv8-aes, rv8-primes, sglib-hashtable-libc and sglib-rbtree-libc.

**Boots** (`board/objb/`). Board 2, after B's third check boot (b4-t1c) ends. `objb-first.sh <log> <sealed .bit>
<sha256>`:
- waits for b4-t1c's `TC DONE`, then for the .bit file, and checks its sha256;
- checks the stages (pilot armed, six main kits disarmed) and every kit's SHA256SUMS;
- waits for the board, checks that bitstream 3 stays registered as the fallback, flashes, and reads the resident name
  back on a fresh session;
- makes the board kit `bkit15-objb-b2`: bkit12-perf-b3-b2 with exactly one change, `BOARD_BITSTREAM=<sealed name>`.
  The module (9c60fb53f80ce67e, bench counters), launcher, firmware recipe, defines and DTB stay. The variant changes no
  memory constant (ariane_pkg.sv's generated hash equals e7af48457's);
- boots through `board/objb/chain-board.sh` and `spec-board.sh`. These are copies of the SPEC lane's chain.
  `spec-board.sh` takes the kit and the bitstream name as parameters, refuses bitstream 3's name, and runs the SPEC
  lane's run-stage.py by hash (734ad25cf901d4da). The known-good control is crc.K0 T1 (bd909cc6eba4bf2d), as in every
  bitstream-3 perf boot. Every boot's driver HARD-STOPs unless the resident name equals the kit's;
- the stage wrapper hashes every kit file on the board against SHA256SUMS and exits 3 on a mismatch, before any
  process; then it runs the kit's run script unchanged.

**The pilot goes first and alone.**
- objbh1 boots on its own. `check-objb-boot.py` (C1-C6 below) must read it VALID before
  `make-objb-stages.py --arm` arms the six main kits and the rest of the list runs.
- Otherwise the run stops with the main kits disarmed. spec-board.sh refuses a stage without plan.json, so a chain
  started by hand stops at the first main kit too.
- List lines: objbh1..3 at step 3600 s and 90 min booked; objba1..4 at 10800 s and 210 min (the campaign's bounds).
- make-kits estimates 0.26 + 0.23 + 0.23 h and 1.02, 1.02, 1.02, 0.99 h of steps, about 6 h of board time with the
  boots. The campaign's three-arm kits took 24-29 and 78-84 min of step; these have two arms.

**Dry run** (`board/objb/dryrun/`, record `dryrun-objb.txt`). The real objb-first.sh, make-objb-kit.sh, chain-board.sh,
spec-board.sh, run-stage.py, check-objb-boot.py and make-objb-stages.py --arm ran. The busy checks, store, hub, flash,
readback and board were stubbed, and the stub transcripts are the campaign's bitstream-3 blocks for the same program,
arm and process.
- **Positive.** All seven boots ended rc 0 with `RUN control PASSED`, and every gate read VALID: 54, 40, 40, 190, 190,
  190 and 176 processes, each equal to its steps.tsv.
- **Negatives:**
  - a wrong sha256: STOP before the flash;
  - the .bit never appears: STOP;
  - b4-t1c never ends: STOP;
  - the flash fails: STOP;
  - a pilot cut after 20 processes: VOID C2/C3/C5, HOLD, main kits disarmed;
  - a failed control: CB STOP, VOID C1, HOLD;
  - the chain run by hand without the gate: SB STOP at the first disarmed kit;
  - a kit copy with one byte flipped: the wrapper exits 3 with no process run.

**VOID.** A boot is VOID, and none of its rows enter any ratio, unless `check-objb-boot.py` passes all of:
- C1 `RUN control PASSED`;
- C2 `@@OBJB-SUMS OK`, and the board-side hash block equal to the kit's SHA256SUMS line for line;
- C3 `@@PERF-<KIT>-DONE` and `@@SPEC-DONE objb-<kit>`;
- C4 parse-board.py exit 0;
- C5 the parsed processes exactly the kit's steps.tsv;
- C6 no `@@STEP-BOUND`, runner TIMEOUT, kernel oops or OOM.
Also VOID: any boot whose resident bitstream is not the sealed name. A run in which prediction 2 misses by more than
0.1 % on more than one program is held: the platform is not the one this section describes, and no ratio from it is
used until that is explained.

**Predictions.** "MapLife" means the bitstream-3 T1 rows already on record (`results/2026-10-27-t3all-bitstream3/`
for the 88, `results/2026-10-28-t3heap-bitstream3/` for the 15); "ObjBounds" and "ObjLife-v" mean this run's T1 and
T3 rows. Cycles are per timed iteration, medians as in perf-analyze.
1. **Correctness.** Every process in both arms exits 0, with W + I BENCH lines, no VERIFY-FAILED and BENCH-DONE
   failed=0. Within each program check= is one value across the arms, and it equals the bitstream-3 records'. A
   correct program does no stale access, so removing the check changes no value.
2. **instret.** Per program and arm, the median instret per iteration equals the bitstream-3 run's for the same image
   within 0.1 %. Same images, same instructions; a replayed access retires once on bitstream 3, so the replays it saves
   are not instructions.
3. **Cycles: ObjBounds ≤ MapLife.** The variant can only remove stalls, so MapLife/ObjBounds ≥ 0.99 on every program
   (point estimate; the 1 % allows for cross-board and boot-to-boot placement of the heap and of the kernel).
   - **The model.** Per iteration, MapLife − ObjBounds ≈ 3.96 × N_LDC + 4.67 × N_STC + R. N_LDC and N_STC are the
     dynamic capability loads and stores. R is the replays of plain loads and stores whose node missed the 256-entry
     cache (after every escape, and on working sets above 256 live nodes). The per-op numbers are rtl's simulation
     at memory delay 12 on a resident node: a pattern, not a silicon value. A record read that misses costs more.
   - **No dynamic LDC/STC census exists.** So the bands below come from program class, not from counts. Static counts
     are not used: in one image (sglib-listsort) 2,694 ldc and 2,444 stc sit mostly in libc and the runtime.
   - **Pointer-chasing programs move most.** Each node visit loads a pointer with LDC and each link with STC: sglib
     upstream and -libc, dijkstra, ctl-*, wikisort, levenshtein, rv8-qsort, huffbench. At the campaign's CPI of 4.5 to
     5.5 and ~10-20 instructions per node, ~4 cycles per LDC is 4 to 9 %. Band per program: MapLife/ObjBounds
     in [1.01, 1.15].
   - **Integer and FP kernels move least.** These have few pointers loaded from memory: crc32, matmult-int, nettle-*,
     aha-mont64, rv8 sha512/primes/norx, the FP kernels. Band: [0.995, 1.01].
   - **Suite geomeans of MapLife/ObjBounds:**
     - BEEBS upstream (80): [1.000, 1.04];
     - BEEBS -libc (13): [1.01, 1.10];
     - RV8 (7): [1.000, 1.04];
     - CoreMark: [1.000, 1.04] (its list phase chases pointers).
   - **Falsifiers.**
     - Any program below 0.99 means ObjBounds is slower somewhere. That is reported as a finding.
     - Any program above 1.25 means the check cost more than its queries and replays could. It is reported, with
       instret checked first.
4. **ObjLife-v / ObjBounds, in the same boot: the software side of per-object lifetimes.** It isolates the CDERIVE per
   malloc and CREVOKE per free from the hardware check.
   - For each of the 15 allocating programs, it lies at or below bitstream 3's ObjLife/MapLife point estimate. On
     bitstream 3 ObjLife also pays the LDC/STC query, and the LSU replays on its many per-object nodes, which overflow
     the 256-entry cache.
   - For the five sglib-*-libc, it is expected 0.5 to 3 points lower (bitstream 3: 1.047 to 1.064).
   - For the 88 programs, it lies within [0.995, 1.005].
5. **The ladder's headline, ObjLife (bitstream 3) / ObjBounds.**
   - For every program it is ≥ bitstream 3's ObjLife/MapLife. It is that ratio times MapLife/ObjBounds, and the
     second factor is ≥ 0.99 by prediction 3.
   - For the 8 allocation-heavy programs (≥ 1,400 heap calls per iteration), its geomean exceeds 1.044.
   - This is a cross-bitstream, cross-board ratio of identical images. Its interval is reported with that named.

**Analysis.**
- Per boot: `check-objb-boot.py <out>/perf-objb-<kit> perf/objb/kits/<group>/<kit>` writes `runs-<kit>.tsv` (dev's
  parse-board.py).
- In-boot: `perf-analyze.py report --arms T1,T3` on this run's rows, giving prediction 4.
- Cross-bitstream (predictions 2, 3 and 5): this run's T1/T3 rows are relabelled ObjBounds/ObjLife-v by a script
  committed BEFORE the first main kit's results are read. They are joined with the bitstream-3 rows by (program, arm
  image, process, K), and reported per program with Welch intervals and per suite as geomeans. Nothing from this
  bitstream enters a bitstream-3 geomean, and the reverse.
