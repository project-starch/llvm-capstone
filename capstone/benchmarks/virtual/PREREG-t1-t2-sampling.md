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

## Plain: the same programs with no capabilities at all, bitstream 3 (2026-10-30, before any Plain boot)

**Why.** MapLife and ObjLife are Capstone programs: 16-byte capability pointers, capability instructions, hardware
checks on every access, a context with its own escapes, and (ObjLife) per-object lifetimes. Plain builds the same
103 programs as ordinary RISC-V Linux executables. So ObjLife/Plain is the whole cost of the Capstone virtual profile
as built today, and MapLife/Plain the same without per-object lifetimes.

Arm names: Plain is new (record arm `Plain`). MapLife is this document's T1 and ObjLife its T3 (lifetimes, no scrub).
The Plain kits keep the record labels `T1` and `T3` for the Capstone arms.

**Images.**
- **Toolchain.** The campaign SDKs' own clang (cc-prmerged, sha256 b90e13006a01…), targeting
  `riscv64-unknown-linux-musl -march=rv64gc -mabi=lp64d`. Flags, sources and iteration counts are build.py's for the
  campaign (`build.py --mode plain`). Every program's iteration count equals the `iterations` of the image booted on
  bitstream 3.
- **libc.** musl 1.2.5 from the SDKs' tarball, stock mallocng, built by musl's own configure at -O1 (the Capstone
  libc's level). The generic musl-capstone patches 0001-0003 are applied; the capability patches 0004-0010 are not.
  libgcc comes from the board's buildroot gcc 12.3. Links are static, `ld.lld --gc-sections`.
- **Layout.** A Plain image cannot share the Capstone arms' addresses, and the K ladder measured code placement at up
  to ~8 % per program at identical instret. So every Plain program is linked at K = 0..3: a retained, never-executed
  `.text` pad of K × 64 bytes after crt1/crti. Every text symbol after the pad moves by exactly K × 64 in all 103
  programs, and the check flags 1,118 of 1,118 symbols when given the wrong shift. A program's 4 Plain processes
  take K0..K3. Per-program ratios therefore carry the Capstone-vs-Plain layout difference, which no pad spans.
  **Suite geomeans are the result**, and a per-program cycle difference below ~5 % is not read as a codegen effect.
- **Images:** `perf/T0b/build/Plain`, 103 programs × 4 pads = 412 images, rebuilt byte for byte.
- **QEMU twin** (`-icount`, `1 2`):
  - all 412 run rc 0, every program's check= equals every bitstream-3 board record of T1/T2/T3 (103 of 103);
  - Plain ctl-loop's user instret, after the kernel subtraction, is 19,999,937 against ObjLife's in-window
    20,001,989 (both 2N+2 = 20,000,002 of user loop).

**Counting.** Plain has no CSRUNV window, so it counts with user-mode `rdcycle`/`rdinstret` around the same bracket.
- **Enabled.** The firmware's OpenSBI (wt-wrap-s26 aab63d5, `lib/sbi/sbi_hart.c:55`, `:63`, `:67`) writes
  SCOUNTEREN = 7, MCOUNTEREN = −1 and MCOUNTINHIBIT = 0xFFFFFFF8. The board kernel (linux-6.4.14 `.config`:
  `# CONFIG_PERF_EVENTS is not set`) never rewrites scounteren. The silicon (capstone-ariane e7af48457,
  `core/csr_regfile.sv:3006-3012`) faults a user read only on `~mcounteren | ~scounteren`.
- **What the readings include.** Every kernel entry inside the bracket: timer ticks (`CONFIG_HZ=100`, periodic, so
  ~1 per 250,000 cycles), other interrupts, page faults, and the heap's mmap/munmap/brk syscalls. The Capstone arms
  count only inside their windows: they escape about once per 2 M cycles, at ~14 k in-window cycles each, and carry a
  0.3 M-cycle bracket per iteration.
- **Harness.** `-DBENCH_USER_COUNTERS`; the Capstone build is byte-identical without it. Each BENCH line is followed
  by `BENCH-KERN … timer= irq= minflt= majflt= nvcsw= nivcsw=`, from /proc/interrupts and getrusage taken outside the
  counter reads. `entries` = interrupts + faults inside the bracket.
- **Correction** (`plain-correct.py`, per boot, from that boot's own controls; every kit carries ctl-empty and
  ctl-loop):
  - k_ins = (Plain ctl-loop instret − Plain ctl-empty − 20,000,002) summed, divided by the interrupts.
  - k_cyc = the probe's median gap cycles: each gap is one interrupt seen between two back-to-back rdcycle reads.
  - per-fault cost = the probe's first-touch fault cycles and instret.
  - Plain rows lose irq × k and faults × fault cost.
  - The ctl-loop least-squares slope of cycles on interrupts is printed with its SE as a cross-check only. It has one
    interrupt of leverage: on planted data with 8,000 cycles of noise it read 10,935 ± 2,476 for a true 9,000.
- **Stated bias.** The correction removes the handler and a tiny working set's refill, not a real program's refill
  after each tick, and not the syscalls. So, per program, **ObjLife/Plain(raw) ≤ ObjLife/Plain(corrected) ≤ the
  interrupt-free value.** Both raw and corrected are reported. `--subtract-empty` also removes each arm's own
  ctl-empty bracket (the "work only" ratio), reported beside them.

**The frame-pointer finding** (QEMU, instructions). The capstone64 driver defaults to `-mframe-pointer=all`, while
Plain's Linux triple omits the frame. Rebuilt with `-fno-omit-frame-pointer` (PlainFP), ObjLife/PlainFP instret falls:
- BEEBS 1.296 → 1.189;
- RV8 1.119 → 1.075;
- CoreMark 1.101 → 1.079;
- BEEBS -libc 2.246 → 2.202.

So about a third of BEEBS's instruction overhead against Plain is a toolchain default, not a capability cost. In
single programs it is most of it: fibcall 2.76 → 1.09, bs and janne_complex 3.00 → 1.09, rv8-qsort 1.51 → 1.14.
Adding `-fno-jump-tables` changes the suites by < 0.015. PlainFP is built (`build/Plain-fp`) and is not booted: the
lead's decision, 2026-10-30.

**Kits.** `make-kits.py`, arms `Plain,T1,T3` alternating ABC CBA. Dimensions:
- P = 4 × (W = 1, I = 5) per arm;
- crc32, huffbench, huffbench-libc and wikisort at P = 11 × (0, 1);
- seed 20261030.

The T1 and T3 images are byte-identical to images staged in the bitstream-3 campaign kits that booted (148 per arm,
0 refused; `tools/merge-images.py`). Every kit starts with `plain-tick-probe` (rv64imafd, zero 2-byte instructions,
sha256 8e405ba6c42ef433) and contains ctl-empty and ctl-loop. Launcher 978c09d61f774315.

| kit | programs | processes | run script | SHA256SUMS | stage wrapper run.sh | stage bytes (gzip) |
|---|---|---|---|---|---|---|
| pp1 (pilot) | 8 | 96 | 1aa34b3d96ea7fbe | 152f067e8cc5bcde | f1dc3d7d542075fd | 11.05 M (4.69 M) |
| p1 | 19 | 249 | 149060bc317f7861 | 8bc7adc9ce247939 | 52d5c47e3c6dbeae | 16.87 M (7.10 M) |
| p2 | 19 | 249 | 8935bf94c6fac406 | 3656c96bb282904a | 4a9d54efa3cdeccc | 15.54 M (6.62 M) |
| p3 | 19 | 249 | 5e8c4fc3858bde19 | 8262664d3cf279f3 | 785baa5b35b3b28f | 17.48 M (7.49 M) |
| p4 | 18 | 237 | ceb5289d893c300a | d7b6f4fc142a8670 | 823bfc71a8db1537 | 16.53 M (7.04 M) |
| p5 | 19 | 228 | 9d9dce27f635fd3d | e846e27a42a8d8c6 | f66f4e746bec2081 | 16.27 M (7.00 M) |
| p6 | 19 | 228 | 7f033c9f8cba5e60 | 9db4cfa0d9cb36e8 | 6ddbe71ea509a045 | 16.18 M (6.97 M) |

pp1 holds ctl-empty, ctl-loop, coremark, janne_complex, sglib-listsort, rv8-aes, rv8-miniz and sglib-rbtree-libc.

**Boots.**
- **Board and route.** Board 1, bitstream 3 (caplifive_vp2c_e7af48457), kit `bkit12-perf-b3-b1`, through the SPEC
  lane's chain, driven by `plain-chain.sh 1 plain-b1.list <root>`:
  - chain-board.sh runs spec-board.sh, which runs `run-stage.py run --backend board`;
  - spec-board.sh checks the launcher and module by hash and runs the known-good control first (crc.K0 T1,
    bd909cc6eba4bf2d, W 1, I 5).
- **The stage.** `perf/T0b/board/stages/perf-plain-<kit>/` is a thin wrapper (`make-plain-stages.py`).
  - Its run.sh hashes every kit file on the board against the kit's SHA256SUMS and stops with exit 3 on a mismatch.
    On the host, a kit copy with one byte flipped exited 3 with no process run.
  - It then runs the kit's own run script unchanged.
- **The pilot goes first and alone.** plain-chain.sh runs the chain over the pilot line only, then
  `check-plain-boot.py --pilot`. It arms p1..p6 and runs the rest of the list only on exit 0.
  - p1..p6 are written disarmed (`plan.json.disarmed`), so spec-board.sh refuses them ("not a make-stage.py output",
    SB STOP) and chain-board.sh stops. A chain started on the whole list without the gate cannot reach the main kits.
- **List lines** (stage, step_s, booking_min, twin):
  - pp1 at 3600 s and 90 min: 96 processes, estimated 0.54 h. The campaign's t3k kits (60-81 processes) took 24-29
    min of step and 33-38 min wall.
  - p1..p6 at 10800 s and 210 min: 228-249 processes, estimated 1.26-1.33 h each. The campaign's t3a kits (285
    processes) took 78-84 min of step and 87-94 min wall.
  - The bounds are the campaign's (3600 s for t3k, 10800 s for t3a). Total about 9 h of board time on board 1,
    plus one boot (~6 min) per kit.
- **Firmware size.** The main stages are 15.5-17.5 MB, 6.6-7.5 MB gzipped, against 16.0 MB / 6.2 MB for the campaign's
  t3a2. That stage gave a 23,759,368 B payload, which pads in 2 MiB steps. Even with the whole 1.3 MB of extra
  compressed bytes, the next step is 25,856,520 B, under BOARD_FW_CAP 26,000,000.
- **Dry run** (`perf/T0b/board/dryrun/`):
  - Stubs: busy checks, hub and BOARDVM_STUB_TRANSCRIPT. The REAL chain-board.sh, spec-board.sh and run-stage.py ran.
  - Stub transcripts: the 519.lbm_r-test board-1 frame, with each kit step taken from the campaign's bitstream-3
    block of the same program, arm and process, renumbered. Plain steps use the campaign's T2 block, relabelled,
    with BENCH-KERN lines added.
  - All seven lines reached END rc 0 with `RUN control PASSED` and the gate read PILOT PASS. check-plain-boot.py read
    every boot VALID: 96, 249, 249, 249, 237, 228 and 228 processes, each set equal to the kit's steps.tsv.
  - The runner was invoked with step_s 3600 for pp1 and 10800 for p1..p6, and booked 90 and 210 min.

**Pilot gate** (`check-plain-boot.py --pilot`; the main kits run only on exit 0):
- **STOP (exit 3): the user-mode counters are not usable on this board.**
  - P1: the monitor's illegal-instruction report appears anywhere in the transcript (`ILLX:`, `INJS:` or `INJL:`).
    None of the 22 passing bitstream-3 campaign and SPEC-chain transcripts carries one.
  - P2: the probe block lacks `PLAINPROBE access=ok cycle_advances=1 instret_advances=1`, `PLAINPROBE done` or
    `@@PLAINPROBE-END rc=0`.
- **HOLD (exit 4): the counters work, but the two instruments disagree.**
  - P3: cycles_per_us outside [24.5, 25.5]; gaps per second outside [90, 110]; |gaps − proc_interrupts_delta| > 3.
  - P4: ctl-loop's k_ins not within ±15 % of the probe's gap_instret median, or any iteration's ratio outside ±10 %
    of k_ins.
- A VOID pilot (exit 1) or no data (exit 2) also stops the chain.

**What a non-working counter looks like on this board.**
- The bitstream-3 monitor (csbi-batch d9da397, `handle_exception`) takes the user rdcycle's illegal-instruction trap
  in M. It prints `ILLX:0000E003`, `MTVL:` (the instruction word, `C0002573` for `rdcycle a0`) and `MEPC:`.
- Since f8bfde5 (2026-10-17) it then injects the trap into Linux (`INJS:00000002`). The probe's SIGILL handler prints
  `PLAINPROBE access=trapped at=rdcycle`, exits 2, and every Plain process prints `counters=absent`. The boot
  completes.
- If the injection would land on the faulting pc, the monitor prints `INJL:` and spins. The transcript then stops
  after the monitor lines. No heartbeat follows, the driver's idle limit (`BAKED_IDLE_S=200`) ends the session about
  200 s later, and the step's text is only the last heartbeat's tail plus `[runner] TIMEOUT`.
- run-stage.py reads that step as NO DATA, and chain-board.sh by itself would go on, because a failed benchmark is a
  result. That is why the gate, and the disarmed stages, exist.
- Dry-run negatives (pilot stubs), each of which ended the chain with p1..p6 not run and still disarmed:
  - trapped: STOP exit 3, P1 and P2;
  - wedge: STOP exit 3, P1 and P2, VOID C2-C6;
  - instruments disagree (ctl-loop excess 2× the probe's): HOLD exit 4, P4;
  - failed control: CB STOP and the gate VOID (C1);
  - chain-board.sh run directly over the whole list: pp1 ran, then SB STOP at the disarmed p1, then CB STOP.

**VOID.** A boot is VOID, and none of its rows enter any ratio, unless `check-plain-boot.py` passes:
- C1 `RUN control PASSED`;
- C2 `@@PLAIN-SUMS OK`, and the board-side hash block equal to the kit's SHA256SUMS line for line;
- C3 `@@PERF-<KIT>-DONE` and `@@SPEC-DONE plain-<kit>`;
- C4 parse-board.py exit 0 (with the Plain arm, counters=present, and one BENCH-KERN per BENCH line);
- C5 the parsed processes exactly the kit's steps.tsv (a transcript cut after process 100 of 249: parse-board alone
  exit 0, the gate VOID);
- C6 no `@@STEP-BOUND`, runner TIMEOUT, kernel oops or OOM.

Also VOID:
- a boot in which spec-board.sh printed `SB STOP`;
- a boot whose MapLife or ObjLife instret misses prediction 5 by more than 0.1 % on more than one program. The
  platform is then not the T3 runs', and no ratio from that boot is used until it is explained.

**Predictions.** Ratios are ObjLife/Plain and MapLife/Plain (perf-analyze `--arms Plain,T3` and `--arms Plain,T1`).
Cycles are the corrected, bracket-subtracted reading, unless named raw.

1. **Correctness.**
   - Every process in every arm exits 0, with W + I BENCH lines, no VERIFY-FAILED and BENCH-DONE failed=0.
   - Within each program, check= is one value across the three arms and equals the bitstream-3 records'. It already
     does on QEMU, 103 of 103.
2. **Plain's instruction count is the ISA's.**
   - For each of the 88 non-allocating programs, Plain's board instret after the boot's interrupt and fault
     correction equals its QEMU user instret within 0.5 % (`qemu-instret.tsv`, plain_user_instret). It is the same
     static binary.
   - For the 15 allocating programs, within 3 %: their heap syscalls are inside the window and are not corrected.
3. **Instructions.** ObjLife/Plain and MapLife/Plain per suite equal the QEMU table within ±0.01:

   | suite | n | MapLife/Plain | ObjLife/Plain |
   |---|---|---|---|
   | BEEBS upstream | 80 | 1.296 | 1.296 |
   | BEEBS -libc | 13 | 2.232 | 2.246 |
   | BEEBS, all 93 | 93 | 1.398 | 1.399 |
   | RV8 | 7 | 1.119 | 1.119 |
   | CoreMark | 1 | 1.101 | 1.101 |
   | ctl-loop | 1 | 1.000 | 1.000 |

   Per program class (ObjLife/Plain, geomean and range):
   - pointer-chasing (sglib upstream, dijkstra, ctl-*, wikisort, levenshtein; 12): 1.199, 1.010-1.631;
   - register-bound (crc32, nettle, aha-mont64, rv8 sha512/primes/norx/miniz and others; 12): 1.069, 0.938-2.053;
   - call-overhead (fibcall, bs, janne_complex, rv8-qsort): 2.476, which frame pointers alone reduce to 1.09-1.14;
   - sglib -libc (5): 3.506, 2.695-3.957. That is 724-834 extra instructions per malloc/free (ctl-vector-libc 2,671):
     heap.c's ownership work and capability mallocng.
   - miniz (3.77) and nsichneu (2.91) are reported, not predicted. miniz's gap is unexplained; jump tables are ruled
     out by the PlainFPNJT run. nsichneu is consistent with capability materialisation of global accesses: 893
     `shrink`, 900 `stc`, 900 `ldc` in its `benchmark()`; static 2.08× of a dynamic 2.91×.
4. **Cycles.**
   - **Direction: cycles ≥ instructions for most programs.** Capstone's CPI is the higher one:
     - pointer-holding data take twice the D-cache (32 KiB, 8-way, 16-byte lines: one capability per line) and
       twice the bus beats;
     - the hot text is ~2-3× the bytes in a 16 KiB I-cache (no RVC, a frame prologue in every function; coremark
       41,302 four-byte instructions against Plain's 9,424 four-byte plus 7,892 two-byte);
     - every escape (~1 per 2 M cycles) flushes all TLBs inside the window.
     The Capstone arms already run at CPI 3-5.5 on the pointer- and memory-heavy programs (sglib-listsort 4.40,
     miniz 4.93, nsichneu 5.51, coremark 3.19), against 1.27-1.52 on register-bound ones (rv8-sha512, ctl-loop).
   - **Bands (ObjLife/Plain, corrected).**
     - BEEBS upstream geomean in [1.30, 1.75]. Pointer-chasing programs exceed their instret ratio by ≥ 5 %.
       Register-bound kernels lie within ±5 % of theirs.
     - RV8 geomean in [1.10, 1.45]. sha512, primes, norx and miniz have instret ratios of 0.99-1.00, so their cycle
       ratios are the pure CPI effect, in [0.98, 1.15].
     - CoreMark in [1.10, 1.45]. Its list nodes are 32 bytes in every build (CM_LIST_NODE_BYTES), so the list's
       footprint is equal by construction.
     - ctl-loop in [0.99, 1.03]: the same 2N+2 user loop, and ~15 in-window escapes per iteration at ~14 k cycles
       each (+0.7 %). Raw, it reads lower by interrupts × k_cyc / cycles.
     - BEEBS -libc geomean in [2.0, 3.3]; each sglib -libc ≥ 2.5. miniz-libc in [0.6, 1.3]: Plain executes more
       instructions there (0.86×), from mallocng's munmap and re-fault churn inside its window (468 faults per
       iteration on QEMU). rv8 aes, miniz, norx and primes in [0.95, 1.20]; rv8-qsort in [1.15, 1.80].
     - Suites: BEEBS-93 [1.40, 1.90], RV8 [1.10, 1.45], CoreMark [1.10, 1.45].
   - **MapLife/Plain** equals ObjLife/Plain within the T3/T1 ratio of the same boot: ≤ 0.3 % apart on the 88 (T3/T1
     reads 1.0001 there), and apart by the libc group's T3/T1 (~1.027) on the BEEBS -libc programs.
   - **raw ≤ corrected** per program, by about interrupts × k_cyc / cycles (predicted 1-4 %).
5. **Replicate of the existing arms on board 1.** Per program, the median instret per iteration of T1 and T3 equals
   the bitstream-3 T1 and T3 runs' within 0.1 % (`results/2026-10-27-t3all-bitstream3/` and
   `results/2026-10-28-t3heap-bitstream3/`). ObjLife/MapLife's Welch interval overlaps those runs' T3/T1 interval.
6. **The pilot's instruments.**
   - k_ins is constant within ±10 % across the 20 Plain ctl-loop iterations and within ±15 % of the probe's
     gap_instret median (QEMU: 4,301-6,537 instructions per interrupt).
   - The probe's gaps agree with /proc/interrupts within 3.
   - Plain's σ_i of corrected log cycles is of the Capstone arms' order (pilot 0.13-1.0 %). If it exceeds the worst
     pilot program's (miniz, 1.01 %), P is re-derived by this document's rule before p1..p6 run.

A result outside a band is reported as such, with no re-run and no top-up.

**Not predicted:**
- per-program cycle ratios below ~5 %, because of layout (above);
- miniz and nsichneu;
- anything that needs k_cyc before the pilot has measured it.

**Analysis.**
```
cd <work root holding perf/> || exit 1; G=perf/T0b/board; O=<chain out root>
V=<tree with the Plain parse-board.py / perf-analyze.py / plain-correct.py edits>/capstone/benchmarks/virtual
python3 $G/check-plain-boot.py $O/perf-plain-pp1 perf/T0b/kits-pilot/pp1 --pilot --runs $G/runs-pp1.tsv
for k in p1 p2 p3 p4 p5 p6; do python3 $G/check-plain-boot.py $O/perf-plain-$k perf/T0b/kits/$k --runs $G/runs-$k.tsv; done
# per VALID boot (the probe lines are in that boot's step text):
for k in pp1 p1 p2 p3 p4 p5 p6; do
  python3 $V/plain-correct.py $G/runs-$k.tsv $G/runs-$k.tsv.kern.tsv --probe $O/perf-plain-$k/spec.txt --out $G/corr-$k.tsv --report $G/corr-$k.txt
  python3 $V/plain-correct.py $G/runs-$k.tsv $G/runs-$k.tsv.kern.tsv --probe $O/perf-plain-$k/spec.txt --subtract-empty --out $G/work-$k.tsv
  python3 $V/plain-correct.py $G/runs-$k.tsv $G/runs-$k.tsv.kern.tsv --raw --out $G/raw-$k.tsv
done
# concatenate p1..p6 (one header) per reading -> corr.tsv, work.tsv, raw.tsv; the pilot is reported on its own
for f in corr work raw; do for pr in Plain,T3 Plain,T1 T1,T3; do
  python3 $V/perf-analyze.py report $f.tsv --arms $pr --out plain-$f/${pr/,/-} --boot 10000 --seed 20261030; done; done
```
- perf-analyze.py reports TEST/BASE, so `--arms Plain,T3` gives ObjLife/Plain.
- The control rows are renamed `ctl-*@<board>/<boot>` by plain-correct.py, because every kit carries them.
- Predictions 2, 3 and 5 are checked by a script committed with the results. It is written after the boots and before
  any transcript is read, and each check is shown to fire on tampered inputs.
- The arm name `Plain` needs the perf-t0b edits to parse-board.py, perf-analyze.py, perfmain3-check.py and
  tier3-perfpilot.py. dev's parse-board.py reads a Plain step as nothing: one process of three in the dry run.

**Route as run** (coordinator, 2026-10-30, before any Plain boot). Board 1 already runs one chain: chain-board.sh over
`spec/board/b1-cont.list`. plain-chain.sh would start a second chain on the same board, so it is not used. Instead the
Plain lines go into that live list's unread tail:
- pp1 right after the directed perlbench boot, before the seven Caps boots;
- p1..p6 after the Caps boots, before the SPEC train lines.

The pilot gate is `plain-gate-waiter.sh <chain pid>`. It waits for a new `END perf-plain-pp1` line in chain-b1.log,
runs `check-plain-boot.py --pilot` on the chain's out dir, and arms p1..p6 only on exit 0. On any other exit p1..p6
stay disarmed: the chain stops at p1 (SB STOP, CB STOP), and chain-xz.sh does not start the train continuation. The
gate, VOID rules and predictions above are unchanged. Stubbed dry run: `board/plain/dryrun/waiter-dryrun.txt`. PASS
arms the six stages; trapped, wedge, hold and failed-control each leave them disarmed; a chain gone without a pp1 END
stops the waiter.

### Plain addendum 1 (2026-10-30 ~22:00, after the pilot, before any main Plain kit runs): the pilot read HOLD; an exclusion rule, and the main kits armed

**What the pilot read** (board 1, bitstream 3, pp1, 19:36-20:12; out dir spec/board/out/perf-plain-pp1). The gate,
`check-plain-boot.py --pilot`, exited 4 (HOLD) on P4 alone:
- P1, P2, P3 held: no ILLX/INJS/INJL line; `PLAINPROBE access=ok`; cycles_per_us 24.997, 202 gaps in 2 s (101/s),
  proc_interrupts_delta 202. The user-mode counters work on this board.
- P4: ctl-loop k_ins 10,637 instructions per interrupt (a mean over 20 iterations, 3,675..29,262) against the probe's
  gap_instret median 3,817 (mean 3,810).

**Why.** User-mode rdcycle/rdinstret count the whole hart, not the process. In a few iterations another task ran: the
timer count is far above the program's median and the involuntary context switches (nivcsw) jump. Examples: ctl-loop
425 ticks against a median of 134, nivcsw 157-162; rv8-miniz 466-475 against 152, nivcsw 163-172; coremark 194
against 83; sglib-rbtree-libc's iterations 0 and 1 in every process, 80-91 ticks against 45.5 with nivcsw 69-90, which
its 29-36 page faults (~0.8 M cycles at the probe's fault cost) cannot account for. The Capstone arms count inside their
own context only, so the other task's work does not reach their counters. On the 15 ctl-loop iterations outside those
events, k_ins is 3,978 (3,899..4,027): +4.2 % against the probe, and every iteration within ±3 %.

**The change. Written after reading the pilot and stated as such: it is a re-scoping of P4 and an exclusion rule
the original section did not have.**
1. *Contaminated iteration.* A Plain iteration is excluded when its `timer` count exceeds its program's median Plain
   `timer` count in the same kit by more than 10 % plus 2 ticks. The same iteration indices of that program are
   excluded from the MapLife and ObjLife rows of the same kit, so every ratio compares the same iterations. Excluded
   counts are reported per program; a program with more than 2 of its 5 timed iterations excluded in any process is
   reported, not used in a suite geomean.
2. *P4, re-scoped.* k_ins is the median over ctl-loop's non-excluded Plain iterations of (instret - ctl-empty median -
   20,000,002) / irq. It must lie within 15 % of the probe's gap_instret median, and each non-excluded iteration within
   ±10 % of k_ins. On pp1: 3,978 against 3,817, every iteration within ±3 %: PASS.
3. Everything else in the Plain section stands: the fault correction, the bias statement, the VOID rules C1-C6, the
   predictions.

**Consequence.** p1..p6 are armed now (`make-plain-stages.py --arm`), before board 1's chain reaches p1. Their rows are
read under rules 1-2. Rule 1 cannot remove a uniform cost: miniz's and aes's steady ~14 switches per iteration stay in.
That bias is in the direction the section already states (raw <= corrected <= true).

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
arguments and stops on a mismatch. rtl's board pre-registration is land/rtl-objb beb5384abb3b
(`capstone/tests/rtl-smoke/objb-bare-2026-10-30/PREREG.md`: ten bare images and their vectors), with its Addendum 1
at 177c6fd9f6e9 (both arms on board 2, one verdict per arm). Its identity kit runs before this run's pilot, below.

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
- runs rtl's bare identity kit before any Linux boot (`objb-bare.sh 2 <sealed name> <out>/bare`):
  - the ten images, in order: objb-pcc-ctrl, objb-stale, objb-perf-s1..s7, objb-pcc;
  - each one is its own boot through `sup-bare-2026-10-03-ladder/run_sup_bare.py` (102ab84e762cba50), with the images
    checked against beb5384abb3b's SHA256SUMS (8dd2a4812c9da608);
  - graded by value with `objb-bare-grade.py` under the RTL lane's rules:
    - CONFIRMED: objb-stale reads the ObjBounds vector, objb-pcc returns `0xbb5 0x0 0x0`, AND every cost shape's
      non-cycle positions (minstret, value, trap count, cause) equal the table. The last condition is the
      coordinator's ruling of 2026-10-30, as on the bitstream-3 arm: if ObjBounds changes anything but cycles, the
      rung measures the wrong thing;
    - VOID: the control objb-pcc-ctrl, or a returning image's SR;
    - STOP: objb-stale reads bitstream 3's vector, so the variant is not on the board;
    - FINDING: any other mismatch, the cost-shape one included;
  - only CONFIRMED goes on, and every other verdict stops the driver with its grade in the log.
  The bitstream-3 arm of the same kit runs on board 2 before the reflash to ObjBounds (Addendum 1: board 2 against
  board 2, one variable). There objb-pcc wedges the core, so it is alone in the last boot, ended by the runner's
  timeout, with the board powered off and released. `objb-bare-grade.py --compare <bs3 out> <objb out>` reads the cost
  pattern;
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
- **Bare kit** (`run-bare-dryrun.sh`, the runner stubbed with the PREREG's vectors):
  - ObjBounds: CONFIRMED 0, STOP 3 (stale reads bitstream 3's vector), VOID 1 (pcc-ctrl wrong), FINDING 4 (identity
    confirmed, s3's minstret off by one);
  - bitstream 3: AS EXPECTED 0 (objb-pcc does not return), FINDING 4 (stale off), STOP 3 (objb-pcc returns);
  - the compare reads 9 of 9 on the PREREG's vectors and 2 of 9 with the arms swapped (rc 1);
  - an empty output reads NO DATA.
  In the driver, a bare STOP, a bare VOID and the s3-minstret FINDING each stopped it before any Linux boot, with the
  main kits disarmed.
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

## Caps-v: capabilities without the per-access lifetime check (2026-10-31, before the flash and any Caps-v boot)

**Why.** The ladder has no rung with capabilities and no temporal check: Caps runs on bitstream 3, which checks a
lifetime on every capability load and store, and ObjBounds adds per-object bounds. The lead asked for that rung.
Caps-v is the Caps kits (perf/T0a/kits, unchanged, the same seven kits and SHA256SUMS as the Caps section) on the
ObjBounds bitstream (capstone-ariane 603528b8b, sha256 0f03b097bbd1693b..., the kit bkit15-objb-b2), on board 2,
the board the ObjBounds boots used. Arms in each boot: Caps (here Caps-v), MapLife (here an ObjBounds repeat) and
ObjLife (here ObjLife-v). Driver: board/objb/capsv-first.sh, a copy of objb-first.sh (wait for board 2's SPEC chain to
end its current boot; local sha256; fallback registered; booked flash; readback; the RTL lane's bare identity kit,
CONFIRMED only; pilot capsh1 gated by check-caps-boot.py; the other six kits; board 2 reflashed to bitstream 3 by
spec/board/b2-to-bs3.sh on every exit after the flash). Stubbed dry run: board/objb/dryrun/capsv (ok, pilot VOID,
bare FINDING, wrong hash: each branch as designed).

**The expected reading is predictable from the measured ratios.** Per-object bounds cost nothing measurable on
bitstream 3 (MapLife / Caps 1.000, 1.003, 1.000, 1.000), so Caps-v should read where ObjBounds reads. The run is for
the ladder's completeness, at the lead's request, not to discover something.

**Predictions.**
- C1. The bare kit reads CONFIRMED (the ObjBounds vector).
- C2. All seven boots VALID by check-caps-boot.py.
- C3 (identity control). The in-boot MapLife arm is an ObjBounds repeat: same images, bitstream and board. Against the
  31 October ObjBounds boots, per suite geomean 1.000 ± 0.3 % (beebs-up, beebs-libc, rv8, coremark). A miss makes
  every cross-day Caps-v reading unreadable; the in-boot ratios (C4, C5) stand.
- C4. In-boot Caps-v / MapLife-v (= Caps-v / ObjBounds, same boot) per suite within ±0.5 % of the inverse of
  bitstream 3's MapLife / Caps: 1.000, 0.997, 1.000, 1.000. Per program, every ratio within 0.98-1.02.
- C5. Instructions: Caps-v equals the Caps boots on bitstream 3 within 0.02 % for every program.
- C6. Caps-v / Plain, chained through ObjBounds, within ±1 % of ObjBounds / Plain: 1.43, 3.07, 1.23, 1.62.

**Analysis.** The estimator is ladder.py's (per program, process means of log cycles; Welch 95 % for direct ratios,
two-level bootstrap for chained ones and suite geomeans), over the programs Plain kept for C6. The script that joins
the Caps-v boots to the ObjBounds and Plain sources is committed before any Caps-v row is read.
