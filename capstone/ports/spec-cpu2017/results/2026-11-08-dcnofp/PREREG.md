# gcc train with direct calls and no frame pointer (compiler lane's image), board 3: pre-registration

Written and pushed before the board call (8 Nov, 12:1x). The compiler lane measured where gcc train's in-context
instructions exceed Plain-with-cache (13.735 G vs 10.977 G program instructions, +2.76 G) and built three images on
c3f's exact recipe, one change each (c1-gate/gt2, script build-arm.sh). This run takes the combined one:
- dcnofp: `-mllvm -capstone-direct-calls` (llvm-c4 = origin/dev c9611344eeb7 + 331d73414667, branch
  compiler-spec-c4, default off) and `-fomit-frame-pointer` on gcc's compiles; runtime and libc prebuilt as in c3f.
- Stripped exe 736f881f4958ac3b. QEMU twin (kit-profq): specdiff 1/1 PASS, every output hash equal to c3f's,
  in-context instret 12,868,888,921 against c3f's 13,735,649,769 (−6.31 %). Edge check SAFE (slack 1,760 B).
- The lead's trade, taken here as a measurement: without a frame pointer, a fault inside gcc's code gets no frame
  chain in the launcher's fault record (diagnostic only; nothing decides on it). Plain has no frame pointer either.

**Stage** c3f's board stage with only the executable replaced (5642377943d724c6 → 736f881f4958ac3b). **Kit**
bkit27-gtq. **Board 3**, after the tickcost boots there.

| # | Prediction |
|---|---|
| D0 | rc 0, specdiff 1/1, outputs equal c3f's; control passes; in-context instret within ±0.1 % of 12.869 G |
| D1 | inside time 3,514 s → 3,280-3,370 s (the removed instructions are moves, address arithmetic and spills, cheaper than average) |
| D2 | run 4,253 s → 4,030-4,110 s, i.e. 1.23-1.25× Plain-with-cache (3,280.9 s); outside about unchanged |

## Addendum 1 (8 Nov 12:2x, before the board call): on the 32 M quantum kit

The quantum result (results/2026-11-08-quantum: c3f 4,215.0 s with a 32,000,000-cycle supervisor quantum, kit
bkit32-q32) came in first. The run moves to kit bkit32-q32 so that its reference is that run (one variable: the
image). Predictions restated against it: D1 inside 3,504 s → 3,270-3,360 s; D2 run 4,215 s → 3,990-4,075 s
(1.22-1.24× Plain-with-cache).

## Result (8 Nov 13:41, board 3, kit bkit32-q32)

Stage out-dcnofp-b3/502.gcc_r-train01-capstone-lifetimes-devmk-dcnofp (exe 736f881f4958ac3b), reference out-q32-b3
(c3f exe on the same kit and board). The board's clock is 25 MHz (the reference's 87.61 G in-context cycles = 3,504 s).

| | q32 reference | dcnofp | change |
|---|---|---|---|
| run time | 4,214.98 s | 3,924.91 s | −290.1 s (−6.88 %) |
| in-context cycles | 87,610,137,530 | 81,318,547,238 | −7.18 % (inside 3,504.4 → 3,252.7 s) |
| in-context instret | 13,735,699,863 | 12,869,971,613 | −6.30 % |
| outside (run − inside) | 710.6 s | 672.2 s | −38.4 s |
| context entries | 11,710 | 11,545 | −165 |

- **D0 holds**: rc 0, specdiff 1/1 PASS, control passed, train01 .s hash c786c07f51d5 equal to the reference's (only
  the timing .err differs), in-context instret 12,869,971,613 = QEMU's 12,868,888,921 + 0.008 %.
- **D1 failed on the favourable side**: inside 3,252.7 s, below the 3,270-3,360 s band. The removed instructions
  were not cheaper than average: cycles fell 7.18 % for 6.30 % fewer instructions.
- **D2 failed on the favourable side**: 3,924.9 s, below the 3,990-4,075 s band; outside fell 38 s instead of
  "about unchanged" (165 fewer entries).
- **Ratio**: 3,924.9 / 3,280.9 = **1.196× Plain-with-cache**. N = 1; the c3f image's repeated gcc train runs on these
  boards spread by about ±3 s (4,253.3 s board 3, 4,255.8 s board 2, 4,256.3 s board 3 on the exhpm kit), so the margin to 1.20× (13 s) is outside that spread.
- **Fairness check**: Plain's compiler omits the frame pointer by default. In Plain gcc's binary
  (cpugcc_r_base.plain-rv64r), 274 of 12,675 functions set s0 as a frame pointer: the ones with a dynamic stack.
  Control: the same compiler with -fno-omit-frame-pointer emits `addi s0, sp, 16` for a two-line function, and
  without the flag it does not. Direct calls are what Plain does anyway. The change brings Capstone level with
  Plain on both counts. It does not give Capstone an option Plain lacks.

## Addendum 2 (8 Nov 14:0x, before its board call): perlbench test, the same two flags on the unify-nd stack

The compiler lane rebuilt perlbench on the unify recipe (c1-gate/gt2/build-perl-unify.sh): the control reproduces
the board's unify-nd exe 89398ba2be3127ab, and udcnofp (5996ba25b57a303c) adds `-mllvm -capstone-direct-calls
-fomit-frame-pointer`. Edge check SAFE (slack 688 B). QEMU twin (kit-unq, the unify-rg-nd stage, exe swapped):
specdiff 2/2, in-context instret 93,567,228,511 → 91,120,748,598 (−2.61 %).

**Stage** perf/dcnofp/stages/500.perlbench_r-test-capstone-lifetimes-unify-rg-nd-udcnofp: the unify-rg-nd stage
with only the executable replaced. **Kit** bkit30-unify (driver perf/unify/un-run.sh). The reference is unify-nd:
30,120.3 s on **board 2**, with in-context cycles 13,822,001,440 + 532,911,311,666 = 546.73 G (21,869 s at 25 MHz),
instret 93.547 G, outside 8,251 s. This run is on **board 3**, so the image and the board both differ; the c3f gcc
train image on kit bkit27-gtq read 4,255.8 s on board 2 (out-fill-b2) and 4,253.3 s on board 3 (out-ga2-b3).

| # | Prediction |
|---|---|
| U0 | rc 0 on both commands, specdiff 2/2, every output hash but the two .err files equal to unify-nd's; control passes; in-context instret 91.12 G ± 0.2 % |
| U1 | in-context cycles −2.6…−3.2 % (gcc's cycles fell 1.14× its instructions): 529.2-532.5 G, inside 21,170-21,300 s |
| U2 | run 29,400-29,650 s (outside 8,251 s ± 100 s), i.e. 1.344-1.355× Plain-with-cache (21,882.6 s) |
