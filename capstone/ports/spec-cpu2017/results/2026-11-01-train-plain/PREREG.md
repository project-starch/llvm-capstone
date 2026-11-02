# SPEC train: the Plain baseline and the paper-set order (pre-registration, 2026-11-01)

Written and pushed before any Plain stage reaches a board, and before `board/after-train.sh` is armed.

## What changes

The project lead set SPEC train to the configurations the paper most likely needs and dropped further SPEC test runs.
The set is **Plain, MapLife (capstone-spatial), ObjLife (capstone-lifetimes)**:

- ObjLife / Plain is the cost of Capstone as a whole. SPEC had no baseline, so this is the number missing.
- ObjLife / MapLife is the cost of per-object lifetimes. lbm MapLife is done; perlbench MapLife is running.
- Caps, ObjBounds and Caps-v are not run on SPEC: each needs a reflash and about 8 h of a board.

Order on the boards, so that the must-have ratio is complete first:

1. the running stages finish (board 1 perlbench MapLife, board 2 lbm ObjLife);
2. every remaining ObjLife stage (`objlife-rest.list`, shared by both boards);
3. the Plain stages that passed the QEMU twin (`plain-gate.py`);
4. the MapLife stages not yet started (mcf, x264, imagick, nab, xz). Their `out-hfl` directories hold a `DEFERRED`
   file so the current chains skip them; they run into `out-pm`.

## Plain: what it is

The microbenchmark Plain recipe (`perf/T0b/STATUS.md` section 1), applied to SPEC:

| | |
|---|---|
| compiler | the SDKs' clang, sha256 b90e13006a016791271486e7f5c40f70f70b18a9f4bbf49347f2a9160ab6da43 |
| target | `--target=riscv64-unknown-linux-musl -march=rv64gc -mabi=lp64d`, static, ld.lld |
| libc | Plain musl 1.2.5, stock mallocng, libc.a 2d16947e1ca7…; libgcc.a 4dd5073760839b94… |
| config | `plain/plain-rv64.cfg` (sha256 5063622f86ae…): `mkcfg.py` of the kit's `capstone-virtual.cfg`. Same OPTIMIZE, PORTABILITY and src.alt per benchmark; only the compiler and the label differ |
| sources | the same src.alt packages, sha256-checked (`plain/build-plain.sh`) |
| tree | `spec/tree-plain`, a fresh copy of `spec/pristine` |

The buildroot gcc + glibc `native-rv64` build is not used: it changes compiler, libc and malloc at once.

Images (sha256, first 12):

| benchmark | images |
|---|---|
| perlbench | 779a8688f58c |
| mcf | 2e1b522831e9 |
| lbm | bd72cb22753b |
| x264 | 5adf7cac02e0, ldecod 68748eb4e198, imagevalidate 47e170a94b81 |
| imagick | f7f3e78a5a49, imagevalidate 47e170a94b81 |
| nab | d0c47f5c4a14 |
| xz | b889d0f57fcd |

**Stages** (`spec/stages-train-plain`) come from `make-stage.py --plain-arms plain-rv64 --rdctr rdctr`, with each
benchmark's campaign options (perlbench `--keep-lib`, x264 `--ldecod`, xz `--link-inputs --commands 0`). Their
plan.json inputs, commands, generation, host validations, outputs and exclusions equal the ObjLife stages' field for
field. In x264's and imagick's target validations only the reference file's staged name differs: it carries a hash
of the tree path.

**Run path.** A plain arm runs its commands directly, not through capstone-vexec. perlbench keeps argv[0] =
`../sw/perl` through a link. Each command quotes no module bench line (`nbench=0 :: plain`, so the control step's
line cannot be read as the command's). It prints `@@SPEC-CTR`: the hart's rdcycle and rdinstret before and after.

**Negative controls, run.**

- `make-stage.py` before and after the change gives byte-identical run.sh and plan.json for lbm train ObjLife and
  perlbench train MapLife. The lbm one also equals the campaign stage.
- `run-stage.py read` before and after gives an identical result.json and report for lbm train MapLife's board
  transcript.

## Metric

- **Primary: elapsed time per command**, the /proc/uptime span `run.sh` prints in every arm. It covers the same
  thing in all three configurations: the program plus whatever the kernel does for it. The ratio is the sum over a
  benchmark's run commands.
- The Capstone arms' in-context cycles and the Plain arm's hart counters are recorded too. They are not mixed into
  one ratio: hart counters include the kernel and every task on the hart, so they are an upper bound on Plain's own
  work.

## Predictions

- **P1.** On the QEMU twin and on the board, every Plain command and target validation ends rc 0, and every host
  specdiff passes. A Plain stage reaches a board only if its twin run did (`plain-gate.py`).
- **P2.** Board output hashes equal the Plain twin's for perlbench, mcf, xz and x264. For lbm, nab and imagick only
  specdiff is predicted: R-64/65/66 (FP flags, subnormals, NaN-boxing) can move FP bits between QEMU and silicon.
- **P3.** The integer benchmarks' outputs (perlbench, mcf, xz, x264) equal the Capstone arms' outputs.
- **P4.** Plain elapsed is below MapLife's for every benchmark.
- **P5.** ObjLife / Plain elapsed, per benchmark, lies in [1.1, 4.0]. This band is wide: the microbenchmark suites
  give 1.2–3.3 times Plain, and SPEC's allocation rates and working sets are outside what they cover.
  - A ratio below 1.1 is checked for a Plain arm that did extra work (hart counters against the twin) before it is
    believed.
  - A ratio above 4.0 is checked for kernel time (peak pages, collections) before it is attributed to the program.

**Board-time estimate.** Plain takes half to two-thirds of MapLife's time, about 50–70 board-hours for the seven
benchmarks. To be replaced by the twin's instruction counts.

## Gates, checked

- `plain-gate.py` was tested on fixtures before use:
  - a twin result with specdiff FAIL is HELD;
  - a passing one is listed;
  - a missing one is HELD.
- `after-train.sh` was dry-run with chain-board stubbed (`dryrun-after-train.txt`). The run covers:
  - the normal path;
  - a chain that ended without its done marker (nothing launched);
  - an ObjLife part that did not finish (stop before Plain);
  - a twin chain not done by its limit (gates on what has finished).
- `boardvm.py`'s heartbeat follows the boot budget (903cc08e7410), so the 26 h stages are not cut at 19.5 h.

## Addendum 1 (2026-11-01 13:30): Spatial replaces MapLife

Written and pushed before board 2 is flashed and before any Spatial stage boots.

**The set.** The project lead set SPEC train to three configurations, named after the external collaborator's three
steps:

| Name | What it is | Build and bitstream |
|---|---|---|
| Plain | RISC-V, no capabilities | `plain-rv64` stages, bitstream 3 |
| Spatial (ObjBounds) | capabilities, per-object bounds checked on every access; no per-access lifetime check | the `capstone-spatial` (MapLife) stages, unchanged, on `caplifive_vp2c_603528b8b` |
| Full (ObjLife) | Spatial plus a lifetime per malloc object and the hardware lifetime check on every capability load and store | `capstone-lifetimes` stages, bitstream 3 |

The SPEC arm label `capstone-spatial` names MapLife's build, not this Spatial. Spatial is that build on the bitstream
without the check. It still carries mapping-level lifetime state, so it is an upper bound on a Spatial with no
temporal state at all.

**What changes.**
- The deferred MapLife stages (mcf, x264, imagick, nab, xz) are dropped. `maplife-deferred.list` is emptied before
  `after-train.sh` reads it.
- lbm MapLife (done) and perlbench MapLife (running on board 1) are kept. Each pairs with its Spatial run as a
  same-image control, and they split Full / Spatial into the check and per-object lifetimes.
- Board 2's chain is cut after mcf ObjLife: its list is truncated at the byte its reader had reached, so no running
  stage is touched. Its `after-train.sh` is stopped while still waiting.

**What `board/spatial-b2.sh` does on board 2**, in order:
1. waits for that chain to end with its done marker;
2. checks the bitstream's sha256 (0f03b097…) and the kit;
3. checks the board is free, the store-has control and the bitstream-3 fallback;
4. flashes the ObjBounds bitstream, then reads the name back;
5. runs the RTL lane's bare identity kit. Only CONFIRMED goes on;
6. runs lbm Spatial as the pilot, gated by `spatial-gate.py` against lbm MapLife. If the pilot passes, the other six
   run in the order mcf, perlbench, x264, imagick, nab, xz;
7. flashes back to bitstream 3, on every exit after the flash;
8. hands over to `after-train.sh 2` (the ObjLife stages left, then the gated Plain stages), only once board 2 is
   confirmed back on bitstream 3.

**The kit.** `bkit15-objb-b2` is `bkit12-perf-b3-b2` file for file. Launcher 978c09d61f774315, module 9c60fb53f80ce67e,
`board-run.sh`, base and bake manifests are equal by sha256. Only `board.env`'s `BOARD_BITSTREAM` differs.
`spec-board.sh` takes it through `SB_KIT` and `SB_BITSTREAM`, set together. With neither set it behaves as before.
A six-case dry run covers:
- the default;
- each variable alone (STOP);
- both;
- each kit under the other bitstream's name (STOP).

**The pilot gate** (`spatial-gate.py`, the same images as MapLife):
- G1: the run is done and every command ended rc 0.
- G2: every specdiff passes.
- G3: every recorded hash is equal, stderr excepted. The launcher's stderr carries ASLR bases; it differs between
  lbm's MapLife and ObjLife runs.
- G4: each command's in-context instret is within 0.1 %.
- Missing data exits 2.

Controls, run on the recorded lbm results:
- MapLife against itself, and ObjLife against MapLife, both pass;
- each of rc 1, specdiff FAIL, a changed `lbm.out`, instret +0.2 % and done=false gives MISS;
- an empty command list and a missing file give NO DATA.

`dryrun-spatial-b2.txt` runs the driver with every board call stubbed over eight branches:
- the normal path;
- no done marker;
- a wrong store control;
- a wrong bitstream hash;
- a failed flash;
- a bare kit not CONFIRMED;
- a pilot gate MISS;
- a failed flash back.

**Predictions.**
- **S1.** Every Spatial command ends rc 0, and every specdiff passes.
- **S2.** Outputs (stderr excepted) are equal to MapLife's on the board where MapLife ran (lbm, perlbench), and to the
  QEMU twin's for the integer benchmarks. In-context instret is within 0.1 % of MapLife's or the twin's: the images
  are the same, and the bitstream changes timing only.
- **S3.** Spatial is never slower. MapLife / Spatial elapsed, where both exist:
  - lbm in [1.000, 1.02]. Its grid sweep loads and stores doubles, not capabilities, and the check costs 4 cycles per
    capability load and 5 per capability store.
  - perlbench in [1.03, 1.40]. An interpreter loads and stores pointers constantly. The microbenchmark range is
    1.04-1.28.
- **S4.** Full / Spatial per benchmark lies in [1.00, 1.50], and Spatial / Plain in [1.1, 3.5]. Microbenchmarks give
  1.04-1.28 and 1.23-3.07, but SPEC's elapsed time includes the kernel. Full enters the kernel more often, so Full /
  Spatial is expected at or above the microbenchmark range.

**Board time.** Spatial is about MapLife's time less the check, about 95 board-hours for the seven benchmarks. Board 2
starts it after mcf ObjLife, about 04:30 on 2 November, and ends near 6 November. Board 1 runs the remaining Full
stages (about 68 h), then Plain (about 54 h). Board 2 joins whatever is left once it is back on bitstream 3. All three
configurations should complete around 7 November.

## Addendum 2 (2026-11-02 15:00): SPEC test in all three configurations, ahead of the remaining train stages

Written and pushed before any of these test stages boots. The project lead asked for SPEC test in Plain, Spatial and
Full by 3 November, without stopping a running stage.

**What exists.** The board test runs of 28-30 October (`results/2026-10-28-board-test`) are two arms on bitstream 3:
`capstone-lifetimes` (Full) and `capstone-spatial`, which on bitstream 3 is **MapLife**, not Spatial. Neither Plain
nor Spatial has run on test. The images are the pre-fix campaign images. perlbench test's second command fails specdiff
in both Capstone arms; this is the heap defect of Addendum 16 of that run, and it recurs in any arm built from those
images.

**Spatial test** (board 2, already on `caplifive_vp2c_603528b8b`, no extra flash). The same images as the MapLife test:
- perlbench, mcf, xz and x264 use the very stage directories MapLife's test ran (`stages-board/*-test-capstone-spatial`).
- lbm, nab and imagick ran as two-arm stages, so one-arm `capstone-spatial` stages were made from the same tree
  (`stages-test-objb/`, `make-stage.py --arms capstone-spatial`). Each plan.json's `capstone-spatial` executable sha256
  equals the two-arm stage's, and every other field is equal, apart from the new `stripped: false`.
- gcc is left out: it is not in the seven-benchmark set.
- The lines go into `spatial-b2.list` after the lbm train pilot, so they run only if the pilot passes its gate.
  Order, short first: lbm, nab, imagick, xz, mcf, perlbench, x264. The Spatial train stages follow.
- The stubbed dry run of `spatial-b2.sh` was repeated over the new list. It covers all eight branches, including the
  stage-directory check on every line.

**Plain test** (board 1, bitstream 3, after the running perlbench Full stage, about 00:30 on 3 November).
- The stages come from `make-stage.py --plain-arms plain-rv64 --rdctr rdctr --size test` over `tree-plain`, with
  the test options of the Capstone test stages.
- Each one reaches the board only after its QEMU twin run passes (`plain-gate.py`, as for train).
- Its image hashes and twin results go into Addendum 3, before the first Plain test boot.

**Predictions.**
- **T1.** Spatial test against MapLife test, per benchmark, with `spatial-gate.py` (G1-G4):
  - every command ends rc 0;
  - outputs are equal (stderr excepted);
  - in-context instret is within 0.1 %.
  Exceptions:
  - x264's host validation is NO-OUTPUT by design (the board stage has no imagevalidate), so its G2 reads by
    output-hash equality;
  - perlbench's failing command is set off by allocation history (Addendum 16 of the test run), so its output and
    instret may differ from MapLife's, and G2 is predicted MISS for it, as in both Capstone arms.
- **T2.** MapLife / Spatial elapsed:
  - lbm and nab in [1.000, 1.02];
  - every other benchmark in [1.00, 1.40].
- **T3.** Full / Spatial in [1.00, 1.50], and Spatial / Plain in [1.1, 3.5], as S4.
- **T4.** Plain test passes the twin and the board with rc 0 and specdiff PASS for all seven benchmarks, perlbench
  included: Plain does not use the Capstone heap.

**Board time and the cost to train.**
- Spatial test is about MapLife test's 36 h: board 2, from the pilot gate (about 16:00 on 2 November) to about 04:00
  on 4 November. lbm, nab, imagick, xz and mcf land by about 01:00 on 3 November; perlbench and x264 last.
- Plain test is about 0.5-0.7 of MapLife's, 18-25 h: board 1, about 00:30 on 3 November to about 20:00-01:00.
- Both push the remaining train stages back by the same amount, about 1.5 days on board 2 and about 1 day on board 1.

## Addendum 3 (2026-11-02 15:30): the Plain test stages, their twin run, and what is measured for the overhead split

Written and pushed before the first Plain test stage boots (board 1, after perlbench Full, about 00:30 on 3 November).

**Stages** (`spec/stages-test-plain`):
- `runcpu --fake --action=run --size=test` in `tree-plain`. The Plain executables' sha256 is unchanged by it.
- Then `make-stage.py --arms plain-rv64 --plain-arms plain-rv64 --rdctr rdctr --size test`, with the Capstone test
  stages' options: perlbench `--strip --native-harness --keep-lib perl-test-lib.txt`, x264 `--ldecod
  --no-target-validation`.
- Their plan.json commands, generation, host validations, outputs, exclusions, inputs and ldecod equal the Capstone
  test stages' field for field. The exceptions are the staged name of imagick's reference image and the stub path in
  perlbench's pack.t record, which name the stage directory.
- Executables are the Plain train images: lbm bd72cb22753b, nab d0c47f5c4a14, imagick f7f3e78a5a49 (imagevalidate
  47e170a94b81), xz b889d0f57fcd, mcf 2e1b522831e9, x264 5adf7cac02e0. perlbench is bf1813c35f7f: the train image
  stripped, as the Capstone test images are.

**Twin** (`census-plain-test`, kit-q34, 936M), 15:01-15:13: every command ends rc 0.
- lbm, nab, imagick, xz, mcf and perlbench pass specdiff (perlbench 2/2, unlike both Capstone arms) and pass
  `plain-gate.py`.
- x264 is HELD by the gate: its three host validations are NO-OUTPUT. This is by design, as in both Capstone arms:
  the board stage has no imagevalidate.
  - The dumped frames (frame_{0,50,100,150,155}.yuv), BuckBunny.264/.yuv and dataDec.txt are hash-equal to the
    Capstone arms'. x264 Plain goes on the board by that reading, the one the Capstone x264 test was read by.
  - **Prediction miss (P3, x264).** The encoded BuckBunny_New.264 differs from the Capstone arms' (eddbb3318627 vs
    0e453c440e68). Every Capstone build gives 0e453c440e68: MapLife and Full on the board, the q34 twin, and the
    fixed-heap twin.
  - Equal dumped frames do not show that the encoding work is equal: frames that were not dumped can differ.
  - Not yet explained. The candidates are the stream's options SEI and floating-point codegen in rate control.
    Until it is settled, x264's Spatial / Plain and Full / Plain carry this caveat.

**Board order** (board 1, bitstream 3, `b1-resume-oom.list` after perlbench Full): lbm, nab, imagick, xz, mcf,
perlbench, x264.

**Prediction T5.** Board outputs equal the Plain twin's for perlbench, mcf, xz and x264, and specdiff passes for lbm,
nab and imagick. Plain elapsed is below MapLife's test elapsed for every benchmark.

**What is measured for the overhead split.** The board harness is unchanged. Changing it now would break the
comparison with the MapLife and Full test runs already recorded and with the running train stages.

Per command, the board records:
- elapsed time;
- in the Capstone arms: in-context cycles and instret, context entries, peak pages, lifetime nodes and collections;
- in Plain: hart cycles and instret, which include the kernel.

The split:
- **Elapsed ratio = instruction ratio × CPI ratio, plus time outside the context.** Time outside the context is elapsed
  × 25 MHz minus in-context cycles: kernel, launcher and escapes.
- **Spatial against MapLife:** the same images, so instret is equal within 0.1 % (G4), and the cycle difference is the
  hardware check alone.
- **Full against Spatial:** the instret difference is the lifetime runtime in software. The cycles beyond it are the
  check plus memory and kernel effects, read from entries, peak pages and collections.

Offline on the QEMU twin, with the same images and no board time:
- the instruction mix by class (capability loads and stores, plain loads and stores, capability operations);
- the share of malloc, free and the runtime;
- an L1 cache simulation;
- Plain's user/kernel instruction split.

The twin profile describes the board run only where its instret matches the board's.

## Addendum 4 (2026-11-02 18:45): imagick test repeated ten times per configuration

**Why.** One imagick test run is too short to weigh. Its main command runs about 16 s in context and 33 s elapsed, and
about half of a command's elapsed time is launch and I/O. A single reading cannot separate a 4 % configuration
difference from run-to-run noise. The lead asked for repetitions, with a mean, a spread and an interval.

**What runs.** `make-stage.py --repeat N` runs a stage's arms N times in one boot, interleaved: r0 of every arm, then
r1, and so on. Each run is its own arm in the result, labelled `<arm>.r<k>`, with the same executables. The images are
the single-run test stages', by hash: imagick_r c2fa368c2b62 (capstone-spatial) and a7ddf8d556c6 (capstone-lifetimes),
and the Plain f7f3e78a5a49.

| Stage (N = 10) | Board, bitstream | Arms | Slot in the queue |
|---|---|---|---|
| `538.imagick_r-test-r10-spatial-objb` | 2, ObjBounds 603528b8b | Spatial | after xz Spatial test, before mcf |
| `538.imagick_r-test-r10-maplife-full` | 1, bitstream 3 | MapLife and Full, alternating | after perlbench Full train |
| `538.imagick_r-test-r10-plain` | 1, bitstream 3 | Plain | after the MapLife/Full stage, before Plain test |

**Generator unchanged at N = 1.**
- With `--repeat 1`, make-stage.py output is byte-identical to the current generator's for the one-arm Spatial, Plain
  and two-arm imagick test stages.
- The reader change in run-stage.py touches only the twin lookup: a label is compared with the twin's same label,
  or else with its arm.
- With the new reader, the result.json and verdict lines are identical to the old reader's on the recorded nab Spatial,
  imagick MapLife/Full and xz Full test results.

**Dry run on the QEMU twin** (census-repeat, kit-q34, 936M):
- All 40 runs: rc 0, specdiff PASS. Outputs are identical across repetitions and to the single-run twin.
- Run-command instret spread across repetitions is at most 1.1e-5 in each Capstone arm.
- Plain's hart-wide instret spreads 7.9 %. It includes the kernel, so it is not a determinism measure.
- **Positive control.** One repetition's test_output.tga hash is corrupted in a copy of the transcript. The reader
  reports `r0 vs r4: DIFFER` and `r4 vs twin: DIFFER`, rc 1.

**Predictions.**
- **R1.** Every run returns rc 0 and specdiff PASS. Outputs are identical across repetitions and to the twin.
- **R2.** Within each Capstone arm, run-command instret is within 0.1 %. Spatial and MapLife instret are within 0.1 %:
  same images.
- **R3.** Per arm, the run command's coefficient of variation is at most 3 % in elapsed time and at most 1 % in
  in-context cycles.
- **R4.** Run-command in-context cycles: MapLife / Spatial in [1.00, 1.10], Full / MapLife in [0.95, 1.05]. Run-command
  elapsed: Spatial / Plain in [1.1, 3.5].

**Statistics.** Per arm and command (run, val): mean, standard deviation, CV, a 95 % t-interval (t = 2.262, 9 degrees of
freedom), and box plots with 1.5 IQR whiskers. Metrics: elapsed time, in-context cycles, instret, entries; Plain's hart
cycles.

Ratios:
- MapLife / Full alternate in one boot, so their ratio is paired, per repetition. The paired ratio's mean and its
  t-interval are reported.
- The ratios across boots and boards (MapLife / Spatial, Spatial / Plain) are ratios of means. Their 95 % interval
  comes from a bootstrap: 10,000 resamples per arm, drawn independently.

**Caveat.** Spatial runs on board 2 and the other arms run on board 1. A ratio across the boards therefore carries a
board term that this design cannot remove. Each boot's crc.K0 control step is read beside the results as a
cross-board reference; it is not a gate.
