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
