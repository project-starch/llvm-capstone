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
