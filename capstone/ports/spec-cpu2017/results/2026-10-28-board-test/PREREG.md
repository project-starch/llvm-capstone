# SPEC CPU2017 test size on silicon, T1 (capstone-spatial) and T3 (capstone-lifetimes): pre-registration

Written 2026-10-28, before any SPEC boot. Pushed to the lane branch `spec-silicon` before the first boot; its commit
is cited by every result that follows.

## What runs

- **Bitstream 3**, `caplifive_vp2c_e7af48457`, vp2 boards 1 and 2. The kit is the perf evaluation's
  `bkit12-perf-b3-b<board>`: launcher `978c09d61f774315`, module `9c60fb53f80ce67e` (bench counters). spec-board.sh
  checks every one of these by hash before a booking.
- **Arms.** T1 = `capstone-spatial` (heap lifetimes off, scrub off). T3 = `capstone-lifetimes` (lifetimes on, scrub
  off: the default since PR #290). SDKs `perf/sdk-T1` and `perf/sdk-T3`, the bitstream-3 BEEBS SDKs (dev
  4f054964a7d8, compiler cc-prmerged). Both are built with `-movc-keeps-integer-source` (the feature OFF; C-32).
- **Images** (sha256, first 16), one hash per image across every stage, twin and board:

  | image | T1 | T3 |
  |---|---|---|
  | perlbench_r | 2c073498b100b0ca | b05dcf06469c2024 |
  | cpugcc_r (stripped) | 32eee4e451c97a8d | f8d2e7bcb3368bf7 |
  | mcf_r | 8fbf5fe40f1fe8e2 | 9b7507f16a950592 |
  | lbm_r | 943c0a8eeec5e6ee | 7f68a236dd636865 |
  | x264_r | 4f8f79dc83b125aa | 1a9a868754718e32 |
  | imagick_r | c2fa368c2b620dcf | a7ddf8d556c644d6 |
  | imagevalidate (525 and 538 alike) | 1f7ed1fa8247eade | cc25def6222c0793 |
  | nab_r | f4c2ed7debb4f59c | 38c821a59539869c |
  | xz_r | 961dd14555f0b041 | c2f39e5811e45ac2 |

- **One boot = the known-good control, then one stage.** The control is crc.K0 T1 (`bd909cc6eba4bf2d`, args 1 5),
  which passed on bitstream 3 in kit t3a2. A stage is make-stage.py's: runcpu's own test command lines, inputs baked
  into the image, every arm in a fresh copy of the inputs. gcc and perlbench go one arm per boot (payload), xz and mcf
  one arm per boot (time: both arms at CPI 8 would pass the 8-hour booking).
- **Twin reference:** the same stages on kit-q34, the twin (QEMU 46b69c03a4 + the Q-34 alignment trap, module and
  launcher of `perf/kit-twin-bench`) under `-icount shift=0`, 1 GiB, `cma=256M`. Its result lines are
  `twin-reference.json` beside this file (twin-reference.py). Every twin run: rc 0 on every command, specdiff PASS, T1
  and T3 outputs byte-identical, and no `domain fault` line (the Q-34 pin's positive control `misal.dom` faults with
  cause 4 on this kit).

## Twin counts (instructions inside the context, run commands only)

| benchmark | T1 instret | T3 instret | entries | peak pinned | nodes T1 / T3 |
|---|---|---|---|---|---|
| 557.xz_r | 50,821,695,905 | 50,821,712,690 | 1,697,626 | 555 MiB | 81 / 143 |
| 505.mcf_r | 35,301,316,585 | 35,302,835,720 | 393,974 | 681 MiB | 812 / 64,004 |
| 544.nab_r | 7,991,860,754 | 7,992,016,799 | 2,314 | 4 MiB | 504 / 7,194 |
| 519.lbm_r | 7,278,938,612 | 7,278,939,204 | 106,840 | 411 MiB | 34 / 40 |
| 538.imagick_r | 108,677,371 (+ validation) | 108,697,798 | 1,296 | 5 MiB | 205 / 1,195 |
| 502.gcc_r | 30,520,991 | 30,629,201 | 2,879 | 24 MiB | 1,199 / 5,993 |

x264 and perlbench are not registered here. perlbench's stage changed after its twin run (its `$^X` must be a relative
path the shell can run, since the board kernel has no binfmt_misc); both get an addendum, pushed before their boots,
once their twin runs are read.

## Predictions

- **P1. The control passes in every boot.** A boot whose control fails is VOID: it says nothing about SPEC.
- **P2. Every command ends with rc 0, and every host specdiff PASSes,** for lbm, imagick, gcc, nab, xz and mcf T1.
- **P3. T1 and T3 outputs are byte-identical** (.err files aside: they carry the launcher's counters).
- **P4. Every output hash equals the twin's.**
  - For lbm, imagick and nab, a difference with specdiff PASS is recorded as an FP divergence and read against
    R-64/R-65/R-66, which this bitstream carries. It is not a failure.
  - For the integer programs, any difference is a failure.
- **P5. No domain fault.** In particular no cause 4 or 6: the Q-34 twin ran every one of these clean.
- **P6. instret per command is within 0.1 % + 250 × (board entries) of the twin's.** A CSRUNV entry costs about 125
  instructions inside the window on silicon (perf probe, 2026-10-26). This is a characterisation: a miss is
  reported with its size, not read as a defect.
- **P7. CPI, read as bench cycles / instret on the board, is between 2.0 and 8.0 for every run command.** The time
  estimates use 2.7–3.75 (the BEEBS/RV8/CoreMark geomean and speedtest1). The measured value replaces them for train.
- **P8. T3/T1 of in-context cycles is within 0.97–1.05** for lbm, nab, imagick and xz, which allocate little. For
  gcc and mcf no band is registered: they are reported.
- **P9. mcf T3 does not finish.** Its twin run used 64,004 lifetime nodes in a table of 262,144 (the twin's QEMU
  table is 18-bit). The board's table is 16-bit, and its malloc path refuses at 64,001 records (measured 2026-10-22).
  This boot is not in the first lists: it runs after the twin re-run with `node_flat_records=65536` (kit-q34-16), and
  an addendum records that run's reading before the boot.
- **P10. mcf T1 fits in memory, with little margin:** 681 MiB pinned on the twin. A pin or OOM refusal would be a
  capacity finding, not a correctness one.

## Verdict

Per boot, `chain-board.sh` logs spec-board.sh's `SB`, `RUN` and `READ` lines. run-stage.py writes `result.json`
(commands, hashes, specdiff verdicts, `cross_arm`, `vs_twin`). It exits 1 on a failed command or check, and 2 on
missing data, an OOPS or OOM, or a VOID control. Tables come from `census-table.py` over the result directories.
Board results are cited by image hash. Only result lines are committed, never a transcript or a runcpu `.cmd` file.

## Boot lists (chain-board.sh)

Steps are sized from twin instret at CPI 8, both arms; bookings are the step plus 30–40 minutes.

- **Board 1:** imagick (1200 s), gcc T1 (900 s), gcc T3 (900 s), lbm (5400 s), nab (6000 s), xz T1 (16200 s).
- **Board 2:** mcf T1 (12000 s), xz T3 (16200 s).
- **Later, each behind its addendum:** x264, perlbench T1/T3, mcf T3.

Board 2 was held by another lane when this was written. Its chain waits for the board, polling every 120 s for up to
8 h.

## Dry runs (the board-run skill, rule 8)

- **spec-board.sh**, every branch, with stubs: ok, control failed, truncated transcript, busy, 409, and the guard
  against a half-stubbed run. One slip on 2026-10-28: a broken booking stub let a check-only run reach the real
  driver. The other lane's board lock refused it, so there was no boot. The guard was added afterwards.
- **chain-board.sh** on stubs:
  - busy wait;
  - busy refusal, which retries;
  - 409, which waits and stops at its limit (this branch spun without sleeping until fixed);
  - a failed benchmark, after which the chain continues;
  - a VOID control, which stops the chain;
  - `--twin` passed through.
- **The twin comparison fires:** a forged lbm.out hash reads DIFFER.

## Addendum 1 (2026-10-28, before any mcf boot): P9 RETRACTED

P9 rested on a 16-bit flat table. That is the table of vp2 and the batch bitstream, not bitstream 3's.

- The kit's module 9c60fb53 is the 18-bit build. It carries the `index_bits` parameter, which the source exposes only
  when the width is not 16 (`runtime/virtual/module/capstone_vm.c:180-203`).
- The board's own stats line in the imagick boot reads `node_capacity=262144`.
- Bitstream 3 is built on bitstream 2, whose index is 18 bits.
- Bitstream 2 pushes a retired record onto the table's free list. So the bound is likely the LIVE set, not the
  lifetime allocation count. For bitstream 3 this is inferred from its base, not measured.
- The `nodes=` field of the stats line is the context's allocation total (`table[3]`). mcf T3's 64,004 is therefore
  an allocation count, while native mcf test holds at most 5,995 objects live.

**New P9: mcf T3 completes on test**, with the same outputs as T1 and the twin. mcf T3 joins board 2's list after mcf
T1, with step 12000 s and twin census-q34/505.mcf_r-test.

kit-q34-16, which set `node_flat_records=65536` through sysfs, cannot test the old P9 anyway: the parameter is
read-only after insmod (0444), so the batch refuses it.
