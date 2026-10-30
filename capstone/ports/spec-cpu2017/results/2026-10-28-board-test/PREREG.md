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

## Addendum 2 (2026-10-28, before any boot it names): x264 test, and the first train wave

**Infrastructure changed for these boots.** Neither change touches the board, the kit or the images.
- The booking hub accepts bookings of up to 24 h (fpga-lab f0e0654; the old limit was 8 h). A train command that
  cannot be split runs longer than 8 h.
- Both consoles' upload limits go to 256 MiB (fpga1 B15, fpga2 B20).
- The project's 26,000,000-byte bake cap is unchanged, and every stage below fits under it.

**Correctness oracles.**
- Unmodified train commands use host specdiff against SPEC's own train references. The outputs come back in-band,
  gzip+base64 over the UART; gcc's `.s` files go in-band with `--inband-max 8388608`.
- imagick and x264 train also run imagevalidate on the target against the staged references.
- x264 test, one arm per boot, is staged without its references, which would push it over the cap. Its verdict is
  the frame hashes against the twin's `census-q34/525.x264_r-test`, plus T1 = T3. The run reads host specdiff 0/3
  there by construction.

**Boot list.** One arm per boot, the known-good control first. Steps are sized from the native-ratio estimate at
CPI 4.5, which is above the 3.6-3.7 the board read on imagick test, capped below the 24 h booking.

| boot | step s | booking min | estimate at CPI 3.75 |
|---|---|---|---|
| 538.imagick_r train T1 / T3 | 75600 | 1290 | 16.5 h |
| 519.lbm_r train T1 / T3 | 21600 | 390 | 4.5 h |
| 525.x264_r train T1 / T3 | 50400 | 870 | 10.3 h |
| 525.x264_r test T1 / T3 | 61200 | 1050 | 12.8 h |
| 505.mcf_r train T1 / T3 | 36000 | 630 | 7.4 h |
| 502.gcc_r train T1 | 46800 | 810 | 9.6 h |
| 544.nab_r train T1 / T3 | 82800 | 1410 | 20.2 h |

nab is last on each board: at CPI 4.27 or above, its 439e9-instruction command outlives the booking.

**Predictions, train wave:**
- **P11.** Every command ends with rc 0, every specdiff and imagevalidate PASSes, and T1 and T3 outputs are
  byte-identical.
- **P12.** The board time per arm falls within ±35 % of the CPI-3.75 estimate above. The estimate is native x86
  train/test instruction ratio × the twin's exact test count, good to tens of percent.
- **P13.** mcf train fits in memory: native peak 613 MiB, and mcf test pinned 681 MiB on the twin.
- **P14.** gcc train T1 completes. Its native peak is 693 MiB for 200.c, the largest of the three commands. A pin or
  OOM failure there is a capacity reading, not a correctness one.
- **P15.** T3/T1 of in-context cycles is within 0.97-1.05 for lbm, imagick, nab and x264.

Not in this wave:
- **xz train:** 24 MB of input exceeds the bake cap.
- **perlbench train:** waits for the twin check of the lib-trimmed stage and its negative control.
- **gcc train T3:** 816k live objects against 262,144 records.
- **perlbench test T1/T3:** waits for the twin re-run of the relative-`$^X` stage.

## Addendum 3 (2026-10-28 21:20, before any train boot): steps resized to CPI 7

lbm test read CPI 5.07 on the board, against the 2.7-3.75 the estimates assumed. At that CPI lbm train needs 6.1 h,
and its 21600 s step would have cut it off. Every train step and the x264 test steps are now sized at CPI 7, capped
at 84600 s (23.5 h, bookings 1440 min):

| boot | step s | booking min |
|---|---|---|
| lbm train | 32400 | 570 |
| mcf train | 54000 | 930 |
| x264 train | 72000 | 1230 |
| x264 test | 84600 | 1440 |
| gcc train T1 | 68400 | 1170 |
| imagick train | 84600 | 1440 |
| nab train | 84600 | 1440 |

At the cap, imagick fits up to CPI 5.35 and nab up to CPI 4.37. P12 (±35 % of the CPI-3.75 estimate) stands as
registered: lbm's CPI already predicts that it misses for lbm.

## Addendum 4 (2026-10-28 23:20, before the reruns it names): mcf test outlived its step

**What happened.** mcf test T1 (board 2, image 8fbf5fe40f1fe8e2) was cut at its 12000 s step, still inside its only
command. The known-good control passed, so the boot is valid. It carries no SPEC verdict.

**What the run's own diagnostics show.** The module's stall line, `@@STALL spec mod 2 sfences=260748
satp=372469/0 busy=0`, shows that it was still taking escapes when it was cut. Each escape is a satp write, which
flushes the TLB. On lbm, wall time beyond the in-context cycles came to about 5.2 ms per CSRUNV entry, and the twin
counted 394k entries for mcf test.

**What changes.**
- mcf test T1 is rerun with step 30000 s; mcf test T3 is raised to 30000 s before its boot.
- mcf train goes to the 84600 s cap.
- The bounded run's directory is kept as `505.mcf_r-test-capstone-spatial.bound12000`.

**P16.** mcf test completes within 30000 s per arm. Its outputs equal the twin's, and T1 equals T3.

## Addendum 5 (2026-10-28 23:35, before any x264 boot): x264's count, and log.dec

**x264 test on the twin.**
- Both arms: rc 0, imagevalidate specdiff 3/3 PASS, no domain fault.
- The run command is **543,659,713,729 instructions** per arm.
- That is 1.95 × the native x86 count of 278e9. The native-ratio estimate used the median of 1.1, so x264's estimates
  in Addendum 2 were low by about 1.8×. x264 is likely the program x86's vector units shorten most; that is a
  reading, not a measurement.
- Rescaled with x264's own ratio, x264 train is about 439e9 instructions.

**New steps.** The booking hub now accepts up to 48 h (fpga-lab 63e78a0). The steps are:

| boot | step s | covers up to CPI | booking min |
|---|---|---|---|
| x264 test | 129600 | 5.9 | 2190 |
| x264 train | 100800 | 5.7 | 1710 |

**log.dec.** T1 and T3 differed in `log.dec` on the twin. That file is the log of x264's input generator, ldecod,
which runs outside the context. ldecod stamps it with the date and time (`ldecod.c:546-549`), so it differs on every
run. run-stage.py now leaves `log.dec` out of the cross-arm and twin comparisons, as it does `.err`. Re-reading the
twin's x264 transcript gives T1 = T3 identical.

**P17.** x264 test completes on the board within 129600 s per arm. Its frame and output hashes equal the twin's, and
T1 equals T3.

## Addendum 6 (2026-10-29 05:50, before gcc train or any perlbench boot): a heap-lock trap in realloc, seen on the twin

**gcc train T1 on the twin (kit-q34)** (`census-train/stages-trainT1-502.gcc_r-train`):
- train01.c completes, with specdiff PASS.
- 200.c and scilab.c end with rc 139 after 9.86e9 and 5.92e9 instructions.

**perlbench test on the twin, with the native harness:**
- 315 of 318 sub-tests pass.
- Failing in one run: japh/abigail, re/reg_email, re/pat_rt_report.
- In a later run, re/reg_email passes. (That run's lib/deprecate failure was a gap in the trimmed lib list, since fixed.)

**The fault is the same for every one of them:**
- `capstone-exec: domain fault cause=2 ... address=0xc0001073`, which is `unimp`.
- The pc is the single `unimp` inside `realloc` (gcc .text+0x10f94, perl .text+0x11eb0). It is the `__builtin_trap()` of
  `lock_heap` (`runtime/virtual/heap.c:39`).
- The disassembly is `amoswap.w.aq a0,s5,(s4)`; `beqz a0,taken`; `jalr __capstone_vm_wait`; `beqz a0,retry`; `unimp`.
- No other branch reaches that `unimp`.

**Reading.** realloc found the heap lock already held, in a single-threaded program. Two parts of this are not
established:
- why the lock is held;
- why the wait path traps. The wait service sets no result, so a0 may still carry the amoswap's 1 (exec.c:876-882,
  module 2010-2011).

The SPEC kit's own QEMU qualification, on the older runtime, passed gcc train and perlbench test. A runtime regression
is therefore the leading reading. It is UNRESOLVED and is not yet an ISSUES entry.

**Predictions:**
- **P18.** gcc train T1 on the board: train01 completes with specdiff PASS. 200.c and scilab.c end with rc 139 and the
  same `cause=2 ... 0xc0001073` fault at .text+0x10f94.
- **P19.** perlbench test on the board, with the native harness and the trimmed lib:
  - makerand passes;
  - at most japh/abigail, re/reg_email and re/pat_rt_report fail, each with this fault;
  - every other sub-test reads `ok`.

The perlbench test boots are added to the lists after their twin references (census-perl) are read.

## Addendum 7 (2026-10-29 06:40): RETRACTION of Addendum 6's localization

Addendum 6 attributed the `cause=2` trap at .text+0x11eb0 (perl) / +0x10f94 (gcc) to `lock_heap`'s `__builtin_trap()`
(heap.c:39). **That is wrong.** The two traps share one `unimp`.

**What the trace shows.** An exec trace of japh/abigail.t on the twin (kit-q34-trace: every translated block in the
domain range, 70.2M blocks) shows the following:
- `unimp` was reached once, at trace line 81477757.
- The lock-wait path (`jalr` at +0x11ea8, `beqz` at +0x11eac) never ran.
- The trap was reached from `bne a5, a3, <unimp>` at ELF 0x2239c, in realloc's inlined `publish()` →
  `issue_child()`. That branch compares the base of the child it just shrank (`lcc a5, a0, 3`) with the requested
  start (`ld a3, 0x40(r)`, the slot key).

**The check that fires** is issue_child's exactness check in the spatial arm (heap.c:207-210):
`__builtin_capstone_cap_shrink(at, start, start + size)` followed by
`if (get_base(child) != start || get_end(child) != start + size) __builtin_trap();`.

So realloc published a block whose `[key, key + capstone_repr_round(n))` is not exactly representable at that key: the
SHRINK rounded the base.

**Not established:**
- which mallocng realloc path returned the under-aligned key (in-place growth inside the slot's slack, or the
  mremap path);
- what the lifetimes arm does there. It takes the `capstone_cap_derive` path instead of SHRINK, and CDERIVE refuses
  inexact bounds with cause 29.

**What changes.** P18 and P19 keep their observable predictions: the same rc 139 and the same fault line at the same
pc. Only Addendum 6's reading of the cause is withdrawn. How it was missed: the first localization took the
`__builtin_trap` nearest the `unimp` in the disassembly, and never asked which branches target it.

## Addendum 8 (2026-10-29 11:00, before any perlbench boot): perlbench references, the realloc mechanism, list order

**perlbench test, twin references** (kit-q34, native harness, trimmed lib; `census-perl/500.perlbench_r-test-<arm>`):

| arm | image (stripped) | makerand | test.pl | instret (makerand, test) | entries |
|---|---|---|---|---|---|
| T1 | 6b1a30a5b1e71f58 | specdiff PASS | 2 of 316 fail | 4.654e9, 115.966e9 | 1,063,661 |
| T3 | 371553e3b5adffa7 | specdiff PASS | 2 of 316 fail | 4.654e9, 116.460e9 | 1,136,493 |

The two failures are t/japh/abigail and t/re/pat_rt_report in both arms. Each ends in one fault: T1
`cause=2` at code offset 0x11eb0, T3 `cause=29` at code offset 0x124c4. re/reg_email passes in both.

**perlbench train T1, twin** (lib-trimmed stage, image 2c073498b100b0ca): all 5 commands rc 0, 300.46e9
instructions, 380,232 entries. All six output files' sha256 equal SPEC's reference files. The negative control
(lib/Date/Format.pm removed) fails the two commands that load it, rc 2. The reader's "OOPS/OOM" line on that run
matched base64 and is fixed (b46e45a00ca5).

**The realloc mechanism, from a directed probe on the twin** (`realloc-probe/RESULT.md`): this answers Addendum 7's
first open line.
- realloc keeps an object's offset in two routes: the in-place branch, and the no-op mremap of a sizeclass-63
  mapping.
- In both routes it publishes `[p, p + repr_round(n))`. But enframe aligned p only to the granule of the original
  size.
- The pairs 32720→32768 and 261888→262144 fault on exactly the rounds whose base is not a multiple of the new
  granule: T1 with cause=2, T3 with cause=29. The aligned rounds and the moved control pass.
- Which route the SPEC traps take is not established.

**Predictions:**
- **P20, perlbench test on the board.** For each arm:
  - makerand: specdiff PASS;
  - test.pl: exactly japh/abigail and re/pat_rt_report fail, each with the twin's fault (cause and code offset as
    above);
  - every output's hash equals the twin's;
  - instret within +0.3 % of the twin.
  
  This sharpens P19 now that the twin is pinned.
- **P21, perlbench train T1 on the board.**
  - All 5 commands rc 0.
  - All six outputs' sha256 equal SPEC's references. splitmail.out (26 MB) is judged by hash alone.
  - instret within +0.3 % of 300.46e9.

**Steps, at CPI 7 plus 15 ms per entry** (Addendum 3's rule):

| boot | step | booking |
|---|---|---|
| perlbench test (either arm) | 50400 s | 870 min |
| perlbench train T1 | 93600 s | 1590 min |

**List order.** Test boots now precede the remaining train boots. Each list's tail is rewritten after the line its chain
has already read (offsets 868 and 566, checked in /proc), so the running boots are untouched.
- **Board 1**, after imagick train T1:
  1. perlbench test T1;
  2. x264 test T1;
  3. perlbench train T1;
  4. lbm train T1;
  5. x264 train T1;
  6. gcc train T1;
  7. nab train T1.
- **Board 2**, after mcf test T1:
  1. perlbench test T3;
  2. x264 test T3;
  3. imagick train T3;
  4. lbm train T3;
  5. x264 train T3;
  6. mcf train T1;
  7. mcf train T3;
  8. nab train T3.
- perlbench train T3 waits for its twin run, which started at 10:45.

## Addendum 9 (2026-10-29 12:15, before the boot it names): perlbench train T3

**Twin** (kit-q34, lib-trimmed stage, image b05dcf06469c2024):
- diffmail, perfect, suns, validate: rc 0, specdiff PASS.
- scrabbl (rc 1) and splitmail (rc 1) print perl's `Out of memory!` with only 19,047 and 27,114 pages pinned.
- **Classified by the table, not the error.** Both stats lines read `node_capacity=262144 node_growths=0`. scrabbl
  stops at nodes=261,089, which is the lifetime table, not RAM.
- Total: 170.76e9 instructions.

**P22.** perlbench train T3 on the board:
- diffmail, perfect, suns and validate give rc 0, with outputs sha256-equal to the twin's;
- scrabbl and splitmail end with rc 1 and `Out of memory!` at the table's capacity.

**Step:** 54000 s at CPI 7 plus 15 ms per entry, booking 930 min. It is appended to board 2's list, after nab
train T3.

## Addendum 10 (2026-10-29 15:00): the two boards split the work, nothing is repeated

- **mcf train T1** moves from board 2 to board 1, before nab train T1. The two lists then need about 95 h and 91 h of
  expected board time, so both finish around 2 November.
- **No boot repeats a run.**
  - Where a runtime fix (the realloc defect) produces new images before a queued boot starts, that boot's line is
    swapped to the new stage. It is pre-registered here before it runs, and the run carries the new image hash.
  - A finished run on the old images stays the record for those images and is not rerun.

## Addendum 11 (2026-10-30 00:55, before any xz train boot): xz train on the board, and perlbench test T3's reading

**perlbench test T3, board 2 (recorded here because it bears on P20).**
- makerand: specdiff PASS.
- test.pl: three sub-tests fail: japh/abigail, re/pat_rt_report and re/reg_email. P20 predicted exactly the first two.
  - re/reg_email is the extra one. Its sub-run stopped at 0.753e9 instructions against the twin's 5.813e9. No other
    sub-run differs from the twin's instret by more than 2.5 %.
  - The same sub-test failed in one of the earlier twin runs (Addendum 6, before the twin was pinned), with the
    realloc fault.
  - The board console does not carry a sub-run's fault line, so this boot cannot show which fault ended it.
- **P20 does not hold for T3.** The reading is: a nondeterministic member of the known realloc-fault set,
  unconfirmed on the board. Its cause is not established by this boot.

**xz train T1, twin** (kit-q34, `--link-inputs` stage, 936 MiB guest = 963 − 24 MiB of inputs − 3 MiB margin):
- both commands rc 0, specdiff 2/2 PASS;
- peak 222,921 pages;
- 112.65e9 + 50.97e9 instructions.

A T3 twin run at 936 MiB is queued (`census-train/557.xz_r-train-capstone-lifetimes-link-mem936M`). The T3 board boot
goes on the list only after it reads PASS.

**Firmware size.** The stage is 24,427,799 B raw and its inputs do not compress. The firmware is about 2 MiB plus the
Image, and the Image is about 4.9 MB plus the raw stage, so this boot needs about 31.5 MB. That is above the
26,000,000 B cap in the perf kits' board.env. Two bounds set the cap for this one benchmark:
- **Layout.** OpenSBI places the payload at 0x80200000 and the FDT at 0x82200000, and nothing is reserved in
  between. The kernel's footprint, from the Image header's image_size, is the file size plus 357,888 B. A cap of
  33,000,000 B keeps that footprint below 31.27 MB, under the 32 MiB to the FDT.
- **Load.** The JTAG load runs at about 131 KB/s, measured over 12 loads, so 31.5 MB takes about 240 s against the
  driver's 300 s load budget.

The kits are copies, `t3-smoke-fw33/bkit12-perf-b3-b{1,2}`. Each differs from its original only in
`BOARD_FW_CAP=33000000`, with SHA256SUMS regenerated for board.env. They are used only for the xz train lines, via
SB_KITROOT.

**P23. xz train on the board, per arm** (T1 image 961dd14555f0b041, T3 c2f39e5811e45ac2):
- the firmware is below 33,000,000 B and loads within the 300 s budget;
- both commands rc 0, specdiff 2/2 PASS, and output hashes equal to the twin's;
- no OOM kill;
- instret within +0.3 % of the twin's.

If input.combined is OOM-killed (rc 137), the fallback is the reduced first command. That is pre-registered
separately before any further boot.

**Step and booking.** CPI on xz test was 2.8 to 3.3 on the board, so 163.6e9 instructions take about 6 h. The step
is 32,400 s and the booking 570 min.

**Placement.** One successor chain per board runs after that board's continuation list ends, with SB_KITROOT set to
the fw33 kits: T1 on board 1, and T3 on board 2 after its twin PASS.

## Addendum 12 (2026-10-30 02:25): perlbench test T1 on board 1, and a RETRACTION of Addendum 11's reading of re/reg_email

**perlbench test T1, board 1.**
- makerand: specdiff PASS.
- test.pl fails three sub-tests: japh/abigail, re/pat_rt_report and re/reg_email.
- The only difference from the twin's test.out is re/reg_email.
- Its sub-run stops at 0.7523e9 instructions; the twin's runs 5.813e9. T3 on board 2 stopped at 0.7527e9.
- Every other sub-run except one is within 5 % of the twin. That one, index 4, is +10 %, as it was in T3.

**Withdrawn.** Addendum 11 read T3's re/reg_email failure as "a nondeterministic member of the known realloc-fault
set". Both arms, on two boards, stop at the same point, within 0.05 %. The pinned twin passes the sub-test in both
arms. So this is a **deterministic silicon-vs-twin divergence**, and nothing yet connects it to the realloc defect. The
reading rested on a single pre-pin twin run, and it was written before the second arm could test it.

**P20 does not hold in either arm.** The cause is unresolved. The next step is off the board:
- run re/reg_email alone on the twin with the silicon-mimic switches;
- then capture the sub-run's fault line in a directed boot.

## Addendum 13 (2026-10-30 05:40, before the boot it names): a directed boot for re/reg_email

**Stage.** `spec/stages-directed/500.perlbench_r-regemail-both`. Its run.sh is hand-written; make-stage.py does not
produce it.
- Inputs and images are the perlbench test stages' (T1 6b1a30a5b1e71f58, T3 371553e3b5adffa7). The inputs are
  identical across the arms.
- Per arm it runs japh/abigail.t and then re/reg_email.t alone, exactly as t/TEST runs them: cwd t/,
  `../../sw/perl -I.. -MTestInit <test>`.
- It prints:
  - each sub-run's rc and bench line;
  - its stderr lines matching capstone-exec / capstone_vm / Segmentation / Killed / panic / Out of memory /
    recursion;
  - the last 40 lines of its stderr, its stdout tail, and dmesg.

**Twin reference** (kit-q34, `census-directed/500.perlbench_r-regemail-both`). The first run's capture lines were
void, because a sed delimiter broke on the `t/` names. The control showed this, and the rerun is the reference:
- T1 abigail: rc 139, `domain fault cause=2` at code+0x11eb0, address 0xc0001073.
- T3 abigail: rc 139, `domain fault cause=29` at code+0x124c4.
- reg_email, both arms: rc 0, 13/13 ok, 5.813e9 instructions.

**P24, on board 1 (bitstream 3):**
- **Control.** abigail reads the twin's cause and code offset in both arms. If it does not, the capture is broken
  and the boot says nothing about reg_email.
- **reg_email.** It ends after about 0.752e9 instructions, as in both full runs, and is NOT `ok` 13/13. Its stderr
  names the end: a domain fault line (cause, pc offset, address), a signal, or a perl error. That line is the
  reading. No cause is predicted; this boot exists to observe it.

**Placement.** The line goes on board 1's continuation list, right after the x264 test T1 boot that is running now,
with step 3600 s and booking 90 min. Board 1 stays on bitstream 3.

## Addendum 14 (2026-10-30 08:45, before the flash and the boots it names): bitstream 4 (B) on board 2

**The lead's word.** In this lane's session the lead approved the reflash ("Даю добро на перепрошивку, как будет
готово") and ruled that WNS is not an acceptance criterion ("Wns не важен").

**The bitstream.**
- `caplifive_vp2c_325966ee4.bit` is capstone-ariane 325966ee4, option B: 55-bit lifetime ids with a 25-bit slot.
- sha256 `f0893218dcdb53f4ad0b9fe452571f9e5c3bc5fa7ffade0025657c662c4da8cc`, 11,443,722 B. The hash was checked again
  after the copy to this host.
- The rtl lane graded it PASS on every pre-registered key (dev b0046d1c5b84). Routed WNS is -11.964; it was reported
  and not graded.
- `ariane_pkg.sv` diff e7af48457..325966ee4: no memory constant changes; the diff touches only types and comments.
  So the firmware recipe and its DTB stand, with 256 MiB of reusable CMA at 0xac000000. A board boot on bitstream 3
  prints `created CMA memory pool at 0x00000000ac000000, size 256 MiB`.
- Board 1 stays on bitstream 3. Bitstream 3 must still be registered on board 2's console as the fallback.

**The queue change, made before this addendum.**
- Board 2's list was cut in place at its chain's read offset, so its chain ends after the x264 test T3 boot that is
  running now.
- Its six remaining bitstream-3 T3 train lines (imagick, lbm, x264, mcf, nab, perlbench-T1-split) and xz train T3 now
  sit at the end of board 1's xz list. A stubbed dry run read all 8 lines on kit bkit12-perf-b3-b1, rc 0.
- So every bitstream-3 train boot of both arms runs on board 1 and on the same bitstream. Nothing is repeated
  (Addendum 10).

**The kit.**
- `bkit13-b4-b2` is `make-b4-kit.sh`'s copy of bkit11-b3-b2 with exactly three changes, checked by hash:
  - board.env names `caplifive_vp2c_325966ee4`;
  - `adapter/module/capstone_vm.ko` is the flat ids55 board module, 6422805f49533ea5 (module-ids55 3523f8833531,
    named in that branch's PREREG addendum 2 at e08dfaa95f22). It has 0 of 5,268 instructions in 16-bit form, read
    here; the same count reads 572 of 792 on a kernel object, so the count can fire;
  - `adapter/capstone-vexec` is the twin grid's launcher, 62753dd693cc4a25.
- The b0 controls' /capstone.ko (BAKE_REPLACE_KO) is unchanged.

**The procedure.** `b4/b4-first.sh` (0e8d766a954ce205) runs these steps, each gating the next:
1. Wait for board 2's chain to end and for the board to be free.
2. Check the console's bitstream store: the negative control (a missing name reads 1), then that bitstream 3 is
   registered.
3. Check the local hash.
4. Flash with `flash-bitstream.py` (book, flash, mandatory power cycle, name read back on the same session).
5. Read the resident record back on a fresh session (`nv-readback.py`: name, and sha256 if the console exposes it).
6. Make the kit.
7. Boot 1, then boot 2 only if boot 1 reads as predicted.

**Boot 1** (stage `b4-t1a`, run.sh 5d9a56a00d93ce64, judge `b4-judge-t1a.py` 27a13f080fd30591). In order:
- the kit's board controls b0-stats and b0-hello;
- the module's parameters and load lines;
- Tier 1 contract: contract.dom 00428f273c653d9f, `ok` then `stale`;
- the twin grid's live smoke (live-T3 1eb994b5fb979862) at N = 1000, 400000 and 1100000, with 1100000 LAST;
- the ring buffer's CMA and allocation lines.

Predictions are module-ids55 PREREG section 2 (1 and 3) as amended by its addendum 2, read against the twin's F0:
- **Load and allocation.**
  - The parameters read ids=55, index_bits=25, node_flat_records=1048576.
  - Exactly one allocation line: `flat table 1048576 records, 16777216 bytes, from CMA, pa X, linear mapping`. X lies
    inside the CMA region the boot's own reserved-mem line names.
  - There is no `ALLOCATION FAILED` line.
  - **NON-LINEAR on the board contradicts the 6.4.14 source and is a STOP.**
- **Contract.** `ok` prints VIRTUAL_APPLICATION_OK and exits 0. `stale` is denied with cause 25 and exits 139.
- **live 1000 and 400000.** LIVE-OK, bad=0, rc 0, no fault line. At 400000, nodes is in [400000, 430000] (F0: 414,217)
  and capacity is 1048576.
  - **This is the bitstream-identity reading.** Records above 2^18 = 262,144 need slots that bitstream 3's 18-bit
    index cannot hold, so this reading is possible only on the new silicon.
- **live 1100000.** LIVE-NULL at an index in [990000, 1047000] (F0: 1,011,158), rc 3, nodes in [1040000, 1048576]
  (F0: 1,046,942), and no domain fault.

**Boot 2** (stage `b4-t1b`, run.sh 1c2ff626569baa81, judge `b4-check-t1b.py` 8d4cfa27d35b4bc8).
- What runs: b0-stats, b0-hello, then the guest SCRIPT of run.py@69fb6cae3db4 (11c47d62fe9f34ae), the twin's own Tier 1.
  It was generated by `gen-t1b.py` with three asserted substitutions:
  - the working directory is a tmpfs copy of the stage;
  - the launcher is the batch's;
  - insmod is guarded.
- Images, by hash: contract 00428f273c653d9f, sqlite3 5494c1c306b1f1c3, mruby 152ae8dc725d59c7.
- The judge executes run.py's own analysis block over the board transcript.
- **Prediction 2 of section 2:** every one of the 45 cells equals the twin's F0 cell
  (`f55/t1-F0/result.json`: 34 of 45, with 11 named failures). The one exception is `cleanup`, which is false on the
  board by construction, because the board kernel cannot unload a module.

**Dry run** (`b4/dryrun-b4.txt`). Every board and console call was stubbed. The kit maker and the judges ran for real.
- **The pass branch** reaches both judges and reads AS PREDICTED twice. The fixtures are F0's transcripts with only
  the board-side differences applied: the contract section, the linear mapping, and the reserved-mem line for boot 1;
  cleanup for boot 2.
- **Branches that stop or hold:**
  - a failed flash → STOP;
  - a store control that reads present → STOP;
  - a readback mismatch → STOP;
  - NON-LINEAR → HOLD, and boot 2 does not boot;
  - one flipped Tier 1 cell (mruby) → that one cell is named off.
- **Faults the dry run caught:**
  - The boot runner tripped `set -u` on a variable referenced in the same `local` statement that declares it. This is
    fixed.
  - Both judges read NO DATA on a missing or truncated transcript.
  - The twin's own F0 serial log, judged as the board, reads off in exactly `cleanup`.

**Not covered, said here.**
- The board console may not expose `nv_bitstream_sha256`. Then the readback is by name only, and the bitstream's
  identity rests on boot 1's 400000 reading.
- `preempt` waits 5 s for the spin child. On bitstream 3 the board's older vm-suite passed that cell.

## Addendum 15 (2026-10-30 12:35, before the boot it names): the directed reg_email boot, v2

**What changed since Addendum 13.** Off the board, re/reg_email's end is localized to one input. The native x86 perl
(perlbench_r_base.native-x86, `perf stat -e instructions:u`) runs the test with the first k emails:
- k=5: 0.4246e9 instructions;
- k=6: 3.0345e9 instructions;
- all 13: 4.5332e9 instructions;
- email 6 alone: 2.6394e9 instructions.

The twin/native ratio over the whole test is 5.813e9 / 4.533e9 = 1.28, so on RISC-V k=5 is about 0.54e9 and k=6
about 3.9e9. Both board arms ended at 0.752e9, early in email 6: `bammbamm (bam! bam! (bam! bam! (bam!)) bam!) @
flintstones.org`, the only input with nested comments. That makes it the deepest `(?&comment)` recursion. The test
prints its plan last (`done_testing`), so "no leader found" means the process ended before the end of the test,
probably after `ok 1`..`ok 5`. t/TEST does not print subtest stdout. perl grows the regex save stack with `Renew`
(realloc) under `regcppush` (scope.c `savestack_grow`, regexec.c:294 `SSGROW`).

**Candidate, not a finding.** The campaign SDKs carry the two heap defects that the realloc fork fixed on
spec-realloc-fix: realloc kept an offset that was not representable, and free() slots in the power-of-two-stride
classes were bound half a granule off. Whether either fires depends on how the heap address is aligned, and the board
kernel (6.4.14) and the twin's (6.1) may place the heap differently.

**Stage.** `spec/stages-directed/500.perlbench_r-regemail-both-v2`. It is a copy of Addendum 13's stage with these
changes:
- **run.sh.** 0a9aaeb1820a6a60, was 43edc93047b4a535. Its Addendum 13 rows (both arms: abigail, then reg_email)
  are unchanged and run first. After them come `@@DIRV` rows with their own markers, so the runner reads exactly the
  Addendum 13 rows. In order:
  1. camp-k5: the campaign T1 image (6b1a30a5b1e71f58) on `re/regemail_k5.t`, the first 5 emails;
  2. camp-k6: the same image on `re/regemail_k6.t`, the first 6;
  3. camp-e6: the same image on `re/regemail_e6.t`, email 6 alone;
  4. fix2-e6: the fix2 T1 image (2d7952ad24af6bb6) on `re/regemail_e6.t`;
  5. fix2-full: the fix2 T1 image on the full `re/reg_email.t`.
- **The variant tests** are `head -n` cuts of reg_email.t: the same text up to `__DATA__`, then the chosen lines.
- **The fix2 image.** It is tree-fix2's perlbench_r_base.capstone-spatial, stripped the way make-stage.py strips.
  That method reproduces the staged campaign hash 6b1a30a5b1e71f58 exactly. Its SDK is perf-fix2/sdk-T1, built from
  spec-realloc-fix. Against the campaign SDK perf/sdk-T1, compared member by member with llvm-ar x and cmp:
  - libapplication-runtime.a: 26/26 members, only `heap.c.obj` differs;
  - libc-capstone-virtual.a: 1361/1361 members, only `src_malloc_mallocng_malloc.o` differs (musl patch 0011,
    the slot-granule header).

  So the pair varies the heap fixes as one unit. There is no twin reference for the fix2 image; its reading is the
  test's own TAP.
- **Size and placement.** The stage is 6,394,467 B gzip -9, up from 4,826,982. The last perl test bake was
  21,662,216 B with a stage 1.63 MB smaller, so this one is estimated at about 23.4 MB, under BOARD_FW_CAP
  26,000,000. board-run.sh stops before the boot if it is over. The line replaces Addendum 13's on board 1's
  continuation list, in place, with step 5400 s and booking 120 min.
- **Host check.** The script was run with its /tmp paths moved and a fake launcher running native perl. Every row
  returned: k5 `1..5`, k6 `1..6`, e6 `1..1`, fix2-full `1..13`.

**P25, on board 1 (bitstream 3):**
- **Addendum 13's rows read first, and P24 stands as written.** That includes the abigail control: if abigail does
  not read the twin's cause and offset, the capture is broken and the boot says nothing.
- **camp-k5:** rc 0, `1..5`, ok=5, 0.4e9-0.7e9 instructions. The first five inputs pass on silicon.
- **camp-k6:** ends like the full test: no plan line, ok=5, and instret within 1% of 0.7523e9.
- **camp-e6:** no cause is predicted, but both outcomes are informative.
  - It ends before `1..1`, at about 0.2e9-0.35e9: the defect lies in email 6's own processing.
  - It passes `1..1`: the end needs state built by emails 1-5, which points to heap history.
- **Readings that name a cause:**
  - **The heap defects.** fix2-full reads `1..13` with ok=13 at about 5.6e9-6.1e9, and fix2-e6 reads `1..1`, while
    the campaign rows end. The campaign rows' stderr then names a fault in heap code (prepare_free, realloc, malloc).
    A fix2 pass alone does not prove it, because the fix2 images also move the heap layout.
  - **Not the heap defects.** fix2-e6 and fix2-full end at the same point as camp-e6 and camp-k6. The stderr line
    decides:
    - cause 4 or 6 is a misaligned access (Q-34);
    - another cause at a pc outside heap code is a silicon-vs-twin divergence in that code.

## Addendum 16 (2026-10-30 18:10, before the directed reg_email boot runs): the twin reproduces the end; cause named

**Finding, offline, from the QEMU twin (kit-q34), not from the board.** The campaign T1 image (6b1a30a5) ends re/reg_email
with `domain fault cause=2` at code+0x11eb0 (`realloc+0x7c`, an `unimp` trap) in 14 of 15 invocation variants run on
the twin. Those are: the launcher run directly, a harness-like environment, board paths, ASLR off/on, and environment
padding of 8-512 B. The fault comes at 0.75107e9-0.75251e9 instructions; the board read 0.75233e9 (T1) and 0.75266e9 (T3).
With 1024 B of padding it passes. The fix2 image (2d7952ad, the spec-realloc-fix SDK) completes all 15 at
5.8103e9-5.8118e9. Record: spec/perl-regemail/sweep1/guest/serial.log, FINDINGS.md.

**Mechanism** (a model of mallocng's arithmetic, native/sim63.py, checked against a native allocation trace).
- perl's savestack grows by exact-size realloc (scope.c:129-130).
- Its entries are 16 B on Capstone, so it reaches mallocng's large class early in email 6 and crosses 2^18 B there.
- The realloc to 261,888 B moves it to a fresh mapping. Its offset comes from ctx.mmap_counter, stepped to a multiple
  of 256.
- The next realloc, to 263,680 B (granule 512), keeps that offset. A key that is 256 mod 512 cannot be published
  exactly, so the heap's exactness check traps. In T3, CDERIVE refuses the same thing (cause 29, realloc+0x5c8).
- Whether it fires depends on the process's allocation history, which shifts mmap_counter. It does not depend on the
  kernel or ASLR: heap mappings are reserved aligned to their own granule.

So the board/twin difference is environmental, NOT silicon. The census and Addendum-13 twin references passed only
because of their environments.

**Corrections to Addendum 15.**
1. Its candidate "the board kernel (6.4.14) and the twin's (6.1) may place the heap differently" is refuted, as above.
2. The slot-granule fix (a5ad85f49dca) is folded into musl patch 0009, not a patch 0011.
3. P25's "camp-k6 ... ok=5" cannot be read. perl's stdout is block-buffered and lost at the fault, so an ending row
   has 0 output lines.
4. camp-e6 alone does not reach the trapping sequence (its fresh mapping lands at 262,144 B, already granule-512), so
   P25's "camp-e6 ends" branch is not expected.

**Twin references for the v2 stage** (spec/perl-regemail/v2-twin-b; every row passes on the twin, with the twin's
environment). On the twin both T1 abigail and T3 abigail fault as before.

| row | twin result |
|---|---|
| abigail T1 | cause 2 at +0x11eb0, 0.07138e9 |
| abigail T3 | cause 29 at +0x124c4, 0.07175e9 |
| reg_email T1 | `1..13`, 5.8128e9 |
| reg_email T3 | `1..13`, 5.8131e9 |
| camp-k5 | `1..5`, 0.56689e9 |
| camp-k6 | `1..6`, 3.89734e9 |
| camp-e6 | `1..1`, 3.39312e9 |
| fix2-e6 | `1..1`, 3.39323e9 |
| fix2-full | `1..13`, 5.81282e9 |

**P26, on board 1, replacing P25's predictions for the DIRV rows:**
- abigail as P24.
- reg_email T1/T3 and camp-k6: either `1..13`/`1..6`, or an end at 0.7505e9-0.7535e9 with cause 2 at +0x11eb0 (T1)
  and cause 29 at +0x124c4 (T3). The board environment decides which. If it ends, the fault line in its stderr is
  the reading that ties the board to this mechanism.
- camp-k5 `1..5` at ~0.567e9.
- camp-e6 `1..1` at ~3.393e9; an end here would refute the model.
- fix2-e6 `1..1` and fix2-full `1..13` at ~5.813e9, with no fault line in any variant.

## Addendum 17 (2026-10-30 19:00, before the boot it names): B check boot 3 (b4-t1c), Tier 1 with the S-24 fix on

**What boot 2 left.** b4-t1b (results in b4/b4-boots-1-2.result-lines.txt, 24769c569df0) read 40 of 45 cells equal to the
twin F0. Four of the five off cells came from the vm step: it exited 139 with cause 15 at arena + 0x2000, and the step
text carries S-24's bound line. The fifth, spin_entered, is open. That bound line reads `pte rc=-38 val=0x0
present=-1 ... dirty=-1`: resolve_pte was off (its default; `int prc = -ENOSYS` stands), so S-24's defining value,
a PTE with V R W A set and D = 0, was NOT read. **Correction to b4-boots-1-2.result-lines.txt** (its readings paragraph
is amended in the same commit): the attribution to S-24 rests on the shape (cause 15 at arena + 0x2000, exit 139)
and the bound line, with the mechanism's value unread. This boot settles it.

**What changes: the kit, nothing else.**
- t3-smoke/bkit14-b4r-b2 (make-b4r-kit.sh) is bkit13-b4-b2 plus one file, adapter/module/params (ff33a066fb679956):
  `resolve_dirty=1` and `resolve_pte=1`. boardvm writes each through sysfs right after insmod, prints
  `@@PARAM <name>=<read back>` and refuses the run unless both read Y. The module (6422805f49533ea5), launcher
  (62753dd693cc4a25), board.env and recipe are bkit13's, by hash.
- resolve_dirty=1 is the S-24 fix, two-sided on vp2 r13 and the module owner's decision (compiler lane, 2026-10-30).
- resolve_pte=1 only adds the PTE value to S-24's bound line when that line fires (capstone_vm.c, the bound branch).
  It changes no control flow, so it adds no second variable. If the loop still fires, its line now carries the value.
- The stage b4-t1c (gen-t1c.py, run.sh df5b86a3fea4a245) is t1b's Tier 1 SCRIPT byte for byte except:
  - the stage marker reads t1c;
  - the guarded insmod (a fallback; boardvm has already loaded the module) carries both arguments;
  - a must-fire read-back of both parameters (`@@T1C-RD`) follows it.
  After the Tier 1 script's VIRTUAL_RUNTIME_DONE, every line is prefixed `@@T1C-` so the Tier 1 analysis cannot count
  it:
  - a second read-back (`@@T1C-RD2`);
  - a clock check: `date +%s%N` and /proc/uptime, raw, around one `ok` launch, beside its launch_ns/elapsed_ns (R-52);
  - the spin pair: the spin step with a 5 s wait, then with a 15 s wait, each killed with TERM and printing its own
    `@@T1C-SPIN wait=<w> rc=<rc> ready=<count>`.
- The reference is the twin F0r: F0's kit and module with resolve_dirty=1 (module-ids55 PREREG addendum 3,
  3160e7d8e147). It equals F0 in all 45 cells, and vm exits 1 at `VM_CONTRACT_FAIL:execute through mapping PCC`. The
  twin never takes the resolve_dirty branch (QEMU's guest kernel handles the D-clear fault), so F0r shows only that the
  parameter changes nothing there. The silicon evidence for the branch is r13.

**Boot.** Board 2, bitstream 4 resident (read back by name before the boot), no flash: `b4-t1c.sh <log>` ->
`b4r-boot.sh 2 stages-directed/b4-t1c spec/b4-t1c/t1c 5400 120`. t1b's step ran ~46 min (91 heartbeats). Boot timeout =
step + 600 s, inside the 120 min booking. The judge is b4-check-t1c.py.

**Predictions** (exit 0 only if R, P1, P2 and P3 hold):
- R: `@@PARAM resolve_dirty=Y` and `@@PARAM resolve_pte=Y` once each, and both `@@T1C-RD` and `@@T1C-RD2` read Y for
  both. Otherwise READBACK FAIL, and the boot is no record of this configuration.
- P1: every Tier 1 cell equals F0r's, except cleanup (false by construction) and spin_entered (excluded, reported).
- P2, S-24's prediction for this boot:
  - vm reads F0r's own status, exit 1 at `VM_CONTRACT_FAIL:execute through mapping PCC`;
  - the four cells t1b's fault moved (nested_denial_site, vm_denial_sites, linux_pages_released, linux_demand_faults)
    return to F0r's values (inside P1);
  - no S-24 bound line fires.
  Any other vm reading reopens t1b's attribution. If a bound line fires, its PTE value is the reading.
- P3: the board's domain-fault causes are t1b's minus its leading cause 15: [25, 25, 28, 25, 15, 13, 13, 28, 25].

**Spin pair: reported, not predicted.** The readings, fixed before the boot:

| wait 5 s ready | wait 15 s ready | reading |
|---|---|---|
| 0 | 1 | A: slow start. The 5 s wait is shorter than a spin launch here (t1b's launch_ns 3.7-8.4 s against the twin's 8-22 ms); spin works on B, and t1b's spin_entered is a harness timing miss |
| 1 | 1 | B: not a fixed start-up cost. t1b's miss needs another account |
| 0 | 0 | C: no readiness within 15 s. Either still slower (read the clock line) or spin does not reach readiness on B; open, and next is a directed spin run |
| 1 | 0 | D: inconsistent; read the outputs |

Against A: vp2 boots b4c and b5 read spin_entered true with the same 5 s wait. The Tier 1 script's own
spin_entered is reported beside the pair. The clock line is read as elapsed_ns against the uptime delta. A launcher
elapsed time well above the wall time would make launch_ns, and so reading A's premise, a clock artifact (R-52).

**VOID.** The boot is VOID, with no reading, if:
- the kit or stage fails its hash checks (b4-t1c.sh/b4r-boot.sh STOP before booking);
- the resident bitstream does not read back as caplifive_vp2c_325966ee4;
- the board controls b0-stats/b0-hello fail;
- the step is cut by its bound (@@STEP-BOUND);
- the step text lacks VIRTUAL_RUNTIME_DONE (judge NO DATA).
A READBACK FAIL is not VOID: it is a record that the configuration did not take, and no cell is read from it.

**Dry run** (b4/dryrun-t1c.txt): the real b4-t1c.sh, b4r-boot.sh, run-stage.py and boardvm ran on stub transcripts
built from t1b's board.txt (build-t1c-stubs.py):
- pos: AS PREDICTED, exit 0, spin reading A;
- no @@PARAM lines, resolve_dirty N, resolve_pte N: boardvm refuses, the judge reads READBACK FAIL, exit 3;
- t1b's S-24 lines kept: OFF in the four cells, P2 and P3, exit 1;
- no spin and clock lines: graded AS PREDICTED, both reported NO DATA;
- the step's RD2 reading N under a passing @@PARAM (judge only): READBACK FAIL, exit 3.
The built batch carries both sysfs writes in gate.sh and both insmod arguments in spec/run.sh.


## Addendum 18 (2026-10-31 04:35, before any boot it names): the train stages rebuilt on the fixed heap with the code layout controlled

**What changes, and why.** Two findings made the queued train images the wrong ones to spend about 150 board-hours on.
- **Heap defects.** The campaign SDKs carry the two heap defects named in Addendum 16. realloc kept an object's offset
  at a base the new size cannot be published at, and power-of-two-stride slots were bound half a granule off, so free()
  trapped. On silicon this ended perlbench's re/reg_email in both arms (Addendum 16; P26). The fixes are on dev,
  348d9cf2c29c (realloc) and de7cdacda737 (slot header). The SDKs `perf/sdk-heapfix-T1` (MapLife: lifetimes off) and
  `perf/sdk-heapfix-T3` (ObjLife: lifetimes on, scrub off) are built from exactly that runtime tree.
- **Code layout.** An audit found that the campaign images do not control layout. The runtime's heap.c.obj, the one
  file the arms differ in, sits early in `.text`. So every benchmark function after it moves between the arms: mcf's
  main is at 0x212ec in MapLife and 0x21414 in ObjLife. Every ObjLife/MapLife cycle ratio then carries a placement
  term, and on the microbenchmarks that term reached about 8 % at identical instructions.
  - The layout control is on lane spec-layout c1b3cc5d7c7a (`CAPSTONE_SPEC_LAYOUT=heap-last`, tools/layout-sdk.py,
    tools/layout-check.py).
  - It links heap.c.obj last in `.text`, `.data` and `.bss`, and puts its read-only data in its own `.rodata.heap`.
    Each of these goes in a region of fixed size.
  - Every other symbol and section then sits at the same address in both arms.

Nothing else changes: the bitstream (3, e7af48457), the kits (`bkit12-perf-b3-b<board>`; the fw33 copies are retired, below), the
known-good control, and every make-stage.py option. Only the images differ, and so the stages' bin/.

**Build** (`hfl/build-hfl.sh`). A fresh tree `spec/tree-hfl`, record `spec/rec-hfl`, kit = spec-layout c1b3cc5d7c7a
(clean), the seven train benchmarks, both arms. Helpers built in each arm: x264's ldecod_r and imagevalidate_525,
imagick's imagevalidate_538.
- Every benchmark object is byte-identical across the arms (387 files).
- **layout-check: 10 of 10 executable pairs PASS** (`hfl/layout-check-hfl.txt`). That covers 6256 benchmark functions,
  1052 benchmark data, 2971 libc+runtime functions, 1000 libc+runtime data and 130 sections. mcf's main is at 0x1eb84
  in both arms, and `__capstone_mallocng_malloc` at 0x25f50 in both.
- **Negative control:** the campaign build (rec-board, tree-board) still FAILS, 0 of 13 pairs, exit 1. mcf's main is at
  0x212ec against 0x21414 (`hfl/layout-check-neg-rec-board.txt`).
- `runcpu --fake --action=run` for train and test names the layout view (`rec-hfl/layout-sdk/<arm>`) as the README
  requires. Every staged executable's sha256 equals the build record's (36 of 36; the stripped perl test images are not
  compared).

**Images** (sha256, first 16). The `run.sh` of each stage is make-stage.py's current output, the same version that
wrote the xz link stages; the hashes are in `hfl/stage-compare.txt`'s stages.

| image | MapLife (T1, capstone-spatial) | ObjLife (T2, capstone-lifetimes) |
|---|---|---|
| perlbench_r | 13d99c3cfb5ff803 | 05044fa04bc2275d |
| mcf_r | e29f5bf9808d4f5e | c9bb655c5550d1e6 |
| lbm_r | 8698cd834f0b139d | 719af4fde88c23f9 |
| x264_r | ef04ec09fab01572 | 352d9af44c927ea4 |
| imagick_r | 1ad8e4ec5eac7e17 | ff733eb6d46fc845 |
| imagevalidate (525 and 538 alike) | 058701b817197584 | 5675010c762a7c7d |
| nab_r | a027892a2d34405c | c90722ccacc0bf62 |
| xz_r | 6f12dfe4ff05e9cd | c51256d3a3e4760f |

ldecod (x264's input generator, outside the context) is unchanged: b624ecdd5437252b.

**Stages** (`hfl/make-stages-hfl.sh`, `spec/stages-train-hfl/`). Each stage mirrors one queued stage, one for one:
- the same runcpu train commands;
- the same options: perlbench `--keep-lib perl-train-lib.txt`, x264 `--ldecod`, xz `--link-inputs`, and none for mcf,
  lbm, imagick and nab;
- imagick and x264 keep their target-side imagevalidate.

Proof, `hfl/stage-compare.txt`:
- The same script run on the campaign tree reproduces all 14 queued stages file for file and plan field for plan field.
- The only difference is run.sh, which is newer than the campaign's for 12 of them: it adds the module's sysfs counter
  delta. The xz link stages are byte-identical.
- The hfl stages differ from those regenerated stages only in bin/ and `plan.executables`. Staged references are
  compared by content: their names carry the tree path.
- Negative control: mcf against lbm reads MISMATCH.

**Payload.** The firmware is the Image plus 2,097,160 B, and the stage lives in the Image's gzip'd initramfs.
`hfl/fw-estimate.py` gives the firmware size from the stage's gzip -9 size.
- **Fit.** The initramfs is 3,440,665 B plus the stage's gzip size. Four Images, with stages from 0.26 to 6.4 MB, match
  that within 30 KB.
- **Image size.** The Image rounds to 15,370,752 + k × 2 MiB.
- **Control.** All ten past boots of this kit fall inside its range, `hfl/fw-estimate.txt`.

| stage (per arm) | gzip -9 | firmware estimate | cap |
|---|---|---|---|
| perlbench train | 5.31 MB | 21.66-23.76 MB | 26.0 MB, FITS |
| x264 train | 6.52 MB | 23.76 MB | 26.0 MB, FITS |
| imagick train | 1.60 MB | 19.57 MB | 26.0 MB, FITS |
| mcf train | 0.89 MB | 17.47-19.57 MB | 26.0 MB, FITS |
| nab train | 0.51 MB | 17.47-19.57 MB | 26.0 MB, FITS |
| lbm train | 0.14 MB | 17.47 MB | 26.0 MB, FITS |
| xz train, --link-inputs, both commands | 24.13 MB | 42.63 MB | refused (see below) |
| **xz train, --link-inputs --commands 0** (input.combined) | 8.45 MB | **25.86 MB** | 26.0 MB, FITS |

**xz train: the full stage cannot be baked; the board runs command 0 (input.combined) only, in both arms.**

**The bake model.** An audit of the bake path reproduces six past Images exactly with:
- Image = 2 MiB × ceil((6,349,144 + initramfs gzip + 35,320) / 2 MiB) + 4,884,992;
- firmware = Image + 2,097,160.

The initramfs is the base rootfs (about 2.97 MB gzip), plus the batch files (about 0.47 MB), plus the stage.

**The full xz train stage is refused.** It has both commands, and its inputs are already .xz, so they do not compress.
- Its firmware is **42,633,736 B**. board-run.sh refuses it at its cap check before any boot.
- Raising the cap is not a fix:
  - the FDT sits 32 MiB above the payload;
  - the documented upload hard cap is 32 MB (docs/plans/2026-10-16-board-campaign-all.md:25);
  - the largest firmware ever uploaded is 30,050,824 B.

**This corrects Addendum 11.** Its estimate ("about 31.5 MB": the Image as about 4.9 MB plus the raw stage) left out the
kernel prefix and the base rootfs. lbm test's Image is 15,370,752 B with a 0.26 MB stage. So:
- **P23 is withdrawn** as registered. No full xz train boot runs, on the campaign images or on these.
- **The fw33 kits are retired.** Their 33,000,000 B cap was itself above the documented 32 MB upload cap.
- **xz-b1.list is retired.** Its waiter was stopped before it started anything.

**xz train on the board is command 0, input.combined, in both arms, on the standard kits**
(bkit12-perf-b3-b<board>, cap 26,000,000).
- **The stage.** make-stage.py gains `--commands` (spec-silicon, board/make-stage.py). It keeps the selected run
  commands of runcpu's speccmds.cmd, their specdiffs, and only the input files they name; plan.json records the
  selection.
- **Default unchanged.** Without the option, regenerated stages are byte-identical to the existing ones: run.sh,
  plan.json and every file of stage/, for xz link (both arms), x264 and perlbench.
- **The new stages** are `557.xz_r-train-<arm>-link-c0`:
  - one command (`xz_r input.combined.xz 40 ... 8`), one host specdiff;
  - inputs `control` and `input.combined.xz`;
  - `IMG_2560.cr2.xz` dropped;
  - run.sh is the full stage's with command 1's lines removed;
  - stage gzip 8,453,6xx B (spatial 8,453,625, lifetimes 8,453,738).

**Firmware: 25,856,520 B in both arms,** under the 26,000,000 cap. About 0.59 MB of initramfs is left below the next
2 MiB step. Three checks:
- the formula above;
- `hfl/fw-estimate.py`;
- `hfl/bake-size-check.py`, a bake without the buildroot or the board:
  - it builds boardvm's real per-arm batch directory (run-stage.py with a stub transcript, the standard kit);
  - it splices that directory into a past Image's initramfs in place of that boot's batch, and recompresses with
    gzip -9;
  - control: it reproduces the Image and firmware of five past boots exactly. Its gzip runs 5-11 KB above the
    kernel's, so it errs high;
  - it gives the full stage 42,633,736.

**IMG_2560.cr2.xz alone** would bake to 34,245,128 B, above every proven upload. It is a limitation recorded here, not a
run.

**Twin reference.** The c0 stage runs exactly the full stage's first command, in the same fresh run directory, first.
- **The full census is not a usable reference.** Against it (census-train/...-link-mem936M), run-stage.py's twin
  comparison reads `DIFFER ['IMG_2560.cr2-40-4.out']`: an output the c0 run never makes.
  - Shown on the real full transcript with command 1's lines cut out, both arms (`hfl/xz-c0-refcheck.txt`).
- **The derived reference.** `hfl/derive-subset-ref.py` derives one from the same run:
  - command 0, its outputs' hashes and in-band files, and its specdiff;
  - written to census-train/...-link-mem936M-c0, recording the full result's sha256.
- **Against the derived reference:**
  - the same transcript reads `identical`;
  - a forged input.combined-40-8.out hash reads DIFFER;
  - a transcript with that output removed reads specdiff NO-OUTPUT, DIFFER, exit 1.
- run-stage.py is unchanged.

**P23' (replaces P23), per arm.** Images 6f12dfe4ff05e9cd (MapLife) and c51256d3a3e4760f (ObjLife).
- The firmware is 25,856,520 B and loads.
- input.combined ends with rc 0 and specdiff 1/1 PASS.
- The output hash equals the derived twin reference's.
- No OOM kill.
- instret within 0.3 % of the twin's command 0: 112.65e9 MapLife.

**Twin check of the new images** (kit-q34, the campaign twin; `hfl/twin-hfl.sh`, `hfl/twin-hfl2.sh`; table
`hfl/twin-readings.txt` from `hfl/twin-table.py`):

Each row is one arm of a twin run of the new images against the campaign twin's run of the same stage (census-q34
for test, census-heapfix for perlbench test, census-train for train), on the same twin:

```
run                                          arm                  cmds rc!=0 specdiff      instret new      instret ref  new/ref     nodes new/ref  outputs
519.lbm_r-test                               capstone-spatial        1     0      1/1    7,278,940,293    7,278,938,612  1.00000             34/34  vs-ref identical; cross-arm identical
519.lbm_r-test                               capstone-lifetimes      1     0      1/1    7,278,939,501    7,278,939,204  1.00000             40/40  vs-ref identical; cross-arm identical
538.imagick_r-test                           capstone-spatial        2     0      1/1      108,675,796      108,677,371  0.99999           160/160  vs-ref identical; cross-arm identical
538.imagick_r-test                           capstone-lifetimes      2     0      1/1      108,697,138      108,697,798  0.99999         1039/1039  vs-ref identical; cross-arm identical
544.nab_r-test                               capstone-spatial        1     0      1/1    7,991,860,750    7,991,860,754  1.00000           504/504  vs-ref identical; cross-arm identical
544.nab_r-test                               capstone-lifetimes      1     0      1/1    7,992,016,156    7,992,016,799  1.00000         7194/7194  vs-ref identical; cross-arm identical
505.mcf_r-test                               capstone-spatial        1     0      2/2   35,301,304,555   35,301,316,585  1.00000           812/812  vs-ref identical; cross-arm identical
505.mcf_r-test                               capstone-lifetimes      1     0      2/2   35,302,814,867   35,302,835,720  1.00000       64004/64004  vs-ref identical; cross-arm identical
557.xz_r-test                                capstone-spatial       12     0    12/12   50,821,690,317   50,821,695,905  1.00000           972/972  vs-ref identical; cross-arm identical
557.xz_r-test                                capstone-lifetimes     12     0    12/12   50,821,706,424   50,821,712,690  1.00000         1716/1716  vs-ref identical; cross-arm identical
500.perlbench_r-test-capstone-lifetimes      capstone-lifetimes      2     0      2/2  122,111,935,520  122,111,423,984  1.00000       17161/17161  vs-ref identical
500.perlbench_r-test-capstone-spatial        NO DATA (run has no result.json)
525.x264_r-test-capstone-lifetimes           capstone-lifetimes      2     0      0/3  543,659,797,655  543,659,795,476  1.00000         4817/4817  vs-ref DIFFER ['imagevalidate_frame_100.out', 'imagevalidate_frame_155.out', 'imagevalidate_frame_50.out']
525.x264_r-test-capstone-spatial             NO DATA (run has no result.json)
519.lbm_r-train-capstone-lifetimes           NO DATA (run has no result.json)
519.lbm_r-train-capstone-lifetimes-2         capstone-lifetimes      1     0      1/1  106,393,125,104  106,393,124,115  1.00000             40/40  vs-ref identical
505.mcf_r-train-capstone-lifetimes           NO DATA (run has no result.json)
500.perlbench_r-train-capstone-lifetimes-2   capstone-lifetimes      5     2      4/6  170,780,415,093  170,764,041,539  1.00010 10697283/10697283  vs-ref identical
```

- **Instruction counts.** Every run that has data reads new/ref 1.00000 to 1.00010, so the new layout moved no count
  by more than 0.01 %. The node counts are identical, and so is every output except as noted below.
- **x264 test: specdiff NO-OUTPUT on the imagevalidate files** in this row and in the reference comparison. The test
  stages carry no imagevalidate, the same as both boards' x264 test boots. Every real output equals the reference's.
- **perlbench train ObjLife: 2 of 5 commands rc 1, the same two as on the campaign image.** See P29.
- **NO DATA (not run; deferred by the 05:00 lock guard):** perlbench test MapLife, x264 test MapLife, mcf train ObjLife,
  and the xz command-0 pair (twin-hfl3.sh). lbm train ObjLife ran as `-2`. For those lines P32 falls back to the campaign
  twin's count, within 1 %.
- The exit status 2 of twin-table.py is these four NO DATA rows.

**Boot lists** (`spec/board/train-hfl-b1.lines`, `train-hfl-b2.lines`, `train-hfl-xz-c0.lines`).

Steps and bookings are the queued lines', benchmark by benchmark. Every instruction count the twin measured on the new
images is within 5 % of the campaign count (above). The twin column is the queued lines': for perlbench and xz it
compares outputs only, so it holds across images.

The one exception is **perlbench train ObjLife**: 93,600 s and 1,590 min, was 54,000 s and 930 min.
- The queued step was sized for a run cut short at 170.8e9 instructions, when scrabbl and splitmail stop at the
  lifetime table's capacity (Addendum 9).
- MapLife's full run is 300.5e9 instructions.
- If ObjLife now completes, it needs MapLife's step, and an unfinished boot would cost 15 board-hours for no verdict. A
  boot that ends early unbooks, so the larger step costs nothing.

Board time per arm is the train instret (twin, or the native-ratio estimate of Addendum 2 for x264 and nab) times the
board's own CPI on that benchmark's test or train boot:

| benchmark | instret per arm | board CPI | hours per arm |
|---|---|---|---|
| perlbench | 300.5e9 (MapLife); ObjLife 170.8e9 if cut short as in Addendum 9 | 6.4 | 21.3 / 12.3 (21.6 if complete) |
| mcf | 180.6e9 | 8.7 | 17.5 |
| nab | ~439e9 | 3.06 | 14.9 |
| imagick | 287.6e9 | 3.20 | 10.2 |
| lbm | 106.4e9 | 5.07 | 6.0 |
| x264 | ~247e9 | 1.99 | 5.5 |
| xz | 163.6e9 | 2.8 | 5.1 |

**Both arms of a benchmark run on the same board, back to back,** so each cycle ratio is read within one board.
- **Board 1:** perlbench, then nab, then the xz command-0 pair. About 63 h plus 7 h for xz. They follow the Plain
  microbenchmark boots p1-p6 in the same live list (b1-cont.list). They replace the campaign train lines, which were
  never started; the list's unread tail was edited in place.
- **Board 2:** lbm, then mcf, x264, imagick. About 78 h. They follow the ObjBounds microbenchmark kits on the ObjBounds
  bitstream (land/rtl-objb) and the RTL lane's one-image witness. Then `spec/board/b2-after-objb2.sh`:
  - reflashes bitstream 3 (b2-to-bs3.sh: local sha256 d6ed97b6..., booked flash, readback by name);
  - waits until this addendum is on origin/spec-silicon;
  - runs train-hfl-b2.lines.

  Its stubbed dry run covers every branch (`hfl/dryrun-b2after.txt`; the scripts are copied into hfl/), and the live
  gate read "not registered" before this push.
- The long ones are split: perlbench on board 1, mcf and x264 on board 2. Each list is MapLife first, then ObjLife.
- **xz:** the command-0 pair (`train-hfl-xz-c0.lines`, about 3.5 h per arm at 112.65e9 instructions and CPI 2.8)
  runs on board 1, after nab. Step 32,400 s and booking 570 min are the queued xz lines'. The
  full-stage xz lines (`train-hfl-xz.lines`) are not board-ready and are not queued.

**Dry run** (`hfl/dryrun.txt`, the board-run skill's rule 8). chain-board.sh from spec-silicon ran on stubs (busy check,
hub and runner; the kit, control and stage checks are real):
- board 1, kit bkit12-perf-b3-b1, every b1 and b2 line: 12 of 12 END rc=0;
- board 2, kit bkit12-perf-b3-b2, the b2 lines: 8 of 8;
- the xz command-0 lines on the STANDARD kits, as board 1 (bkit12-perf-b3-b1) and as board 2 (bkit12-perf-b3-b2): 2 of
  2 each, with the derived twin reference passed as `--twin`;
- the twin column reaches the runner as `--twin`;
- negative: a line naming a directory that is not a stage reads `SB STOP ... is not a make-stage.py output`, and the
  chain stops.

The stubs never bake, so the dry run cannot see the firmware cap.

**Predictions.** Every boot below runs on bitstream 3 with the registered images. The verdict and VOID rules are the
main text's and Addendum 17's, plus those listed here.
- **P27. Control and payload.** The control passes in every boot. Each firmware lies inside its estimate above, and
  under 26,000,000 B.
- **P28. Correctness, MapLife, six benchmarks:** perlbench, mcf, lbm, x264, imagick and nab. xz command 0 is
  P23' above.
  - Every command ends with rc 0, and every host specdiff PASSes.
  - imagick's and x264's target-side imagevalidate exit 0.
  - No domain fault.
  - perlbench's six outputs are sha256-equal to SPEC's references (splitmail.out by hash).
- **P29. Correctness, ObjLife:**
  - mcf, lbm, x264, imagick and nab: as P28. Every output is byte-identical to the same benchmark's MapLife output (.err
    and log.dec aside).
  - **perlbench:** on bitstream 3 the run is cut short by the lifetime table, as the campaign image's was (Addendum 9).
    The registered ObjLife image on the twin (census-hfl/500.perlbench_r-train-capstone-lifetimes-2):
    - diffmail, perfect and suns end rc 0, with specdiff PASS;
    - scrabbl and splitmail end rc 1 with perl's own "Out of memory!":
      - scrabbl at 261,089 lifetime nodes of the table's 262,144, node_growths 0;
      - splitmail after 495,939 node allocations through the table's recycling. Its live high-water is not printed,
        so its attribution to the table is by the same message and the same table, and is weaker than scrabbl's;
    - outputs hash-identical to the campaign image's twin run, which ended the same two commands the same way;
    - instret 170.78e9 against 170.76e9, ratio 1.0001; node counts identical, 10,697,283.

    Prediction for the board:
    - diffmail, perfect and suns: rc 0, specdiff PASS, outputs equal to the twin's;
    - scrabbl and splitmail: rc != 0 with "Out of memory!", each node count within 1 % of the twin's.

    Those two commands are a capacity reading of bitstream 3's 18-bit table (2^18 = 262,144 records). They carry no
    correctness verdict, and the P31 perlbench ratio uses only the three commands that complete in both arms. The full
    ObjLife run needs bitstream 4 (B), where the 55-bit twin passed 5 of 5 (Addendum 14). It is not part of this
    addendum. The step stays 93,600 s, because a boot that ends early unbooks.
- **P30. No heap trap.** Neither arm of any benchmark ends a command with a domain fault in heap code. That means no
  cause 2 (MapLife) and no cause 29 (ObjLife) at realloc, `__capstone_malloc_prepare_free` or free. In particular,
  perlbench train has no realloc trap in either arm.
- **P31. Cycle ratio ObjLife/MapLife,** in-context cycles summed over the run commands.
  - **Readings so far.** The campaign's uncontrolled test-size pairs on bitstream 3 read lbm 1.0003, xz 0.998,
    x264 0.9987, nab 1.014, mcf 1.0255 and perlbench 1.020. imagick test's 0.4e9-cycle run read about 0.98.
  - **What the control leaves.** Moving the benchmark's own code is now ruled out. What remains is heap.c's own code
    and offsets, the run-time heap addresses, and the per-object lifetime nodes. On the twin, the arms' instruction
    counts differ by under 0.01 % on test for every benchmark except perlbench.
  - **Bands:**
    - lbm, x264, imagick and nab, and xz command 0, which hold few lifetime nodes (test: 40, 1,039 and 7,194 against
      34, 160 and 504; xz test 143 against 81): within **0.985-1.025**;
    - mcf: within **0.99-1.08**. On test it read 1.0255, with 64,004 nodes against 812. On train the twin counts
      168,412 ObjLife allocations;
    - perlbench, per command that completes in both arms: **at least 0.99**. No upper band is registered. On train,
      diffmail allocates 7.9 million nodes through the table's recycling, outside anything test measured.
  - **Not comparable with the campaign's ratios.** These images share one placement, which is not the campaign
    images', so a ratio here is compared with nothing from the campaign images.
- **P32. instret.** Each command is within 0.3 % of the twin's count on the same images, where the twin ran them (see
  the twin table above). Otherwise it is within 1 % of the campaign twin's count.

**VOID and no-reading rules for these boots:**
- **VOID, the known-good control fails.** The boot carries no verdict (main text).
- **NOT RUN, no reading:**
  - spec-board.sh stops on a kit, control or stage hash check;
  - board-run.sh refuses the bake, the firmware build, or the firmware cap.
- **VOID, the wrong bitstream.** The boot ran on a bitstream other than caplifive_vp2c_e7af48457, for example after a
  reflash of its board between queueing and the run.
- **VOID, the images are not the registered ones.** Any staged image whose sha256 differs from the table above.
- **No verdict for the commands left unfinished:**
  - the step is cut by its bound or by run-stage.py's timeout. Those commands carry no verdict and the pair gives no
    ratio;
  - SPEC-DONE is absent, or the reader exits 2 on no data.
- **A kernel OOPS or OOM** is a capacity reading, not a correctness one, as P13 and P14 were.
- **A ratio (P31) is read only when** both arms' boots are valid, every run command ended with rc 0 in both, and both ran
  on the same board.
  - A pair split across the boards is reported as cross-board and not graded against the band.
  - perlbench ObjLife's cut-short commands, if any, are left out of its per-command ratios.
- **No boot repeats a run** (Addendum 10).
  - imagick train MapLife ran on the campaign image (287.6e9 instructions, PASS), and that run stays the record for that
    image.
  - The hfl run is a different image under a different layout, so it is not a repeat.
