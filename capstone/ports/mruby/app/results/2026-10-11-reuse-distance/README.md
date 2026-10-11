# mruby 4.0.0-rc2: reuse distance at the GC object slots, stock against protected, 2026-10-11

The mruby row of the pre-registration `capstone/docs/plans/2026-10-11-nested-reuse-distance.md`
(pushed as 16d2171a3edc, not edited). Stock = mruby with the port's patches; protected = patch 0008
active (`MRBD_SUBLET=1`, `MRB_CAPSTONE_SUBLET`: every GC object slot a Sublet child lifetime of its
heap page). Boundary `gc-slots`: GC object slot hand-out (`mrb_obj_alloc`) and the sweep's free.

**Measured at a reduced AO size, a deviation from the registered workload (see Deviations):**
`bm_ao_render.rb 16`, a 16x16 image, where the registration's workload is the benchmark as shipped,
64x64. N = 3 boots per arm (the repetitions vary, below); boots 2026-10-11 11:56:50Z to 12:29:14Z
UTC.

## Verdicts

| Prediction | Verdict | Numbers |
|---|---|---|
| 1. Equal histograms, bucket for bucket; where repetitions vary, the arms vary alike (overlapping ranges per bucket) | **holds**, in its "vary alike" form | Buckets 1-16 (distances below 65,536 hand-outs: 890,279 reuses, 98.3-98.5 % of each boot's reuses) are identical in all six boots. Buckets 17-19 vary between repetitions of one arm, and every bucket's stock and protected ranges overlap. Stock repetition 2 and protected repetition 3 have identical histograms (share 0). Share of reuses whose bucket differs between arms: 0.105 % (repetition 1 against 1), 0-0.226 % over all nine pairs, against 0.048-0.123 % (stock) and 0.077-0.152 % (protected) between repetitions of one arm |
| Median bucket equal | holds | bucket 10 (distance 512-1,023) in all six boots |
| 3. Figure: the curves overlap | not drawn here (shared plot script) | largest difference between the arms' cumulative fractions at any bucket edge: 0.0023 (any pair); between repetitions of one arm: 0.0012 (stock), 0.0015 (protected) |
| Sanity: AO render output equal across arms, observer error 0 | **holds** | all six PPMs (781 bytes) have sha256 `cfd73bb1...a634`, equal to the native build of the same tree at width 16; exit status 0, `error=0`, no other stderr |

Issues and releases are equal in every boot (916,101 and 913,510). Reuses and distinct start
addresses vary with the repetitions, in both arms, and their ranges overlap:

| counter | stock rep 1 / 2 / 3 | protected rep 1 / 2 / 3 |
|---|---|---|
| attempts = issues | 916,101 (all) | 916,101 (all) |
| releases | 913,510 (all) | 913,510 (all) |
| reuses | 904,859 / 904,424 / 903,738 | 905,812 / 904,446 / 904,424 |
| distinct start addresses | 11,242 / 11,677 / 12,363 | 10,289 / 11,655 / 11,677 |
| observer error | 0 | 0 |
| median bucket | 10 | 10 |

Buckets (k holds distances 2^(k-1) <= d < 2^k, observer `bins[k-1]`; buckets 20-32 are 0 in every
boot):

| bucket k (distance) | stock rep 1 / 2 / 3 | protected rep 1 / 2 / 3 | ranges overlap |
|---|---|---|---|
| 1 (1) | 685 / 685 / 685 | 685 / 685 / 685 | yes |
| 2 (2-3) | 1,370 / 1,370 / 1,370 | 1,370 / 1,370 / 1,370 | yes |
| 3 (4-7) | 2,740 / 2,740 / 2,740 | 2,740 / 2,740 / 2,740 | yes |
| 4 (8-15) | 5,480 / 5,480 / 5,480 | 5,480 / 5,480 / 5,480 | yes |
| 5 (16-31) | 10,960 / 10,960 / 10,960 | 10,960 / 10,960 / 10,960 | yes |
| 6 (32-63) | 21,920 / 21,920 / 21,920 | 21,920 / 21,920 / 21,920 | yes |
| 7 (64-127) | 43,624 / 43,624 / 43,624 | 43,624 / 43,624 / 43,624 | yes |
| 8 (128-255) | 86,463 / 86,463 / 86,463 | 86,463 / 86,463 / 86,463 | yes |
| 9 (256-511) | 171,776 / 171,776 / 171,776 | 171,776 / 171,776 / 171,776 | yes |
| 10 (512-1,023) | 311,298 / 311,298 / 311,298 | 311,298 / 311,298 / 311,298 | yes |
| 11 (1,024-2,047) | 218,075 / 218,075 / 218,075 | 218,075 / 218,075 / 218,075 | yes |
| 12 (2,048-4,095) | 1,888 / 1,888 / 1,888 | 1,888 / 1,888 / 1,888 | yes |
| 13 (4,096-8,191) | 356 / 356 / 356 | 356 / 356 / 356 | yes |
| 14 (8,192-16,383) | 2,288 / 2,288 / 2,288 | 2,288 / 2,288 / 2,288 | yes |
| 15 (16,384-32,767) | 4,556 / 4,556 / 4,556 | 4,556 / 4,556 / 4,556 | yes |
| 16 (32,768-65,535) | 6,800 / 6,800 / 6,800 | 6,800 / 6,800 / 6,800 | yes |
| 17 (65,536-131,071) | 13,466 / 13,466 / 13,457 | 14,312 / 14,165 / 13,466 | yes |
| 18 (131,072-262,143) | 9 / 9 / 0 | 0 / 0 / 9 | yes |
| 19 (262,144-524,287) | 1,105 / 670 / 2 | 1,221 / 2 / 670 | yes |

`reuse.json` holds the same (repetition 1 as `buckets`, all three under `repetitions`; `comparison`
holds the medians and shares; `deviation` the size change).

The varying part is confined to reuses after 65,536 or more hand-outs, and comes with a varying
number of distinct start addresses. A candidate, **not tested**: mruby frees a heap page whose
slots are all dead, and whether a later page lands at a previously used address is up to the system
allocator under address-space randomisation; a slot address reused that way counts as a reuse at a
long distance.

Consistency check, not pre-registered: `capstone-vexec --stats` `nodes_allocated_total` over the
mruby run is 2,436,082 (stock) and 3,352,203 (protected) in every boot; the difference, 916,121, is
20 more than the 916,101 slot hand-outs, each of which patch 0008 makes a lifetime (0008 also derives
one per heap page; this run does not separate the 20). `.text` holds 8 CDERIVE and 2 CREVOKE words
in the protected interpreter and none in the stock one.

## Hook placement

One define, `EXP_REUSE_GAP`, and one patch for both arms
(`capstone/experiments/study/reuse-hooks/mruby/gc-reuse-gap.patch`): patch 0008 is applied in both
trees and is entirely under `#ifdef MRB_CAPSTONE_SUBLET`, so the hooked lines are the same text in
both; the hooks sit outside 0008's `#ifdef` blocks and use the GC's own slot pointer `p` in both.

| | both arms |
|---|---|
| hand-out | `src/gc.c:mrb_obj_alloc`, right after the slot is popped from the page's free list and `gc->live++`, before 0008's `#ifdef` (protected: before `gc_issue` derives the child; stock: before `gc_protect`): `reuse_gap_attempt`, `reuse_gap_issue((uintptr_t)p, sizeof(RVALUE))` |
| release | `src/gc.c:incremental_sweep_phase`, after `obj_free` left the slot `MRB_TT_FREE`, before 0008's `gc_retire` and before the slot is linked to the free list: `reuse_gap_release((uintptr_t)p)` |
| report | `atexit` handler registered at the first hook call: one stderr line `REUSE-GAP gc-slots capacity=... attempts=... issues=... releases=... reuses=... distinct_starts=... error=... bins=b0,...,b31` |

Not hooked: `free_heap` at `mrb_close` (it frees the remaining objects with their pages, no slot is
handed out again). Static table of 2^20 slots, integers only.

## Builds

Same snapshot as the CPython row: `git archive 16d2171a3edc capstone` plus
`capstone/experiments/study/reuse-hooks/` (cpython, mruby) plus `compiler-rt/lib/builtins` of
9bc6f9adf1c9, committed as a fresh repository on the build host. One virtual-profile application SDK
(musl 1.2.5, `build-sdk.sh`, `CAPSTONE_APPLICATION_ARENA_BYTES=67108864`), as the memory-behaviour
run built it, copied once per arm. Then per arm:

    LD_LIBRARY_PATH=$K/toolchain/hostlib CAPSTONE_LLVM_BUILD_DIR=$K/toolchain \
      CAPSTONE_REPO_ROOT=<snapshot> CAPSTONE_TMP_ROOT=<tmp> PYTHONDONTWRITEBYTECODE=1 JOBS=16 \
      bash capstone/experiments/study/reuse-hooks/mruby/build-mruby-reuse.sh stock|protected SDK-COPY ROOT

`build-mruby-reuse.sh` runs `ports/mruby/app/build-mruby-domain.sh` unchanged (`MRBD_PIN=4.0.0-rc2`,
`MRBD_SUBLET=0|1`: clone at 9d523e2f, the port's patches, a full build), applies the hook patch with
`--fuzz=0`, copies `reuse-gap-observer.h` beside `gc.c`, removes `build/`, and runs the same script
from its mruby stage with `MRBD_DEFINES=-DEXP_REUSE_GAP`, so every object is rebuilt with the define
(124 capstone objects, 0 failed, each arm). It fails unless the interpreter contains the report
string. 229 s per arm, in parallel. (The first attempt failed to compile: the hook block lacked
`#include <stdlib.h>` for `atexit`; fixed in the patch, which is the one listed here.)

| item | stock | protected |
|---|---|---|
| `build/capstone/bin/mruby` | `45fc3b81da836ff439a515ba6811a24e68c92c295bf6697ef4ca42973e669cd5` | `8487ecb12b22452069e72d3d5b671b8bc2491c185e3e014fc735d523c8d68716` |
| CDERIVE / CREVOKE words in `.text` | 0 / 0 | 8 / 2 |

Hook patch `0cf22747...f736`, patched `gc.c` `6fe03e52...36ab` (same in both trees), observer header
`627b0fe3...1769`, `bm_ao_render.rb` `e44efffd...180f`, clang `4c6135fe...65d4`.

## Runs

Per boot (`gate.sh`, written by `run-mruby-reuse.py`): `insmod`; `capstone-vexec --stats`;
`./capstone-vexec ./mruby ./bm_ao_render.rb 16`, stdout to a file whose sha256 and gzip+base64
come back on the console; `capstone-vexec --stats`; `rmmod`. On the host, per arm and repetition:

    LD_LIBRARY_PATH=$K/toolchain/hostlib CAPSTONE_QEMU_LOCK=<own lock> \
      python3 capstone/experiments/study/reuse-hooks/mruby/run-mruby-reuse.py --arm ARM --rep N \
      --root ROOT --kit $K/kit-sublet --run-staged capstone/runtime/virtual/run-staged.py \
      --work <fresh> --ao-width 16 --timeout 3600

then `capstone/experiments/study/reuse-hooks/cpython/collect-reuse.py` over the six `rep.json`
files writes `reuse.json`. Platform as pre-registered and as in the CPython row: QEMU
`af37cc32...d27d`, launcher `3975314a...4ad4`, module `a1c6cb6b...e512`, `fw_jump.elf`
`44cd6cbc...f116`, `Image` `e833dfb1...8a4b`, `rootfs.ext2` `62049800...7845`, `run-staged.py`
`d6770b8b...c54c`; one hart, 2 GiB, exact bounds.

Wall times (parallel VMs on a host also running many other guests; not performance measurements):
the AO run in the guest took 1,132 / 1,191 / 663 s (stock) and 1,134 / 1,192 / 692 s (protected);
repetitions 1-2 ran four at a time beside other load, repetition 3 two at a time. A single-boot
probe of the stock interpreter (outside the result) took 126 s at width 8 and 486 s at width 16.
Natively, the same tree's hooked build runs width 16 in 0.59 s and width 64 in 5.97 s.

## Deviations

- **Reduced workload size.** Registered: the AO render benchmark as shipped, 64x64
  (`IMAGE_WIDTH = IMAGE_HEIGHT = Integer(ARGV[0] || 64)`, `NSUBSAMPLES = 2`, `NAO_SAMPLES = 8`).
  Measured: `bm_ao_render.rb 16`, 16x16, `NSUBSAMPLES` and `NAO_SAMPLES` unchanged. Width 64 is
  15,096,713 slot hand-outs (native hooked build) against 916,101 at width 16; a first pair of
  width-64 boots did not finish within its 1-hour boot timeout, and the width-16 probe's rate
  (486 s for 916,101 hand-outs) extrapolates to about 2.2 h per width-64 boot. Width 64 is running
  with a 4-hour timeout; if it completes it is added as an addendum, not in place of this row.
- Repetitions: three per arm, not one, because the first two per arm varied.
- The benchmark runs directly, not under the memory-behaviour run's `mstats.rb` wrapper (which adds
  `ObjectSpace.count_objects` and a `GC.start`).
- Images were built from 16d2171a3edc with the hooks; they are not the memory-behaviour images.

## Method limits

- The observer covers the whole interpreter process up to the C `atexit` handlers; the event index
  counts slot hand-outs only.
- `-DEXP_REUSE_GAP` reaches the native host build too (`MRBD_DEFINES` applies to both), so the host
  `mrbc` also prints a report line during the build; it plays no part in the measurement.
- One workload size, one hart, QEMU.

## Files

- `reuse.json`: the shared-format result.
- Hooks, build wrapper and runner: `capstone/experiments/study/reuse-hooks/mruby/`; collector:
  `capstone/experiments/study/reuse-hooks/cpython/collect-reuse.py`.
