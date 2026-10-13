# CPython 3.13.7: reuse distance at pymalloc, stock against protected, 2026-10-11

The CPython row of the pre-registration `capstone/docs/plans/2026-10-11-nested-reuse-distance.md`
(pushed as 16d2171a3edc, not edited). Stock = pymalloc as released; protected = patch 0014
(`pymalloc-sublet-lifetimes`, `CPY_SUBLET=1`). Boundary `pymalloc`: pymalloc's block hand-out and
free (requests over 512 bytes go to the system allocator and are not observed). N = 3 boots per
arm, the six in parallel VMs, 2026-10-11 10:04:46Z to about 10:05:41Z UTC.

## Verdicts

| Prediction | Verdict | Numbers |
|---|---|---|
| 1. Equal histograms, bucket for bucket; where repetitions vary, the arms vary alike (overlapping ranges per bucket) | **does not hold** | The repetitions vary (by up to 3 per bucket within an arm), and in **8 of the 14 non-empty buckets** the stock and protected ranges do not overlap (table below). The share of reuses whose bucket differs between the arms is 0.082 % (repetition 1 against 1), 0.056-0.082 % over all nine pairs, against 0.004-0.027 % between repetitions of one arm. Counters: issues and releases equal; protected has 6 more reuses and 6 fewer distinct start addresses, the same in every repetition |
| Median bucket equal (stated for SQLite; reported here too) | holds | bucket 2 (distance 2-3) in all six boots |
| 3. Figure: the curves overlap | not drawn here (shared plot script) | largest difference between the two arms' cumulative fractions at any bucket edge: 0.00037 (any stock against any protected repetition); between repetitions of one arm: 0.00020 (stock), 0.00024 (protected) |
| Sanity: CPYTHON-OK 1000, observer error 0 | **holds** | all six: stdout `CPYTHON-OK 1000`, exit status 0, `error=0`, no other stderr, `rmmod` 0 |

The difference is small and it is a difference: the per-bucket ranges of the two arms are disjoint
more often than not, and the between-arm share is 3-20 times the within-arm share. It is the
measurement as taken, not adjusted. The memory-behaviour run of the same day
(`../2026-10-11-memory-behaviour/`) found the arms' blocks in use differing in the 80 B and 96 B
classes from interpreter start-up, with root cause unresolved; this run observes the whole process
including start-up, so it would carry that difference, but whether it is the same cause was **not
tested**.

## Results

| counter | stock (each of 3) | protected (each of 3) |
|---|---|---|
| attempts = issues (pymalloc hand-outs) | 51,670 | 51,670 |
| releases | 47,885 | 47,885 |
| reuses (hand-out at a previously released start) | 25,497 | 25,503 |
| distinct start addresses | 26,173 | 26,167 |
| observer error | 0 | 0 |
| median bucket | 2 | 2 |

Bucket k holds distances d with 2^(k-1) <= d < 2^k (observer `bins[k-1]`); buckets 15-32 are 0 in
every boot.

| bucket k (distance) | stock rep 1 / 2 / 3 | protected rep 1 / 2 / 3 | ranges overlap |
|---|---|---|---|
| 1 (1) | 5,191 / 5,191 / 5,192 | 5,197 / 5,196 / 5,196 | **no** |
| 2 (2-3) | 8,494 / 8,494 / 8,493 | 8,499 / 8,494 / 8,494 | yes |
| 3 (4-7) | 2,826 / 2,826 / 2,827 | 2,820 / 2,823 / 2,823 | **no** |
| 4 (8-15) | 1,706 / 1,706 / 1,707 | 1,704 / 1,705 / 1,705 | **no** |
| 5 (16-31) | 2,029 / 2,030 / 2,032 | 2,032 / 2,032 / 2,033 | yes |
| 6 (32-63) | 1,595 / 1,594 / 1,591 | 1,591 / 1,591 / 1,590 | yes |
| 7 (64-127) | 824 / 824 / 824 | 821 / 821 / 821 | **no** |
| 8 (128-255) | 789 / 789 / 790 | 797 / 798 / 798 | **no** |
| 9 (256-511) | 813 / 813 / 812 | 814 / 814 / 814 | **no** |
| 10 (512-1023) | 630 / 630 / 629 | 626 / 627 / 627 | **no** |
| 11 (1024-2047) | 365 / 365 / 365 | 367 / 367 / 367 | **no** |
| 12 (2048-4095) | 157 / 157 / 157 | 157 / 157 / 157 | yes |
| 13 (4096-8191) | 51 / 51 / 51 | 51 / 51 / 51 | yes |
| 14 (8192-16383) | 27 / 27 / 27 | 27 / 27 / 27 | yes |

`reuse.json` holds the same (repetition 1 as `buckets`, all three under `repetitions`), and under
`comparison` the median buckets and the shares (`reusediff_*`).

Consistency check, not pre-registered: `capstone-vexec --stats` `nodes_allocated_total` over the
interpreter run is 13,762 (stock) and 65,443 (protected) in every boot. The difference, 51,681, is
11 more than the 51,670 hand-outs, each of which patch 0014 makes a lifetime (0014 also derives one
per arena and one per in-place realloc, and the system allocator's own count need not be equal
between arms; this run does not separate those 11), so the protected image did make lifetimes of
the blocks the observer saw. `.text` holds 8 CDERIVE and 2 CREVOKE words in the protected image and none in the
stock image.

## Hook placement

One define, `EXP_REUSE_GAP`, both arms; patch 0014 rewrites the lines the hooks sit next to, so
there are two patch files whose added lines are identical
(`capstone/experiments/study/reuse-hooks/cpython/obmalloc-reuse-gap-{stock,protected}.patch`):

| | stock | protected |
|---|---|---|
| hand-out | `Objects/obmalloc.c:pymalloc_alloc`, after the block is taken (free list or new pool), before `return (void *)bp;`: `reuse_gap_attempt`, `reuse_gap_issue(AS_UINT(bp), INDEX2SIZE(size))` | same function, same place, before `return block_issue(state, bp, nbytes);` -- `bp` is obmalloc's own pointer, not the client's child |
| release | `Objects/obmalloc.c:pymalloc_free`, after `address_in_range` says the block is pymalloc's, before it is linked into the pool's free list: `reuse_gap_release(AS_UINT(p))` | same function, after `CREVOKE` and `p = arena_inside(ao, p); pool = POOL_ADDR(p);`, before the free-list link, with obmalloc's own `p` (same address as the client's) |
| report | `atexit` handler registered at the first hook call: one stderr line `REUSE-GAP pymalloc capacity=... attempts=... issues=... releases=... reuses=... distinct_starts=... error=... bins=b0,...,b31` | identical |

Static table of 2^20 slots (32 MiB of BSS, outside every arena); integers only. In-place
`pymalloc_realloc` hands nothing out in either arm and is not hooked (in the protected arm it
replaces the client's lifetime at the same block, which the observer does not see as a reuse).
`_PyObject_Calloc` and moving reallocs reach the hooks through `pymalloc_alloc`/`pymalloc_free`.

## Builds

From one snapshot: `git archive 16d2171a3edc capstone` plus `capstone/experiments/study/reuse-hooks/`
(cpython, mruby) plus `compiler-rt/lib/builtins` of 9bc6f9adf1c9, committed as a fresh repository on
the build host; `build.py` linked both images at that repository's second commit (which only fixed
the mruby hook patch) with `runtime_dirty: false`. Both built in parallel, 225 s and 224 s.

    LD_LIBRARY_PATH=$K/toolchain/hostlib CAPSTONE_LLVM_BUILD_DIR=$K/toolchain \
      CAPSTONE_REPO_ROOT=<snapshot> CAPSTONE_TMP_ROOT=<tmp> PYTHONDONTWRITEBYTECODE=1 \
      CPY_ARCHIVE=<Python-3.13.7.tgz, sha256 6c9d8083...53b2> CPY_TEST_CAPI=1 CPY_JOBS=16 JOBS=16 \
      bash capstone/experiments/study/reuse-hooks/cpython/build-cpython-reuse.sh stock|protected OUT

`build-cpython-reuse.sh` is `build-virtual.sh`'s cpython recipe with one inserted step after
`prepare-cpython-capstone.sh` (unpack, port patches, configure): apply the arm's hook patch with
`--fuzz=0`, copy `reuse-gap-observer.h` beside `obmalloc.c`, prepend `-DEXP_REUSE_GAP` to the
configured Makefile's `CONFIGURE_CPPFLAGS`; then survey, link and `build.py --nested none|cpython
--stack-bytes 8388608`. It fails unless the image contains the report string.

| item | stock | protected |
|---|---|---|
| `cpython.dom` (= manifest `image_sha256`) | `dcad934a920d46162d222150ce26dd2463dfc077261eea589737bb845b88d0c2` | `9fa4d381bd0573efeada0d38a2d202202787599d03b3a50ee597bd176fe21bde` |
| manifest `nested` | none | cpython |
| applied port patches | 0001-0013, 0015, 0016 | 0001-0016 |
| hook patch sha256 | `689a732d...49cf` | `5e4f24eb...528b` |
| patched `obmalloc.c` sha256 | `2a46f69f...4e96` | `58337a5c...8c09` |

Observer header `627b0fe3...1769`; clang `4c6135fe...65d4`. These images are not the
memory-behaviour images (those had no hooks, and were built from another snapshot).

## Runs

Per boot (`gate.sh`, written by `run-cpython-reuse.py`): `insmod`; `capstone-vexec --stats`;

    PYTHONHOME=/mnt/vm/cpy ./capstone-vexec ./cpython.dom -S -c 'import json,gc; a=[{"n":n} for n in range(1000)]; assert json.loads(json.dumps(a))==a; del a; gc.collect(); print("CPYTHON-OK 1000")'

`capstone-vexec --stats`; `rmmod`. Stage: each arm's image, its build's own `Lib/` as
`/mnt/vm/cpy/lib/python3.13` with its `_sysconfigdata__linux_.py` and an empty `lib-dynload`,
compiled by that build's native 3.13.7 (`compileall -f -x /test/ -d /mnt/vm/cpy/lib/python3.13`,
rc 0), as in the memory-behaviour run. On the host, per arm and repetition:

    LD_LIBRARY_PATH=$K/toolchain/hostlib CAPSTONE_QEMU_LOCK=<own lock> \
      python3 capstone/experiments/study/reuse-hooks/cpython/run-cpython-reuse.py --arm ARM --rep N \
      --build OUT --kit $K/kit-sublet --run-staged capstone/runtime/virtual/run-staged.py --work <fresh>

then `collect-reuse.py` (same directory) over the six `rep.json` files writes `reuse.json`.

Platform as pre-registered: QEMU `af37cc32...d27d`, launcher `capstone-vexec` `3975314a...4ad4`,
module `a1c6cb6b...e512`, `fw_jump.elf` `44cd6cbc...f116`, `Image` `e833dfb1...8a4b`,
`rootfs.ext2` `62049800...7845`, `run-staged.py` `d6770b8b...c54c`. CPU
`rv64,sstc=false,h=false,sv48=false,sv57=false,x-capstone-u-mode=true,x-capstone-exact-bounds=true`,
one hart, 2 GiB.

Wall times (six VMs in parallel on an 80-core host also running other guests; not performance
measurements): interpreter run in the guest 36, 36, 37 s (stock) and 37, 37, 37 s (protected);
boot 45-46 s each.

## Method limits

- The observer covers the whole interpreter process: start-up, the workload, and finalisation up to
  the C `atexit` handlers. The event index counts pymalloc hand-outs only.
- Addresses are obmalloc's own; with address-space randomisation active, absolute addresses differ
  between boots, but the histograms depend only on which block is reused when.
- One workload, one hart, QEMU.

## Deviations

- The workload is the verbatim one-liner alone (one process per boot), not the memory run's
  `mem_driver.py`; the pre-registration names the run-ports.py JSON/GC workload, which is the
  one-liner.
- Images were rebuilt from 16d2171a3edc with the hooks, not the memory-behaviour images.
- `stdlib_py_tree_sha256` in the repetitions differs between arms because it includes each build's
  `_sysconfigdata__linux_.py` (build paths, `CPPFLAGS`); the stdlib sources are the same release.

## Files

- `reuse.json`: the shared-format result (counters, buckets, repetitions, comparison, images,
  platform, build).
- Hooks, build wrapper, runner and collector: `capstone/experiments/study/reuse-hooks/cpython/`.
