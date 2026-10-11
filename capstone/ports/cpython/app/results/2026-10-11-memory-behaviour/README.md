# CPython 3.13.7: protected pymalloc memory behaviour, 2026-10-11

The CPython row of the pre-registration
`capstone/docs/plans/2026-10-11-protected-memory-behaviour.md` (pushed as 82c1607119ec, not edited).
Stock = pymalloc as released; protected = patch 0014 (`pymalloc-sublet-lifetimes`, `CPY_SUBLET=1`).
N = 3 per arm, the six boots in parallel VMs. Run 2026-10-11 09:34:09Z to 09:35:12Z UTC.

## Verdicts

| Prediction | Verdict | Numbers (stock vs protected) |
|---|---|---|
| Arenas in use equal | **holds** | 4 allocated, 4 current, 4 high water, 0 reclaimed, both arms |
| Pools in use per size class equal | **holds** | all 32 classes equal; 61 unused pools both |
| Blocks in use per size class equal | **does not hold** | 30 of 32 classes equal; class 4 (80 B) 5,003 vs 4,995 (-8); class 5 (96 B) 2,521 vs 2,523 (+2). Bytes in allocated blocks 2,810,688 vs 2,810,240 (-448). These two classes also moved between repetitions of one arm in the cross-check (below) |
| No pymalloc footprint difference (arena bytes) | **holds** | 4,194,304 B in arenas, pool headers 15,280, quantization 23,744, alignment 65,536, both arms |
| The radix tree's leaf grows by two fields | **holds; measured** | `bytes lost to arena map bot` 131,072 vs 262,144 (+131,072 B, one bot node in both arms). This is system-allocator memory, outside the arenas |
| Protected allocates more lifetimes | **holds** | `nodes_allocated_total` +13,796 vs +71,105 (+57,309, 5.15x) |
| ... and reclaims them | **holds** | `nodes_reclaimed` +10,450 in 1 collection vs +30,784 in 3 collections; `nodes_live` and `nodes_retired` 0 after exit in both |
| Sanity: CPYTHON-OK 1000 | **holds** | printed by the driver and by the verbatim run-ports.py one-liner, 3/3 boots per arm, exit status 0 |

In the pre-registered boots every reading was **identical across the three repetitions** of each
arm: the full `sys._debugmallocstats()` text byte for byte, every `--stats` field before and after,
and both VmHWM readings, although hash randomisation and address-space randomisation were active.
That identity is **not general**: in the `time -v` cross-check boots (below) one protected
repetition differed from its two siblings in the same two classes, 80 B 4,992 vs 5,001 (-9) and
96 B 2,518 vs 2,522 (-4), bytes in allocated blocks 2,809,520 vs 2,810,624; the same repetition is
the one whose `reclaimed` (30,782 vs 30,769) and module high water (41,502 vs 41,509) differ. The
three stock cross-check repetitions were identical. Ranges over all repetitions: see the
cross-check table. Other repetition-to-repetition variation: `nodes_reclaimed` of the verbatim run
(below) and wall times.

## Comparison (each cell identical in all three repetitions)

### Allocator statistics after the workload, `sys._debugmallocstats()`

| class | size | pools stock | pools prot. | blocks in use stock | blocks in use prot. | avail stock | avail prot. |
|---|---|---|---|---|---|---|---|
| 0 | 16 | 1 | 1 | 42 | 42 | 977 | 977 |
| 1 | 32 | 1 | 1 | 74 | 74 | 435 | 435 |
| 2 | 48 | 3 | 3 | 678 | 678 | 339 | 339 |
| 3 | 64 | 5 | 5 | 1136 | 1136 | 134 | 134 |
| **4** | **80** | 25 | 25 | **5003** | **4995** | 72 | 80 |
| **5** | **96** | 15 | 15 | **2521** | **2523** | 14 | 12 |
| 6 | 112 | 18 | 18 | 2447 | 2447 | 163 | 163 |
| 7 | 128 | 11 | 11 | 1370 | 1370 | 27 | 27 |
| 8 | 144 | 28 | 28 | 3125 | 3125 | 39 | 39 |
| 9 | 160 | 8 | 8 | 799 | 799 | 9 | 9 |
| 10 | 176 | 3 | 3 | 210 | 210 | 66 | 66 |
| 11 | 192 | 2 | 2 | 139 | 139 | 29 | 29 |
| 12 | 208 | 5 | 5 | 384 | 384 | 6 | 6 |
| 13 | 224 | 2 | 2 | 95 | 95 | 49 | 49 |
| 14 | 240 | 2 | 2 | 85 | 85 | 49 | 49 |
| 15 | 256 | 1 | 1 | 56 | 56 | 7 | 7 |
| 16 | 272 | 1 | 1 | 44 | 44 | 15 | 15 |
| 17 | 288 | 3 | 3 | 149 | 149 | 19 | 19 |
| 18 | 304 | 1 | 1 | 35 | 35 | 18 | 18 |
| 19 | 320 | 19 | 19 | 918 | 918 | 32 | 32 |
| 20 | 336 | 1 | 1 | 25 | 25 | 23 | 23 |
| 21 | 352 | 1 | 1 | 26 | 26 | 20 | 20 |
| 22 | 368 | 6 | 6 | 244 | 244 | 20 | 20 |
| 23 | 384 | 3 | 3 | 88 | 88 | 38 | 38 |
| 24 | 400 | 6 | 6 | 221 | 221 | 19 | 19 |
| 25 | 416 | 5 | 5 | 176 | 176 | 19 | 19 |
| 26 | 432 | 3 | 3 | 68 | 68 | 43 | 43 |
| 27 | 448 | 3 | 3 | 72 | 72 | 36 | 36 |
| 28 | 464 | 3 | 3 | 72 | 72 | 33 | 33 |
| 29 | 480 | 2 | 2 | 55 | 55 | 11 | 11 |
| 30 | 496 | 2 | 2 | 51 | 51 | 13 | 13 |
| 31 | 512 | 2 | 2 | 48 | 48 | 14 | 14 |

| Total line | stock | protected | difference |
|---|---|---|---|
| arenas allocated total / reclaimed / high water / current | 4 / 0 / 4 / 4 | 4 / 0 / 4 / 4 | 0 |
| bytes in arenas (4 x 1,048,576) | 4,194,304 | 4,194,304 | 0 |
| bytes in allocated blocks | 2,810,688 | 2,810,240 | -448 |
| bytes in available blocks | 279,632 | 280,080 | +448 |
| unused pools (x 16,384 B) | 61 (999,424) | 61 (999,424) | 0 |
| bytes lost to pool headers / quantization / arena alignment | 15,280 / 23,744 / 65,536 | same | 0 |
| arena map mid / bot nodes | 1 / 1 | 1 / 1 | 0 |
| bytes lost to arena map root / mid | 524,288 / 524,288 | same | 0 |
| bytes lost to arena map bot | 131,072 | 262,144 | +131,072 |
| arena map total | 1,179,648 | 1,310,720 | +131,072 |
| free lists (dict, keys, float, list, tuple 1-20) | 1 list (80 B), 1 1-tuple (64 B), rest 0 | identical | 0 |

-448 B = 8 x 80 - 2 x 96. Both differing classes stay in the same number of pools.

### Lifetime counters, `capstone-vexec --stats` after minus before the driver run

| field | stock | protected |
|---|---|---|
| nodes_allocated_total | 13,796 | 71,105 |
| nodes_reclaimed | 10,450 | 30,784 |
| collections | 1 | 3 |
| nodes_high_water (module-wide maximum since insmod; before = 0) | 13,054 | 41,495 |
| nodes_live, nodes_retired (after the process exited) | 0, 0 | 0, 0 |
| live_domains, live_regions, live_bytes, tag_pages, cached_bytes, poisoned_blocks | 0 | 0 |
| node_capacity, node_bytes | 1,024, 28,672 before and after (delta 0) | same |

Before the run every counter was 0 except node_capacity 1,024 and node_bytes 28,672.

### Peak resident memory of the interpreter process

| reading | stock | protected | difference |
|---|---|---|---|
| VmHWM, read in-process from `/proc/self/status` after the statistics | 33,440 kB | 33,572 kB | +132 kB |
| VmHWM, last of the external samples of `/proc/<pid>/status` | 33,588 kB | 33,720 kB | +132 kB |
| kernel peak at exit, busybox `time -v` "Maximum resident set size" / 4 (cross-check boots) | 33,588 kB (printed 134,352) | 33,720 kB (printed 134,880) | +132 kB |
| VmRSS / VmSize / VmPeak at the in-process read | 33,392 / 43,680 / 52,556 kB | 33,524 / 43,808 / 52,556 kB | +132 / +128 / 0 kB |
| launcher `pages` / `peak` (CAPSTONE_VM_STATS, 4 kB pages) | 7,611 / 8,208 | 7,641 / 8,241 | +30 / +33 pages (+120 / +132 kB) |

All values identical in every repetition of each arm. The sampler read `$!` of the backgrounded
`capstone-vexec` itself (no wrapper shell, no spawner pid), and its last value equals the kernel's
peak at exit, so it sampled the interpreter process. Busybox `time -v` prints 4 x ru_maxrss; the
printed value is exactly 4 x the sampled VmHWM in all six runs. The +128 kB of VmSize matches the
+131,072 B arena-map bot node; the remaining 4 kB of the VmHWM difference was not attributed.

### Cross-check boots: `time -v` and `CAPSTONE_VM_STATS=1` (3 per arm, run after the above)

`gate-timev.sh`: same stage, the driver wrapped in busybox `/usr/bin/time -v` with
`CAPSTONE_VM_STATS=1`, no sampler and no verbatim run. The extra environment variable and wrapper
change the interpreter's inputs, so these allocator readings are not comparable with the boots
above, only arm against arm.

| reading (min-max over 3 repetitions where they vary) | stock | protected |
|---|---|---|
| lifetime table at exit: `node_capacity` / `node_bytes` / `node_growths` | 13,312 / 225,280 B / 3 | 41,984 / 684,032 B / 10 |
| `nodes` / `collections` / `reclaimed` | 13,788 / 1 / 10,444 | 71,101 / 3 / 30,769-30,782 |
| `arenas` / `faults` | 22 / 2,522 | 22 / 2,554 |
| `--stats` delta: nodes_allocated_total / nodes_reclaimed / nodes_high_water | 13,788 / 10,444 / 13,054 | 71,101 / 30,769-30,782 / 41,502-41,509 |
| pymalloc class 4 (80 B) / class 5 (96 B) blocks in use | 5,001 / 2,523 | 4,992-5,001 / 2,518-2,522 |
| bytes in allocated blocks | 2,810,720 | 2,809,520-2,810,624 |
| arenas / unused pools / arena map bot bytes | 4 / 61 / 131,072 | 4 / 61 / 262,144 |
| elapsed (time -v) | 27.14-27.44 s | 28.14-28.39 s |

CPYTHON-OK 1000 and exit status 0 in all six; no domain faults; boots 34-36 s.

### Wall times

Boot (QEMU start to `VIRTUAL_STAGED_DONE`, two interpreter runs per boot): stock 61 s, protected
63 s, each of 3. Driver run in the guest (1 s resolution, sampler running beside it): stock 30, 31,
30 s; protected 32, 32, 32 s. Six VMs in parallel on an 80-core host that was also running other
QEMU guests; these are not performance measurements.

## Supplementary localisation of the blocks-in-use difference (not pre-registered)

Run after the measurement above, one boot per arm, N = 1, same stages with a different gate
script. These do not change the verdicts; they say where the difference is not.

| probe (`-S -c ...`, then `sys._debugmallocstats()`) | class 4 (80 B) stock / protected | class 5 (96 B) stock / protected | bytes in allocated blocks stock / protected |
|---|---|---|---|
| interpreter start only | 3,166 / 3,164 | equal | 1,505,344 / 1,505,184 |
| after `import json,gc,sys` | 4,960 / 4,957 | 2,520 / 2,521 | 2,836,272 / 2,836,128 |
| after building the 1,000-dict list | 4,964 / 4,959 | equal | 3,191,456 / 3,191,056 |
| after `json.dumps` of it | 4,953 / 4,960 | 2,523 / 2,520 | 3,190,560 / 3,190,832 |
| interpreter start, `PYTHONHASHSEED=0` | 3,166 / 3,164 | equal | 1,505,344 / 1,505,184 |
| full workload, `PYTHONHASHSEED=0` | 4,970 / 4,959 | 2,518 / 2,517 | 2,816,720 / 2,815,744 |

`sys.getallocatedblocks()` around the workload steps (start, built, dumped, loaded, after
`gc.collect()`): stock 21,675 / 24,411 / 24,415 / 27,154 / 21,529; protected 21,673 / 24,409 /
24,413 / 27,151 / 21,526. Every step's increment is equal or within 1 between arms; the offset of
2 blocks is already present at interpreter start.

What this shows:
- The difference exists **at interpreter start**, before the workload runs, and is confined to the
  80 B and 96 B classes at every probe point. Its sign varies between probe points.
- It is **not hash randomisation**: `PYTHONHASHSEED=0` gives the same start-up difference
  (`hash("abc")` equal in both arms under seed 0). Without a seed the hash differs from run to run,
  and so do absolute addresses, yet the three pre-registered repetitions of each arm read
  identically.
- It is **not purely a function of the arm**: one protected cross-check repetition moved by -9 / -4
  in these two classes against its siblings (above), so part of it varies from run to run.
- It is **not a realloc decision**: patch 0014's `pymalloc_realloc` keeps upstream's stay-or-move
  test on the size class (`4 * nbytes > 3 * size`); the bounded length only limits the copy.
- The two images place their static objects at different offsets: `id(object)` ends in
  `...adfae0` (stock) and `...adfe00` (protected) under every address-space base, while heap objects
  (`id(sys)`, a fresh `object()`) keep equal low-order offsets. That address-dependent behaviour
  (for example hashing of type objects) causes the difference is a candidate, **not tested**.
  Root cause: **unresolved**.

## Images and platform

| item | sha256 |
|---|---|
| stock `cpython.dom` (manifest `nested: none`) | `6760ff8773f450b044fbdce3ab6432cde3988c5c51aff25142eb4b1af61fec13` |
| protected `cpython.dom` (manifest `nested: cpython`) | `36cececaab98fb19aee0b905331a768c63c1a19341addc24539c881639cbeeea` |
| QEMU `qemu-system-riscv64` | `af37cc3269d1cfb47a72eb79cc3dfd2397848c613ca703e43625c68ab50ad27d` |
| launcher `capstone-vexec` | `3975314a7a6da119c4bff6a0ab0c601fcd96d305f59a0a68fe5b36b9bd544ad4` |
| module `capstone_vm.ko` | `a1c6cb6b79a98cf1f09e55a37adf2320e2b9787922b7e10bf4088c27e396e512` |
| `fw_jump.elf` / `Image` / `rootfs.ext2` | `44cd6cbc...f116` / `e833dfb1...8a4b` / `62049800...7845` (full in `raw-counters.json`) |
| `run-staged.py` | `d6770b8b9c55ef8a4e136a17f5411a2fe2f139af9fee47f5e02328810acfc54c` |
| `mem_driver.py` | `bb39f831e2228e42971d350ca79b50c38fa9ad58d587acc80b0c5b3b77d5b819` |
| `gate.sh` | `02191c4e338dad4ca4d5c9ce0d7f5d7d501f2a87841952b060fe330ad9902b33` |

Both images are the ones the same-answers CPython run built (recipe and runtime revision
`e801018c031c`, a snapshot of `9bc6f9adf1c9`; both manifests say `runtime_dirty: true`). Each
copy's sha256 was checked against `image_sha256` in its build's `manifest.json` before staging.
CPU: `rv64,sstc=false,h=false,sv48=false,sv57=false,x-capstone-u-mode=true,x-capstone-exact-bounds=true`,
one hart, 2 GiB.

Stdlib: each arm stages its own build's `Lib/` as `/mnt/vm/cpy/lib/python3.13` (the run-ports.py
layout) plus its `_sysconfigdata__linux_.py` and an empty `lib-dynload`, compiled with native
CPython 3.13.7 (`compileall -f -x /test/ -d /mnt/vm/cpy/lib/python3.13`). The two `.py` trees
without sysconfigdata are identical (`1bb077b7...4197`); the two trees differ only in
`_sysconfigdata__linux_.py` and its `.pyc` (build paths and `CPPFLAGS`), which the driver reports
as not loaded. The driver also checks that the staged `.pyc` of `json`, `json.decoder`,
`json.encoder`, `json.scanner` and `encodings` match their sources (all `valid`, every run).

## Commands

Per boot (`gate.sh`, in this directory): `insmod`; `capstone-vexec --stats`;
`PYTHONHOME=/mnt/vm/cpy ./capstone-vexec ./cpython.dom -S /mnt/vm/mem_driver.py` in the background
with a VmHWM sampler; `capstone-vexec --stats`; the run-ports.py one-liner verbatim;
`capstone-vexec --stats`; `rmmod`. On the host, per arm and repetition:

    LD_LIBRARY_PATH=$K/toolchain/hostlib CAPSTONE_QEMU_LOCK=<own lock per VM> \
      python3 capstone/runtime/virtual/run-staged.py --qemu $K/kit-sublet/qemu-system-riscv64 \
      --images $K/kit-sublet/images --stage <stage-ARM> --work <fresh dir> --timeout 1800

`mem_driver.py` runs the run-ports.py JSON/GC statements, then `sys._debugmallocstats()`, and only
afterwards takes the readings that allocate (VmHWM, bytecode check, module check).

## Method limits

- **Allocator statistics** are pymalloc's own, in the same process, after `gc.collect()` and before
  interpreter finalisation. The arena-map bytes and the `arena_object` array (patch 0014 adds
  `arena_life` to each entry) are system-allocator memory; `_debugmallocstats` reports the former
  and not the latter.
- **Lifetime table size** comes only from the cross-check boots' `CAPSTONE_VM_STATS` line, which the
  launcher prints for its own context at exit. `node_capacity` and `node_bytes` in `--stats` sum
  the contexts open at the time of the call, which is only the `--stats` process's own (1,024 nodes,
  28,672 B); the interpreter's context is gone by then, so the pre-registered boots' `--stats`
  readings of these two fields say nothing about the workload. `nodes_reclaimed` counts
  identities recovered by collections while the process ran; the rest of a context's identities
  are discarded with it at exit and appear in no counter, so allocated minus reclaimed is not a
  leak count. `nodes_high_water` is a module-wide maximum.
- The number of pymalloc hand-outs was not counted, so "about one lifetime per hand-out" is not
  tested here; only the direction (more) is.
- **VmHWM** is of the Linux process hosting the domain (launcher, image, domain heap, stack and
  exchange region), as Linux counts resident pages. The in-process read precedes interpreter
  finalisation; the external sampler (busybox shell, nominal 0.2 s sleep, 83 samples per run)
  shares the one hart with the workload. Its last sample equals the kernel's peak at exit
  (`time -v` / 4), so it missed no later growth here.
- Single workload, single hart, QEMU. The supplementary probes are N = 1 per arm.

## Deviations

- The driver is a staged file run with `-S`, not the `-c` one-liner; the one-liner was also run
  verbatim in every boot (CPYTHON-OK 1000, exit 0). Its lifetime deltas: stock +13,790 allocated,
  +10,390 reclaimed, 1 collection (all 3 boots); protected +65,472 allocated, 3 collections, and
  reclaimed +30,811 / +30,800 / +30,811 (the only counter that varied between repetitions).
- Images were copied from the same-answers build, not rebuilt; their manifests record a dirty
  runtime tree at `e801018c031c`.
- The localisation probes and the `time -v` / `CAPSTONE_VM_STATS` cross-check boots were added
  after the pre-registered runs and are reported separately. The cross-check found the
  repetition-to-repetition variation in the 80 B and 96 B classes that the first three boots per
  arm did not show.

## Files

- `mem_driver.py`: the driver staged as `/mnt/vm/mem_driver.py`.
- `gate.sh`: the in-guest script of every pre-registered boot.
- `gate-timev.sh`: the in-guest script of the cross-check boots (sha256 `feee6020...0bda7`).
- `raw-counters.json`: every repetition's raw counters (full `_debugmallocstats` text, parsed size
  classes and totals, all three `--stats` readings and their deltas, VmHWM readings, exit statuses,
  bytecode checks), the stage hashes, the supplementary probes' parsed readings, and the
  cross-check boots' `CAPSTONE_VM_STATS` fields, `time -v` values (printed and divided by 4),
  `--stats` readings and full `_debugmallocstats` text.
