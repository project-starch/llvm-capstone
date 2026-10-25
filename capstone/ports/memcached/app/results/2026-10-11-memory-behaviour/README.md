# memcached 1.6.45: memory behaviour with and without patch 0006, 2026-10-11

The memcached row of the pre-registration
[`docs/plans/2026-10-11-protected-memory-behaviour.md`](../../../../../docs/plans/2026-10-11-protected-memory-behaviour.md)
(pushed as `82c1607119ec` before any build or run below). Two arms, three repetitions each, every
repetition its own ephemeral one-hart VM, the six VMs in parallel; three native repetitions on the
build host as the reference. `repetitions.json` holds every repetition's raw counters; `driver/`
holds every script and the stats client the runs used. No serial log is kept here.

## What ran

**Source.** One snapshot of `82c1607119ec` (`git archive` of `capstone/`, plus
`compiler-rt/lib/builtins` from `9bc6f9adf1c9`), committed in a scratch repository on the build
host. Both arms come from one `host/build-virtual.sh` run (`driver/build-mc.sh`, `JOBS=16`): wall
time **436 s** for the whole recipe (libevent with its native `make verify`, the plain, marker,
safety and native builds, and `host/build-sublet.sh`), in parallel with the mruby row's build.
libevent's native check failed `util/monotonic_prc_fallback` once and passed it alone 6 of 6, the
recipe's own rule (loaded host).

| arm | image | sha256 | CDERIVE / CREVOKE words in `.text` |
|---|---|---|---|
| stock | `domain/memcached.dom` | `c070a61b05f87b98bb7e916befa943d21dcb79e3c168309625649c3090696ccb` | 0 / 0 |
| protected | `sublet/memcached.dom` (`-DMC_CAPSTONE_SUBLET`, patch 0006) | `0c76ef0615c6c061014533787c84180b348225a6ca9edf0e1b92e5eacfefaba4` | 18 / 5 |
| native | `native/bin/memcached` (host gcc) | `e00f86eb4ad82e8354450bd239b0c9a8943c8a3ca8a9130537292ff66546f071` | -- |

Manifest: the same configured tree for both Capstone images (patches 0001-0006 applied to both;
0006 compiles to nothing without its define), the virtual SDK of `deps/env.sh` key
`1cbd443b6104146d` (`CAPSTONE_APPLICATION_VIRTUAL=ON`, musl mallocng, arena 64 MiB, data 32 MiB),
compiler `clang-22` sha256 `4c6135fe...`. Patch sha256 prefixes: 0001 `bbc8a297f05f19b6`, 0002
`66c14e4e8124985c`, 0003 `d0762121af3c024f`, 0004 `62d8344a1333de47`, 0005 `75b9b63fea562fe5`,
0006 `1c26d4659940635d`. `build-sublet.sh`'s gates passed (instruction words in both Sublet
images, none in the plain ones; links clean). The instruction counts above were re-read with
`driver/insn-count.py`.

**Platform** (sha256 of the files booted, recorded per repetition): QEMU `af37cc32...`, launcher
`capstone-vexec` `3975314a...`, module `capstone_vm.ko` `a1c6cb6b...`, `capstone-job`
`fb6192b8...`, kernel `e833dfb1...`, firmware `44cd6cbc...`, rootfs `62049800...`; exact bounds,
one hart (`runtime/virtual/run-staged.py` from the snapshot). These are the platform hashes of the
pre-registration and of the same-answers runs.

**Workload and readings** (`driver/mm-run.py`, one boot per repetition, gate sha256
`ed090b02a445125b...`, the same for all six boots):

1. `capstone-vexec --stats` (reading `before`).
2. The server under `capstone-job --user 65534:65534` with the oracle gate's flags
   `-l 127.0.0.1 -p 21299 -U 0 -m 64 -t 4` and `CAPSTONE_VM_STATS=1`.
3. The unchanged protocol oracle `mc-harness` (source sha256 `1e569a8b...`, the prebuilt static
   guest binary `63cf0442...` that commit `bd6a7dcac627`'s server gate used) runs its four phases
   against it: phase 1 every command family, phase 2 values of 1 B to 900 kB (chunked items) and
   one too large, phase 3 eight concurrent connections of 40 set/get pairs, phase 4 `stats`. Its
   own SERVER-COMMAND is a `sleep` it stops at the end, so the real server stays up.
4. The launcher process (the only child of `capstone-job`, pid confirmed per run) is read from
   `/proc/<pid>/status`: `VmHWM`, `VmRSS`, `VmPeak` (`after_workload`).
5. `capstone-vexec --stats` (`during`).
6. `driver/mc-stats.c` (static guest build sha256 `0755e28c...`, host gcc 12.3 riscv64 glibc)
   sends `stats`, `stats slabs`, `stats items`. The process is read again (`after_stats`) and
   `--stats` again (`during2`).
7. TERM to `capstone-job`; it forwards it; the server exits 0 (`job.json` `exit 0` every run).
8. `capstone-vexec --stats` (`after`).

Native: `driver/mm-native.sh memcached`, the same sequence on the build host (port 23917), host
builds of `mc-harness` and `mc-stats` from the same sources, three repetitions one after another.

## Sanity: the workload's output

The oracle's normalised transcript is byte-identical in all nine runs: sha256
`ee1522c8e0eceec97a20b0372e888be774e019178e3a3c45cead03663d934651` for stock 1-3, protected 1-3 and
native 1-3. It is also the transcript of the earlier server gate's native and stock runs (the
`native-null1` and `normal1` transcripts of that gate carry the same hash).

## Allocator statistics (`stats`, `stats slabs`, `stats items`)

Every reading below was identical in all three repetitions of its arm; where native is shown it
was identical in its three runs too.

| statistic | stock | protected | native | prediction | verdict |
|---|---:|---:|---:|---|---|
| `curr_items` | 328 | 328 | 328 | equal | holds |
| `total_items` | 341 | 341 | 341 | equal | holds |
| `cmd_set` / `cmd_get` | 345 / 342 | 345 / 342 | 345 / 342 | equal | holds |
| `get_hits` / `get_misses` | 335 / 7 | 335 / 7 | 335 / 7 | equal | holds |
| `evictions`, `slabs_moved` | 0, 0 | 0, 0 | 0, 0 | equal | holds |
| `bytes` (sum of item sizes) | 1,947,537 | 1,952,785 | 1,937,041 | -- | protected +5,248 = 328 items x 16 B: one capability per item header |
| `active_slabs` | 9 | 9 | 10 | -- | equal between arms |
| `total_malloced` | 9,437,184 | 9,437,184 | 10,485,760 | -- | equal between arms: nine 1 MiB slab pages |
| `pointer_size` | 128 | 128 | 64 | -- | |

Per slab class (`stats slabs`). `stats items` counts 321, 1, 1, 1, 1, 1 and 2 items in classes 1,
3, 10, 20, 31 (stock) or 30 (protected), 35 and 37 on both arms; the used chunks of classes 29 and 36
have no item of their own, which fits their being the tails of the two chunked values (600 kB and
900 kB, whose headers are the two items of class 37); this reading is inferred, not traced.
Each cell: chunk size / total chunks / used chunks; every class used has `total_pages` 1.

| class | stock | protected | native |
|---:|---|---|---|
| 1 | 128 / 8,192 / 321 | 144 / 7,281 / 321 | 96 / 10,922 / 321 |
| 2 | -- | -- | 120 / 8,738 / 2 |
| 3 | 208 / 5,041 / 3 | 240 / 4,369 / 3 | -- |
| 4 | -- | -- | 192 / 5,461 / 1 |
| 10 | 1,104 / 949 / 1 | 1,200 / 873 / 1 | -- |
| 12 | -- | -- | 1,184 / 885 / 1 |
| 20 | 10,560 / 99 / 1 | 11,360 / 92 / 1 | -- |
| 22 | -- | -- | 11,104 / 94 / 1 |
| 29 | 78,848 / 13 / 1 | 84,768 / 12 / 1 | -- |
| 30 | -- | **105,968 / 9 / 1** | -- |
| 31 | **123,200 / 8 / 1** | -- | 82,792 / 12 / 1 |
| 32 | -- | -- | 103,496 / 10 / 1 |
| 35 | 300,800 / 3 / 1 | 323,440 / 3 / 1 | -- |
| 36 | 376,000 / 2 / 1 | 404,304 / 2 / 1 | -- |
| 37 | 524,288 / 2 / 2 | 524,288 / 2 / 2 | 315,872 / 3 / 1 |
| 38 | -- | -- | 394,840 / 2 / 1 |
| 39 | -- | -- | 524,288 / 2 / 2 |

The movement, as predicted it might: the item header grows by one capability (16 B), so the
smallest chunk is 144 B instead of 128 B and the whole size-class ladder (factor 1.25) shifts; every
class's chunk size is larger and its chunks per 1 MiB page fewer. Eight of the nine occupied classes
keep their index and their occupancy. One item moves: the 100,000-byte value of phase 2 sits in
class 31 (123,200 B chunks) on stock and in class 30 (105,968 B chunks) on protected, because the
shifted ladder has a class between 100,000 B plus header and the next stock class. Pages and
`total_malloced` do not move. The 64-bit native server has its own ladder (96 B first chunk) and is
shown for orientation only.

## Lifetime counters (`capstone-vexec --stats`)

Identical in all three repetitions of each arm (`--stats` is module-wide; nothing else ran in the
VM). The `before` reading was the module's initial state in every run (all counters 0, node
capacity 1,024).

| counter | stock | protected | protected - stock |
|---|---:|---:|---:|
| `nodes_allocated_total`, whole run (`after` - `before`) | 868 | 1,408 | +540 |
| ... of which by the end of the oracle workload (`during`) | 820 | 1,355 | +535 |
| ... during the three stats commands | 39 | 44 | +5 |
| `nodes_live` at the end of the workload | 739 | 1,125 | +386 |
| `nodes_retired` at the end of the workload | 81 | 230 | +149 |
| `collections`, whole run | 0 | 0 | 0 |
| `nodes_reclaimed`, whole run | 0 | 0 | 0 |
| `nodes_live` / `nodes_retired` after the server exited | 0 / 0 | 0 / 0 | |
| `tag_pages` at the end of the workload | 4,110 | 4,112 | +2 |
| `live_regions` / `live_bytes` at the end of the workload | 100 / 16,834,560 | 100 / 16,842,752 | 0 / +8,192 |
| node table (`node_capacity` / `node_bytes`) at the end of the workload | 6,144 / 122,880 | 6,144 / 122,880 | 0 |

The launcher's own exit line (`CAPSTONE_VM_STATS`, per process) agrees: `nodes` 868 vs 1,408,
`collections` 0 and `reclaimed` 0 on both, `pages` 4,115 vs 4,117, `peak` 4,118 vs 4,120 pages,
`faults` 3,270 vs 3,272, `arenas` 100 on both, `node_growths` 1 on both. `steps`, `rounds` and
`bytes_out` vary between repetitions of the same arm (6,657-6,932 steps; 193 or 197 rounds) by the
same amounts on both arms.

## Peak resident memory

`VmHWM` of the launcher process, whose address space holds the domain's arenas, read from
`/proc/<pid>/status` while the server was alive. The pid is the one child of `capstone-job`; the
launcher's spawner is the launcher's own child (`runtime/linux/spawner.c` clones it), so it is not
the process read. The values (19 MB) are of the right order for the 16.8 MB of registered arenas
the module reports, not the spawner's ~120 kB.

| | stock (rep 1, 2, 3) | protected (rep 1, 2, 3) | native (rep 1, 2, 3) |
|---|---|---|---|
| `VmHWM` after the workload, kB | 18,944 / 18,896 / 18,948 | 18,948 / 19,028 / 19,168 | 14,272 / 14,256 / 14,252 |
| `VmHWM` after the stats commands, kB | 18,988 / 18,952 / 18,996 | 18,988 / 19,076 / 19,212 | 14,284 / 14,268 / 14,264 |

Not identical across repetitions (spread up to 52 kB on stock, 220 kB on protected); the
protected range lies at or above the stock range, by 0-220 kB. With N = 3 and this spread the
difference is not resolved by `VmHWM`. The module's page counts resolve it: +2 pinned pages
(8 KiB), the same in every repetition.

## Verdicts

| prediction (pre-registration) | verdict |
|---|---|
| Items stored and retrieved equal | **holds**: `curr_items` 328, `total_items` 341, `get_hits` 335, `get_misses` 7, all command counts equal, every repetition, and equal to native |
| Per-class chunk counts may move between classes | **holds, with the movement measured**: the class ladder shifts (first chunk 128 -> 144 B), chunks per page fall in every class, one item (the 100,000 B value) moves from class 31 to class 30; occupancy of the other nine classes, pages (9) and `total_malloced` (9 MiB) are equal |
| Decisions equal, every repetition (general rule) | **holds** for counts of items, commands, pages and per-class occupancy except the one item above, which the memcached row named in advance |
| Footprint differs only by the documented metadata | **holds** on what is measured: `bytes` +16 B per item; +2 tag pages; `VmHWM` difference unresolved within its repetition spread |
| Protected allocates more lifetimes | **holds**: 1,408 vs 868 per run (+540) |
| ... and reclaims them | **does not hold as a collection**: no collection ran on either arm (`collections` 0, `nodes_reclaimed` 0); protected ends the workload with 230 retired lifetimes (stock 81) that stay unreclaimed while the server lives, and all lifetimes are released at process exit, which the counters do not count as reclaimed |
| Readings identical across repetitions | **yes** for every allocator statistic and every lifetime counter; **no** for `VmHWM` (ranges above) and the launcher's `steps`/`rounds`/`bytes_out` |

## Method limits

- `--stats` is module-wide; the readings are attributable to the server because nothing else
  that uses the module ran in the VM (mc-harness and mc-stats are ordinary Linux programs).
- The stats commands are part of the run: they add 39 (stock) / 44 (protected) lifetimes and
  20 KiB of `live_bytes`; the workload-only numbers are the `during` column.
- `VmHWM` counts what Linux maps for the launcher; the module's pinned tag pages are counted
  separately (`tag_pages`, `pages`).
- One workload (the oracle's script, ~340 items). It never fills a slab page, so it cannot show
  evictions, slab rebalancing or LRU behaviour under pressure.
- A first launch of the six boots is void: `mc-stats` was staged without its execute bit and
  exited 126 in every run, so no `stats` reply was recorded. Its lifetime counters were the same
  as the valid runs' (`during` 820 vs 1,355). The six boots were rerun unchanged otherwise; only
  the rerun is reported and kept.
- The pre-registration says images are those of the same-answers runs where they exist; the
  protected image here (`0c76ef06...`) is a fresh build from the snapshot this measurement was
  told to build from, not that run's `ddcdea85...`.
- Boot wall time per repetition: 23-24 s (launch to `VIRTUAL_STAGED_DONE`), 30-32 s per job.
