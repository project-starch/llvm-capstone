# mruby 4.0.0-rc2: memory behaviour with and without patch 0008, 2026-10-11

The mruby row of the pre-registration
[`docs/plans/2026-10-11-protected-memory-behaviour.md`](../../../../../docs/plans/2026-10-11-protected-memory-behaviour.md)
(pushed as `82c1607119ec` before any build or run below). Two arms, three repetitions each, every
repetition its own ephemeral one-hart VM, the six VMs in parallel; three native repetitions on the
build host as the reference. `repetitions-ao-width16.json` holds every repetition's raw counters;
`driver/` holds every script the runs used. No serial log is kept here.

**Deviation from the registered workload.** The registered workload is the AO render benchmark
as shipped (`benchmark/bm_ao_render.rb`, 64 x 64 pixels). On this platform it does not finish in
the time available: natively it takes 6.75 s (three runs, 6.74-6.78 s), and guest probes took 36 s
at 4 x 4 and 135 s at 8 x 8, about 1,300 times native, which puts 64 x 64 at about 2.4 h per
boot. The numbers below are for the **same script with `IMAGE_WIDTH` = `IMAGE_HEIGHT` = 16** (its
own `ARGV[0]`; `NSUBSAMPLES` 2 and `NAO_SAMPLES` 8 unchanged), 16 times fewer pixels, 0.42 s
natively. The full-size runs (64 x 64) are still running on six further VMs; their result will be
an addendum to this file. The GC smoke ran unchanged.

## What ran

**Source.** One snapshot of `82c1607119ec` (`git archive` of `capstone/`, plus
`compiler-rt/lib/builtins` from `9bc6f9adf1c9`), committed in a scratch repository on the build
host. `driver/build-mruby.sh` builds one virtual application SDK (`CAPSTONE_APPLICATION_PROFILE=virtual`,
musl mallocng, arena 64 MiB, data 32 MiB; 21 s), copies it twice, and runs
`build-mruby-domain.sh` for both arms in parallel against the copies (`MRBD_PIN=4.0.0-rc2`, mruby
`9d523e2f74f2`, patch set sha256 `bd250b03...`, `JOBS=16`): stock 117 s, protected 116 s, all
138 s; 124 Capstone objects each, none failed. The memcached row's build ran at the same time.

| arm | image | sha256 | CDERIVE / CREVOKE words in `.text` |
|---|---|---|---|
| stock | `stock/src/mruby/build/capstone/bin/mruby` | `7b5872bc7141cbadd9e1bf3057e025ad25040ad6a50542ada1214f899c172a61` | 0 / 0 |
| protected | `prot/.../bin/mruby` (`MRBD_SUBLET=1`, patch 0008) | `321dc9b98691cc0abf177da42f335fdc58cbb2e97aa2138592b59313975ac0da` | 8 / 2 |
| native | `stock/.../build/native/bin/mruby` (host gcc) | `905ff685f8edba81b047b964b34066eeb3ca0bc3108475ea8abb0dd4eb6076a3` | -- |

Both Capstone images are byte-identical to the images of the 2026-10-11 release-corpus run
(`bug-corpora/mruby/release-differential/results/2026-10-11-virtual`: `7b5872bc7141cbad` stock,
`321dc9b98691cc0a` protected), so these are the same-answers images. The instruction counts were
read with `driver/insn-count.py`. Compiler `clang-22` sha256 `4c6135fe...`.

**Platform** (sha256 of the files booted, recorded per repetition): QEMU `af37cc32...`, launcher
`capstone-vexec` `3975314a...`, module `capstone_vm.ko` `a1c6cb6b...`, kernel `e833dfb1...`,
firmware `44cd6cbc...`, rootfs `62049800...`; exact bounds, one hart
(`runtime/virtual/run-staged.py` from the snapshot). These are the pre-registration's hashes.

**Workload and readings** (`driver/mm-run2.py` via `driver/launch2.sh` with
`MM_AO_ARGS=16`, gate sha256 `548319825cc6bbe0...`, the same for all six boots). Per boot, two
programs, each as `CAPSTONE_VM_STATS=1 ./capstone-vexec ./mruby ./mstats.rb WORKLOAD [ARGS]`:

1. `ao`: `bm_ao_render.rb 16` (sha256 `e44efffd...`, the pin's own file).
2. `smoke`: the port's `scripts/smoke.rb` (sha256 `16718df7...`, 13 checks, M6 builds 20,000
   three-object entries and calls `GC.start`), with a writable directory under `/tmp`. This is the
   pre-registration's "GC smoke".

`driver/mstats.rb` (sha256 `ffa49482...`) reads the workload, prints
`ObjectSpace.count_objects` before it, right after it, and after a `GC.start`, all to stderr, and
leaves stdout to the workload. `capstone-vexec --stats` is read before and after each program. The
launcher process is the shell's `$!` (the background command execs `capstone-vexec`; its spawner
is a clone the launcher makes, `runtime/linux/spawner.c`, so a different pid); its
`/proc/<pid>/status` is sampled every 0.5 s. The reading reported is the one taken after the
driver printed `MM_DONE` and while it sleeps 4 s (`done`); the last sample before exit is not
usable (it caught the process in exit, without memory lines) and is not reported.
`CAPSTONE_VM_STATS=1` makes the launcher print its own counters at exit.

Native: the same `mstats.rb` with the native interpreter on the build host, three runs each
(`driver/mm-native.sh mruby` for the smoke, `driver/native-ao16.sh` for AO at 16 x 16).

## Sanity: the workload's output

| output | stock 1-3 | protected 1-3 | native 1-3 |
|---|---|---|---|
| AO 16 x 16 PPM (stdout) | `cfd73bb1...` | `cfd73bb1...` | `cfd73bb1...` |
| smoke stdout (`M1`-`M13`, `SMOKE_DONE`) | `791d0649...` | `791d0649...` | `791d0649...` |

Every program exited 0. Full hashes in the JSON.

## Allocator statistics (`ObjectSpace.count_objects`)

Every value identical in all three repetitions of both arms **and** in the three native runs.

| reading | `TOTAL` slots (heap pages of 1,024) | `FREE` | `T_PROC` | `T_SCLASS` | `T_CLASS` | `T_ENV` | `T_MODULE` | `T_STRING` | others |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| AO, before | 1,024 (1) | 295 | 484 | 93 | 77 | 14 | 10 | 24 | `T_ARRAY` 1, `T_DATA` 3, `T_EXCEPTION` 2, `T_HASH` 1, `T_ICLASS` 17, `T_ISTRUCT` 1, `T_OBJECT` 2 |
| AO, after | 2,048 (2) | 1,256 | 531 | 101 | 83 | 15 | 11 | 24 | same |
| AO, after `GC.start` | 1,024 (1) | 232 | 531 | 101 | 83 | 15 | 11 | 24 | same |
| smoke, before | 1,024 (1) | 295 | 484 | 93 | 77 | 14 | 10 | 24 | same |
| smoke, after | 2,048 (2) | 1,315 | 487 | 93 | 77 | 15 | 10 | 24 | same |
| smoke, after `GC.start` | 1,024 (1) | 291 | 487 | 93 | 77 | 15 | 10 | 24 | same |

GC runs: **not measured**. mruby 4.0.0-rc2 has no `GC.stat` or GC count (its `GC` module offers
`start`, `enable`, `disable`, the two ratios and `generational_mode`; the readings were
`interval_ratio` 200, `step_ratio` 200, generational on, on every arm). Heap pages are measured
only at the three readings (`TOTAL` / 1,024); the peak page count during a run is not exposed.

## Lifetime counters

`capstone-vexec --stats`, `after` minus `before` for each program; every counter identical in all
three repetitions of its arm. The launcher's exit line (`CAPSTONE_VM_STATS`) gives the same
`nodes`, `collections` and `reclaimed`, and the per-process page and node-table figures.

| counter | AO stock | AO protected | smoke stock | smoke protected |
|---|---:|---:|---:|---:|
| lifetimes allocated (`nodes_allocated_total`) | 2,443,922 | 3,360,624 (+916,702, +37.5 %) | 131,762 | 253,213 (+121,451, +92.2 %) |
| reclaimed by collections (`nodes_reclaimed`) | 2,440,800 | 3,341,341 | 123,040 | 74,698 |
| collections | 258 | 244 | 2 | 4 |
| `nodes_live` / `nodes_retired` after exit | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| node table at exit (`node_capacity`) | 17,408 | 25,600 | 99,328 | 197,632 |
| node table bytes (`node_bytes`) | 290,816 | 421,888 | 1,601,536 | 3,178,496 |
| node-table growths | 4 | 6 | 24 | 48 |
| pages pinned at exit / peak (`pages` / `peak`) | 1,748 / 2,072 | 1,748 / 2,100 | 3,560 / 6,595 | 3,553 / 6,942 |
| capability faults served (`faults`) | 1,806 | 1,878 | 5,759 | 6,106 |
| `arenas` | 15 | 15 | 18 | 18 |

The rest of each process's lifetimes (AO: 3,122 stock, 19,283 protected; smoke: 8,722 stock,
178,515 protected) were neither live nor retired after exit: released with the process, which
`nodes_reclaimed` does not count. `steps`, `rounds`, `bytes_in`, `bytes_out` and the times vary
between repetitions of one arm (e.g. AO `steps` 143,029-143,811 stock, 146,201-148,039 protected).

## Peak resident memory

`VmHWM` of the launcher process at the `done` reading (after the workload and the final
`GC.start`, before exit). Identical to the kB in all three repetitions of each arm.

| | stock | protected | protected - stock | native |
|---|---:|---:|---:|---:|
| AO 16 x 16, `VmHWM` kB | 8,992 | 9,104 | +112 | 4,368 / 4,376 / 4,396 |
| smoke, `VmHWM` kB | 27,000 | 28,436 | +1,436 | 9,244 / 9,244 / 9,252 |

The module's peak pinned pages move by the same order: AO +28 pages (112 KiB, equal to the
`VmHWM` difference), smoke +347 pages (1,388 KiB). Patch 0008 adds `life` and `issued[1024]`
(16 B each) to every heap page, 16,400 B per page; the smoke's M6 holds 60,000 objects at once,
some 60 pages, so about 1 MB of the smoke's difference is of the documented kind. That split is an
estimate, not a measurement: the peak heap-page count is not exposed, and the protected arm's
larger lifetime table (above) is in the same address space.

## Verdicts

| prediction (pre-registration) | verdict |
|---|---|
| Object counts per type equal | **holds**: every type's count equal at all three readings of both programs, every repetition, and equal to native |
| Heap pages equal | **holds** at the readings (1 / 2 / 1 pages for both programs on both arms and natively); the peak during a run is not measured |
| GC runs equal | **not measured**: no GC run count is exposed by this mruby |
| Heap memory higher on protected (`issued[]` per page) | **holds**: `VmHWM` +112 kB (AO) and +1,436 kB (smoke), every repetition; peak pinned pages +28 and +347 |
| Protected allocates more lifetimes | **holds**: +916,702 (AO 16 x 16) and +121,451 (smoke) per run |
| ... and reclaims them | **holds for AO** (3,341,341 reclaimed by 244 collections, 99.4 % of those allocated); **partly for the smoke**: protected reclaimed 74,698 by collections, fewer than stock's 123,040, and released 178,515 only at exit |
| Readings identical across repetitions | **yes** for every object count, every lifetime counter, `pages`/`peak` and `VmHWM`; **no** for the launcher's `steps`, `rounds`, `bytes_in`/`bytes_out` and times |

Two findings beyond the predictions: the protected AO run needs **fewer** collections (244 vs
258) because its lifetime table grew larger (25,600 vs 17,408 nodes), and the protected smoke
doubles the lifetime table (48 vs 24 growths) and reclaims fewer lifetimes before exit.

## Method limits

- The registered 64 x 64 AO render was replaced by 16 x 16 (above). The full-size runs are still
  in progress; nothing here is from them.
- A first attempt at the full-size runs (six boots under `driver/mm-run.py`, the same gate
  without the width argument and without `CAPSTONE_VM_STATS`) had a 3,600 s boot timeout, which
  the probe showed it could not meet; those six VMs were stopped after about 41 min with no
  reading and are void. The current full-size runs use `driver/mm-run.py` with
  `MM_TIMEOUT=14400` and `driver/launch.sh`.
- `--stats` is module-wide; each program ran alone in its VM, so the deltas are its own.
- `VmHWM` counts what Linux maps for the launcher. Whether the module's node table is in it is not
  separated here; `node_bytes` is listed apart.
- The sampler forks a few busybox processes every 0.5 s on the VM's one hart; that costs time,
  not memory decisions, and the same sampler ran on both arms.
- Boot wall time per repetition: 986-1,005 s stock, 1,017-1,032 s protected (launch to
  `VIRTUAL_STAGED_DONE`, both programs).
