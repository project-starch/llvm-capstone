# FFmpeg 9.0.1: protected pools, same memory behaviour? (2026-10-11)

The FFmpeg row of the pre-registration
[`docs/plans/2026-10-11-protected-memory-behaviour.md`](../../../../../docs/plans/2026-10-11-protected-memory-behaviour.md)
(82c1607119ec, pushed before this run). Workload: the decoder application on the run-of-record
clip, `capstone-vexec ffmpeg.dom input.mkv 1` (30 frames), as in
[`../2026-10-11-virtual-pools/`](../2026-10-11-virtual-pools/README.md). Arms: **stock**
(`vstock`, FFmpeg's pools as released) and **protected** (`vpools`, the buffer-pool patch 0003
in libavutil). N = 3 per arm, one VM per repetition, all 12 VMs of this and the Perl row booted
in parallel.

Registered predictions for this row: allocator statistics none exposed; footprint difference
small (pool entries gain two pointers), its size measured; protected allocates more lifetimes
than stock, about one per nested-allocator hand-out, and reclaims them.

## Outcome

| | stock (3 reps) | protected (3 reps) | protected - stock |
|---|---|---|---|
| allocator statistics | not exposed | not exposed | -- |
| **VmHWM** (kB) | 7,276 (all 3) | 7,272 (all 3) | **-4** |
| ru_maxrss at exit (kB, cross-check) | 7,276 (all 3) | 7,272 (all 3) | equal to VmHWM in every run |
| VmHWM reached, s after start | 6.49-6.70 | 6.49-7.05 | -- (decode phase) |
| VmRSS max (kB, sampled) | 7,264 (all 3) | 7,260-7,268 | -4 to +4 |
| **lifetimes allocated** (`nodes_allocated_total`) | 2,436 (all 3) | 2,761 (all 3) | **+325** (x1.13) |
| **lifetimes reclaimed** (`nodes_reclaimed`) | 0 (all 3) | 0 (all 3) | 0 |
| collections | 0 | 0 | 0 |
| `nodes_high_water` | 2,436 (all 3) | 2,761 (all 3) | +325 |
| `nodes_live`, `nodes_retired`, `live_domains`, `live_regions`, `live_bytes`, `tag_pages`, `node_capacity`, `node_bytes`, `cached_bytes`, `poisoned_blocks` (delta) | 0 | 0 | 0 |
| lifetime table at exit (launcher, bytes / capacity / growths) | 94,208 / 5,120 / 1 | 94,208 / 5,120 / 1 | 0 |
| domain peak pinned pages (launcher, 4 KiB) | 1,659 (all 3) | 1,658 (all 3) | -1 |
| domain arenas | 13 | 13 | 0 |
| exit, frames | 0, 30 frames equal native (all 3) | 0, 30 frames equal native (all 3) | equal |

Deltas are `capstone-vexec --stats` after minus before in the same VM; each VM loads the module
fresh, so "before" is zero for every counter except the idle `node_capacity` 1,024 /
`node_bytes` 28,672, which are equal before and after. Raw readings of every repetition:
[`results.json`](results.json).

## Verdicts

- **Allocator statistics: not exposed** (as registered).
- **Footprint difference small: holds; size -4 kB.** VmHWM is 7,276 kB on stock and 7,272 kB on
  protected, identical within each arm: protected is 4 kB (0.05 %) lower, not higher. The
  domain's peak pinned memory differs by one page (1,659 against 1,658), in the same direction.
  The two pointers each pool entry gains do not show at this page granularity; the measured
  difference is one page, of the opposite sign to the added metadata.
- **Protected allocates more lifetimes: holds.** 2,761 against 2,436, +325, identical in every
  repetition. FFmpeg exposes no count of pool hand-outs, so "about one per hand-out" is not
  checked.
- **Protected reclaims them: does not hold on this workload.** No collection ran on either arm
  (`collections` 0, `nodes_reclaimed` 0); `nodes_high_water` equals `nodes_allocated_total` on
  both, so no lifetime slot was reused during the decode. The protected arm's extra lifetimes
  were dropped with the domain's table at exit, not reclaimed by the processor.
- **Output check: holds.** Exit 0 and 30 frames, every frame hash equal to the native reference
  (native.out 3eebf311c9fa...), on all six runs. stdout's other lines (`STAGE M1`-`M5`,
  `EXP-OK ffmpeg 1`) are identical on all six.

**Repetitions identical?** Within each arm: VmHWM, lifetimes allocated, high water, collections,
reclaimed, table size, peak pinned pages and the frames are identical in all three repetitions.
Only the sampled VmRSS maximum varies on protected (7,260-7,268 kB), which is a sampling-time
effect.

## What ran

| | sha256 |
|---|---|
| stock image (`vstock/image/ffmpeg.dom`) | 385f958631d81b528f255b95549ce80a4940b945898761ddbd21b23998c20351 |
| protected image (`vpools/image/ffmpeg.dom`) | 6993865e22c4907a4f4d2f36677501b67088af47f2ed394aae3ddb0236427110 |
| input.mkv | 7c0964440ad178a0c9a00dba72060322e68bea340715bb247e05d6a815844362 |
| native.out (frame reference) | 3eebf311c9fae322ef74b8a9bad371eda975852299d620bbf3b3bd7ef466f43a |

The images are those of `../2026-10-11-virtual-pools/` (same hashes as its `decode-result.json`),
rehashed on the host before the boot and again in each guest. Only the run-of-record clip ran;
the flipped control of that run was not repeated.

Platform (the registered one: Sublet-lifetime QEMU af37cc32, launcher 3975314a, module a1c6cb6b,
exact bounds, one hart, 2 GiB):

| | sha256 |
|---|---|
| qemu-system-riscv64 | af37cc3269d1cfb47a72eb79cc3dfd2397848c613ca703e43625c68ab50ad27d |
| fw_jump.elf | 44cd6cbc2d225c2fea6c6000ea0b9eab83bc34e769a7c95c036cf17b6fe0f116 |
| Image | e833dfb12bccf8a39bd6a81e8e42e4622f0e76bcc6cc09624a1fbe7198c18a4b |
| rootfs.ext2 | 62049800ed0fa8827959c0060bbd74f0435aacb577dd521e14a14a099d167845 |
| capstone-vexec | 3975314a7a6da119c4bff6a0ab0c601fcd96d305f59a0a68fe5b36b9bd544ad4 |
| capstone_vm.ko | a1c6cb6b79a98cf1f09e55a37adf2320e2b9787922b7e10bf4088c27e396e512 |
| run-staged.py (snapshot of 9bc6f9adf1c9) | d6770b8b9c55ef8a4e136a17f5411a2fe2f139af9fee47f5e02328810acfc54c |

Script and parser (shared with the Perl row):
[`run.sh`](../../../../perl/musl/results/2026-10-11-memory-behaviour/run.sh),
[`parse.py`](../../../../perl/musl/results/2026-10-11-memory-behaviour/parse.py). Each VM boots
through `runtime/virtual/run-staged.py --timeout 3600` with its own `CAPSTONE_QEMU_LOCK` and a
fresh `--work`. In the guest: `insmod`, `capstone-vexec --stats`, then from `/mnt/vm`

    CAPSTONE_EXEC_DIAGNOSTICS=1 CAPSTONE_VM_STATS=1 \
      time -o /tmp/mb.time -v sh wrap.sh /mnt/vm/capstone-vexec /mnt/vm/prog.dom /mnt/vm/input.mkv 1 &

(`wrap.sh` records its pid and `exec`s, so the sampled pid is the `capstone-vexec` process),
the sampler below until the process is gone, `capstone-vexec --stats` again, `rmmod` (exit 0 in
all runs). Frames compared on the host with the regex of `ffapp-decode.sh`.

## Method and its limits

- **VmHWM**: a guest shell loop reads `VmHWM` and `VmRSS` from `/proc/<pid>/status` of the
  `capstone-vexec` process with shell builtins, then `usleep 100000`. Effective interval: median
  0.34-0.35 s, 29-30 samples per run. `VmHWM` is the kernel's high-water mark, so the last read
  covers all earlier peaks; the last 0.2-1.9 s of each run went unsampled. `ru_maxrss` from
  `wait4` (busybox `time -v`) equals the last sampled `VmHWM` to the kB in all six runs, so
  nothing after the last read raised the peak.
- **busybox `time` unit**: busybox prints "Maximum resident set size" as 4x `ru_maxrss` (it
  converts the value as if it counted 4 KiB pages). The printed value is exactly 4x the sampled
  VmHWM in all 12 runs and in a probe boot; `results.json` keeps the printed value and the
  corrected one.
- **What VmHWM covers**: the `capstone-vexec` process, i.e. the domain's memory plus the
  launcher's 2 MB image mapping. Not covered: the launcher's spawner helper (a separate process;
  116 kB in a diagnostic boot of the stock image) and the lifetime table, which the module
  allocates from kernel pages. Page granularity (4 KiB) bounds what VmHWM can show of a
  two-pointer-per-entry change.
- **`--stats` after the run** sums only live contexts plus the running totals of finished ones
  (allocated, collections, reclaimed). A finished domain's live/retired counts and table size
  therefore read 0 after; the table size above comes from the launcher's exit line
  (`CAPSTONE_VM_STATS=1`, not registered, stderr only). stderr also carries two `MEMPHASE` lines
  (`before-0`, `released-0`), identical on both arms.
- **Wall times** (`time -v` elapsed): stock 12.1-12.2 s, protected 12.0-12.7 s; launcher launch
  1.3-1.4 s on both. Boots (run-staged.py start to end): 21 s each; the whole batch of 12 VMs
  71 s. Not a performance comparison: 12 VMs plus other sessions' QEMUs on one host, and the
  sampler shares the guest's hart.

## Deviations

- Two boots that are not measurements: a probe before the run (`time -v`, `usleep`, `--stats`,
  the sampler) and a diagnostic boot of the stock image afterwards with an all-process `/proc`
  scan (to find the source of the 4x `ru_maxrss`). No number above comes from either.
- `CAPSTONE_VM_STATS=1` added to the registered environment (launcher's exit counters on
  stderr; stdout unchanged).
- N = 3 per arm as registered. Raw serial logs not committed.
