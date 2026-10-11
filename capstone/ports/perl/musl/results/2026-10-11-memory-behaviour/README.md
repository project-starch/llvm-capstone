# Perl 5.36.3: protected build, same memory behaviour? (2026-10-11)

The Perl row of the pre-registration
[`docs/plans/2026-10-11-protected-memory-behaviour.md`](../../../../../docs/plans/2026-10-11-protected-memory-behaviour.md)
(82c1607119ec, pushed before this run). Workload: `scripts/smoke.pl` at 2,000 objects
(`CAPSTONE_PERL_SMOKE_OBJECTS=2000`). Arms: **stock** (virtual image, Perl's SV arenas as
released) and **protected** (`PERLD_SUBLET=1`, patch `sublet/patches/5.36.3/0001`). N = 3 per
arm, one VM per repetition, all 12 VMs of this and the FFmpeg row booted in parallel.

Registered predictions for this row: allocator statistics none exposed; `VmHWM` higher on
protected by the arena trailer, one capability per SV head; protected allocates more lifetimes
than stock, about one per nested-allocator hand-out, and reclaims them.

## Outcome

| | stock (3 reps) | protected (3 reps) | protected - stock |
|---|---|---|---|
| allocator statistics | not exposed | not exposed | -- |
| **VmHWM** (kB) | 29,116 / 29,116 / 29,116 | 29,084 / 29,148 / 29,148 | -32 to +32 |
| ru_maxrss at exit (kB, cross-check) | 29,116 (all 3) | 29,084 / 29,148 / 29,148 | equal to VmHWM in every run |
| VmHWM reached, s after start | 3.27-3.35 | 3.24-3.44 | -- (inside the launch, see below) |
| VmRSS max after the launch drop (kB, sampled) | 26,140-26,176 | 26,392-26,420 | +216 to +280 |
| **lifetimes allocated** (`nodes_allocated_total`) | 59,973-60,014 | 140,504-140,541 | +80,490 to +80,568 (x2.34) |
| **lifetimes reclaimed** (`nodes_reclaimed`) | 17,574-17,608 | 89,416-89,542 | +71,808 to +71,968 |
| collections | 1 (all 3) | 3 (all 3) | +2 |
| `nodes_high_water` | 45,822 (all 3) | 74,494 (all 3) | +28,672 |
| `nodes_live`, `nodes_retired`, `live_domains`, `live_regions`, `live_bytes`, `tag_pages`, `node_capacity`, `node_bytes`, `cached_bytes`, `poisoned_blocks` (delta) | 0 | 0 | 0 |
| lifetime table at exit (launcher, bytes / capacity / growths) | 749,568 / 46,080 / 11 | 1,208,320 / 74,752 / 18 | +458,752 bytes |
| domain peak pinned pages (launcher, 4 KiB) | 6,371 / 6,373 / 6,371 | 6,433 (all 3) | +60 to +62 (+240 to +248 KiB) |
| domain arenas | 139 / 140 / 139 | 149 (all 3) | +9 to +10 |
| exit, stdout | 0, sha256 a4863fc7bf3ca182... (all 3) | 0, the same (all 3) | equal |

Deltas are `capstone-vexec --stats` after minus before in the same VM; each VM loads the module
fresh, so "before" is zero for every counter except the idle `node_capacity` 1,024 /
`node_bytes` 28,672, which are equal before and after. Raw readings of every repetition:
[`results.json`](results.json).

## Verdicts

- **Allocator statistics: not exposed** (as registered).
- **VmHWM higher on protected: does not hold.** Stock reads 29,116 kB in all three repetitions,
  protected 29,084, 29,148 and 29,148 kB: -32 to +32 kB, inside protected's own spread of 64 kB.
  The peak is set 3.2-3.4 s after start, inside the launcher's launch phase (its own launch
  time is 5.1-5.2 s, before the domain runs Perl; the image is 20 MB; which part of the launch
  sets the peak is not determined here). VmRSS then falls by about 12 MB and the heap phase never
  climbs back to it. So `VmHWM` on this workload measures the launch, not the
  SV arenas, and cannot show the predicted trailer cost. Two unregistered readings that do cover
  the heap phase move in the predicted direction: VmRSS after the launch drop is 216-280 kB
  higher on protected, and the domain's peak pinned memory is 60-62 pages (240-248 KiB) higher,
  with 9-10 more arenas. They are reported as found; they do not turn the registered verdict.
- **Protected allocates more lifetimes: holds.** 140,504-140,541 against 59,973-60,014,
  +80,490 to +80,568. Perl exposes no hand-out count, so "about one per hand-out" is not
  checked.
- **Protected reclaims them: holds.** 89,416-89,542 reclaimed against 17,574-17,608, in 3
  collections against 1; nothing live or retired is left after the run on either arm.
- **Output check: holds.** Exit 0 and stdout sha256 a4863fc7bf3ca182... (the same-answers smoke
  hash at 2,000 objects, native = stock = protected) on all six runs, computed in the guest and
  again on the host from the serial text.

**Repetitions identical?** Within each arm: `nodes_high_water`, collections, table size and
exit/stdout identical. Not identical: stock lifetimes allocated (59,973-60,014), reclaimed
(17,574-17,608), peak pages (6,371-6,373), arenas (139-140); protected VmHWM (29,084-29,148),
lifetimes allocated (140,504-140,541), reclaimed (89,416-89,542). Stock VmHWM is identical. The
spread is small against the arm difference; its source is not determined here (Perl's hash seed
is randomised per run, which this run did not pin).

## What ran

| | sha256 |
|---|---|
| stock image (`perl-stock.dom`) | e4c27891c419ebd92a157745a7174b98d97c4e5b52098839ca8fe0104d998e7a |
| protected image (`perl-protected.dom`) | 20856abd9cfbb5c40a249f28c06f753bbfe85b370c2bc73f408a74e3c15045b8 |
| smoke.pl | 15bf158879a2f5432b183ed944bd52d928e0e4bbf40a5f93a8d9d0f2c6af1f44 |

The images are those of the same-answers run (`../2026-10-11-same-answers/`, its 2,000-object
follow-up stage), rehashed on the host before the boot and again in each guest.

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

Script: [`run.sh`](run.sh) (generates the 12 stages and gates, boots them in parallel with
`runtime/virtual/run-staged.py --timeout 3600`, a distinct `CAPSTONE_QEMU_LOCK` and a fresh
`--work` per VM); parser: [`parse.py`](parse.py). Per VM, in the guest: `insmod`,
`capstone-vexec --stats`, then

    cd /tmp/run   # tmpfs copy of smoke.pl
    CAPSTONE_PERL_SMOKE_OBJECTS=2000 CAPSTONE_EXEC_DIAGNOSTICS=1 CAPSTONE_VM_STATS=1 \
      time -o /tmp/mb.time -v sh wrap.sh /mnt/vm/capstone-vexec /mnt/vm/prog.dom smoke.pl /tmp/run &

(`wrap.sh` records its pid and `exec`s, so the sampled pid is the `capstone-vexec` process),
the sampler below until the process is gone, `capstone-vexec --stats` again, `rmmod` (exit 0 in
all runs).

## Method and its limits

- **VmHWM**: a guest shell loop reads `VmHWM` and `VmRSS` from `/proc/<pid>/status` of the
  `capstone-vexec` process with shell builtins, then `usleep 100000`. Effective interval on the
  loaded single hart: median 0.36-0.37 s. `VmHWM` is the kernel's high-water mark, so the last
  read covers all earlier peaks. The last 10-11 s of each run (from 0.5 s in protected rep 1) went unsampled (the process is in
  its exit/teardown; in protected rep 1 one read landed there after a 9.9 s stall of the
  sampler). The exit value closes that gap: `ru_maxrss` from `wait4` (busybox `time -v`) equals
  the last sampled `VmHWM` to the kB in all six runs, so nothing after the last read raised the
  peak.
- **busybox `time` unit**: busybox prints "Maximum resident set size" as 4x `ru_maxrss` (it
  converts the value as if it counted 4 KiB pages). The printed value is exactly 4x the sampled
  VmHWM in all 12 runs and in a probe boot; `results.json` keeps the printed value and the
  corrected one.
- **VmRSS** between samples is not seen; "VmRSS max after the launch drop" is the largest sample
  after VmRSS first falls by more than 25 % between two samples.
- **What VmHWM covers**: the `capstone-vexec` process, i.e. the domain's memory plus the
  launcher's image mapping. Not covered: the launcher's spawner helper (a separate process; 116
  kB in a diagnostic boot) and the lifetime table, which the module allocates from kernel pages.
- **`--stats` after the run** sums only live contexts plus the running totals of finished ones
  (allocated, collections, reclaimed). A finished domain's live/retired counts and table size
  therefore read 0 after; the table size above comes from the launcher's exit line
  (`CAPSTONE_VM_STATS=1`, not registered, stderr only).
- **Wall times** (`time -v` elapsed): stock 50.0-50.3 s, protected 57.8-58.5 s; launcher
  launch 5.1-5.2 s on both. Boots (run-staged.py start to end): stock 62-63 s, protected 70-71 s;
  the whole batch of 12 VMs 71 s. Not a performance comparison: 12 VMs plus other sessions'
  QEMUs on one host, and the sampler shares the guest's hart.

## Deviations

- Two boots that are not measurements: a probe before the run (checks `time -v`, `usleep`,
  `--stats`, the sampler on a `perl -e` one-liner) and, after it, a diagnostic boot of the
  FFmpeg stock image with an all-process `/proc` scan (to find the source of the 4x `ru_maxrss`).
  No number above comes from either.
- `CAPSTONE_VM_STATS=1` added to the registered environment (launcher's exit counters on
  stderr; stdout unchanged).
- N = 3 per arm as registered. Raw serial logs not committed.
