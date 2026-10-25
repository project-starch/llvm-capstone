# SQLite 3.22.0: memory behaviour of protected memsys5 (2026-10-11)

The SQLite row of the pre-registration
[`docs/plans/2026-10-11-protected-memory-behaviour.md`](../../../../../docs/plans/2026-10-11-protected-memory-behaviour.md)
(82c1607119ec). Workload: speedtest1 `main --size 1` over an 8 MiB memsys5 heap
(`--heap 8388608 64`), the configuration of the
[same-answers run](../2026-10-11-same-answers/README.md), now with `--stats`.

- **stock**: `SQLITE_HEAP=memsys5`.
- **protected**: `SQLITE_HEAP=memsys5-sublet`, i.e. with `../../../sublet/sublet-3220000-memsys5.patch`.

N = 3 per arm, one arm per boot, all boots in parallel.


> **Correction, 2026-10-11, same day.** busybox `time -v` prints "Maximum resident set size" as
> 4x `ru_maxrss` (it converts the value as if it counted 4 KiB pages). The Perl and FFmpeg row of
> this measurement established it: in all twelve of its runs the printed value is exactly 4x the
> sampled VmHWM of the launcher process, and a diagnostic boot read 7,248 kB where busybox printed
> 28,992 (`perl/musl/results/2026-10-11-memory-behaviour/README.md`). The peak-RSS figures below
> are therefore given as printed and divided by 4; `memory-behaviour.json` keeps the printed
> values. The stock-versus-protected comparison is unchanged in direction and ratio.

## Outcome

| Prediction | Verdict |
|---|---|
| malloc count equal | **not measured as named**: speedtest1 `--stats` prints no cumulative malloc count. The only allocation count it prints is `SQLITE_STATUS_MALLOC_COUNT` ("Outstanding Allocations"), judged on the next line. |
| outstanding allocations equal | **does not hold.** At the end both are 0. The high-water is 238 on stock and 254 on protected. |
| lookaside counts equal | **does not hold.** Hits (24,639), slots used (1, max 82) and OOM faults (0) are equal. Size faults are 159 on stock and 163 on protected. |
| memory-used high-water lower or equal on protected | **holds.** 796,216 bytes against 1,117,760 (-321,544, -28.8%). |
| protected allocates more lifetimes than stock and reclaims them | **holds.** 28,256 against 30 allocated. Protected reclaims 27,923 in 6 collections; stock reclaims none in 0 collections. |
| repetitions identical | **yes, for every counter in this table**, on both arms. Across repetitions only timing fields vary (see below). |

Both arms passed every check, in all six boots:
- they exit 0, and busybox `time` reports exit status 0;
- no `domain fault` appears;
- all 32 phases are equal to the same-answers phases (rows and hash);
- `capstone_vm` unloads with status 0.

## Allocator statistics (speedtest1 `--stats`)

Every value is the same in all 3 repetitions of its arm.

| counter (speedtest1 label) | stock | protected | protected - stock |
|---|---|---|---|
| Memory Used (bytes), cur / max | 0 / 1,117,760 | 0 / 796,216 | max -321,544 |
| Outstanding Allocations, cur / max | 0 / 238 | 0 / 254 | max +16 |
| Lookaside Slots Used, cur / max | 1 / 82 | 1 / 82 | 0 |
| Successful lookasides | 24,639 | 24,639 | 0 |
| Lookaside size faults | 159 | 163 | +4 |
| Lookaside OOM faults | 0 | 0 | 0 |
| Pager Heap Usage (bytes) | 326,928 | 325,016 | -1,912 |
| Page cache hits / misses / writes | 33,653 / 0 / 263 | 33,653 / 0 / 263 | 0 |
| Schema Heap Usage (bytes) | 16,640 | 12,064 | -4,576 |
| Statement Heap Usage (bytes) | 0 | 0 | 0 |
| Pcache Overflow Bytes, cur / max | 0 / 528,384 | 0 / 378,656 | max -149,728 |
| Largest Allocation (bytes) | 120,000 | 120,000 | 0 |
| Largest Pcache Allocation (bytes) | 4,592 | 4,592 | 0 |

The byte counters come from `sqlite3MallocSize`, i.e. `memsys5Size`. The patch makes that function
report the request rounded to 8 bytes rather than the power-of-two block. This explains why Memory
Used, Pager Heap, Schema Heap and Pcache Overflow are lower. It is the documented change, and these
are reporting differences, not placement differences.

The allocation-count differences (outstanding high-water +16, lookaside size faults +4) contradict
the "decisions are equal" prediction, and they stand as a finding. A likely mechanism, from reading
the source and not measured: the 3.22.0 amalgamation sizes several buffers from
`sqlite3MallocSize`/`sqlite3DbMallocSize`. Examples are `StrAccum` `nAlloc`, `Mem.szMalloc`,
`szOpAlloc`, `WhereClause.nSlot`, the hash table's `htsize` and the sorter's `nMemory`. On stock
these callers see the power-of-two slack; on protected they see only the request. The engine then
grows and reallocates on a different schedule. Either way, the reported size feeds back into
SQLite's own allocation decisions.

`--stats` also tries `/proc/<pid>/io`. Inside the domain that open fails, so those lines are absent
on both arms.

## Lifetime counters (`capstone-vexec --stats`, after minus before)

The module was freshly loaded in each boot. Before the run, every counter is 0, except
`node_capacity` 1024 and `node_bytes` 28672.

| field | stock | protected |
|---|---|---|
| nodes_allocated_total | 30 | 28,256 |
| nodes_reclaimed | 0 | 27,923 |
| collections | 0 | 6 |
| nodes_high_water | 30 | 4,862 |
| nodes_live / nodes_retired after | 0 / 0 | 0 / 0 |
| live_domains / live_regions / live_bytes / tag_pages after | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |

The launcher's per-process line (`CAPSTONE_VM_STATS`, printed at the domain's exit) adds:

| field | stock | protected |
|---|---|---|
| node_capacity / node_bytes / node_growths | 1,024 / 28,672 / 0 | 5,120 / 94,208 / 1 |
| arenas | 9 | 9 |
| pinned pages at exit / peak pinned pages | 869 / 2,918 | 869 / 2,918 |
| faults / rounds | 2,066 / 15,223 | 2,066 / 15,223 |
| bytes_in / bytes_out | 1,987,376 / 773,987 | 1,987,375 / 773,987 |

The node table is the processor's lifetime table. It grew once on protected, to five times its
initial capacity.

On "about one per nested-allocator hand-out": the 28,226 extra lifetimes exceed the 24,639
lookaside hits. The number of memsys5 hand-outs is not exposed, so the ratio cannot be checked.

## Peak resident memory

| method | stock | protected |
|---|---|---|
| busybox `time -v` maximum RSS of the launcher process, printed (/4 = `ru_maxrss`) | 49,104 (12,276) KiB, all 3 | 49,104 (12,276) KiB, all 3 |
| peak pinned domain pages (`CAPSTONE_VM_STATS peak`) | 2,918 pages = 11,672 KiB | 2,918 pages = 11,672 KiB |

`ru_maxrss` is the kernel's RSS high-water mark (`hiwater_rss`) of the process at its exit, the
same counter `/proc/<pid>/status` shows as `VmHWM`. The domain's arenas are private anonymous pages
of the launcher process, so the domain is included. So is the launcher itself, and so is the
8 MiB heap as far as it is touched. The pool holds about 6% fewer atoms on protected, but the
heap's size is the same, so the footprint is too.

**The direct VmHWM sampler failed.** It read `/proc/<pid>/status` once a second for every
`capstone-vexec` pid and kept the last line written. The launcher also runs a small spawner child,
and the last line came from a pid with VmHWM 116 kB: not the launcher main process, which `time -v`
puts at 49,104 KiB as printed, 12,276 KiB after the /4 correction. The sampled values are kept in the JSON, labelled invalid, and are not used.
`time -v` is the peak-memory measurement.

## Repetitions

Identical across the 3 repetitions of each arm:
- every speedtest1 `--stats` value;
- every launcher `--stats` delta;
- every `CAPSTONE_VM_STATS` field except `steps`, `launch_ns` and `elapsed_ns`;
- `ru_maxrss`;
- all 32 phases.

The fields that vary:

| field | stock min-max | protected min-max |
|---|---|---|
| CAPSTONE_VM_STATS steps | 35,091-35,170 | 35,176-35,208 |
| speedtest1 TOTAL (guest s) | 28.846-29.732 | 30.231-30.641 |
| launcher elapsed_ns (s) | 31.93-33.14 | 33.40-33.74 |
| guest wall, `date` around the run (s) | 34-35 | 35-36 |
| host wall per boot, QEMU start to done (s) | 43-44 | 45 |

The guest times come from 12 VMs running at once on one host. They are not a performance comparison.

## What ran

**Images.** These are the same-answers images, with their sha256 checked before staging; staging
stops on a mismatch.
- stock `speedtest1.dom` `385a7fbb38fcbbd6d2558e9491b4d5e0609ef658132853c0ee280c53c18a37a0`
- protected `speedtest1.dom` `9981f0bef468e8002dcb021ee65be2cd75ad414673532768c47a46c41760d405`

The build provenance is in the same-answers README.

**Platform.** This is the pre-registered platform. The kernel, firmware and rootfs are the same as
in the same-answers run.

| file | sha256 |
|---|---|
| qemu-system-riscv64 | `af37cc3269d1cfb47a72eb79cc3dfd2397848c613ca703e43625c68ab50ad27d` |
| capstone-vexec | `3975314a7a6da119c4bff6a0ab0c601fcd96d305f59a0a68fe5b36b9bd544ad4` |
| capstone_vm.ko | `a1c6cb6b79a98cf1f09e55a37adf2320e2b9787922b7e10bf4088c27e396e512` |
| Image | `e833dfb12bccf8a39bd6a81e8e42e4622f0e76bcc6cc09624a1fbe7198c18a4b` |
| fw_jump.elf | `44cd6cbc2d225c2fea6c6000ea0b9eab83bc34e769a7c95c036cf17b6fe0f116` |
| rootfs.ext2 | `62049800ed0fa8827959c0060bbd74f0435aacb577dd521e14a14a099d167845` |
| run-staged.py | `d6770b8b9c55ef8a4e136a17f5411a2fe2f139af9fee47f5e02328810acfc54c` |
| gate.sh stock / protected | `908a82bb...47c81` / `67a65f41...9113a` |

**Boots.** Each repetition was its own boot of `run-staged.py`, with these properties:
- one hart, 2 GiB, U-mode, exact bounds (its defaults);
- its own `CAPSTONE_QEMU_LOCK` file and a fresh `--work` directory;
- `--timeout 3600`.

Six SQLite and six PostgreSQL boots ran concurrently. The gate (stock shown; protected differs only
in the label):

```sh
#!/bin/sh
cd /mnt/vm || exit 1
dmesg -n 1
insmod capstone_vm.ko || exit 1
export CAPSTONE_EXEC_DIAGNOSTICS=1 CAPSTONE_VM_STATS=1
echo STATS_BEFORE:$(/mnt/vm/capstone-vexec --stats)
cd /tmp && rm -f st.db st.db-journal
( while [ ! -e /tmp/sampler.stop ]; do for p in $(pidof capstone-vexec); do v=$(grep VmHWM /proc/$p/status 2>/dev/null); [ -n "$v" ] && echo "$p $v" > /tmp/hwm.last; done; sleep 1; done ) &
SAMPLER_PID=$!
echo PORT_BEGIN:sqlite-stock
t0=$(date +%s)
/usr/bin/time -v /mnt/vm/capstone-vexec /mnt/vm/speedtest1.dom --stats --heap 8388608 64 --size 1 st.db
rc=$?
echo PORT_WALL:sqlite-stock:$(($(date +%s) - t0))
echo PORT_END:sqlite-stock:$rc
touch /tmp/sampler.stop; wait $SAMPLER_PID
echo HWM_SAMPLE:$(cat /tmp/hwm.last 2>/dev/null)
cd /mnt/vm
echo STATS_AFTER:$(./capstone-vexec --stats)
rmmod capstone_vm
echo RMMOD:$?
echo VIRTUAL_STAGED_DONE
```

## Deviations and method limits

- **One arm per boot.** The same-answers run booted both arms in one VM. This run gives each arm
  its own boot, so the module's global counters, including `nodes_high_water`, cover one program
  run only.
- **`--stats` added.** Compared with the same-answers command, speedtest1 also prints statistics,
  runs `PRAGMA compile_options` at the end and tries `/proc/<pid>/io`. The phases still equal the
  same-answers phases.
- **`CAPSTONE_VM_STATS=1` set.** The same-answers SQLite gate did not set it. It only prints a line
  at exit.
- **The run is wrapped in busybox `/usr/bin/time -v`.** It forks the launcher and waits for it.
- **The VmHWM sampler is invalid** (see above). It was not rerun, because `time -v` reads the same
  kernel high-water mark exactly at exit.
- **No cumulative malloc count** is available from speedtest1. memsys5's internal counters (`nAlloc`,
  `totalAlloc`, ...) are not exposed through `sqlite3_status`.
- N = 3 per arm, one host.

Every repetition's raw counters are in [`memory-behaviour.json`](memory-behaviour.json): the
speedtest1 stats, launcher before/after/delta, the `CAPSTONE_VM_STATS` fields, `time -v` fields,
the phases and the walls.
