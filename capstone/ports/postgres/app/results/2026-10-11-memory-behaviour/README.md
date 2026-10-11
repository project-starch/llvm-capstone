# PostgreSQL 17.5: memory behaviour of protected memory contexts (2026-10-11)

This is the PostgreSQL row of the pre-registration
[`docs/plans/2026-10-11-protected-memory-behaviour.md`](../../../../../docs/plans/2026-10-11-protected-memory-behaviour.md)
(82c1607119ec). The workload is `work.sql` in the single-user backend, as in the
[same-answers run](../2026-10-11-same-answers/README.md). One query reading
`pg_backend_memory_contexts` is appended to a copy of `work.sql`.

- **stock**: `PGSU_NESTED` unset.
- **protected**: `PGSU_NESTED=sublet`, memory-contexts patch 0003.

N = 3 per arm. Each arm ran in its own boot, and all boots ran in parallel.

## Outcome

| Prediction | Verdict |
|---|---|
| context set (names, levels, count) equal | **holds.** 136 contexts in both arms. The multiset of (level, name, ident, parent) is identical, and so are the per-level counts (0:1, 1:27, 2:101, 3:2, 4:3, 5:1, 6:1). |
| total bytes: no difference | **does not hold.** Summed `total_bytes` is 2,857,264 on protected against 2,737,232 on stock: +120,032 (+4.39%). Blocks are 399 against 341 (+58). 73 of 136 contexts have a larger `total_bytes`, and none has a smaller one. |
| protected allocates more lifetimes than stock and reclaims them | **holds.** Protected allocates 68,727-68,737 lifetimes against 6,507 on stock. It reclaims 53,531-53,787 of them in 5 collections; stock reclaims 0 in 0 collections. |
| 22 result rows still match (sanity) | **holds.** All six boots give 22 rows equal to native 17.5. They hash to `280000084d6b...`, the same-answers rows hash. |
| repetitions identical | **contexts: yes.** All 136 rows, every column, are identical across the 3 repetitions of each arm. **Lifetime counts on protected: no** (see Repetitions). |

Every boot passes these checks:
- it exits 0, and busybox `time` reports exit status 0;
- no `domain fault` appears;
- the `VIRTUAL_STAGED_DONE` marker is present;
- `capstone_vm` unloads with status 0.

## Allocator statistics (`pg_backend_memory_contexts` at the end of the session)

Sums over all 136 contexts. Each value is the same in all 3 repetitions of its arm.

| column | stock | protected | protected - stock | contexts that differ |
|---|---|---|---|---|
| contexts | 136 | 136 | 0 | none (same set) |
| total_bytes | 2,737,232 | 2,857,264 | +120,032 | 73 (all higher) |
| total_nblocks | 341 | 399 | +58 | 58 (all higher) |
| used_bytes | 2,162,672 | 2,240,160 | +77,488 | 136 (all higher) |
| free_bytes | 574,560 | 617,104 | +42,544 | 136 (58 higher, 78 lower) |
| free_chunks | 251 | 449 | +198 | 96 (82 higher, 14 lower) |

Where the difference comes from:
- **Context headers.** Every context's `used_bytes` is higher, by at least 32 bytes. For an empty
  context the difference is exactly 32: `ErrorContext` 432 → 464, `TupleSort sort` 432 → 464,
  `Caller tuples` 64 → 96.
- **Contexts with chunks** grow by more. `CacheMemoryContext` uses 938,848 → 965,360,
  `MessageContext` 73,264 → 82,320, and `ExprContext` 2,704 → 3,504.
- **The extra blocks.** All 58 extra blocks are in `index info` contexts, and 58 of the 96 `index
  info` contexts take one more block. Typically they go from 2,048 bytes in 2 blocks to 4,096 bytes
  in 3.
- **The largest contexts' totals do not move.** `CacheMemoryContext` (1,048,576) and
  `MessageContext` (131,072) stay at the same totals; only their used/free split moves.

The pre-registration's reason for predicting no difference was that "the generation is stored in
the existing chunk header". That is not what the patch does. Its own description says: "The chunk
header grows from 16 to 32 bytes, and every block header by two capabilities"
(`memory-contexts/patches/postgresql-17.0-0003-memory-contexts-sublet-lifetimes.patch`, sha256
`8b36b576...`, header text). The measured direction matches that description: every context's
`used_bytes` rises, by 32 bytes even in an empty context. Whether the rise is exactly 16 per chunk
plus 32 per block was not decomposed. The prediction stands as written and does not hold; this is
a finding about the pre-registration, not an adjustment of the reading.

## Lifetime counters (`capstone-vexec --stats`, after minus before)

The module is freshly loaded in each boot. Before the run every counter is 0, except
`node_capacity` 1024 and `node_bytes` 28672.

| field | stock (all 3) | protected rep 1 | protected reps 2, 3 |
|---|---|---|---|
| nodes_allocated_total | 6,507 | 68,727 | 68,737 |
| nodes_reclaimed | 0 | 53,787 | 53,531 |
| collections | 0 | 5 | 5 |
| nodes_high_water | 6,507 | 17,150 | 17,150 |
| nodes_live / nodes_retired after | 0 / 0 | 0 / 0 | 0 / 0 |
| live_domains / regions / bytes / tag_pages after | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |

The launcher's per-process line, `CAPSTONE_VM_STATS`, is printed at the domain's exit:

| field | stock | protected |
|---|---|---|
| node_capacity / node_bytes / node_growths | 9,216 / 159,744 / 2 | 17,408 / 290,816 / 4 |
| arenas | 86 | 92 |
| pinned pages at exit / peak pinned pages | 5,156 / 6,768 | 5,205 / 6,823 |
| faults | 9,075 | 9,131 |
| rounds / bytes_out | 2,698 / 3,314,508 | rep 1: 2,699 / 3,314,524; reps 2-3: 2,698 / 3,314,508 |
| bytes_in | 2,131,522-2,131,526 | 2,131,604 |

Stock's 6,507 lifetimes are the platform's own, with no nested-allocator lifetimes; protected
adds about 62,220. Protected also has 6 more arenas and 55 more peak pinned pages (220 KiB).

## Peak resident memory

| method | stock | protected |
|---|---|---|
| busybox `time -v` maximum RSS (wait4 `ru_maxrss`) | 126,416 / 126,416 / 126,400 KiB | 126,688 / 126,688 / 126,448 KiB |
| peak pinned domain pages (`CAPSTONE_VM_STATS peak`) | 6,768 pages = 27,072 KiB (all 3) | 6,823 pages = 27,292 KiB (all 3) |

Protected is higher by 32 to 288 KiB in `ru_maxrss`. In peak pinned pages it is higher by 220 KiB
in every repetition.

`ru_maxrss` is the kernel's RSS high-water mark (`hiwater_rss`) taken at exit, the same counter as
`VmHWM`. It covers the launcher process (domain arenas are its private anonymous pages) and its
reaped children. `time` runs inside `su nobody -c`, so `su` and `sh` are not included.

**The direct VmHWM sampler failed.** It polled `/proc/<pid>/status` once a second for every
`capstone-vexec` pid and kept the last line written. That line came from a pid with VmHWM 120 kB,
which is not the launcher main process (the launcher also runs a small spawner child). The values
are kept in the JSON, labelled invalid, and not used.

## Repetitions

- **Identical across the 3 repetitions of each arm:**
  - all 136 memory-context rows;
  - the 22 work rows;
  - every stock launcher delta.
- **Protected lifetime counts differ** between rep 1 and reps 2-3:
  - allocated: 68,727 against 68,737;
  - reclaimed: 53,787 against 53,531.

  Rep 1 also made one more service round (2,699 against 2,698) and wrote 16 more output bytes. The
  likely source of the 16 bytes is the checkpoint log line, whose timing text varies in length. The
  cause of the lifetime difference is not established.
- **Other varying fields:**

| field | stock min-max | protected min-max |
|---|---|---|
| ru_maxrss (KiB) | 126,400-126,416 | 126,448-126,688 |
| CAPSTONE_VM_STATS steps | 19,153-19,318 | 19,297-19,491 |
| bytes_in | 2,131,522-2,131,526 | 2,131,604 (all) |
| launcher elapsed_ns (s) | 41.46-43.56 | 42.25-44.42 |
| guest wall, `date` around the run (s) | 48-51 | 49-51 |
| host wall per boot, QEMU start to done (s) | 63-66 | 64-67 |

The times come from 12 VMs running at once on one host. They are not a performance comparison.

## What ran

**Images.** These are the same-answers images, with sha256 checked before staging; staging stops on
a mismatch.
- stock `postgres.dom` `f03bcc1fc5a912f63a1df0326c2234a67e0f3fb1ba80f1c82e1a443ec5e4f4ba` (nested=none)
- protected `postgres.dom` `9c2098f09cc6d6affc3e391158769cc49e2087afc5305626bebffde3d5bf73a5` (nested=postgres)

Build provenance and the pristine cluster are in the same-answers README.

**Stage per arm:**
- `<arm>/bin/postgres`;
- `<arm>/share`, the kit's share tree;
- the kit's pristine cluster;
- `work-mem.sql`.

`work-mem.sql` is the repo `work.sql` (sha256 `f4310b75...`, unchanged) with one appended line
(file sha256 `258d0ddeb557c4780316f3aa821362df04dd3c71dd25758ca8b55aa1a6b2fd4e`). Single-user mode
ends a command at a newline, so the query is on one line:

```sql
SELECT name, ident, parent, level, total_bytes, total_nblocks, free_bytes, free_chunks, used_bytes FROM pg_backend_memory_contexts ORDER BY level, name, ident NULLS FIRST, parent NULLS FIRST, total_bytes, total_nblocks, free_bytes, free_chunks, used_bytes;
```

PostgreSQL 17.5's view has the `parent` column (checked in `pg_proc.dat` of the built source). The
query was first run natively on p13 against a copy of the same cluster (exit 0) before any guest
boot.

Rows are extracted as follows:
- **Work rows:** run-ports.py's `pg_rows`, applied to the output before the query's header, hashed
  as `\n`-joined rows plus a trailing `\n`. That is the convention of the same-answers
  `result.json`, verified by reproducing its native hash `280000084d6b...` from the native stdout.
- **Context rows:** parsed per attribute. A NULL `ident` or `parent` is not printed by the
  single-user backend and is recorded as null.

**Platform.**

| file | sha256 |
|---|---|
| qemu-system-riscv64 | `af37cc3269d1cfb47a72eb79cc3dfd2397848c613ca703e43625c68ab50ad27d` |
| capstone-vexec | `3975314a7a6da119c4bff6a0ab0c601fcd96d305f59a0a68fe5b36b9bd544ad4` |
| capstone_vm.ko | `a1c6cb6b79a98cf1f09e55a37adf2320e2b9787922b7e10bf4088c27e396e512` |
| Image | `e833dfb12bccf8a39bd6a81e8e42e4622f0e76bcc6cc09624a1fbe7198c18a4b` |
| fw_jump.elf | `44cd6cbc2d225c2fea6c6000ea0b9eab83bc34e769a7c95c036cf17b6fe0f116` |
| rootfs.ext2 | `62049800ed0fa8827959c0060bbd74f0435aacb577dd521e14a14a099d167845` |
| run-staged.py | `d6770b8b9c55ef8a4e136a17f5411a2fe2f139af9fee47f5e02328810acfc54c` |
| gate.sh stock / protected | `95f9f0f6...4b909` / `b71ddb9c...5b7` |

**Boots.** Each repetition is one boot of `run-staged.py`:
- virt-capstone, one hart, 2 GiB, U-mode and exact bounds (its defaults);
- its own `CAPSTONE_QEMU_LOCK` file and a fresh `--work` directory;
- `--timeout 3600`.

Six PostgreSQL and six SQLite boots ran concurrently. The gate, for stock (protected differs only in
the arm name):

```sh
#!/bin/sh
cd /mnt/vm || exit 1
dmesg -n 1
insmod capstone_vm.ko || exit 1
export CAPSTONE_EXEC_DIAGNOSTICS=1 CAPSTONE_VM_STATS=1
chmod 666 /dev/capstone-vm
cp -a pgcluster /tmp/pgcluster-stock
chown -R nobody /tmp/pgcluster-stock
echo STATS_BEFORE:$(/mnt/vm/capstone-vexec --stats)
( while [ ! -e /tmp/sampler.stop ]; do for p in $(pidof capstone-vexec); do v=$(grep VmHWM /proc/$p/status 2>/dev/null); [ -n "$v" ] && echo "$p $v" > /tmp/hwm.last; done; sleep 1; done ) &
SAMPLER_PID=$!
echo PORT_BEGIN:postgres-stock
t0=$(date +%s)
su nobody -s /bin/sh -c "cd /mnt/vm/stock; /usr/bin/time -v /mnt/vm/capstone-vexec ./bin/postgres --single -D /tmp/pgcluster-stock -c shared_buffers=4MB -c max_connections=10 -c timezone=GMT -c log_timezone=GMT -c dynamic_shared_memory_type=sysv -c exit_on_error=true postgres < /mnt/vm/work-mem.sql"
rc=$?
echo PORT_WALL:postgres-stock:$(($(date +%s) - t0))
echo PORT_END:postgres-stock:$rc
touch /tmp/sampler.stop; wait $SAMPLER_PID
echo HWM_SAMPLE:$(cat /tmp/hwm.last 2>/dev/null)
cd /mnt/vm
echo STATS_AFTER:$(./capstone-vexec --stats)
rmmod capstone_vm
echo RMMOD:$?
echo VIRTUAL_STAGED_DONE
```

## Deviations and method limits

- **One arm per boot.** The same-answers run booted both arms in one VM. Here the module's global
  counters, including `nodes_high_water`, cover exactly one program run.
- **The appended query.** It changes the session: it adds the query's own contexts (portal,
  executor, tuplesort) to the reading, the same on both arms, and makes the lifetime and RSS
  figures larger than those of plain `work.sql`. The same-answers `CAPSTONE_VM_STATS` line, with
  `nodes` 6,203 / 65,233, is for `work.sql` without it.
- **The command is wrapped in busybox `/usr/bin/time -v`**, inside `su`.
- **The VmHWM sampler is invalid** (see above). It was not rerun, because `ru_maxrss` reads the same
  kernel high-water mark at exit.
- **The reading is taken once**, at the end of the session, from inside the session. It shows the
  live contexts at that point. Peaks during the session are not captured by
  `pg_backend_memory_contexts`.
- N = 3 per arm, one host.

Every repetition's raw counters are in [`memory-behaviour.json`](memory-behaviour.json): all 136
context rows, the launcher before/after/delta, the `CAPSTONE_VM_STATS` fields, the `time -v`
fields, the row hash and the walls.
