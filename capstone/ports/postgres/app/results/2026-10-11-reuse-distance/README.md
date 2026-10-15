# PostgreSQL 17.5: reuse distance at the memory contexts, stock against protected (2026-10-11)

The PostgreSQL row of the pre-registration
[`docs/plans/2026-10-11-nested-reuse-distance.md`](../../../../../docs/plans/2026-10-11-nested-reuse-distance.md)
(16d2171a3edc). Reuse distance is the number of chunk hand-outs between the release of a start
address and its next hand-out, over all memory contexts together (boundary `contexts`); bucket k
holds distances d with 2^(k-1) <= d < 2^k (observer `bins[k-1]`). Workload: `work.sql` in the
single-user backend, run-ports.py's command, as in the
[same-answers run](../2026-10-11-same-answers/README.md).

- **stock**: `PGSU_NESTED` unset (port patches 0001-0017; 0017's own observer not compiled).
- **protected**: `PGSU_NESTED=sublet` (0001-0016 and memory-contexts patch 0003: every chunk a
  CDERIVE child of its block's generation, revoked by pfree/repalloc).

Both arms carry the same observer hooks, compiled in with `-DEXP_REUSE_GAP`. Numbers are in
[`reuse.json`](reuse.json).

## Outcome

| Prediction | Verdict |
|---|---|
| 1. stock and protected give the same histogram, bucket for bucket; where repetitions vary, the arms vary alike (overlapping ranges per bucket) | **does not hold.** The program is not deterministic at this boundary (three boots per arm differ), so the overlap rule applies: in 13 of the 16 occupied buckets the stock and protected ranges over three boots each are disjoint. The difference sits mostly at distances of 256 and more (buckets 9-15). |
| median bucket (reported for every program) | **equal**: bucket 3 (distance 4-7) in all six boots. |
| share of reuses whose bucket differs (`reusediff`) | **0.0065** for the planned pair of boots; **0.0045-0.0066** over all nine stock-protected pairs, against **0.0004-0.0007** between stock boots and **0.0005-0.0020** between protected boots. |
| 3. the figure (stock and protected cumulative curves overlap) | not drawn here. For the plotting script: the largest gap between the two cumulative fractions is 0.0056, at bucket 10 (0.9790 stock, 0.9733 protected), planned pair. |

The difference is small: 0.65% of reuses fall in another bucket. It is still a difference the
pre-registration said would not exist. It is larger than the boot-to-boot variation: the smallest
stock-protected `reusediff` (0.0045) is more than twice the largest within either arm (0.0020).
It is reported as measured.

Every boot passed every check: exit 0, no `domain fault`, `VIRTUAL_STAGED_DONE`, `capstone_vm`
unloaded with status 0, observer `error=0`, and 22 rows equal to the same-answers rows (run-ports.py
`pg_rows`, sha256 `280000084d6b...`).

## Results

Boot 1 is the planned boot per arm (4 boots at once, with SQLite); boots 2 and 3 are the
repetitions (8 at once).

| arm, boot | hand-outs (attempts = issues) | releases (reset / pfree and moving repalloc) | reuses | distinct starts | median bucket |
|---|---|---|---|---|---|
| stock 1 | 54,083 | 51,228 (14,832 / 36,396) | 44,219 | 9,864 | 3 |
| stock 2 | 54,093 | 51,238 (14,832 / 36,406) | 44,245 | 9,848 | 3 |
| stock 3 | 54,093 | 51,238 (14,832 / 36,406) | 44,222 | 9,871 | 3 |
| protected 1 | 54,085 | 51,230 (14,832 / 36,398) | 44,390 | 9,695 | 3 |
| protected 2 | 54,093 | 51,238 (14,832 / 36,406) | 44,395 | 9,698 | 3 |
| protected 3 | 54,093 | 51,238 (14,832 / 36,406) | 44,321 | 9,772 | 3 |

Reuses per bucket (bucket k: distance 2^(k-1) to 2^k - 1):

| k | stock 1 | stock 2 | stock 3 | protected 1 | protected 2 | protected 3 | ranges |
|---|---|---|---|---|---|---|---|
| 1 | 13,772 | 13,772 | 13,772 | 13,772 | 13,772 | 13,772 | overlap |
| 2 | 5,856 | 5,855 | 5,855 | 5,855 | 5,854 | 5,854 | overlap |
| 3 | 6,980 | 6,981 | 6,981 | 6,982 | 6,983 | 6,983 | disjoint |
| 4 | 12,733 | 12,732 | 12,732 | 12,735 | 12,734 | 12,735 | disjoint |
| 5 | 1,838 | 1,839 | 1,839 | 1,840 | 1,840 | 1,840 | disjoint |
| 6 | 1,069 | 1,069 | 1,069 | 1,065 | 1,065 | 1,066 | disjoint |
| 7 | 354 | 355 | 355 | 345 | 351 | 352 | disjoint |
| 8 | 300 | 303 | 303 | 296 | 298 | 296 | disjoint |
| 9 | 265 | 271 | 274 | 223 | 226 | 230 | disjoint |
| 10 | 123 | 135 | 118 | 94 | 94 | 100 | disjoint |
| 11 | 302 | 311 | 304 | 395 | 382 | 316 | disjoint |
| 12 | 165 | 165 | 165 | 361 | 360 | 352 | disjoint |
| 13 | 255 | 255 | 255 | 258 | 266 | 256 | disjoint |
| 14 | 56 | 58 | 56 | 41 | 42 | 40 | disjoint |
| 15 | 80 | 80 | 80 | 56 | 56 | 58 | disjoint |
| 16 | 71 | 64 | 64 | 72 | 72 | 71 | overlap |

`reusediff` = sum over k of |stock_k/N_s - protected_k/N_p| / 2, with N the arm's reuses.

What the readings show, and what they do not:
- **The event sequence's size is the same in both arms.** Boots 2 and 3 of both arms make exactly
  54,093 hand-outs and 51,238 releases, 14,832 of them by context reset or delete. Boot 1 of each
  arm makes 10 and 8 fewer; the cause of that boot-to-boot difference is not established.
- **Which addresses come back differs.** The difference is in the reuses (+171 on the planned pair)
  and in the distinct start addresses (-169), and it is concentrated at long distances: buckets
  9-15, 256 to 32,767 hand-outs.
- **A likely mechanism, from reading patch 0003 and not measured:** the protected chunk header is
  32 bytes instead of 16 and every block header grows by two capabilities. Chunks then fill
  blocks differently and the backend uses more blocks (the memory-behaviour run measured 58 more,
  all in `index info` contexts), so the addresses at which new blocks and their chunks start differ.
  The pre-registration's reason for equality ("the patches leave placement and free-list order
  alone") does not hold for this patch. Its own header text says the header grows.
- The stock hooks do not change the lifetime count: the launcher reports 6,203 lifetimes on stock,
  as in the same-answers run; protected reports 65,235 (same-answers: 65,233).

## Hook placement

`hooks-stock.patch` and `hooks-protected.patch` in
[`experiments/study/reuse-hooks/postgres/`](../../../../../experiments/study/reuse-hooks/postgres/),
generated by `make-patches.py`: the same hook text at the same places in
`src/backend/utils/mmgr/mcxt.c`. **There are two variants because patch 0003 rewrites the free and
realloc call lines**, which are the anchors there. The call placement is identical: the old chunk is
read before the arm's free or realloc sequence and the outcome is recorded after it.

| event | mcxt.c function | stock | protected |
|---|---|---|---|
| hand-out | `MemoryContextAlloc`, `MemoryContextAllocZero`, `MemoryContextAllocExtended`, `palloc`, `palloc0`, `palloc_extended`, `MemoryContextAllocHuge` (7 sites) | after `ret = context->methods->alloc(...)` | same |
| release | `pfree` | chunk base and method read before `MCXT_METHOD(pointer, free_p)`, release after it | read before `pointer = SubletRelease(pointer)` (before the revoke), release after `free_p` |
| moving repalloc | `repalloc`, `repalloc_extended` | read before the realloc call; after it, same base: resize (no event); new base: release the old start, hand out the new | read before `SubletRelease`, recorded after `SubletReissue(...)` |
| reset, delete | `MemoryContextResetOnly`, `MemoryContextDeleteOnly` | before `methods->reset` / `methods->delete_context`: release every chunk still live in that context | same |

This covers every manager (aset, generation, slab, bump), since all four hand out and take back
chunks only through these mcxt.c dispatch points; a manager's realloc that calls its own alloc and
free is seen as the repalloc's outcome. Bump has no pfree: its chunks are released by reset and
delete only, and a `pfree` whose method is bump is not recorded (bump raises an error there). An
aligned allocation is counted once, as the ordinary chunk `MemoryContextAllocAligned` takes through
`MemoryContextAllocExtended`; the outer `pfree`/`repalloc` of the aligned pointer is skipped and its
inner `pfree`/`palloc_aligned` recorded. The start recorded is the address of the pointer the
manager returns: `MemoryChunkGetPointer(chunk)` on stock, and on protected the capability
`MemoryChunkIssue` derives at that same address; only the integer address is kept.

`exp-reuse-gap.h` (61897b98...) is patch 0017's observer (`toolchain/spatial-reuse-gap.c`) with the
same logic under new names: the reuse-gap observer (`experiments/study/reuse-gap-observer.h`,
627b0fe3..., unchanged; its unit test passes) with 2^17 slots, per-context lists of live chunks
(2^15 contexts) for resets, all in static storage outside the memory contexts, plus counters for
reset releases and in-place resizes, and an `atexit` report on stderr:
`REUSE-GAP contexts attempts=.. issues=.. releases=.. reuses=.. distinct=.. capacity=.. error=..
reset_releases=.. resizes=.. bins=..`.

**In-place resize is untested here**: `resizes=0` in all six boots, so no `repalloc` in `work.sql`
kept its chunk, and that path of the hook never ran.

## Builds

Source: `git archive 16d2171a3edc capstone` with the `reuse-hooks/sqlite` and `reuse-hooks/postgres`
files (uncommitted here) added, and llvm-capstone `compiler-rt/lib/builtins` at 9bc6f9adf1c9,
committed on the build host as one snapshot commit (621df9fc). Both image manifests record that
commit and `runtime_dirty: false`. The `collect.py` scripts were written after the builds and ran
beside the snapshot, not from it.

`experiments/study/reuse-hooks/postgres/build.sh <arm> <new out>` is the postgres branch of
`ports/common/application/build-virtual.sh` with one step added; no port script is changed:

1. `build-domain.sh` with `PGSU_ONLY="musl runtime"`: musl, the SDK runtime and the patched source,
   exactly as the port prepares them. Its link step then stops on the missing objects. That nonzero
   exit is expected, and the wrapper checks the source tree, the runtime and the arm's `nested.mode`.
2. `hooks-<arm>.patch` (`--fuzz=0`), with `exp-reuse-gap.h` and `reuse-gap-observer.h` copied into
   `src/backend/utils/mmgr/`.
3. `build-domain.sh` with `PGSU_FROM=configure` and `CPPFLAGS=-DEXP_REUSE_GAP`. Configure takes it
   from the environment into `Makefile.global`, which the wrapper checks.
4. `build.py --app postgres --profile virtual --nested {none|postgres} --stack-bytes 1048576`, the
   relink of build-virtual.sh. The wrapper checks that the image holds the report string.

Checks on the inputs:
- reversing the hook patch from each arm's compiled `mcxt.c` gives exactly the same-answers build's
  `mcxt.c` (stock e392fef9..., protected f3957c9b...);
- the runtime archive is e5693f5c... and the builtins archive a72ed0bf..., as in the same-answers
  images;
- clang is 4c6135fe....

| file | sha256 |
|---|---|
| stock `postgres.dom` (nested=none) | `9cd1e63df3744a101f9fdbc24b5df555b802a34c18eebc1f3378393ca44b2217` |
| protected `postgres.dom` (nested=postgres) | `e4735dd71638d2acffc42fc78eb3f929f12f99333ce038f4fca7c77112063b03` |
| stock `mcxt.c` (hooked) | `207d9efe443757bd52aa0a238211528f7adf63131174df1d736044e07d27c650` |
| protected `mcxt.c` (hooked) | `6b80f930cc2c2501e555703310cfb09850bcf9775d719d33e87fa1b51dad2f74` |
| `hooks-stock.patch` / `hooks-protected.patch` | `ae9335f4...` / `57bea3a4...` |

## Runs

The pre-registered platform: QEMU af37cc32, launcher 3975314a, module a1c6cb6b (full hashes in
`reuse.json`; kernel, firmware and rootfs as in the same-answers run).
- **Boots:** one boot per arm of `capstone/runtime/virtual/run-staged.py --qemu
  $K/kit-sublet/qemu-system-riscv64 --images $K/kit-sublet/images --stage <stage> --work <fresh dir>
  --timeout 3600`, each with its own `CAPSTONE_QEMU_LOCK` file.
- **Stage:** `<arm>/bin/postgres`, the kit's share tree, the kit's pristine cluster and the repo's
  `work.sql` (f4310b75...).
- **Image check:** the images inside each boot's `stage.ext4` were extracted afterwards (`debugfs
  dump`) and match the hashes above.

The gate (stock; protected differs only in the arm name):

```sh
#!/bin/sh
cd /mnt/vm || exit 1
dmesg -n 1
insmod capstone_vm.ko || exit 1
export CAPSTONE_EXEC_DIAGNOSTICS=1 CAPSTONE_VM_STATS=1
chmod 666 /dev/capstone-vm
cp -a pgcluster /tmp/pgcluster-stock
chown -R nobody /tmp/pgcluster-stock
echo PORT_BEGIN:postgres-stock
t0=$(date +%s)
su nobody -s /bin/sh -c "cd /mnt/vm/stock; /mnt/vm/capstone-vexec ./bin/postgres --single -D /tmp/pgcluster-stock -c shared_buffers=4MB -c max_connections=10 -c timezone=GMT -c log_timezone=GMT -c dynamic_shared_memory_type=sysv -c exit_on_error=true postgres < /mnt/vm/work.sql"
rc=$?
echo PORT_WALL:postgres-stock:$(($(date +%s) - t0))
echo PORT_END:postgres-stock:$rc
cd /mnt/vm
echo STATS_AFTER:$(./capstone-vexec --stats)
rmmod capstone_vm
echo RMMOD:$?
echo VIRTUAL_STAGED_DONE
```

`collect.py` (in the reuse-hooks directory) parses the serial logs into `reuse.json` and refuses to
write it if any check above fails. The JSON's `boundaries` hold boot 1; `repetitions` hold boots 2
and 3.

## Wall times

| step | seconds |
|---|---|
| build per arm, both arms and SQLite at once | 136 |
| boot 1, QEMU start to done, stock / protected (4 boots at once) | 56 / 57 |
| boots 2-3 (8 boots at once) | 68-70 |
| guest wall around the backend: boot 1 stock / protected; boots 2-3 | 42 / 44; 49 |

## Deviations and limits

- **Three boots per arm, not one.** The instruction was one boot per arm, on the grounds that the
  memory-behaviour runs were deterministic. They were not for PostgreSQL: protected lifetime counts
  varied there. The planned boots showed a histogram difference, so two more boots per arm were
  run to compare it with boot-to-boot variation, which is the pre-registration's own rule for
  programs that vary. Both arms vary here, stock included.
- **Plain `work.sql`.** The memory-behaviour run appended a `pg_backend_memory_contexts` query;
  this run uses the pre-registered `work.sql` unchanged.
- **The hooks sit at the mcxt.c dispatch, not inside each manager.** These are the points where
  every manager's chunk enters and leaves; patch 0017 placed its observer at the same points.
  Resets and deletes count as releases (14,832 per boot), as in 0017.
- **Two-pass build.** The port's `build-domain.sh` has no hook for an extra patch or define, so
  the wrapper runs it twice and passes the define through `CPPFLAGS`. The prepare pass ends with
  an expected nonzero exit.
- **A second copy of the run script started by mistake**, about 17 s after the first: a process
  check grepped for the run directory, which the script's command line does not contain. It copied
  identical stage files 14 s after the boots had packed their `stage.ext4` images, and its four
  `run-staged.py` calls stopped at once on the existing work directories. It overwrote those calls'
  stdout files (not the serial logs). The images inside the packed `stage.ext4` files were checked
  against the build hashes afterwards.
- **In-place repalloc** never occurred, so the resize path is unexercised (see Hook placement).
- One host; 3 boots per arm.
