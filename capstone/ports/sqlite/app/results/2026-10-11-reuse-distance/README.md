# SQLite 3.22.0: reuse distance at memsys5 and the lookaside, stock against protected (2026-10-11)

The SQLite row of the pre-registration
[`docs/plans/2026-10-11-nested-reuse-distance.md`](../../../../../docs/plans/2026-10-11-nested-reuse-distance.md)
(16d2171a3edc). Reuse distance is the number of hand-outs at one allocator boundary between the
release of a start address and its next hand-out; bucket k holds distances d with
2^(k-1) <= d < 2^k (observer `bins[k-1]`). Workload: speedtest1 `main --size 1` over an 8 MiB
memsys5 heap (`--heap 8388608 64`), the command of the
[same-answers run](../2026-10-11-same-answers/README.md).

- **stock**: memsys5, as in the same-answers stock arm.
- **protected**: memsys5 with `../../../sublet/sublet-3220000-memsys5.patch` (blocks and lookaside
  slots as CDERIVE/CREVOKE lifetimes).

Both arms carry the same observer hooks, compiled in with `-DEXP_REUSE_GAP`. Numbers are in
[`reuse.json`](reuse.json).

## Outcome

| Prediction | Verdict |
|---|---|
| 2. SQLite differs: the allocation sequence at memsys5 changes | **holds.** memsys5 hands out 3,118 blocks on stock and 3,583 on protected (+465, +14.9%); every bucket from 1 to 12 differs. |
| 2. the median bucket is equal | **holds.** memsys5: bucket 3 (distance 4-7) on both arms. Lookaside: bucket 2 (distance 2-3) on both. |
| 2. share of reuses whose bucket differs (`reusediff`) | **reported: memsys5 0.0607** (6.07%); **lookaside 0.0000**. |
| (lookaside, the second boundary) | **identical bucket for bucket**: 24,630 hand-outs, 24,630 releases, 24,548 reuses, 82 distinct slots on both arms. |
| 3. the figure (stock and protected cumulative curves) | not drawn here. For the plotting script: the largest gap between the two cumulative fractions is 0.0434 at memsys5 (at bucket 2: 0.4249 stock, 0.4683 protected) and 0 at the lookaside. |

Every boot passed every check: exit 0, no `domain fault`, `VIRTUAL_STAGED_DONE`, `capstone_vm`
unloaded with status 0, observer `error=0` at both boundaries, and all 32 phases (row count and
FNV-1a hash) equal to the native phases of the same-answers run (`phases.json`).

## Results

Planned boot, one per arm. The two extra boots per arm (see Repetitions) gave identical numbers.

| boundary | arm | hand-outs (attempts = issues) | releases | reuses | distinct starts | median bucket |
|---|---|---|---|---|---|---|
| memsys5 | stock | 3,118 | 3,118 | 2,584 | 534 | 3 |
| memsys5 | protected | 3,583 | 3,583 | 3,096 | 487 | 3 |
| lookaside | stock | 24,630 | 24,630 | 24,548 | 82 | 2 |
| lookaside | protected | 24,630 | 24,630 | 24,548 | 82 | 2 |

Reuses per bucket (bucket k: distance 2^(k-1) to 2^k - 1; empty buckets omitted):

| k | memsys5 stock | memsys5 protected | lookaside stock | lookaside protected |
|---|---|---|---|---|
| 1 | 714 | 913 | 6,822 | 6,822 |
| 2 | 384 | 537 | 9,391 | 9,391 |
| 3 | 402 | 438 | 4,632 | 4,632 |
| 4 | 396 | 440 | 1,490 | 1,490 |
| 5 | 231 | 232 | 742 | 742 |
| 6 | 189 | 210 | 962 | 962 |
| 7 | 98 | 103 | 82 | 82 |
| 8 | 55 | 72 | 19 | 19 |
| 9 | 72 | 52 | 36 | 36 |
| 10 | 37 | 66 | 38 | 38 |
| 11 | 2 | 12 | 165 | 165 |
| 12 | 4 | 21 | 105 | 105 |
| 13 | | | 64 | 64 |

`reusediff` = sum over k of |stock_k/N_s - protected_k/N_p| / 2, with N the arm's reuses.

Releases equal hand-outs on both boundaries: everything handed out was returned by the end of the
run (memsys5Shutdown releases no block).

**The protected hand-outs account for the protected lifetimes.** The launcher's per-process line
gives 30 lifetimes on stock and 28,247 on protected. The protected excess, 28,217, is the 3,583
memsys5 plus 24,630 lookaside hand-outs counted here (28,213) plus 4. Besides hand-outs the patch
derives 4 lifetimes in this run (the heap and its pool in `memsys5Init`, once; the lookaside buffer
and its generation in `setupLookaside`, called once, at `sqlite3_open`, since speedtest1 configures
no lookaside of its own). The exact match is what the hooks seeing every protected hand-out
predicts, and a missed or double-counted hand-out would break it.
This settles the open point of the memory-behaviour README ("the number of memsys5 hand-outs is not
exposed"); that run's 28,256 differs by 9 because it also ran `--stats`.

The memsys5 difference is the mechanism the pre-registration names (memory-behaviour run,
ce31d2a10e6e): the patch makes `memsys5Size` report the request rounded to 8 instead of the
power-of-two block, and SQLite sizes buffers and decides reallocations from that; the protected
arm asks memsys5 for 465 more blocks. Why the lookaside sequence stays identical is not measured;
a reading of the source suggests that its hit/miss decision compares the request with the fixed
slot size and does not consult `sqlite3MallocSize`. Both are consistency with the named mechanism,
not a separate measurement of it.

## Hook placement

`hooks-stock.patch` and `hooks-protected.patch` in
[`experiments/study/reuse-hooks/sqlite/`](../../../../../experiments/study/reuse-hooks/sqlite/),
generated by `make-patches.py` from the same five insertions at the same anchor lines. They differ
only in line offsets and in the context lines next to the two lookaside returns, which the
lifetime patch rewrites (`return (void*)pBuf;` against `return lookasideIssue(db, pBuf);`). Both
arms, file `sqlite3-capstone.c`:

| event | function | placement | start, size |
|---|---|---|---|
| memsys5 hand-out | `memsys5MallocUnsafe` | after `mem5.aCtrl[i] = iLogsize;` (block taken, before the return) | `&mem5.zPool[i*szAtom]`, `iFullSz` (the power-of-two block) |
| memsys5 release | `memsys5FreeUnsafe` | after `iBlock` is computed, before the buddy merge | `&mem5.zPool[iBlock*szAtom]` |
| lookaside hand-out | `sqlite3DbMallocRawNN` | after `anStat[0]++`, in the `pFree` and the `pInit` branch | the slot `pBuf`, `lookaside.sz` |
| lookaside release | `sqlite3DbFreeNN` | before `pBuf->pNext = db->lookaside.pFree;` | the slot `pBuf` |

The starts are the addresses memsys5 and the lookaside compute themselves (pool plus atom index,
the free-list slot), not the caller's capability; in the protected arm the slot `pBuf` in
`sqlite3DbFreeNN` is `lookasideRelease`'s pointer at the same address. A `memsys5Realloc` that
moves is seen as its own `memsys5MallocUnsafe` and `memsys5FreeUnsafe`; one that stays in place,
and an in-place `sqlite3DbRealloc` of a slot, are not events. Frees that SQLite only measures
(`db->pnBytesFreed`) return before the release hook.

The observer is `experiments/study/reuse-gap-observer.h` (sha256 `627b0fe3...`, unchanged; its unit
test `test-reuse-gap-observer.c` passes). `exp-reuse-gap.h` (78636426...) holds one observer per
boundary with tables in static storage (memsys5 2^17 slots, lookaside 2^12), registers the report
with `atexit` at the first event, and prints one `REUSE-GAP <boundary> attempts=.. issues=..
releases=.. reuses=.. distinct=.. capacity=.. error=.. bins=..` line per boundary on stderr.

A native pre-check (gcc, the stock hook patch, memsys5 over the same heap) gave `error=0` at both
boundaries before any Capstone build. Its counts are a different platform's and are not used.

## Builds

Source: `git archive 16d2171a3edc capstone` with the `reuse-hooks/sqlite` and `reuse-hooks/postgres`
files (uncommitted here) added, and llvm-capstone `compiler-rt/lib/builtins` at 9bc6f9adf1c9,
committed on the build host as one snapshot commit (clean tree when building). The `collect.py`
scripts were written after the builds and ran beside the snapshot, not from it.

`experiments/study/reuse-hooks/sqlite/build.sh <sdk> <sources> <out> <arm>` runs the memsys5 steps
of `ports/sqlite/app/build-domain.sh` (not modified) with one added step: copy `sqlite3.c`, apply the
lifetime patch for protected (`patch -F0`), apply `hooks-<arm>.patch` (`-F0`), compile with the
port's flags (`-O2 -DHAVE_MALLOC_H=1 -DHAVE_MALLOC_USABLE_SIZE=1 -DSQLITE_ENABLE_MEMSYS5
-I capstone/runtime/include`) plus `-DEXP_REUSE_GAP`. It builds `speedtest1.dom` only (not the shell).

```sh
SQLITE_322_CACHE_ROOT=$S/cache bash capstone/ports/sqlite/app/prepare-sources.sh $S/src
MUSL=$(MUSL_CACHE_ROOT=$S/tmp/musl-src bash capstone/ports/musl-capstone/prepare-musl-capstone.sh | tail -1)
bash capstone/runtime/virtual/build-sdk.sh $S/vrt "$MUSL"
bash capstone/experiments/study/reuse-hooks/sqlite/build.sh $S/vrt/sdk $S/src $S/stock stock
bash capstone/experiments/study/reuse-hooks/sqlite/build.sh $S/vrt/sdk $S/src $S/protected protected
```

Checks on the inputs:
- prepared `sqlite3.c` 7fc92e19... and `speedtest1.c` 16750ede..., as in the same-answers run;
- reversing the hook patch from each arm's compiled copy gives exactly the same-answers engine
  sources (stock 7fc92e19..., protected b297fc3c...);
- `libapplication-runtime.a` 4748adba... and `libc-capstone-virtual.a` a0f5c573..., as in the
  same-answers run.

| file | sha256 |
|---|---|
| stock `speedtest1.dom` | `147fdcd3074bbcb0e3224595f4821b7023c28103a068e9e244bbb06c7f9a27de` |
| protected `speedtest1.dom` | `c6a1a0f85f7e38360a7dd9acef6206aff8d09345cdbe301579722a34534db4a5` |
| stock `src/sqlite3-capstone.c` (hooked) | `01d2f0fb2378d2cf1b351e83938921fd7d125cdc678d7dc63b49b0e865ec1a08` |
| protected `src/sqlite3-capstone.c` (hooked) | `a0bd2e89f4c2bd72c2da1c500bc68bbcca5577cc010cd7ed50203d20c2bc743d` |
| `hooks-stock.patch` / `hooks-protected.patch` | `e662978c...` / `1ddad841...` |
| clang | `4c6135feafae20c23edf01a4964d51459d7acb85469cd50166704e1f9a3365d4` |

## Runs

The pre-registered platform: QEMU af37cc32, launcher 3975314a, module a1c6cb6b (full hashes in
`reuse.json`; kernel, firmware and rootfs as in the same-answers run). One boot per arm of
`capstone/runtime/virtual/run-staged.py --qemu $K/kit-sublet/qemu-system-riscv64 --images
$K/kit-sublet/images --stage <stage> --work <fresh dir> --timeout 3600`, each with its own
`CAPSTONE_QEMU_LOCK` file; the two SQLite and two PostgreSQL boots ran at once. The images inside
each boot's `stage.ext4` were extracted afterwards (`debugfs dump`) and match the hashes above. The
gate (stock; protected differs only in the label):

```sh
#!/bin/sh
cd /mnt/vm || exit 1
dmesg -n 1
insmod capstone_vm.ko || exit 1
export CAPSTONE_EXEC_DIAGNOSTICS=1 CAPSTONE_VM_STATS=1
cd /tmp && rm -f st.db st.db-journal
echo PORT_BEGIN:sqlite-stock
t0=$(date +%s)
/mnt/vm/capstone-vexec /mnt/vm/speedtest1.dom --heap 8388608 64 --size 1 st.db
rc=$?
echo PORT_WALL:sqlite-stock:$(($(date +%s) - t0))
echo PORT_END:sqlite-stock:$rc
cd /mnt/vm
echo STATS_AFTER:$(./capstone-vexec --stats)
rmmod capstone_vm
echo RMMOD:$?
echo VIRTUAL_STAGED_DONE
```

`collect.py` (in the reuse-hooks directory) parses the serial logs into `reuse.json` and refuses to
write it if any check above fails.

## Repetitions

Two more boots per arm from the same stages (8 boots at once, with the PostgreSQL ones): every
counter and every bucket at both boundaries is identical to the planned boot, on both arms
(`repetitions` and `repetitions_identical_to_planned_boot` in `reuse.json`).

## Wall times

| step | seconds |
|---|---|
| prepare sources / virtual SDK / build per arm | 1 / 4 / 18 |
| planned boot, QEMU start to done, stock / protected (4 boots at once) | 37 / 37 |
| guest wall around the launcher call, stock / protected | 29 / 29 |
| repetition boots (8 at once) | 36-39 |

## Deviations and limits

- **No `--stats`.** The command is the same-answers one; the memory-behaviour run added `--stats`,
  which runs extra statements (its lookaside hits were 24,639 against 24,630 hand-outs here) and
  makes SQLite measure-only frees.
- **Two extra boots per arm**, beyond the single planned boot, as a variation check; they changed
  nothing.
- **A second copy of the run script started by mistake**, about 17 s after the first: a process
  check grepped for the run directory, which the script's command line does not contain. It copied
  identical stage files 14 s after the boots had packed their `stage.ext4` images, and its four
  `run-staged.py` calls stopped at once on the existing work directories. It overwrote those calls'
  stdout files (not the serial logs). The images inside the packed `stage.ext4` files were checked
  against the build hashes afterwards.
- **The wrapper builds only `speedtest1.dom`**, and its compile step re-states the port's flags
  instead of calling `build-domain.sh`, which has no hook for an extra patch or define.
- N = 1 planned boot per arm on one host; the boundary counts are deterministic across 3 boots.
