# Nested-allocator workloads: the cost of protecting an application's own allocator

## Status (2026-11-14)

| Item | State | Configuration |
|---|---|---|
| speedtest1 `--size 1` | **images built**; QEMU twin pre-registered below, not yet run | FRCDKMS, native route, ids55 |
| speedtest1 on silicon | pending (after the twin; a board slot from the SPEC lane) | bitstream `a47b588a1`, kit bkit73 |
| speedtest1 `--size 100` | pending (after `--size 1` has been looked at) | as above |
| CPython, PostgreSQL, tshark, FFmpeg, memcached | pending, in that order, after speedtest1 | as above |
| CREVOKE worst case (subtree 1 .. 10^6) | pending: RTL simulation | capstone-ariane `a47b588a1` |
| record-cache counters | event 23 (verdict-cache miss replays) exists; lookups / raw misses / walk-busy cycles proposed as events 29-31 for the next bitstream, **waiting on the lead** | - |

Nothing in this file is a measurement yet unless a row says "measured".

## What is compared

Two arms per application, as agreed with the project lead on 2026-11-14:

- **protected**: the application's nested-allocator lifetime patch is applied. Each sub-allocator object (a memsys5
  block, a pymalloc block, a memory-context chunk, a wmem block, a pool buffer, a slab item) is its own lifetime
  node, a CDERIVE child of its pool/arena/context, and freeing or resetting the parent ends the subtree with CREVOKE.
- **not protected**: the same source tree with the patch not applied.

Everything else is identical: the same Sublet processor (bitstream), SDK, compiler and flags, kit, kernel, board and
inputs. malloc lifetimes (musl mallocng, `heap_lifetimes: true`) are ON in both arms, so the difference is the nested
protection alone. There is no pure-RISC-V arm and no LIFETIMES=OFF arm.

## Configuration

- **Compiler and SDK (FRCDKMS, the SPEC table's image configuration):**
  - driver `perf/cclever/arms/FRCDKMS/sdk/capstone-cc`, `sdk.json` cc = llvm-c9, clang 22.0.0git llvm-capstone
    `29027126f39b` (compiler-capbase-rvc-c8);
  - runtime/libc/builtins from `perf/memset/sdk` (llvm-capstone `beb1bbabe26a`; heap LIFETIMES=ON, OBJECT_BOUNDS=ON,
    SCRUB_FREE=OFF, MAPPING_REUSE=ON, MEMSET_CAP_ZERO=ON);
  - flags: `-O2` plus build-arm.sh's set and `arms/FRCDKMS/extra-flags` (listed in the build script).
- **Route:** native Linux task (Option B, 55-bit lifetime ids, K2N kernel).
- **Silicon:** `caplifive_ids55-native-tagcache-mlp-fpieee-hpmreplay-dynbypass3-dtlb64-shrinkfix-ic32k-r72-stcnoclaim_a47b588a1`,
  sha256 `ff36870de12a133ebdf0a5fcc254a8bdbf8bacab2373666c79fcb2f0286076e1`, kit `bkit73-k2n-native-batch-hpm`.
- **QEMU twin:** `perf/faultesc/run.sh` (qemu55, `--ids55 --arm direct`), kernel images `perf/natio/images`,
  `node_flat_records` 262144 (the board kit's value).

## speedtest1 (SQLite 3.22.0)

**Patch.** `capstone/ports/sqlite/sublet/sublet-3220000-memsys5.patch` (dev, last changed f1d522d7fff1): memsys5
blocks and lookaside slots become CDERIVE children; free and lookaside teardown CREVOKE them. The build selects it with
`SQLITE_HEAP=memsys5-sublet` (`capstone/ports/sqlite/app/build-domain.sh`); the unprotected arm compiles the same
amalgamation copy with `SQLITE_ENABLE_MEMSYS5` and no patch.

**Workload.** speedtest1 `main --size 1` over an 8 MiB memsys5 heap (`--heap 8388608 64`), the configuration of every
earlier speedtest1 run in this repository, with the result oracle (`-DSQLITE_STUDY_ORACLE`: rows and an FNV-1a hash per
phase, 32 phases). Reference answers: `capstone/ports/sqlite/app/results/2026-10-11-same-answers/phases.json`.

**Images (built 2026-11-14).**

| arm | image (strip.dom) sha256 | sqlite3.o sha256 | CDERIVE / CREVOKE in sqlite3.o |
|---|---|---|---|
| not protected | `817b165232d3ca10…` | `a74891aa603db1e4…` | 0 / 0 |
| protected | `945fb69f44373154…` | `1205b73b9cbd43a2…` | 14 / 5 |

The static count is `capstone/ports/common/application/lifetime-instructions.py`, checked two-sided: `--expect present`
passes on the protected object and fails on the unprotected one, and `--expect absent` does the opposite. Sources:
sqlite3.c `7fc92e19a3af61f3…`, speedtest1.c `16750ede41d01935…`, patch `f16132a5c03e6316…`.

### Pre-registration: QEMU twin, speedtest1 `--size 1` (written before the run)

One twin boot, four commands in this order, each in a fresh run directory:
1. not protected, file database `st.db`;
2. protected, file database;
3. not protected, `--memdb`;
4. protected, `--memdb`.

The file database is the earlier runs' configuration; `--memdb` is the fallback, because a native-route speedtest1 has
failed at its first CREATE TABLE on a file database before (fcntl lock `-ENOSYS`, nat-fair-sdk).

Predictions:
- **Answers:** both arms produce 32 oracle lines equal to the reference `phases.json` (`--heap 8388608 64 --size 1`
  entry), for every command that exits 0.
- **File database:** works on this kit (its command line sets `native_locks=1`). If it fails with a lock or I/O error in
  both arms, that is a route limitation, not a protection result, and the memdb commands carry the comparison.
- **memdb in 8 MiB:** UNCERTAIN. The database itself then lives in the memsys5 heap; an out-of-memory failure in both
  arms would mean the heap must grow for memdb, again in both arms equally.
- **Lifetime counts (protected minus not protected):** on the order of the 2026-10-11 virtual-route measurement,
  28,256 against 30 lifetimes allocated, i.e. about 28,000 more CDERIVEs and a similar number of ended nodes.
- **Cost:** the protected arm executes more instructions, within +0.5 % to +5 %. The older Sublet design measured
  +4.09 % instructions on the same workload (`docs/ref/fpga-silicon-measurements-for-paper.md` §4g); the current patch
  issues one CDERIVE and one CREVOKE per block instead of that design's split/mrev/delin/revoke sequence, so it should
  cost less. A result above +5 % would be a finding to explain, not to absorb.
- **Control:** the static count above (two-sided). In the run itself, the protected arm must show more ended lifetimes
  than the unprotected arm; if the two arms report equal lifetime counts, the patch did not take effect and the run says
  nothing about protection.
