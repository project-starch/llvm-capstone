# Nested-allocator workloads: the cost of protecting an application's own allocator

## Status (2026-11-14)

| Item | State | Configuration |
|---|---|---|
| speedtest1 `--size 1`, QEMU twin | **measured**: answers equal; program +0.63 % instructions; 28,216 CREVOKEs, subtree mean 1.0001 / max 3; peak table +4.8 KB; kernel +3 % from the native syscall page scan (cause of the +3 % open) | FRCDKMS, native route, ids55, QEMU qemu55 a0501ebc39 |
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

### Results: QEMU twin, speedtest1 `--size 1` (measured 2026-11-14)

Two boots on slot 28. The pre-registration is commit 0bfc5c7ebec4, pushed before the second boot.
- `nst1-q1`: plain twin, the four commands as pre-registered.
- `nst1-q2`: the file-database pair in ABBA order (u, p, p, u), under `capmixn`. That plugin is the SPEC lane's capmix3
  with CDERIVE and CREVOKE counted as their own classes. Its window runs from the stage's `mark1` to its `mark2`, around
  each command.

The program-instruction rule is capmix3-read.py's: user buckets below 0x40_0000_0000 that execute capability
instructions, with the firmware range excluded.

**Pre-registration check.**

| Prediction | Verdict |
|---|---|
| Answers equal to the reference | **holds**: 32/32 phases equal `phases.json` in all six file-database runs, both arms |
| File database works on this kit | **holds**: every file-database run exits 0 |
| `--memdb` | **void; my error in the plan.** Both arms exit 1 with `unknown option: --memdb`, because SQLite 3.22.0's speedtest1 has no such option (the earlier `--memdb` runs used the 3.53 port). The file database is the configuration, as in every earlier 3.22 run. |
| About 28,000 more lifetimes | **holds**: CDERIVE 28,226 vs 9, CREVOKE 28,216 vs 2 (executed, in the program window) |
| Program instructions +0.5 % to +5 % | **holds**: +0.63 % |
| Control: more ended lifetimes on protected | **fired**: 28,216 CREVOKEs vs 2 |

**Measured.**

| Arm | Run | Program instructions | CDERIVE | CREVOKE | Kernel instructions in the window |
|---|---|---|---|---|---|
| not protected | u1 | 190,752,302 | 9 | 2 | 2,030,194,185 |
| protected | p1 | 191,946,580 | 28,226 | 28,216 | 2,090,533,368 |
| protected | p2 | 191,946,569 | 28,226 | 28,216 | 2,090,831,007 |
| not protected | u2 | 190,752,280 | 9 | 2 | 2,029,918,641 |

- Program instructions: **+1,194,284 (+0.626 %)**. Repeats agree within 22 instructions. That is about 42 instructions
  per CDERIVE/CREVOKE pair, the two lifetime instructions included.
- Kernel instructions in the same window: **+60.6 M (+3.0 %)**. The native counters agree: `between_cycles`, the time
  outside the context, is +61 M. Kernel instructions are ten times the program's on this twin. They are attributed
  below.
- Twin wall time per command, about 2.25 s in both arms: QEMU time, not a measurement.

**Where the cost comes from (twin, `nst1-q3u` / `nst1-q3p`, one boot per arm).**

These two boots used a private, measurement-only QEMU: `nested/qemu-nst` = qemu55 `a0501ebc39`, plus counters in
`target/riscv/cap_rev_tree.c` for allocations, CREVOKE subtree sizes and peak live records. They also ran `capmixn` at
64-byte granularity, mapped to `perf/natio/linux-twin/System.map`. Both arms again gave 32/32 reference answers.

| Quantity | not protected | protected |
|---|---|---|
| CDERIVE (allocations by derive), whole boot | 9 | 28,226 |
| CREVOKE, successful / refused | 2 / 0 | 28,216 / 0 |
| Nodes ended by CREVOKE | 2 | 28,219 |
| Revoked subtree size: mean / max | 1 / 1 | **1.0001 / 3** (28,214 of size 1, 2 of size 2-3) |
| Peak live records (all allocations minus all invalidations, whole boot) | 95 | 398 |
| Peak table memory at 16 B per live record | 1.5 KB | **6.4 KB** (+4.8 KB) |

- **The protection itself.** Each memsys5 block or lookaside slot is a leaf: CREVOKE at its free ends one node.
  - The only larger subtrees are the pool and lookaside-generation teardowns at close, and they end at most 3 nodes,
    because by then almost every block has been freed.
  - So on speedtest1, CREVOKE's subtree walk costs nothing beyond a single record.
  - In the program, the cost is +1.19 M instructions, about 42 per protected allocation, the lifetime pair included.
- **The kernel delta is not lifetime work.** In both arms the kernel's time is dominated by the native route's
  syscall path, which has nothing to do with the patch:
  - `pin_remaining` (1.20 G instructions not protected, 1.22 G protected) and `next_resident` (0.55 G / 0.57 G);
  - called from checked copy (`checked-copy-dom.h:190`) on every syscall that passes a user buffer, 15,151 per run in
    both arms;
  - each call scans the arena's page array linearly and walks the page tables for pages not yet held.
- **The protected arm's extra +60 M kernel instructions:**
  - +33.5 M in those two functions;
  - +25 M in the spinlock helpers they take;
  - +1.4 M in `native_rw`;
  - the node-record functions (`retire`, `resolve`) move by under 40 k.
- **Why the scan costs 2-3 % more when protected is not attributed yet.** My working hypothesis is different page
  residency in the 8 MiB memsys5 arena: protected memsys5 rounds blocks to the bounds granule, and its peak "Memory
  Used" differs.
- **The scan is real on silicon too.** The board kernel's `capstone_vm.c`
  (`perf/k2n/linux-board-k2n-native-hpm`) differs from the twin's only by the HPM reads.

### Pre-registration: silicon, speedtest1 `--size 1` (written before the boot)

**Configuration.**
- Bitstream `caplifive_ids55-native-tagcache-mlp-fpieee-hpmreplay-dynbypass3-dtlb64-shrinkfix-ic32k-r72-stcnoclaim_a47b588a1`,
  sha256 `ff36870d…`, on board 1. It is the board's resident bitstream, so no flash.
- Kit `t3-smoke/bkit73-k2n-native-batch-hpm`: K2N native kernel with HPM counters, monitor 4f51541, node_flat_records
  262144.
- Stage `nested/st1/board/b1-stage` (`make-train-stage.sh` of the twin's `stages/q2`). Its `train/run.sh` sha256 is
  `6cd49b361db37dd9…`, and the images are the two above.
- Budget: 900 s × BOARD_TIMEOUT_MULT 4 = 3,600 s. The kit's stub dry run prints `BAKED_TIMEOUT=3600`.

**Order.**
1. The kit's controls `b0-stats` and `b0-hello`.
2. The native smoke.
3. The four commands u1, p1, p2, u2 (ABBA), each in a fresh run directory.

If a control fails, the boot is VOID.

**Readings per command:** rc, the 32 oracle lines, and the native counters' deltas:
- `run_cycles`: in-context cycles;
- `between_cycles`: kernel and other work between context entries;
- `runs`;
- hpm3..8: L1I miss, L1D miss, DTLB miss, mispredict, issue stall, scoreboard full;
- the uptime span.

**Predictions.**
- **Answers:** every command exits 0 with 32/32 oracle lines equal to the reference. Anything else stops the
  comparison. A protected-only failure would be a silicon finding, recorded as such.
- **Repeats agree.** The two runs of each arm agree within 1 % on `run_cycles` and within 2 % on `between_cycles`.
  If they do not, the arm difference below is not interpretable at N = 2.
- **In-context cost (`run_cycles`, protected minus not protected): +0.5 % to +3 %.**
  - The twin's instruction delta is +0.63 %.
  - Each of the 56,442 lifetime instructions also pays the rev-node unit's record traffic: a header load, a record
    read and write, and a header publish for CREVOKE.
  - Taken as 20-60 cycles each, against roughly 300-400 M program cycles, that adds +0.3 % to +1 %.
- **Kernel (`between_cycles`): +1 % to +5 %.** The twin reads +3 %, from the native syscall page scan. The board kernel
  runs the same code.
- **Whole command (uptime span): +1 % to +4 %.** The kernel scan dominates both arms.
- **Duration:** each command takes 1-2 min; the boot about 15-20 min in all.
