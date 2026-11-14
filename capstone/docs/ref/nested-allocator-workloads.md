# Nested-allocator workloads: the cost of protecting an application's own allocator

## Status (2026-11-14)

| Item | State | Configuration |
|---|---|---|
| speedtest1 `--size 1`, QEMU twin | **measured**: answers equal; program +0.63 % instructions; 28,216 CREVOKEs, subtree mean 1.0001 / max 3; peak table +4.8 KB; kernel +3 % from the native syscall page scan, all of it from the patch's `aReq` array (lifetimes: kernel ±0, in-context +0.39 %) | FRCDKMS, native route, ids55, QEMU qemu55 a0501ebc39 |
| speedtest1 `--size 1` on silicon | **measured** (board 2, 15:00-15:30): 32/32 answers in every command; in-context +1.04 %, wall-clock +1.90 % (both inside the pre-registered bands); kernel repeat spread 2.47 % > 2 %, so the kernel difference is not read. Boot 1 on board 1 was VOID (unbaked empty `train/in`) | bitstream `a47b588a1`, kit bkit73 |
| speedtest1 `--size 1` on silicon, bitstream F, with event 23 | **queued**: board 3 after its x264 run (~03:00 15 Nov), pre-registered below; both arms again (ABBA), hpm8 = event 23 (verdict-cache miss replays) | bitstream F `a948e4134` (`ec675c2b…`), kit bkit75n |
| speedtest1 `--size 100` | **deferred** (SPEC-lane default, 14 Nov 21:40: breadth first; the lead may override): about 6.5 h per command on silicon | - |
| CPython (pyperformance subset), PostgreSQL (single-user pgbench), tshark (pcap), FFmpeg (decode), memcached (load) | **queued, in that order**: each first on the QEMU twin, then one silicon ABBA boot. Patches in tree: cpython-3.13.7-0014, postgresql-17.0-0003, wireshark-4.6.8-0001, ffmpeg-9.0.1-0003, memcached-1.6.45-0006. CPython twin preparation is next | FRCDKMS, native route; silicon on F |
| CREVOKE worst case (subtree 1 .. 10^6, latency, interruptibility; PostgreSQL context reset) | **queued**: RTL simulation, after the CPython twin build is under way | capstone-ariane `a948e4134` (F) |
| record-cache counters | event 23 is on hpm8 in kits bkit73n (batch) and bkit75n (F). Events 29-31 (lookups, raw misses, walk-busy cycles) would need a new bitstream; that is **the lead's call** and is not requested | - |

Only rows marked "measured" hold measurements; the rest say what runs next and where.

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
| Peak live records (all allocations minus all invalidations, every context in the boot, including 86 that are not derives) | 95 | 398 |
| Live-record memory at 16 B per record | 1.5 KB | **6.4 KB** (+4.8 KB) |

- **Table memory.** On the native route the flat lifetime table is one fixed allocation of node_flat_records x 16 B
  = 262,144 x 16 B = 4 MiB per context, whichever arm runs. Nothing grows. The +4.8 KB above is the memory the extra
  live records occupy inside that table, not a change in its size.
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
- **Why the scan costs 2-3 % more when protected: the patch's `aReq` array, not the lifetimes.** This bullet replaced
  an earlier reading (c0440089f41c) that put the kernel delta on the protected arm's +69 resident heap pages. The pages
  are real, but they are not what costs.
  - Diagnostic boot `nst1-q4`: `diag_admit=1`, vmstat around each command. It was not used for any cost number. At
    retire, the 8 MiB memsys5 arena (`+0x808000`, 2,056 pages) has 268 pages resident not protected and 337 protected
    (+69). Every other arena of the process is identical between the arms.
  - The patch changes memsys5 in four ways. Three are not lifetimes:
    - it carves a smaller pool, (nByte-8)/(szAtom+5) atoms instead of nByte/(szAtom+1);
    - it keeps `aReq`, one u32 per atom at the heap's tail (475 KB), written at every malloc;
    - it reports the 8-byte-rounded request instead of the power-of-two block as the usable size.
    The fourth is the per-block CDERIVE/CREVOKE.
  - Three attribution arms, each one step from the last, measurement only (`nested/st1/build-var.sh` and `bin/preq`):

    | arm | what it is | image sha256 |
    |---|---|---|
    | `ucarve` | not protected, pool carved as protected | `3d4854e6…` |
    | `preq` | plus `aReq` written and the block size reported, no per-block derive or revoke (13 CDERIVEs per boot) | `c578ff71…` |
    | `ppow2` | plus the per-block lifetimes (27,761 CDERIVEs); the block size is still reported | `5e537407…` |
    | protected | plus the request size reported (the full patch) | `945fb69f…` |

  - Every arm gave 32/32 reference answers.
  - Kernel instructions (`capmixn`, one profiled boot per arm, `nst1-q3u`, `q6uc`, `q6pr`, `q6pp`, `q3p`), and
    memsys5 pages swept at retire (`nst1-q5`, one boot running not protected, `ucarve`, `ppow2` and protected):

    | step | kernel instructions | of which `next_resident` + `pin_remaining` | in-context cycles | pages swept |
    |---|---|---|---|---|
    | not protected | 2,030.2 M | | 196.45 M | 268 |
    | pool carve | -13.6 M | ~0 | ±0 | 264 |
    | `aReq` | **+66.1 M (+3.3 %)** | +39.4 M; the spinlock helpers carry most of the rest | +0.06 % | not measured |
    | per-block lifetimes | **-0.6 M** | ~0 | **+0.39 %** | 278 (`ppow2`) |
    | request-size reporting | +8.4 M | +5.5 M | +0.17 % | **337** |
    | total | +60.2 M | | +0.62 % | +69 |

    In-context cycles come from the profiled boots for the middle arms (`q6pr`, `q6pp`). The first and last rows are
    the unprofiled `nst1-q5` boot, where `ucarve` and `ppow2` read the same as their profiled boots within 0.001 %.
  - **The protection's lifetimes cost the kernel nothing.** They cost the program +0.39 % in-context.
  - **The kernel's +3 % is the `aReq` array.** Most of the extra pages come from request-size reporting, which costs the
    kernel little. So the scan's cost is not proportional to the resident page count. Which part of the scan grows
    (more absent runs, or a page table instantiated where the walk used to skip a whole 2 MiB level) is not resolved.
  - The scan is the native kernel's (`capstone_vm.c`, `pin_remaining` → `next_resident`, on every syscall that passes
    a user buffer). Without it, the protection's whole cost on speedtest1 would be the in-context +0.6 %.
  - An alternative the patch could take, untested: `memsys5Size` could read the request from the block capability's
    bounds length instead of `aReq`, which would remove the array.
  - A host x86 build (lifetimes stubbed, residency by `mincore`) shows the same direction, 243 vs 298 resident heap
    pages, but splits it differently, because its pointers are 8 bytes rather than 16. The twin is the authority here.
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

**What this boot cannot show.**
- The kit's HPM set is hpm3..8 = events 1, 2, 4, 10, 22, 14. It does not include event 23 (verdict-cache miss replays),
  so if the in-context cost reads above its band, this boot cannot attribute it. A follow-up boot would carry event 23.
- No counter on this boot counts CDERIVE or CREVOKE. The evidence that the protected construct runs is image identity:
  both `.strip.dom` files in the stage have the same sha256 as the twin's, where 28,226 CDERIVEs executed.

The chain runs on board 1 as train PREREG Addendum 109 (spec-silicon 92b7b2662c97), which cites c3663f0362a1. The work
dir is `spec/board/out-nat-batch-nested-st1-b1/b1-stage`.

**Addendum 1 (2026-11-14, after boot 1): boot 1 is VOID, and the stage is fixed.**
- Boot 1 ran on board 1 from 14:27 to 14:35 (work dir `nested/st1/board/b1-run/b1-stage`). The native smoke passed all
  9 steps (`@@NS-RESULT 70`).
- No command ran. Right after `@@SPEC-ARM-BEGIN u1`, the console reads
  `cp: can't stat '/test-domains/vbatch/train/in/.': No such file or directory`.
- Cause: speedtest1 takes no input files, so the stage's `train/in` was an empty directory. The kit builds its bake
  manifest from `find . -type f`, so it never stages an empty directory. The baked `rootfs.cpio` lists
  `test-domains/vbatch/train/` with `bin/`, `mark1`, `mark2`, `plan.json` and `run.sh`, and no `in/`.
  - The QEMU twin copies the stage directory whole, which is why it did not show this.
- Fix: `train/in/NO-INPUTS`, a one-line text file, sha256 `298a9444…`. It lands in the run directory and speedtest1
  never reads it. The stage generator (`stage-gen/make-stage.py`) now writes it for every stage.
  - Nothing else changed: `train/run.sh` is still `6cd49b361db37dd9…`, and both images are the same.
  - `b1-stage.SHA256SUMS` gains exactly that one line.
- Checked off the board, on a real bake of the fixed stage (kit `bake-manifest.sh`, then the kit's `BOARD_STUB=2` deep
  dry run, which relinks the native kernel on that cpio):
  - The new cpio (`8ab3e5d7…`) carries `test-domains/vbatch/train/in/NO-INPUTS`.
  - The stub's native relink rebuilt the initramfs from that cpio, and the smoke classifier ran (PASS on boot 1's transcript).
  - run.sh's own setup line, run on the tree extracted from each cpio, succeeds on the new one: the run directory
    holds `NO-INPUTS` and `.capstone-tmp`, and `b/` holds both images. On boot 1's cpio it fails with the board's error,
    `cp: cannot stat '…/train/in/.'`.
  - The kit's `spec-classify.py` reads boot 1 as `NO RESULT` (rc 2), so the gate itself fired.
- Every prediction above stands unchanged for the re-run.

### Results: silicon, speedtest1 `--size 1` (measured 2026-11-14)

**Configuration.**
- Bitstream `…_a47b588a1`, sha256 `ff36870d…`, on **board 2**, not board 1 as pre-registered. The board-1 slot went to
  another lane's runs, and the SPEC lane placed the re-run on board 2 as train PREREG Addendum 111 (spec-silicon cbb22e752fd4).
  That addendum cites 9b325ad6fcac and records boot 1 as void. Board 2 was flashed once to this bitstream.
- Kit bkit73, the fixed stage (`train/run.sh` `6cd49b361db37dd9…`, the two images above). Boot 15:00-15:30.
- Transcript `nested/st1/board/b2-run/b1-stage/board.txt` (sha256 `9a2c1a61ee4ec18d…`), read by
  `stage-gen/parse.py` into `nested/st1/b2.json`.

**Controls.**
- `b0-stats` and `b0-hello` returned 0.
- The native smoke passed under the kit's documented waiver (`classify.py --waive-accounting`). Step 70 (task) fails
  only its two user-tick checks, "compute charged to user" and "compute resumes" (`native task failures=2`).
  Domain time is system time on this kernel.
- The kit's `spec-classify.py` reads this boot as `NO RESULT`. That is a format mismatch, not a failure: it expects
  `@@SPEC-DONE <n>` and three fields before `rc=`, and this stage prints `@@SPEC-DONE speedtest1 nested` and four.
  All four commands printed `@@SPEC-END`, and the stage reached `@@SPEC-DONE`.

**Readings.** The counters are summed over the same in-context windows as `run_cycles`. The uptime span runs from
before `mark1` to after `mark2`, so it covers the whole command.

| | u1 | p1 | p2 | u2 | repeat spread, u / p | protected vs not (ABBA means) |
|---|---|---|---|---|---|---|
| rc, oracle lines equal to the reference | 0, 32/32 | 0, 32/32 | 0, 32/32 | 0, 32/32 | | |
| `run_cycles` (in-context) | 807,968,123 | 813,284,712 | 813,924,141 | 802,440,785 | 0.69 % / 0.08 % | **+1.04 %** |
| `between_cycles` (kernel and other) | 5,036,804,504 | 5,169,234,617 | 5,248,671,297 | 5,162,802,037 | **2.47 %** / 1.53 % | +2.14 % (not interpretable, see below) |
| uptime span, s | 235.56 | 240.69 | 243.90 | 239.99 | 1.86 % / 1.32 % | **+1.90 %** |
| runs (context entries) | 15,393 | 15,443 | 15,441 | 15,393 | | +0.32 % |
| hpm3 L1I miss | 6,035,763 | 6,249,153 | 6,250,741 | 6,074,924 | 0.65 % / 0.03 % | +3.21 % |
| hpm4 L1D miss | 2,982,717 | 3,195,543 | 3,163,910 | 2,980,011 | 0.09 % / 0.99 % | +6.65 % |
| hpm5 DTLB miss | 86,981 | 109,138 | 111,151 | 86,079 | 1.04 % / 1.83 % | +27.3 % (+23.6 k) |
| hpm6 mispredict | 7,072,042 | 7,251,661 | 7,250,507 | 7,075,927 | 0.05 % / 0.02 % | +2.50 % |
| hpm7 issue stall | 157,130,907 | 162,271,990 | 163,067,601 | 157,143,349 | 0.01 % / 0.49 % | +3.52 % (+5.5 M cycles) |
| hpm8 scoreboard full | 10,041,036 | 9,835,725 | 9,674,390 | 10,219,890 | 1.77 % / 1.65 % | -3.71 % |

The SPEC lane's wall-clock cross-check from the same `@@SPEC-END` lines (235.56 / 240.69 / 243.90 / 239.99 s) agrees.

**Against the pre-registration.**
- Answers: **met.** Every command exits 0 with 32/32 reference answers, protected arm included.
- Repeats: **met for `run_cycles` (0.69 % and 0.08 %, within 1 %). Missed for `between_cycles`: the unprotected
  pair is 2.47 % apart, over the 2 % band.** By the pre-registered rule, the arm difference in `between_cycles` is
  therefore not interpretable at N = 2.
  - The kernel time rises through the boot (u1 < p1 < p2) and falls back for u2, which is not a linear drift that
    ABBA would cancel. The ABBA mean, +2.14 %, is recorded and not read.
- In-context cost: **met. +1.04 %**, inside the +0.5 % to +3 % band. On the twin it was +0.62 %.
  - That is 8.4 M cycles per protected run, or 149 cycles per lifetime instruction (56,442 per run on the twin) if all
    of it were charged to them. This boot cannot attribute it: it carries no event 23 (verdict-cache miss replays).
  - The in-domain counters move together: issue stall +5.5 M cycles, L1D misses +6.7 %, DTLB misses +27 %.
  - On the twin, 0.39 points of the in-context cost were the lifetimes, 0.17 the request-size reporting and 0.06 the
    `aReq` array (see the cost attribution above). That split is a twin reading, not a silicon one.
  - The pre-registration called for a follow-up boot with event 23 only above the band, so none is triggered.
- Kernel (`between_cycles`): **not interpretable** (the repeat rule above). The twin's +3.3 % is the `aReq` array.
- Whole command (uptime span): **met. +1.90 %**, inside +1 % to +4 %.
- Duration: **missed.** Each command took 236-244 s, not 1-2 min, and the boot took 31 min, not 15-20. The
  prediction had no measured basis; the budget, 3,600 s, held with room.

**So, on silicon:** protecting SQLite's nested allocators (memsys5 blocks and lookaside slots as Sublet lifetimes)
costs speedtest1 `--size 1` +1.0 % in-context cycles and +1.9 % wall-clock, with every answer unchanged. This file
is compared below with the SPEC temporal increment, which the brief expects it to be close to. The kernel share is
not read from this boot.

**Against the SPEC temporal increment.** The increment is Capstone temporal (FRCDKMS, heap lifetimes ON) against
Capstone spatial (FRCDKMSsp, lifetimes OFF; its SDK differs only in the allocator objects), on the same bitstream,
board and kit. So far it has one silicon row: gcc train01, bitstream `a47b588a1`, bkit73, board 1 (spec-silicon train
PREREG Addendum 108, 07d80585e4b2). Its figures, read from the two transcripts
(`spec/board/out-nat-batch-hpm-frcdkms{,sp}-b1/502.gcc_r-train01-native/board.txt`, end minus begin):

| | gcc train01: temporal vs spatial | speedtest1 `--size 1`: protected vs not |
|---|---|---|
| in-domain cycles (`run_cycles`) | 58.377 G vs 58.016 G, **+0.62 %** | **+1.04 %** |
| wall-clock | 2,446.30 s vs 2,431.58 s, **+0.61 %** | **+1.90 %** |

- Protecting the nested allocator costs speedtest1 about 1.7 times gcc's temporal increment in-domain, and about 3
  times on wall-clock.
- The wall-clock gap includes the kernel time this boot cannot read. On the twin, that time is the patch's `aReq` array
  working through the per-syscall page scan, not the lifetimes.
- The two are different programs measured one way each: N = 1 SPEC row, ABBA N = 2 here. Nothing is normalised for
  allocation rate: speedtest1 runs 56,442 lifetime instructions per run, and gcc's count is not in this file.
- The other seven spatial rows land in the SPEC lane's table (spec-silicon
  `capstone/ports/spec-cpu2017/results/2026-11-14-three-arms/README.md`).

### Pre-registration: silicon on bitstream F, speedtest1 `--size 1`, with event 23 (written before the boot)

**Why this boot.**
- The lead's standing ruling: experiments run only on the best bitstream, and F is now the table's.
- This boot re-measures both arms on F.
- It also reads how many verdict-cache misses the protection adds. That is the only record-cache event this RTL has.
  The batch boot could not attribute its +1.04 %.

**Configuration.**
- Bitstream F: `caplifive_ids55-native-tagcache-mlp-fpieee-hpmreplay-dynbypass3-dtlb64-shrinkfix-ic32k-r72-stcnoclaim-twc16k-merge-hpmtag_a948e4134`,
  sha256 `ec675c2b…`. It is the batch plus a 16 K-entry tag-word cache.
- Board 3, after its x264 run, placed by the SPEC lane.
- Kit `t3-smoke/bkit75n-k2n-native-twcf-hpmnest`: bkit75 (bkit73 with F's bitstream) plus one firmware change. hpm8
  counts event 23 (R-43 verdict-cache miss replays, `perf_counters.sv` `5'b10111`, the same on a948e4134 and
  a47b588a1) instead of 14 (scoreboard full). hpm3..7 are unchanged (L1I, L1D, DTLB, mispredict, issue stall).
- Firmware `e246852282202790`. A real build passes the dom_stack gate. It differs from bkit73's board firmware
  (`38284328eff704a3`, reproduced by a rebuild) in the mhpmevent8 immediate only (`csrwi mhpmevent8, 0xe` -> `0x17`;
  2 bytes).
- Stage `nested/st1/board/f3-stage`: a byte-identical copy of the board-2 stage (`f3-stage.SHA256SUMS`, `train/run.sh`
  `6cd49b361db37dd9…`, the same two images), ABBA u1 p1 p2 u2, `--heap 8388608 64 --size 1 st.db`.
- Checked off the board, as rule 8 asks:
  - a real bake (cpio `f0243639…`, carries `train/in/NO-INPUTS`);
  - the kit's `BOARD_STUB=2` run on that bake: the native relink rebuilt the initramfs, the smoke classified PASS, and
    `BAKED_TIMEOUT=3600`.
  - Record: `nested/st1/board/f3-recheck`.

**Readings:** as on the batch boot (rc, 32 oracle lines, `run_cycles`, `between_cycles`, runs, hpm3..8, uptime span),
with hpm8 now event 23.

**Predictions.**
- **Answers:** every command exits 0 with 32/32 reference answers. Anything else stops the comparison.
- **Repeats:** within 1 % on `run_cycles` and within 2 % on `between_cycles`, or that difference is not read.
  - The batch boot missed the kernel band (2.47 %).
- **In-context cost on F:** +0.5 % to +3 %, the same band as on batch, where it read +1.04 %. F's change is the
  tag-word cache, not the lifetime records, so there is no reason to expect a different cost.
- **Whole command:** +1 % to +4 % (batch: +1.90 %).
- **Event 23, protected minus not protected, per command: +28,000 to +200,000.**
  - The lower bound is the compulsory miss of each new lifetime. The twin counts 28,226 CDERIVEs per run, and each
    new block or slot gets a fresh record whose first checked access cannot hit.
  - The cache is 256 entries (64 x 4), and the protected arm's peak live records on the twin are 398. So capacity and
    conflict misses are expected to add at most a few times the compulsory count.
  - Below 28,000, either a new lifetime's first access is not a checked access, or the event does not count what this
    file assumes. Either way, the miss count does not explain the cost.
  - Above 200,000, the verdict cache thrashes under the protected heap.
- **Event 23, not protected:** small next to the protected arm. Its whole 8 MiB heap is one lifetime.
- **What will be derived, not predicted:**
  - an upper bound on cycles per verdict-cache miss: (`run_cycles` delta) / (event-23 delta);
  - the share of the in-context cost that misses could explain at a cost per miss taken from the RTL replay path
    (to be read from the RTL before the boot ends, not fitted to the result).
- **Duration:** 236-244 s per command on batch, so about 4 min each and about 31 min per boot. Budget 3,600 s per
  rung.
- **VOID** if `b0-stats` or `b0-hello` fails, or if the smoke fails outside the kit's accounting waiver.
 On the twin it is the patch's `aReq` array working through the native route's per-syscall page
scan, not the lifetimes.

