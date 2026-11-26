# Nested-allocator workloads: the cost of protecting an application's own allocator

## Status (2026-11-15)

| Item | State | Configuration |
|---|---|---|
| speedtest1 `--size 1`, QEMU twin | **measured**: answers equal; program +0.63 % instructions; 28,216 CREVOKEs, subtree mean 1.0001 / max 3; peak table +4.8 KB; kernel +3 % from the native syscall page scan, all of it from the patch's `aReq` array (lifetimes: kernel ±0, in-context +0.39 %) | FRCDKMS, native route, ids55, QEMU qemu55 a0501ebc39 |
| speedtest1 `--size 1` on silicon | **measured** (board 2, 15:00-15:30): 32/32 answers in every command; in-context +1.04 %, wall-clock +1.90 % (both inside the pre-registered bands); kernel repeat spread 2.47 % > 2 %, so the kernel difference is not read. Boot 1 on board 1 was VOID (unbaked empty `train/in`) | bitstream `a47b588a1`, kit bkit73 |
| speedtest1 `--size 1` on silicon, bitstream F, with event 23 | **measured** (board 3, 15 Nov 02:26-02:51): 32/32 answers in every command; in-context +1.67 %, kernel +2.10 % (repeats in band this time), wall-clock +1.95 %; event 23 +46,588 verdict-cache miss replays per run, which at the RTL's estimated 17-20 cycles each explain 7-8 % of the in-context cost | bitstream F `a948e4134` (`ec675c2b…`), kit bkit75n, board 3 |
| speedtest1 `--size 100` | **deferred** (SPEC-lane default, 14 Nov 21:40: breadth first; the lead may override): about 6.5 h per command on silicon | - |
| CPython 3.13.7, pyperformance subset (7 benchmarks), QEMU twin | **measured**: 28/28 answers equal to native; in-context +0.66 % (richards) to +2.84 % (json_dumps), +1.63 % summed; kernel ±0.5 %; 2.99 M CDERIVEs, subtree max 2, peak live records 76,767 (1.23 MB) | FRCDKMS SDK, -O2, trimmed built-ins, linked --no-relax; QEMU qemu55 a0501ebc39 |
| CPython on silicon, bitstream F | **measured** (board 2, 15 Nov 07:55-09:58): 28/28 answers equal to native; richards' protected repeats differ 1.22 % (band 1 %), so it is not read; the six read sum to in-context **+5.17 %** (above the pre-registered +1.3 % to +5 %; twin +2.06 % over the same six), kernel +0.30 %, wall-clock +4.43 %; event 23 +5.41 M, explaining 10.5-12.3 % of the cost; L1D misses +14.7 % | F `a948e4134`, kit bkit75n, board 2 |
| PostgreSQL 17.5, single-user pgbench-derived workload, QEMU twin | **measured** (15 Nov 04:21-04:42): 4/4 commands give the native answers; in-context **+4.04 %**, kernel +0.43 %, repeats within 0.02 %; 2.31 M CDERIVEs and 668 k CREVOKEs per run, subtree mean 3.45 / **max 34,509** (a context reset), peak live records 105,043 (1.68 MB). Bring-up found a port defect, **ISSUES.md M-15** (int2vector/oidvector read 8 bytes off at MAXALIGN 16, caught as CAP_OOB by the protected arm). Harness changes, the same in both arms: root (the loader admits euid 0 only), shims over refused or unserved calls; patch 0003 with the Q-15 placement repair | FRCDKMS SDK, -O2, native route, image set 10, QEMU qemu55 a0501ebc39 (counts: qemu-nst) |
| PostgreSQL on silicon, bitstream F | **measured** (board 1, 15 Nov 08:34-10:00): every command gives the native answers; the protected pair's in-context repeats differ 1.10 % (band 1 %), so the in-context difference is **not read** (ABBA mean +6.67 %); kernel +1.07 %, wall-clock **+4.38 %**; event 23 +2.04 M per command; L1D misses +11.4 %, DTLB +30 % | F `a948e4134`, kit bkit75n, board 1 |
| FFmpeg 9.0.1 decode (10 s clip, 300 frames), QEMU twin | **measured** (15 Nov 05:21): all commands decode the stock reference frames; in-context **-0.030 %**, kernel -0.36 %. The protection adds about 3,300 lifetimes per decode (pools recycle), subtree max 2, peak 474 live records. Needs this lane's Q-15 placement repair of 0003 (unrepaired, CDERIVE refuses a frame-plane buffer), and on llvm-c9 `-fno-builtin` in both arms (else cause 24 in both: C-69). With llvm-c15's C-69 fix and builtins on, all commands match and the cost reads -0.031 % | FRCDKMS SDK, -O2 -fno-builtin, native route, set w4, QEMU qemu55 a0501ebc39; builtins-on reading: FRCDKMSc15 SDK, set w6 |
| FFmpeg on silicon, bitstream F | **measured** (board 2, 15 Nov 10:11-10:36): all commands decode the stock frames and the flipped control fires; the unprotected pair's repeats miss both bands (in-context 1.04 %, kernel 2.71 %), so neither difference is read (ABBA means -0.32 % / -0.34 %); wall-clock -0.74 %; event 23 +1,388 per command. Not resolved from noise, as on the twin | F `a948e4134`, kit bkit75n |
| tshark 4.6.8, `-r -V -n` on 9,000 frames, QEMU twin | **measured** (15 Nov 05:45-06:45): every output equals stock tshark's byte for byte; in-context **+2.69 %**, kernel +0.02 %; 7.18 M CDERIVEs and 2.73 M CREVOKEs per run, with one packet-scope reset of 256-1,023 nodes per frame, subtree max 29,134, peak live records 33,998 | FRCDKMS SDK and flags on tshark and its seven dependencies, `--no-relax`, native route, QEMU qemu55 a0501ebc39 |
| tshark on silicon, bitstream F | **protected arm did not return** (board 1, 15 Nov 10:01-13:40, four one-arm boots U P P U): both unprotected commands give stock's output in 1,112-1,125 s; both protected commands printed nothing for at least 3,327 s (2.98 times that) before the 3,600 s rung timeout. The residue commit pc maps to the CDERIVE in the patched allocator. Not read as a cost and not called a wedge. A heartbeat boot to separate slow from stopped was pre-registered below and **not run**: the nested work was paused on 15 Nov evening for the final SPEC test run (the lead's direction), so the question stays open | F `a948e4134`, kit bkit75n, board 1 |
| memcached (load) | **deferred**: the native route cannot run it. The table refuses `socket`, `socketpair`, `epoll_create1` and `epoll_ctl` (`native-syscall-table.c:92-101`, `NS_REFUSE`). A twin probe (`nested/mcprobe`, 15 Nov) confirms it after a passing control (`open` in the run directory): `socket(AF_INET)` and `socket(AF_UNIX)` give ENOSYS, `epoll_create1` ENOSYS, `pthread_create` EAGAIN. memcached needs all three (four worker threads). Running it elsewhere (e.g. the virtual route) is a different instrument; the lead's call | - |
| CREVOKE worst case (subtree 1 .. 10^6, latency, interruptibility; PostgreSQL context reset) | **measured** (RTL sim, memory delay 0): 6 cycles per node up to 2,048 nodes, about 11 beyond; at most 65,532 nodes per subtree (10^6 cannot be built), so the worst single CREVOKE is 720,892 cycles; **not interruptible**: a timer interrupt waits for the whole walk (180,518 cycles at 32,768 nodes, against 1-6 without). PostgreSQL context resets (twin): the largest ends **34,509 nodes** in one CREVOKE, about 0.38 M cycles at delay 0 and 2.1 M at delay 12. CDERIVE: 21 cycles alone, 27 / 157 back to back at delay 0 / 12; CREVOKE about 60 cycles per node at delay 12. **On silicon (F):** 28 cycles per node from 10,000 nodes; 100,000 nodes in 2.82 M cycles (about 113 ms); PostgreSQL's 34,509-node reset about 0.97 M cycles (about 39 ms) | capstone-ariane `a948e4134` (F), `--sv_seed 1` |
| CDERIVE / CREVOKE timed on silicon (lone, back to back, subtrees 1 .. 10^5, a probe to 10^6) | **measured** (board 2, 15 Nov 09:58-10:11): CDERIVE (the runtime's helper) **63** cycles lone, 98 back to back; CREVOKE of 1 node **55**; per node 16.7 at 1,000 nodes, **28** at 10,000 and 100,000; a 100,000-node subtree revokes in **2.82 M cycles (about 113 ms at 25 MHz)**; 10^6 cannot be built (refused after 196,609). The lifetime ops are 29-38 % of speedtest1's F cost | F `a948e4134`, kit bkit75n |
| record-cache counters | event 23 is on hpm8 in kits bkit73n (batch) and bkit75n (F). Events 29-31 (lookups, raw misses, walk-busy cycles) would need a new bitstream; that is **the lead's call** and is not requested | - |

Only rows marked "measured" hold measurements; the rest say what runs next and where.

## Summary across applications (twin unless marked; each section below carries the configuration)

| application, workload | what the arms differ in | in-context cost: twin | silicon (F) | added CDERIVE per run | per M in-context cycles | CREVOKE per run (protected) | max subtree | peak live records (protected) |
|---|---|---|---|---|---|---|---|---|
| speedtest1 `--size 1` | memsys5 + lookaside patch | +0.6 % | **+1.67 %** (batch: +1.04 %) | 28,217 | 41 (silicon cycles) | 28,216 | 3 | 398 |
| CPython 3.13.7, 7 pyperformance benchmarks | pymalloc patch 0014 | +1.63 % (summed) | **+5.17 %** (six summed; twin +2.06 % on those six) | 2,866,016 (all seven) | 345 | 2,936,354 | 2 | 76,767 |
| PostgreSQL 17.5, pgbench-derived | 0003 **with lane q15-placement's Q-15 repair**; harness (root, shims) in both arms | **+4.04 %** | not read (repeats 1.10 %; ABBA mean +6.67 %); wall-clock +4.38 % | 2,276,557 | 810 | 668,247 | **34,509** | 105,043 |
| tshark 4.6.8, 9,000 frames | wmem 0001 (dev, Q-15-repaired) | **+2.69 %** | **no reading**: the protected arm did not return in 3,600 s (2 of 2 boots) | 4,506,984 | 598 | 2,727,471 | 29,134 | 33,998 |
| FFmpeg 9.0.1, 300-frame decode | 0003 **with this lane's Q-15 repair**; 0001 hooks and `-fno-builtin` in both arms | 0.03 % either way, not attributed | not read (repeats 1.04 %; ABBA -0.32 %); wall-clock -0.74 % | 3,327 | 3.7 | 8,974 | 2 | 474 |
| memcached | - | deferred: the native route has no sockets, epoll or threads | - | | | | | |
| *SPEC temporal increment, gcc train01* | *heap lifetimes on vs off* | | *+0.62 % (batch)* | | | | | |

- The cost follows the added lifetime work. The three apps that add the most per cycle (PostgreSQL 810, tshark 598,
  CPython 345 per M) pay +1.6 % to +4 % on the twin, and FFmpeg, at 3.7, pays nothing measurable. speedtest1 costs
  more than its 41 per M suggests: on the twin only 0.39 of its 0.62 points are the lifetimes; 0.17 are the patch's
  request-size reporting and 0.06 its `aReq` array (speedtest1 section).
- On silicon, speedtest1's cost is 1.7-2.7 times its twin cost, and 1.7 times the SPEC temporal increment. CPython's
  is 2.5 times (six benchmarks summed), and about a third of it is issue stalls, with L1D misses up 14.7 %. PostgreSQL's
  wall-clock cost is +4.38 % (its in-context repeats missed the band). FFmpeg's
  is not resolved from noise on silicon either. tshark's protected arm did not return on silicon in either boot (at
  least 3x the unprotected time); whether it is slow or stops is open: the heartbeat boot pre-registered below was not run when the work was paused.
- Both large-subtree applications reset whole pools in one CREVOKE: PostgreSQL's memory contexts, up to 34,509 nodes,
  and tshark's packet scope, every frame. That is where CREVOKE's walk matters: uninterruptible, and 28 cycles per
  node on silicon for large subtrees, so PostgreSQL's largest reset holds the core for about 39 ms.

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
not read from this boot. On the twin it is the patch's `aReq` array working through the native route's per-syscall
page scan, not the lifetimes.

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

**Addendum 1 (2026-11-14 22:30, before the boot): the RTL undercuts the event-23 lower bound; the band is not edited.**
- Read from a948e4134: the node unit's record writes are broadcast to the LSU's verdict cache, and a write with
  `valid = 1` installs the entry live.
  - Install: `load_store_unit.sv` `rvc_ins_valid = (revnode_fill_valid_i && revnode_fill_nodeval_i) || …`.
  - Tap: `ex_stage.sv:2014-2016`, `revnode_fill_valid = mem_rev_wr_req_valid && !vp_hdr_wr`,
    `revnode_fill_nodeval = node_wr_req[31]`, id `{generation, slot}`.
  - CDERIVE writes the new node with `valid = 1'd1` (`capstone_rev_node.anvil`, the `new_node` literal) through
    `send_revnode_update`.
  - So a freshly derived child is installed, and its first access is not a compulsory miss, unless its entry was
    evicted first (256 entries, round-robin) or a namespace flush emptied the cache.
  - CREVOKE writes `valid = 0`, which clears liveness and keeps the entry, so a later access through the dead child
    is denied at once with no replay.
  - Not checked by simulation: that the installed id equals the child capability's `revnode_id` as the LSU compares it.
- So the prediction's lower bound (+28,000, one compulsory miss per CDERIVE) rests on a premise the RTL contradicts.
  - A reading below it is consistent with the RTL. It is read as "the protection's misses are capacity or conflict
    misses", not as a broken counter.
  - The pre-registered band and every other prediction stand as written above.
- The cost of one miss, read from the RTL (`commit_stage.sv:906-914`, `frontend.sv:385`): a pipeline flush and a refetch
  of the same pc. It is not a local LSU stall.
  - With the node unit idle and the record read hitting the L1D: about 17-20 cycles. The source's own "5 typical"
    cycles from accept to resolve, plus about 3 to reach commit, plus the refill.
  - On an L1D miss for the record, add the memory latency.
  - While a revoke walk runs, the probe waits, without a bound short of a 1 M-cycle timeout.
  - The refill depth, the idle accept delay and whether the record region is cacheable were not read. These figures
    are estimates from the source, not measurements.
- The share of the in-context cost that misses explain is therefore read as (event-23 delta x 17-20 cycles) against the
  `run_cycles` delta, a lower bound on the misses' share.

### Results: silicon on bitstream F, speedtest1 `--size 1`, with event 23 (board 3, 2026-11-15 02:26-02:51)

**Configuration.** As pre-registered above: bitstream F `a948e4134` (`ec675c2b…`, read back resident), board 3, kit
bkit75n, stage `nested/st1/board/f3-stage` (the board-2 stage, byte-identical). Board firmware `21951b56b89cced0`
(19,566,088 B; the kit relinks the native kernel `c6e0a4d07274cbff` into it at bake time), monitor `4f51541`.
Transcript `spec/board/out-nat-twcf-hpmnest-nested-st1-b3/f3-stage/board.txt` (sha256 `6264fcd227d07ec7…`), read by
`stage-gen/parse.py`.

**Controls.** `b0-stats` and `b0-hello` returned 0. The native smoke passed all 9 steps under the kit's waiver (step 70,
the two user-tick checks, as on board 2). The driver's rc=1 is that waived step's retval 70, as on board 2's boot. The
kit's `spec-classify.py` again reads `NO RESULT` for the format mismatch described above: all four commands printed
`@@SPEC-END` and the stage reached `@@SPEC-DONE`. Not VOID.

| | u1 | p1 | p2 | u2 | repeat spread, u / p | protected vs not (ABBA means) |
|---|---|---|---|---|---|---|
| rc, oracle lines equal to the reference | 0, 32/32 | 0, 32/32 | 0, 32/32 | 0, 32/32 | | |
| `run_cycles` (in-context) | 690,770,499 | 700,409,377 | 699,492,998 | 686,082,488 | 0.68 % / 0.13 % | **+1.67 %** (+11.52 M) |
| `between_cycles` (kernel and other) | 5,012,130,102 | 5,113,802,329 | 5,106,738,817 | 4,998,316,847 | 0.28 % / 0.14 % | **+2.10 %** (+105 M) |
| uptime span, s | 229.86 | 233.92 | 233.61 | 228.71 | 0.50 % / 0.13 % | **+1.95 %** |
| runs (context entries) | 15,384 | 15,432 | 15,432 | 15,385 | | +0.31 % |
| hpm3 L1I miss | 6,097,496 | 6,268,797 | 6,265,791 | 6,098,602 | 0.02 % / 0.05 % | +2.78 % |
| hpm4 L1D miss | 3,143,886 | 3,277,722 | 3,283,124 | 3,082,790 | 1.96 % / 0.16 % | +5.37 % (+167 k) |
| hpm5 DTLB miss | 87,328 | 106,775 | 110,638 | 86,849 | 0.55 % / 3.55 % | +24.8 % (+21.6 k) |
| hpm6 mispredict | 7,071,579 | 7,249,691 | 7,249,537 | 7,072,391 | 0.01 % / 0.00 % | +2.51 % |
| hpm7 issue stall | 123,406,716 | 127,552,618 | 126,952,636 | 122,261,592 | 0.93 % / 0.47 % | +3.60 % (+4.42 M cycles) |
| **hpm8 event 23** (verdict-cache miss replays) | 65,472 | 111,877 | 111,938 | 65,168 | 0.47 % / 0.05 % | **+46,588** (+71 %) |

**Against the pre-registration.**
- Answers: **met**, 32/32 in every command.
- Repeats: **met** on both: `run_cycles` 0.68 % / 0.13 % (band 1 %), `between_cycles` 0.28 % / 0.14 % (band 2 %).
  Unlike the batch boot, this boot's kernel difference is readable.
- In-context cost on F: **met, +1.67 %**, inside +0.5 % to +3 %. On batch it read +1.04 %; the two are different
  bitstreams and boards, N = 2 each, and the difference between them is not attributed.
- Whole command: **met, +1.95 %**, inside +1 % to +4 % (batch +1.90 %).
- Event 23, protected minus not: **met, +46,588 per command**, inside +28,000 to +200,000. Addendum 1 had already
  removed the premise of the lower bound (CDERIVE installs the new record live), so the reading is read as capacity and
  conflict misses: 1.65 per CDERIVE at the twin's 28,226 derives per run.
- Event 23, not protected: 65,320 per command (mean). That is not "small next to the protected arm": the protected
  arm adds 71 % to it. The unprotected arm's misses come from mallocng's own lifetimes and everything else the verdict
  cache serves; this boot does not split them.
- Duration: **met**: 229-234 s per command, 25 min for the boot.

**Derived, as pre-registered.**
- Upper bound on cycles per verdict-cache miss, if the whole in-context delta were misses: 11,524,694 / 46,588 =
  **247 cycles**.
- At the RTL estimate of 17-20 cycles per miss (Addendum 1, from source, an estimate), the added misses account for
  0.79-0.93 M cycles: **6.9 % to 8.1 % of the in-context cost**. So verdict-cache misses explain under a tenth of the
  protection's in-context cost on F. The rest moves with issue stall (+4.42 M cycles, 38 % of the delta), L1D misses
  (+167 k) and DTLB misses (+21.6 k). This boot cannot attribute it further: events 29-31 (lookups, raw misses,
  walk-busy cycles) are not in this bitstream.
- Kernel: +2.10 % (+105 M cycles per command). On the twin the whole kernel rise was the patch's `aReq` array through
  the native route's per-syscall page scan, not the lifetimes. This boot reads the size, not the cause.

**So, on bitstream F:** protecting SQLite's nested allocators costs speedtest1 `--size 1` **+1.67 % in-context cycles,
+2.10 % kernel cycles and +1.95 % wall-clock**, with every answer unchanged. The protection adds 46.6 k verdict-cache
miss replays per run, which at the RTL's estimated cost explain about 7-8 % of the in-context cost.

## CREVOKE worst case (RTL simulation, measured 2026-11-14)

**Configuration.**
- capstone-ariane `a948e4134` (bitstream F's RTL), Verilator testharness, bare M-mode, the physical node pool.
- `--sv_seed 1` on every run; testbench memory delay at its default, 0.
- Worktree `nested/ariane-crevoke`. Tests are `verif/tests/custom/capstone/crevoke-cost.S`, parameterised by N, shape,
  cold/warm cache, interrupt arm and witness, and `crevoke-cap.S`.
- Logs in `.crv/<A..H>/`. A log counts only if it reached its end marker below the cycle ceiling with every reading
  present.
- The cycles are the bracket around the one CREVOKE minus an empty bracket. In every run that bracket retired exactly
  one instruction with zero exceptions.
- Every run checks that the subtree existed before the revoke: N derives succeeded, and head and tail were live before
  and dead after. The witness arm counts all 32,768 stored capabilities, 32,768 live before the revoke and 0 after.

**Latency against subtree size** (cycles, warm cache; cold reads the same from N = 2, because memory delay is 0):

| N | 1 | 4 | 16 | 64 | 256 | 1,024 | 2,048 | 4,096 | 8,192 | 16,384 | 32,768 | 65,532 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chain (each node derived from the last) | 45 | 56 | 128 | 416 | 1,568 | 6,176 | 12,351 | 40,608 | 89,319 | 180,256 | 360,485 | 720,892 |
| fan (N children of one node) | 45 | 61 | 133 | 421 | 1,573 | 6,181 | 12,340 | 37,223 | 82,534 | 172,566 | 353,030 | - |

- **Up to 2,048 nodes: 6 cycles per node, exactly.** That is 6N + 32 for the chain (N = 2..2,048) and 6N + 37 for
  the fan. It fits one record read and one write per node, plus an O(1) splice.
- **Above 2,048 nodes: about 11 cycles per node.** 2,048 records of 16 B are 32 KiB, which is where the node
  records stop fitting in the L1 data cache.
  - With memory delay 0, the extra 5 cycles per node are the miss path alone, so silicon pays more per node beyond
    this size. These are cycle counts at zero memory latency, not silicon times.
- **The largest subtree is 65,532 nodes.** `crevoke-cap.S` derives a chain until CDERIVE refuses:
  - 65,532 derives succeed, and the next one raises cause 30 (out of nodes);
  - one CREVOKE of the whole chain takes 720,892 cycles with no exception;
  - afterwards 65,532 fresh derives succeed again, so every record went back to the free list exactly once.
  - The 16-bit slot head explains the number: it starts at 3, and 65,535 means full.
  - **So a subtree of 10^5 or 10^6 nodes cannot be built on this RTL.** The largest single CREVOKE is about 0.72 M
    cycles at zero memory latency.
  - The native route's flat table (node_flat_records 262,144) is a different record store, and its limit was not
    measured here.

**Can it be interrupted?** No.
- The test: chain N = 32,768, with a machine-timer interrupt pending at 49.9 % of the walk (4.000 cycles per mtime
  tick, measured by a polling arm).
- The interrupt is taken only after the CREVOKE retires:
  - mepc = CREVOKE pc + 4;
  - the handler enters 180,518 to 180,523 cycles after the interrupt became pending, 13 cycles after the walk's
    measured end;
  - the control arm, with no CREVOKE in flight, takes the same interrupt in 1 to 6 cycles.
- The CREVOKE is neither restarted nor resumed: the trace holds one retirement of it, no exception, and every node dead.
- The reason, by reading `core/decoder.sv:2046-2090` before the run: interrupts are attached to the instruction in
  decode, and the CREVOKE was already issued.
- **So the worst-case interrupt latency the lifetime unit adds is one whole CREVOKE:** up to about 0.72 M cycles at
  65,532 nodes, plus memory latency on silicon. There is no abort path.
- Read with the twin's subtree sizes: speedtest1's CREVOKEs end 1 node (mean 1.0001, max 3), so they cost about 45
  cycles each at memory delay 0. That is a lower bound; the delay-12 readings below are several times higher.

**CDERIVE latency, and both instructions at memory delay 12 (measured 2026-11-15).**
- Same tree and seed. `crevoke-cost.S` gained a knob, `DERIVE_BRACKET`: it brackets c1's CDERIVE, and the chain
  build's n-1 derives in their loop (5 instructions per iteration; reading 27, the loop's instret, is 1,281 = 256 x 5 + 1
  at n = 257, as designed). Logs `.crv/J2` (delay 0) and `.crv/I2` (delay 12). Every run is valid with 0 build and 0
  bracket exceptions.
- Delay 12 means the testbench memory delay `+define+S12_MEM_DELAY=12`, under the knob's mod-16 wrap. The model's
  `verFiles.dat` records `S12_MEM_DELAY=12` for I2 and no define for J2. Calibration: J2's CREVOKE chain controls
  repeat the delay-0 table exactly (45, 416 and 6,176 cycles at n = 1, 64, 1,024), and every delay-12 reading is higher.
- A first attempt (`.crv/J`, `.crv/I`) is void: its derive loop copied the parent with an integer `mv`, which drops the
  tag, so all 256 loop derives trapped (`build_exc` 256). The loop now copies with `MOVC`.

| cycles | memory delay 0 | memory delay 12 |
|---|---|---|
| CDERIVE alone, after a fence | 21 | 21 |
| CDERIVE back to back, per loop iteration (the CDERIVE plus 4 one-cycle instructions) | 31.0 | 161.1 |
| CREVOKE of 1 node, after a fence / right after the derives | 38 / 45 | 42 / 244 |
| CREVOKE of 64 nodes, cold / warm | 416 | 3,511 / 3,692 |
| CREVOKE of 257 nodes (the n = 257 chain), cold / warm | 1,574 | 15,117 / 15,191 |
| CREVOKE of 1,024 nodes, cold / warm | 6,176 | 61,111 / 61,301 |

- The very first CDERIVE after reset reads 3,716 (delay 0) and 3,085 (delay 12) cycles without a fence before it, and 21
  with one. It waits for something still in flight at program start. That is a start-up effect, not a per-instruction
  cost.
- A lone CDERIVE or CREVOKE whose predecessor's record write has drained costs 21-50 cycles at either delay. Back to
  back, each one waits for the previous record write: about 157 cycles per CDERIVE and 60 per revoked node at delay 12.
- So the walk costs 6 cycles per node at delay 0 and about 60 at delay 12. The node unit's record traffic goes to
  memory, and these costs scale with memory latency.

**What these give for speedtest1 on F** (28,226 CDERIVE and 28,216 one-node CREVOKE per protected run, from the twin):
- Every op drained, about 21 + 42 cycles: 1.8 M cycles, 15 % of the +11.5 M in-context cost.
- Delay 0, back to back (27 + 45): 2.0 M, 18 %.
- Delay 12, back to back (157 + 244): 11.3 M, 98 %.
- The bracket is too wide to attribute the cost. Where speedtest1 falls inside it depends on how often one lifetime op
  follows another before its record write drains, and on the board's real memory latency. A simulation at delay 0 or
  12 gives neither.
- What would settle it: time CDERIVE and CREVOKE on the board itself, in a native-route domain on F (a loop of each,
  back to back and spaced). It is a few seconds of board time in any future boot. Not pre-registered yet.

### Pre-registration: CDERIVE and CREVOKE timed on silicon (bitstream F, board 2; written before the boot)

**Why.** The RTL brackets speedtest1's lifetime-op cost at 15-98 % of its in-context cost, because the per-op cost
depends on memory latency, which simulation does not have. Timing both instructions on the board settles it. It also
gives the lead's worst-case question (subtree 1 .. 10^6) its silicon latency.

**Configuration.**
- Bitstream F `a948e4134` (`ec675c2b…`), kit bkit75n, board 2, chained by the SPEC lane after the CPython boot
  (nat-run3, switch park).
- Stage `nested/crvt/board/c2-stage` (`c2-stage.SHA256SUMS`): the f3 stage's native-smoke skeleton, byte-identical, plus
  `train/`, written by `nested/crvt/make-stage-crv.py`.
- One native-route domain, `crvtime` (`crvt/crvtime.c`, FRCDKMS SDK -O2; stripped `44b1492101016471…`). One test per
  command, each in its own process. Times are `csrr cycle` deltas; the domain runs at machine privilege.
- The parent of every derive is a 4 KiB `malloc` block, a mallocng lifetime.

| command | test |
|---|---|
| t0-control | empty bracket, and an empty loop of 100,000 (sentinel `0x5ca1ab1e`) |
| t1-lone | 10,000 times: bracketed CDERIVE of a 64-byte child, spin 200, bracketed CREVOKE of it, spin 200 |
| t2-b2b | 10,000 back-to-back sibling CDERIVEs under Q, then one bracketed CREVOKE of Q (10,001 nodes) |
| t3-chain1 .. t3-chain100k | a chain of N = 1, 10, 100, 1,000, 10,000, 100,000, then one bracketed CREVOKE |
| t4-probe1m | derive a chain toward 10^6 with progress lines every 65,536; the table must refuse first |

- Checked on the twin (`ncrv-w2`): every test completes, and t4 dies on cause 30 after reporting 196,609. The twin's
  cycle counter counts instructions, so its latencies are not readings.
- Checked off the board, as rule 8 asks (`nested/crvt/board/c2-recheck`, via `recheck-c2.sh`):
  - a real bake: cpio `d7afda9720b976da` holds all 24 manifest files, and the baked `crvtime` and `run.sh` match the
    stage hashes. `run.sh`'s setup line succeeds on the tree extracted from that cpio.
  - The kit's `BOARD_STUB=2` run on that bake relinked the native kernel (`9809b4295d967487`) and classified the smoke
    PASS, with `BAKED_TIMEOUT=3600`.
  - A real firmware build on that Image: `9ff675d9b2ff4d23…`, 19,566,088 B (cap 26,000,000), dom_stack gate PASS.

**Predictions.**
- t0: completes, with the sentinel. Anything else voids the timing lines.
- **Lone CDERIVE: 15-150 cycles. Lone CREVOKE of 1 node: 30-300.** The RTL reads 21 and 38-50 drained, at either delay.
- **Back-to-back CDERIVE: 20-400 cycles each.** The RTL reads 27 (delay 0) and 157 (delay 12); the board's memory may
  be slower than delay 12.
- **CREVOKE per node, N >= 1,000: 5-200 cycles.** The RTL reads 6 (delay 0) and about 60 (delay 12).
- t3-chain100k completes: the native table holds 262,144 records. If it is refused with cause 30, that is a
  per-subtree cap on silicon. It is read as such, and the largest N is taken from t4's progress.
- t4 is refused before 262,144 nodes (twin: after 196,609). A 10^6-node subtree cannot exist on this configuration.
- **What will be derived:** speedtest1's per-run lifetime-op cycles on F, as 28,226 x (CDERIVE) + 28,216 x (CREVOKE of 1),
  for lone and for back-to-back costs, against its +11.5 M in-context delta. Also the latency of PostgreSQL's
  34,509-node reset at the per-node cost read here.
- Duration: under a minute of commands. Budget 3,600 s for the rung.
- **VOID** if `b0-stats` or `b0-hello` fails, or the smoke fails outside the kit's accounting waiver.

**Addendum 1 (2026-11-15 05:40, before the boot): the stage is replaced by `nested/crvt/board/c3-stage`, because whether
a native-route domain may read a cycle counter on silicon is not established.**
- The record is mixed:
  - `tests/rtl-smoke/README.md:20-30, :176-190` carries a collaborator's note that the board gates the unprivileged
    `cycle`;
  - QEMU domains read it;
  - a CSRUNV context runs at raw M (R-60).
  - On the twin, `cycle` reads and `mcycle` traps (cause 2).
- In c2-stage every test read `cycle` first, so a trap would have cost every timing line.
- The program now takes `CRVT_CLOCK`, and the stage has 22 commands:
  - t0-control on `mono` (`clock_gettime(CLOCK_MONOTONIC)`, ns), which cannot trap;
  - one-read probes of `cycle` and `mcycle`;
  - every test on `mono`;
  - every test again on the first counter the probes found readable. `run.sh` decides and prints `@@CRVT-COUNTER`;
    with none, it skips those commands with rc 255;
  - the probe toward 10^6.
- Each command is its own process, so a counter that traps loses only its probe.
- A test is added: `lonefence`, which is `lone` with a fence before each op, comparable to the RTL's fence-drained
  21 / 42 cycles.
- Checked:
  - on the twin (`ncrv-w3`): `cycle` readable, `mcycle` cause 2, counter tests on `cycle`, the probe refused (30) after
    196,609;
  - the no-counter branch, with a stub binary on the host: 22 `@@SPEC-END` lines, 9 skips;
  - rule 8 again (`c3-recheck`): cpio `1458c4e07f84a0aa`, 24 files with the baked binary and `run.sh` hash-matching,
    stub smoke PASS (`BAKED_TIMEOUT=3600`), firmware `3da21cf758582af2…` at 19,566,088 B, gate PASS.
  - Binary `a1b9475179b25875…`.
- **Predictions added (the bands above are unchanged and apply to the counter tests):**
  - **Mono resolution:** `clock_gettime` advances in steps of the kernel's timebase, about 1 µs on this board, so a
    lone mono bracket reads 0 or one step. Only the means over 10,000 brackets, `b2b`'s loop and chains of 10,000 or
    more are read from mono. Cycle figures from mono use the board's core clock, which the transcript does not state;
    they are labelled as such.
  - **Refusal cause:** on silicon, node exhaustion refuses with **cause 30** (ISSUES.md R-12). The silicon README's
    table names 30 differently (ILLEGAL_OPERAND_VALUE), and R-12's reading is the one used.
  - R-12 also records a 65,536-node pool in the RTL. If the board's limit is that pool and not the 262,144-record
    table, **t3-chain100k is refused with 30**, and t4's last progress line is 65,537 or below. That outcome is read as
    the per-subtree limit, not as a failure.

**The real large revoke: PostgreSQL's context resets** (twin, `w2-main`, details in the PostgreSQL section). 668,247
CREVOKEs per run. 92 % end one node, 52,685 end a context's chunks, and the largest ends 34,509 nodes. That one costs
about 0.38 M cycles at memory delay 0 and about 2.1 M at delay 12, without interruption.

### Results: CDERIVE and CREVOKE timed on silicon (board 2, 2026-11-15 09:58-10:11)

**Configuration.** As pre-registered, with Addendum 1:
- Stage `nested/crvt/board/c3-stage`, binary `a1b94751…`, bitstream F (read back resident by name), kit bkit75n.
- Transcript `spec/board/out-nat-twcf-hpmnest-crvt-b2/c3-stage/board.txt`.
- `b0-stats` and `b0-hello` returned 0, and the smoke passed under the kit's waiver. Not VOID. The kit's classifier
  printed MALFORMED for this stage's format; all 22 commands printed `@@SPEC-END`.

**What a figure is.** Each latency is a `cycle` delta minus the empty bracket (6 cycles), read in `crvt/crvtime.c`.
- "CDERIVE" is the runtime's `capstone_cap_derive` (`capstone/capability.h`): three field reads, a range check, a
  cursor set and the CDERIVE itself, about eleven instructions. The heap patches call the same sequence.
- "CREVOKE" is the single instruction.

**Counters.** `cycle` is readable from the native domain on silicon (`p-cycle` rc 0; the empty bracket reads 6
cycles). `mcycle` faults (`p-mcycle` rc 139), as on the twin. `run.sh` chose `cycle`.

| test | reading (cycles) | band | RTL (delay 0 / 12) |
|---|---|---|---|
| lone CDERIVE of a 64-byte child, mean of 10,000 | **63.0** (min 63, max 128) | 15-150, met | 21 (CDERIVE alone) |
| lone CREVOKE of 1 node, mean of 10,000 | **55.0** (min 55, max 111) | 30-300, met | 38-50 drained |
| the same, fenced before each op | 63.0 / 55.0 | - | - |
| back-to-back sibling CDERIVEs, per derive | **98.2** | 20-400, met | 27 / 157 |
| CREVOKE of those 10,001 siblings | 258,541 (25.9 per node) | - | - |
| CREVOKE of a chain of 1 / 10 / 100 nodes | 100 / 164 / 1,489 | - | - |
| CREVOKE of a chain of 1,000 | 16,705 (**16.7 per node**) | 5-200 per node, met | 6 / ~60 |
| CREVOKE of a chain of 10,000 | 279,394 (**27.9 per node**) | met | |
| CREVOKE of a chain of 100,000 | **2,815,062** (**28.2 per node**) | met | |
| building a chain of 100,000 (each derive from the last child) | 9,008,132 (90 per derive) | - | - |

- **A 100,000-node subtree builds and revokes**, so silicon has no 65,536-node per-subtree cap. The pool recorded in
  R-12 is not this configuration's limit.
- **10^6 cannot be built.** The probe reports 65,537, 131,073 and 196,609 nodes, then dies (rc 139) before 262,145,
  as on the twin. The board console does not carry the kernel's fault line, so the silicon cause is not read. The
  twin's is 30, node exhaustion. The native table's 262,144 records bound every subtree, including the records other
  lifetimes hold.
- Per node, CREVOKE costs 16-17 cycles up to 1,000 nodes and 28 from 10,000. The rise is consistent with the subtree
  outgrowing a cache level. This boot does not show which.

**Clock rate and the mono clock.**
- On the native route a `clock_gettime(CLOCK_MONOTONIC)` pair costs about 1.95 ms, against the 1 µs step predicted
  in Addendum 1. So every short mono bracket is void: the lone means wrap below zero.
- The long mono readings give the cycle rate. The 100,000-chain revoke takes 114.68 ms minus a 1.94 ms bracket for
  2,815,062 cycles, and the 10,000-chain gives the same within 4 %. So the core clock is about **25 MHz**, which agrees
  with the 24 M cycles/s used in the duration estimates.

**Against the pre-registration.**
- **met:** every latency band (lone, back to back, per node), t3-chain100k completes, and t4 is refused before
  262,144.
- **wrong:** the mono resolution prediction, as above. No cycle reading depends on it.

**Derived.**
- **speedtest1 on F:** 28,226 derives x 63-98 cycles + 28,216 single-node revokes x 55 cycles = 3.3-4.3 M cycles. Its
  in-context delta is +11.52 M, so **the lifetime ops themselves are 29-38 % of speedtest1's cost**. That narrows the
  RTL's 15-98 % bracket. Event 23's replays are 7-8 %. The rest is not attributed here.
- **CREVOKE worst case:**
  - a 100,000-node subtree revokes in **2.82 M cycles, about 113 ms at 25 MHz**, in one uninterruptible instruction
    (interruptibility from the RTL: a timer interrupt waits for the whole walk);
  - the largest subtree this configuration can hold is under 262,144 nodes, so the worst single CREVOKE is under
    about 7.4 M cycles, about 0.3 s, at 28 per node.
- **PostgreSQL's real case:** its largest context reset, 34,509 nodes, costs about **0.97 M cycles, about 39 ms**, at
  28 per node. The RTL estimate at delay 12 was 2.1 M.
- **tshark:** its 9,001 per-frame packet-scope resets of 256-1,023 nodes cost about 4-17 k cycles each at 16-17 per
  node.

## CPython 3.13.7: pyperformance subset

**Patch.** `capstone/ports/cpython/app/patches/cpython-3.13.7-0014-pymalloc-sublet-lifetimes.patch` (sha256
`6c2c170e…`). With it, obmalloc hands out each block as a CDERIVE child of its arena, bounded to the request, and
CREVOKEs it on free. The two arms' source trees differ in exactly `Objects/obmalloc.c` and
`Include/internal/pycore_obmalloc.h` (`diff -rq`). In both arms, malloc lifetimes are ON underneath (mallocng).

**Workload.**
- pyperformance 1.14.0 (tarball sha256 `2eb90812…`): `json_dumps`, `json_loads`, `pickle` (C accelerator),
  `richards`, `deltablue`, `regex_v8` and `nbody`. These are the subset `docs/plans/application-benchmark-study.md`
  names.
- Each body runs unchanged under a stand-in for `pyperf` (`nested/cpy/pyperf-run/pyperf.py`, sha `2b058b90…`). The
  stand-in calls the body a fixed number of times, `CPY_LOOPS=1` times pyperformance's own inner_loops, with no
  calibration, no worker processes and no timing.
- The driver (`run.py`, `ff1b0dd9…`) then prints one `CHECK` digest read from the benchmark module's own state:
  - nbody's planet positions;
  - richards' result;
  - the json and pickle round trips of the benchmarks' own data;
  - regex matches over the benchmark's strings.
- The reference is native CPython 3.13.7 from the same tarball (`native-ref.txt`, `bfc163bc…`). Host Python 3.12 gives
  the same seven digests. Each check is stable across runs and hash seeds, and nbody's changes when the work doubles.
- `PYTHONHASHSEED=0`. Each command runs in a fresh directory.

**Build** (`nested/cpy/build-native.sh` with `prepare-native-trim.sh`; arms `n2-arm0` / `n2-arm1`).
- The port's own recipe (prepare, survey, link), with these changes, all identical on both arms:
  - **SDK:** FRCDKMS, the SPEC and speedtest1 images' SDK (llvm-c9 `29027126f39b`; the `wt-ms` runtime and libc with
    native I/O; heap lifetimes ON). The FRCDKMS `-mllvm` flags go in through `CFLAGS`, and `OPT=-DNDEBUG -O2`.
    - The port's own SDK is dev's runtime, which has no native-route syscall path. Its images fault at the first musl
      `mmap` (cause 26, a `delin` on the virtual-VM delegation reply).
  - **Linked `-Wl,--no-relax`.** lld's Capstone TPREL relaxation rewrites C-47's `addi a0,a0,%tprel_lo` into
    `addi a0, tp, lo`, and the following `cincoffset a1, tp, a0` then adds tp twice.
    - CPython faults on it in `_PyThreadState_Attach` (cause 24).
    - A 4-line reproducer is in `nested/cpy/tprel-repro/`. It went to the compiler lane through the SPEC lane.
    - The deviation from FRCDKMS's link is the same on both arms.
  - **35 `Setup.stdlib` built-ins disabled.** The twin never loads them for this subset (`modules-twin-mu.txt`): the
    CJK codecs, `unicodedata`, `_decimal`, `pyexpat`, `_socket` and so on. This keeps one arm's board firmware at
    21.66 MB, under the kit's 26 MB cap.
    - The trimmed build's 14 twin commands loaded none of them, checked from each command's `sys.modules`.
- 214 of 214 objects compiled, and the link has no undefined symbol.
- The stripped image is 6,369,128 B in both arms: unprot sha256 `b0b29e89…`, prot `eabba900…`.
- **stdlib:** the 54 pure-Python modules the twin loads, as sourceless `.pyc` (1.43 MB, `nested/cpy/cpyhome2`).
- **Start:** `-S -c "runpy.run_path('bench/run.py')"`. Running a script file by path returns ENOSYS on the native
  route; `open`, `stat` and `import` work.

**Results, QEMU twin** (qemu55 `a0501ebc39`, native route, ids55; one boot per benchmark, ABBA u1 p1 p2 u2). Every
command exits 0 with its CHECK equal to native. The repeats agree within 0.01 % in-context.

| benchmark | in-context, not protected | protected vs not | kernel (`between_cycles`) |
|---|---|---|---|
| json_dumps | 458.4 M | **+2.84 %** | -0.23 % |
| json_loads | 387.1 M | **+2.37 %** | -0.44 % |
| pickle | 327.8 M | **+1.30 %** | +0.26 % |
| richards | 2,551.5 M | **+0.66 %** | +0.27 % |
| deltablue | 370.8 M | **+1.16 %** | -0.49 % |
| regex_v8 | 1,115.6 M | **+2.28 %** | -0.12 % |
| nbody | 3,101.4 M | **+2.01 %** | +0.32 % |
| all seven (sum) | 8,312.6 M | **+1.63 %** | |

- The kernel does not move, unlike speedtest1's: CPython's arenas carry no array like memsys5's `aReq`.
- **A twin reading must be ABBA within one boot.** A single pass of all 14 commands in one boot (`ncpy-c6`) read
  richards at -2.22 %, while its own ABBA boot reads +0.66 % with repeats 30 k apart. Position in the boot moves a
  command's count by several percent (boot history), so the single-pass readings are not used.

**Lifetime counts** (counting QEMU `qemu-nst`; one boot per arm runs all seven benchmarks, each in its own process):

| | not protected | protected |
|---|---|---|
| CDERIVE (allocations by derive), whole boot | 121,554 (mallocng's own heap lifetimes) | 2,987,570 |
| CREVOKE, successful / refused | 119,466 / 0 | 2,936,354 / 0 |
| revoked subtree size: mean / max | 1.00003 / 2 | **1.000001 / 2** |
| peak live records | 5,681 | **76,767** |
| live-record memory at 16 B per record | 91 KB | **1.23 MB**, inside the fixed 4 MiB flat table |

- The peak of 76,767 live records exceeds the RTL's physical node pool (65,532, measured above). On the native route
  the records live in the flat table (262,144 per context), so this is not a limit there. It would be one on the
  physical pool.
- 76,767 live records is about 300 times the verdict cache's 256 entries. So the silicon reading of event 23 is
  expected to show capacity misses, which speedtest1 (398 live records) could not.

### Pre-registration: silicon, CPython pyperformance subset (written before the boots)

**Configuration.**
- Bitstream F (`…-twc16k-merge-hpmtag_a948e4134`, sha256 `ec675c2b…`), the table's.
- Kit `t3-smoke/bkit75n-k2n-native-twcf-hpmnest`: hpm8 = event 23 (verdict-cache miss replays), hpm3..7 = L1I, L1D,
  DTLB, mispredict, issue stall.
- Board and slot: placed by the SPEC lane. All four boots on one board.
- **One arm per boot, four boots U P P U.** Both arms in one boot exceed the console's upload limit.
- Each boot runs the seven benchmarks once, in the same order (json_dumps, json_loads, pickle, richards, deltablue,
  regex_v8, nbody), so the boot history is the same for both arms.
- Board stages:
  - `nested/cpy/board/cpy-u-stage` (unprot image `b0b29e89…`, `train/run.sh` `3a7263ea…`);
  - `nested/cpy/board/cpy-p-stage` (prot image `eabba900…`, `train/run.sh` `b7165ae6…`);
  - both made by the kit's `make-train-stage.sh`, with manifests `cpy-{u,p}-stage.SHA256SUMS`.
- Checked off the board for the U stage: a real bake (cpio 19,208,704 B), the kit's `BOARD_STUB=2` relink, and a real
  firmware build of 21,663,240 B that passes the dom_stack gate.

**Readings:** per command: rc, the CHECK against native, `run_cycles`, `between_cycles`, hpm3..8 and the uptime span.

**Predictions.**
- **Answers:** 28 of 28 commands exit 0 with CHECK equal to native. Anything else stops the comparison; a
  protected-only failure is a silicon finding.
- **Repeats across boots** (U1 against U2, P1 against P2, per benchmark): within 1 % on `run_cycles` and 2 % on
  `between_cycles`. A benchmark that misses is not read.
- **In-context cost, protected vs not, all seven summed: +1.3 % to +5 %.**
  - The twin reads +1.63 %.
  - On speedtest1, silicon read 1.7 times the twin (+1.04 % against +0.62 %).
  - CPython's 76,767 live records against a 256-entry verdict cache can add capacity misses the twin does not model.
  - Per benchmark: positive for each of the seven, with json_dumps, json_loads and regex_v8 among the four largest, as
    on the twin.
- **Kernel:** within ±2 % (twin ±0.5 %).
- **Whole command (uptime span), all seven: +1 % to +5 %.**
- **Event 23, protected minus not protected, summed over the seven: positive.**
  - No lower bound is claimed. CDERIVE installs the new node live (Addendum 1 of the F speedtest1 pre-registration), so
    the 2.87 M extra derives are not 2.87 M compulsory misses.
  - The size is the reading. With 76,767 peak live records against 256 entries, capacity misses are expected.
  - The cycles misses explain are read as (event-23 delta) x 17-20 cycles, a lower bound from the RTL replay path,
    against the `run_cycles` delta.
- **Duration:** about 25 min of commands per boot, from the twin-to-silicon ratios of speedtest1 (in-context x4.1,
  kernel x2.5, 24 M cycles/s). Budget 3,600 s per rung; book 50 min per boot.
- **VOID** if a control (`b0-stats`, `b0-hello`) fails, or the smoke fails outside the kit's accounting waiver.

### Results: silicon, CPython pyperformance subset (board 2, 2026-11-15 07:55-09:58)

**Configuration.** As pre-registered above:
- Bitstream F, read back resident on board 2 by name (`…-twc16k-merge-hpmtag_a948e4134`; the board does not expose its
  sha256), kit bkit75n.
- Four one-arm boots, U P P U: `cpy-u1` 07:55, `cpy-p1` 08:25, `cpy-p2` 08:57, `cpy-u2` 09:28.
- The SPEC lane's stage copies (`perf/natstage/board-cpy/cpy-*`) carry the pre-registered images and `run.sh`:
  unprotected `b0b29e89…` / `3a7263ea…`, protected `eabba900…` / `b7165ae6…`.
- Transcripts `spec/board/out-nat-twcf-hpmnest-cpy-b2/cpy-*/board.txt` (sha256 `5e733822…`, `8d860992…`, `77f5ffb6…`,
  `de6c436a…`), read by `cpy/parse-cpy.py`.

**Controls.** In every boot `b0-stats` and `b0-hello` returned 0, and the smoke passed all 9 steps under the kit's
waiver (the two accounting checks). The kit's classifier printed `MALFORMED: @@SPEC-DONE 7 but 0 commands ended`, the
same format mismatch as the speedtest1 boots: all seven commands printed `@@SPEC-END` in every boot. Not VOID.

**Answers.** 28 of 28 commands exit 0 with CHECK equal to native.

| benchmark | in-context, not protected (mean) | repeat spread u / p | **in-context cost** | twin | silicon / twin | kernel cost | event 23, p − u |
|---|---|---|---|---|---|---|---|
| json_dumps | 1,569.3 M | 0.17 % / 0.02 % | **+5.82 %** | +2.84 % | 2.0 | -0.07 % | +722,942 |
| json_loads | 1,424.1 M | 0.41 % / 0.07 % | **+5.05 %** | +2.37 % | 2.1 | -0.03 % | +744,642 |
| pickle | 1,315.4 M | 0.25 % / 0.50 % | **+4.20 %** | +1.30 % | 3.2 | +0.31 % | +800,276 |
| richards | 8,314.3 M | 0.05 % / **1.22 %** | not read (+3.22 %) | +0.66 % | - | +0.30 % | +1,754,032 |
| deltablue | 1,463.0 M | 0.00 % / 0.05 % | **+3.18 %** | +1.16 % | 2.7 | +0.42 % | +778,657 |
| regex_v8 | 3,619.5 M | 0.27 % / 0.02 % | **+4.02 %** | +2.28 % | 1.8 | +0.24 % | +929,342 |
| nbody | 7,591.5 M | 0.06 % / 0.19 % | **+6.15 %** | +2.01 % | 3.1 | +0.92 % | +1,431,908 |

Summed over the six that are read (richards' protected repeats differ by 1.22 %, over the 1 % band, so it is left out
as pre-registered):

| | not protected | protected | protected vs not |
|---|---|---|---|
| in-context (`run_cycles`) | 16,982.9 M | 17,860.4 M | **+5.17 %** (+877.6 M) |
| kernel and other (`between_cycles`) | 2,595.6 M | 2,603.4 M | +0.30 % |
| uptime span | 802 s | 838 s | +4.43 % |
| hpm4 L1D miss | 45.66 M | 52.38 M | +14.7 % |
| hpm7 issue stall | 3,059.8 M | 3,336.6 M | +9.05 % (+276.8 M cycles) |
| hpm8 event 23 (verdict-cache miss replays) | 674,577 | 6,082,344 | **+5.41 M** (x9.0) |

With richards included, the seven sum to +4.53 % in-context and +7.16 M event 23. That figure is not the reading.

**Against the pre-registration.**
- Answers: **met**, 28/28.
- Repeats: `run_cycles` within 1 % for 13 of 14 pairs; **missed for richards' protected pair (1.22 %)**, so richards is
  not read. `between_cycles` within 2 % for all 14 pairs (largest 1.84 %).
- In-context cost, summed: **missed, +5.17 %, above the +1.3 % to +5 % band** by 0.17 points. It is 2.5 times the twin's
  +2.06 % over the same six. The ratio sits at the top of speedtest1's 1.7-2.7.
- Per benchmark: **met for "positive for each"**: all six read are positive, +3.2 % to +6.2 %. **Missed for the
  ordering**: regex_v8 is fifth, not among the four largest (nbody, json_dumps, json_loads, pickle).
- Kernel: **met**, +0.30 % (band ±2 %).
- Whole command: **met**, +4.43 % (band +1 % to +5 %).
- Event 23: **met (positive)**, +5.41 M over the six, at a 76,767-record peak against 256 entries.

**Derived.**
- Event 23 at 17-20 cycles per replay explains **10.5-12.3 %** of the in-context cost over the six: 92-108 M of
  877.6 M cycles. Per benchmark it explains 5 % (nbody) to 34 % (deltablue).
- The L1D miss count rises 14.7 % and issue stalls rise by 277 M cycles, 32 % of the cost.
  - Event 23 explains about a ninth of the cost, so most of it is something else.
  - Candidates are the lifetime records competing for the L1D (16 B each, 1.23 MB at the peak) and the patched
    pymalloc's layout. This boot cannot separate them, and the twin models no caches, so it cannot show either.
- Silicon in-context cycles are 2.5-4.0 times the twin's per benchmark, against 4.1 for speedtest1.

## PostgreSQL 17.5: single-user, pgbench-derived workload

**Patch.** `capstone/ports/postgres/memory-contexts/patches/postgresql-17.0-0003-memory-contexts-sublet-lifetimes.patch`
as repaired for Q-15 on lane `q15-placement` (57215acb2dbb; sha256 `657c0e1fed593ca0…`, with
`capstone/runtime/include/capstone/representable.h` from the same commit, `7c41a409acb7a897…`). With it, aset, generation,
slab and bump hand out each chunk as a CDERIVE child of its block's generation lifetime, and pfree, reset and delete
CREVOKE them. dev's 0003, without the repair, cannot run here: its protected backend faults at startup with
`ILLEGAL_OP_VAL` (29) on the CDERIVE of a large chunk's block, which the compact format cannot hold exactly. That is
Q-15's recorded refusal (ISSUES.md Q-15; the same instance as the 2026-10-18 `virtual-pg-pools` reading). The repaired
patch places such blocks on their granule and pads children at the tail only. It reports the padding at exit: on the
small workload, `Q15_PAD chunks=112762 padded=573 pad_bytes_total=12832 pad_bytes_max=992`.

**Build.** `nested/pg/build.sh` with the port's `build-domain.sh` forked as `build-domain-native.sh`. It compiles and links
with the FRCDKMS SDK and flags at -O2, as the other apps here, and `PGSU_NESTED=none|sublet` selects the arm. The two
arms' trees differ only in `utils/mmgr/{aset,bump,generation,mcxt,slab}.c`,
`include/utils/memutils_memorychunk.h`, the runtime include path and configure's record of the flags.

**The native route needed harness changes. They are the same in both arms and touch nothing on the query path.**
- **Root.** The native route's domain loader runs an image only for euid 0
  (`linux-twin/drivers/misc/capstone/binfmt-dom.h:392`; the SPEC lane confirms this is the module's design). PostgreSQL
  refuses root. `harness-patches/pg-17.5-native-route-harness.patch` (`d5243294605547fd…`) therefore:
  - drops the root refusal (`main.c` check_root);
  - takes the `--single` role name from `PGUSER` (default `postgres`), because a native task cannot read
    `/etc/passwd`.
  - A standalone backend authenticates as `BOOTSTRAP_SUPERUSERID` whatever that name is (`postinit.c:897-899`). So
    **both arms and the native reference run as the bootstrap superuser**, and every permission and RLS check passes as
    superuser in all three.
- **Calls the route refuses or does not serve**, given replacements in one object linked into both images
  (`harness-src/pg-native-shims.c`, `e7d3f31a99ea4e7a…`). It overrides musl members that define only those symbols:
  - `access` (faccessat is NS_REFUSE) on stat;
  - `rename` (renameat2 is NS_REFUSE) as link then unlink;
  - `realpath`, lexical: musl's walks every prefix from `/`, and a native task may not look above its directory;
  - `setitimer` (ENOSYS) arms nothing, and every timeout setting is 0;
  - `posix_fallocate` returns EOPNOTSUPP, so PostgreSQL writes zeros itself (fd.c:2401);
  - `preadv` and `pwritev` are pg_iovec.h's own per-iovec fallback over `pread`/`pwrite`.
  - Each was found by a twin run failing on it, in that order. A table-driven pass over the image's libc members
    (`native-syscall-table.c` against the linked syscall wrappers) is the faster way next time.
- **Inputs.** The fixture `tmp/fix2` is the port's `build-native16-fixture.py` recipe with `initdb -U postgres`, so the
  catalog carries no host account name. It is trimmed: templates dropped, 1 MiB WAL segments. The stage carries it
  xz-packed and unpacks it per command, outside the marks. On the twin the unpack takes 3.0 s per command.
- **Settings**, all three runs: `shared_buffers=4MB max_connections=10 timezone=GMT log_timezone=GMT
  dynamic_shared_memory_type=sysv exit_on_error=true fsync=off log_startup_progress_interval=0`.

**Workload.** `work/gen.py` (`f67f7fc6871f4755…`), scale 1 as `pgbench -i -s 1` builds it, in pgbench's step order:
tables, data, `VACUUM ANALYZE` of the four tables, primary keys. Then select-only, simple-update and tpcb-like
transactions from a fixed-seed PRNG, with a fixed mtime, and four CHECK queries. `w2-small.sql` (200/200/200,
`a5c4e910d13893c2…`) and `w2-main.sql` (500/500/1000, `03465ac436fdda41…`). The native reference is the fixture's
own MAXALIGN-16 build, through `work/mkref.py`.

**A defect the protected arm found: generic array code reads `int2vector`/`oidvector` 8 bytes off on the
16-byte-MAXALIGN port.**
- The first workload ended its init with a database-wide `VACUUM ANALYZE`. The protected arm faulted there with
  `CAP_OOB` (28), in `array_iter_next`, called from `array_cmp` during `ANALYZE pg_catalog.pg_index`. QEMU's `-d int`
  gives `tval 0x3ff61751a4`.
- An instrumented `array_iter_next` (debug build only; its first 3 calls print unconditionally, as the check's
  positive control) reads element 0 at that address. The pointer's capability ends at `0x3ff61751a0`: the array's
  header is inside it, and its data starts 4 bytes past the end.
- **Mechanism, from the source:**
  - Generic array code finds the data at `ARR_OVERHEAD_NONULLS(1)` = `MAXALIGN(16 + 8)` (`utils/array.h:310`).
  - `int2vector` and `oidvector` fix `values[]` at offset 24 (`c.h:715-734`), and their size is 24 plus the elements.
  - With MAXALIGN 8 both are 24. With the port's 16 (`pg_config.h`, forced by its build), generic array code reads
    from offset 32.
- **Without any capability machine** (native host, same query):

  | query | stock 17.5, MAXALIGN 8 | the port's MAXALIGN-16 build |
  |---|---|---|
  | `'1 2'::int2vector = '1 3'::int2vector` | f | **t** |
  | `('1'::int2vector)::int2[]` | `[0:0]={1}` | **`[0:0]={0}`** |

- So the unprotected arm reads the same wrong bytes silently, and the protected arm stops at them. It is a defect of
  the port's MAXALIGN choice, not of upstream PostgreSQL and not of the lifetime patch. It is reported to the SPEC
  lane for the port and bug-corpus owners. Earlier MAXALIGN-16 results that compare or cast these types through
  generic array code have not been audited.
- The measurement workload's vacuum step now does what `pgbench -i` does: the four tables by name, which never compares
  a catalog vector. The native answers are unchanged.

- Filed as ISSUES.md **M-15** (lane branch), with the audit of corpus case 04 that this file does not repeat.

**Twin bring-up** (QEMU qemu55 a0501ebc39, slot 28, native route):
- Image set 9 on `w-small`: the unprotected arm gave the native answers exactly; the protected arm stopped at M-15.
- Image set 10 (clean rebuild, byte-identical to set 9: none `e1a8c8083ff41a34…`, sublet `03e0c47ead40882e…`;
  stripped `1ea084ab…` / `4cc5f247…`) on `w2-small` (stage `pg/stages/s12`): **both arms give the native reference
  answers** (4 CHECK lines, 600 rows, rc 0). The protected arm reports `Q15_PAD chunks=842636 padded=1899
  pad_bytes_total=53280 pad_bytes_max=8160`: 0.23 % of its chunks carry tail padding, 53 KB in all.

**Twin results (2026-11-15 04:21-04:42), image set 10, `w2-main`, one ABBA boot `npg-a1` (stage `pg/stages/a1`).**
Every command exits 0 with the native reference answers (4 CHECK lines, 2,000 rows).

| | u1 | p1 | p2 | u2 | repeat spread, u / p | protected vs not (ABBA means) |
|---|---|---|---|---|---|---|
| in-context (`run_cycles`) | 2,809,800,924 | 2,923,410,930 | 2,923,569,480 | 2,810,038,007 | 0.008 % / 0.005 % | **+4.04 %** |
| kernel and other (`between_cycles`) | 1,655,094,160 | 1,662,098,571 | 1,662,081,963 | 1,654,764,826 | 0.02 % / 0.001 % | **+0.43 %** |
| context entries (`runs`) | 83,025 | 83,137 | 83,137 | 83,025 | | +0.13 % |

- The cost is in the program, not the kernel: +113.6 M in-context cycles per run against +7.2 M kernel.
- It is the largest in-context cost of the three applications on the twin: speedtest1 +0.63 % in program instructions,
  CPython +1.63 % summed, PostgreSQL +4.04 %. PostgreSQL also does the most lifetime work per run (below).
- Twin time per command, about 4.5 s, and 3.0 s to unpack: QEMU time, not a measurement.

**Lifetime counts** (counting QEMU `qemu-nst`, one boot per arm, `pg/stages/c1u` and `c1p`; both runs give the native
answers):

| | not protected | protected |
|---|---|---|
| CDERIVE (allocations by derive), whole boot | 35,153 (mallocng's own heap lifetimes) | **2,311,710** |
| CREVOKE, successful / refused | 34,518 / 0 | 668,247 / 0 |
| nodes ended by CREVOKE | 34,518 | 2,307,406 |
| revoked subtree size: mean / max | 1 / 1 | **3.45 / 34,509** |
| peak live records | 1,073 | **105,043** |
| live-record memory at 16 B per record | 17 KB | **1.68 MB**: 40 % of the 262,144-record (4 MiB) flat table |

**The real case for CREVOKE: PostgreSQL's context resets.** The protected run's subtree sizes, from `hist_log2`:

| nodes ended by one CREVOKE | 1 | 2-3 | 4-7 | 8-15 | 16-31 | 32-63 | 64-127 | 128-1,023 | 1,024-8,191 | 8,192-32,767 | 32,768-65,535 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CREVOKEs | 615,562 | 6,956 | 9,078 | 10,547 | 5,035 | 15,525 | 5,515 | 14 | 10 | 4 | **1** |

- 92 % of CREVOKEs end a single chunk (pfree). The rest end a context's chunks at once (MemoryContextReset or
  Delete): 52,685 of them, mostly up to 127 nodes. 29 end 128 or more, and the largest ends **34,509 nodes** in one
  instruction.
- Its latency from the RTL table (CREVOKE worst case section) is about 0.38 M cycles at memory delay 0, between the
  32,768-node 360,485 and the 65,532-node 720,892. At delay 12 (about 60 cycles per node) it is about 2.1 M cycles.
  CREVOKE is not interruptible, so a timer interrupt waits that long.
- Every other app here is at subtree max 2-3. PostgreSQL's resets are the case where CREVOKE's walk matters.

### Pre-registration: PostgreSQL on silicon, bitstream F, both arms in one boot (written before the boot)

**Configuration.**
- Bitstream F `a948e4134` (`ec675c2b…`), kit bkit75n (hpm8 = event 23), a board the SPEC lane slots.
- Rung budget **1800**: `BAKED_TIMEOUT=7200` at the kit's x4, as the stubbed run prints.
- Stage `nested/pg/board/p2-stage` (`p2-stage.SHA256SUMS`): the f3 native-smoke skeleton, byte-identical, plus
  `train/`. Its `run.sh` is byte-identical to the twin's ABBA stage `pg/stages/a1`. ABBA u1 p1 p2 u2 on `w2-main`;
  image set 10, stripped `1ea084ab…` (not protected) and `4cc5f247…` (protected); fixture `tmp/fix2`.
- Everything in the PostgreSQL section above applies: the harness changes in both arms, the Q-15-repaired 0003, and
  the settings.
- Checked off the board, as rule 8 asks (`nested/pg/board/p2-recheck`, `recheck-p2.sh`):
  - a real bake: cpio `5fcd7fd2aba2d1b0` holds all 26 manifest files. The baked `run.sh`, `in.tar.xz` and both
    `.xz` images match the stage hashes. `run.sh`'s setup line, run on the extracted tree, unpacks the cluster, the
    share directory and `w.sql`. The two images it unpacks hash to the measured `4cc5f247…` and `1ea084ab…`.
  - The board image's BusyBox is a different build from the twin's (`27f322b7…` against `365a0d98…`; libc differs
    too). Its applet table is identical: 279 applets, including `xz` and `tar`, BusyBox 1.36.1. `/usr/bin/xz` and
    `/bin/tar` link to it. The twin ran this exact unpack, so the board's unpack rests on that table, not on a board run.
  - The kit's `BOARD_STUB=2` run relinked the native kernel (`121b267274cf524f`) and classified the smoke PASS, with
    `BAKED_TIMEOUT=7200`.
  - A real firmware build: `5b11bde4c5a1c21e…`, **25,857,544 B against the 26,000,000 cap (142 KB to spare)**,
    dom_stack gate PASS. At about 115 KiB/s the upload takes about 220 s of its 300 s budget.

**Predictions.**
- **Answers:** every command exits 0 with the native reference answers (`work/nref4-main/ref.txt`: 4 CHECK lines,
  2,000 rows). Anything else stops the comparison.
- **Repeats:** within 1 % on `run_cycles` and 2 % on `between_cycles`, or that difference is not read.
- **In-context cost: +2 % to +12 %.** The twin reads +4.04 %. speedtest1 read 1.7-2.7 times its twin cost on silicon
  (+0.62 % twin; +1.04 % batch, +1.67 % F).
- **Kernel: -2 % to +4 %** (twin +0.43 %). **Whole command (uptime span): +2 % to +10 %.**
- **Event 23, protected minus not, per command: +0.5 M to +10 M.** The protected run derives 2.31 M lifetimes and peaks
  at 105,043 live records against a 256-entry verdict cache. speedtest1, at 28 k derives and 398 peak records, read
  1.65 misses per derive.
- **Derived, not predicted:** the share of the in-context cost that event 23 explains at 17-20 cycles per miss.
  Also the lifetime-op cycles: 2.31 M CDERIVE plus 668 k CREVOKE (2.31 M nodes) at the per-op costs the timing boot
  reads.
- **Duration:** about 650 s per command, from the twin and speedtest1's ratios (in-context x4.1, kernel x2.5,
  24 M cycles/s). Each command first unpacks for one to five minutes, outside the marks, and is silent while it
  runs. So 50-70 min of commands, under the 7,200 s rung budget. For a `spec.sh` stage the kit sets the idle limit to the
  same 7,200 s (`board-run.sh:155`), far above one silent command.
- **VOID** if `b0-stats` or `b0-hello` fails, or the smoke fails outside the kit's accounting waiver.

### Results: PostgreSQL on silicon, bitstream F (board 1, 2026-11-15 08:34-10:00)

**Configuration.** As pre-registered above (87b463da4d10):
- Bitstream F, read back resident on board 1 by name (`…-twc16k-merge-hpmtag_a948e4134`; the board does not expose
  its sha256), kit bkit75n.
- Stage `nested/pg/board/p2-stage`: it still matches `p2-stage.SHA256SUMS`, and its images unpack to `1ea084ab…`
  (not protected) and `4cc5f247…` (protected).
- Board firmware `764b4194…`. Transcript `spec/board/out-nat-twcf-hpmnest-pg-b1/p2-stage/board.txt` (sha256
  `753cbed87d9b208b…`), read by `pg/parse-pg.py` with `PG_REF=pg/work/nref4-main/ref.txt`.

**Controls.** `b0-stats` and `b0-hello` returned 0. The smoke passed all 9 steps under the kit's waiver. The kit's
classifier printed MALFORMED for this stage's format, as for the other nested stages; all four commands printed
`@@SPEC-END`. Not VOID. The board's BusyBox unpacked the cluster and images, as the applet-table check predicted.

| | u1 | p1 | p2 | u2 | repeat spread, u / p | protected vs not (ABBA means) |
|---|---|---|---|---|---|---|
| rc, answers equal to native (4 CHECK, 2,000 rows) | 0, yes | 0, yes | 0, yes | 0, yes | | |
| `run_cycles` (in-context) | 12,674,463,103 | 13,541,252,270 | 13,393,561,130 | 12,576,864,815 | 0.78 % / **1.10 %** | not read (+6.67 %, +842 M) |
| `between_cycles` (kernel and other) | 8,720,015,275 | 8,847,970,498 | 8,908,895,659 | 8,848,799,845 | 1.48 % / 0.69 % | **+1.07 %** |
| uptime span, s | 860.6 | 900.5 | 897.1 | 861.6 | 0.12 % / 0.39 % | **+4.38 %** |
| context entries | 83,343 | 83,487 | 83,487 | 83,343 | | +0.17 % |
| hpm3 L1I miss | 160.71 M | 171.68 M | 171.69 M | 160.81 M | | +6.79 % |
| hpm4 L1D miss | 66.36 M | 73.90 M | 74.00 M | 66.45 M | | +11.36 % |
| hpm5 DTLB miss | 1.968 M | 2.561 M | 2.570 M | 1.972 M | | +30.2 % |
| hpm6 mispredict | 103.77 M | 110.37 M | 110.27 M | 103.69 M | | +6.35 % |
| hpm7 issue stall | 1,988.5 M | 2,230.6 M | 2,212.8 M | 1,980.3 M | | +11.96 % (+237 M cycles) |
| **hpm8 event 23** | 1,384,173 | 3,421,271 | 3,435,300 | 1,385,128 | 0.07 % / 0.41 % | **+2.04 M** (x2.5) |

**Against the pre-registration.**
- Answers: **met**, every command gives the native reference.
- Repeats: **missed on `run_cycles` for the protected pair** (1.10 %, band 1 %), so the in-context difference is not
  read. Met for the unprotected pair (0.78 %), and for `between_cycles` (1.48 % / 0.69 %, band 2 %).
  - Both arms fall from their first run to their second (u1 to u2 -0.78 %, p1 to p2 -1.10 %), which is a drift
    across the boot. ABBA cancels a linear drift, so the ABBA mean, +6.67 %, sits inside the +2 % to +12 % band.
  - It is still not the reading. The rule was written to cover exactly this case.
- Kernel: **met, +1.07 %** (band -2 % to +4 %).
- Whole command: **met, +4.38 %** (band +2 % to +10 %).
- Event 23: **met, +2.04 M per command** (band +0.5 M to +10 M), 0.88 per added derive (2.31 M on the twin).

**Derived.**
- Against the ABBA in-context mean, which is not read, event 23 at 17-20 cycles per replay would explain 4.1-4.9 %
  of the difference.
- As on CPython, the L1D misses (+11.4 %), DTLB misses (+30 %) and issue stalls (+237 M cycles) rise more than the
  verdict cache explains.
- The protection's lifetime ops (2.31 M CDERIVE; 668 k CREVOKE ending 2.31 M nodes, the largest 34,509) are costed
  from the timing boot when it reads.
- Silicon in-context cycles are 4.5 times the twin's (12.63 G against 2.81 G per unprotected run).

## FFmpeg 9.0.1: decode (matroska, mpeg4, single thread)

**Patch.** `capstone/ports/ffmpeg/buffer-pool/patches/ffmpeg-9.0.1-0003-libavutil-pools-as-sublet-lifetimes.patch`
(`5d3cd35a00ae199c…`). With it, every `AVBufferPool` and `AVRefStructPool` entry a pool hands out is a CDERIVE child of
the entry, and returning it to the pool CREVOKEs that child. It is written against the buffer-pool port's 0001 (pool
event hooks). The app port prepares it with `prepare-source.sh --sublet`, together with 0001, `trace.h` and the app's
no-op recorder and observer.

**Arms.** Both arms carry the port's own patches and the 0001 hooks with the no-op recorder and observer. The hooks
cost a call per pool event, and that cost belongs in both arms. The protected arm adds 0003 and the repair below. The
unprotected arm is the protected tree with 0003 reverse-applied at fuzz 0. The trees differ in exactly
`libavutil/{buffer.c,buffer_internal.h,refstruct.c}`.

**Build.** `nested/ffmpeg/build-domain-native.sh` (`c25dd62a6aba91a9…`), derived from the port's `build-domain.sh`:
- the FRCDKMS SDK and flags at -O2, as the other apps here, with `capstone-cc` supplying libc;
- the input path relative (a native task opens nothing above its run directory);
- only the full decoder and its flipped-input control image;
- the C-50 scan gate and the undefined-weak gate pass, and no input object carries a TPREL relocation. The check
  finds 4 in the known TLS reproducer, as its control.

**`-fno-builtin` in both arms, as the port's qualified build has it.** Without it, this compiler's FFmpeg faults in
`avformat_open_input` with cause 24 (an untagged pointer reaching `__libc_free`). That is so in both arms, so it is not
the protection. Bisection, one twin run each, unprotected arm:

| added to -O2 + FRCD | result |
|---|---|
| nothing | cause 24 |
| -O1 instead of -O2 | cause 24 |
| -O2 without FRCD | cause 24 |
| `-fno-jump-tables` | cause 24 |
| `-fno-builtin` | 300/300 frames match |
| `-ffreestanding` | 300/300 frames match |
| the port's qualified flags (-O1, frame pointer, freestanding, no builtins, no jump tables), no FRCD | 300/300 match |

It is not C-79: that fix is in llvm-c9's source (a14ed97ccfa1 is an ancestor of 29027126f39b). The compiler lane
identified it as **C-69**, the existing under-aligned capability byte-copy defect
(`compiler-repros/C69-underaligned-capability-bytecopy`):
- with builtins on, SROA folds `av_freep`'s pointer-typed `memcpy` into an align-1 capability load;
- the backend expands that load bytewise and drops the tag, and `free` then receives an integer.
Their run-time dispatch fix (llvm-c14/c15) changes no SPEC image: their census over the 8 SPEC rows finds the shape in
0 functions, with a positive control on `av_freep`.

**Builtins on, with the C-69 fix (twin, 2026-11-15).** The compiler lane's llvm-c15 (29027126f39b plus
`capstone-misaligned-cap-access` c77dd6e705ad) is SDK arm `FRCDKMSc15`, which differs from FRCDKMS only in `sdk.json`'s
cc/ld. Built with -O2, FRCD and builtins on (`FFAPP_ARM=FRCDKMSc15 FFAPP_NOBUILTIN=-fbuiltin`), both arms otherwise as
set w4:
- The pass's census (`-capstone-misaligned-cap-log`) lists 10 functions per arm, among them
  `src/libavutil/mem.c av_freep tested 2 realigned 0`.
- Set w6: unprotected `8ba6e3fc…`, protected `7900e008…`. In one ABBA boot plus the flipped-input control
  (`nff-c15a1`), u1 p1 p2 u2 each decode 300 frames equal to stock, and fp FIRES (3 hashes change).
  - In-context, protected vs not: **-0.031 %** (protected 885,160,252 against unprotected 885,431,613, the ABBA means: 271 k fewer). That equals the
    `-fno-builtin` set's -0.030 %. Kernel: -0.22 %.
  - Builtins on are 0.44 % faster in-context than set w4, in both arms. That is a different compiler as well as a
    different flag, so the 0.44 % is not attributed to builtins.
- **Matched control:** the same compiler, flags and source with the fix switched off
  (`-mllvm -capstone-misaligned-cap-dispatch=false`; census `… av_freep tested 2 realigned 0 off`; set w7, unprotected
  `316c0573…`). It faults as llvm-c9 did: cause 24 at `__libc_free+0x28`, a segmentation fault, 0 frames
  (`nff-c15off`). So the fix, not the compiler change around it, is what lets FFmpeg run with builtins on.
- The silicon boot stays on set w4 (`-fno-builtin`, llvm-c9), as pre-registered.

**Q-15 placement repair of 0003, by this lane** (`nested/ffmpeg/ffmpeg-9.0.1-0003-q15-pool-placement.patch`,
`4e33c7e6ee4a76fc…`; protected arm only, applied after 0003 at fuzz 0). Unrepaired, the protected arm stops twice
on the twin:
- At the first frame, CDERIVE refuses a frame-plane pool buffer (29 in `av_buffer_pool_get`). A 57,600-byte child
  at malloc's 16-byte alignment cannot be held exactly by the compact format (Q-15). No branch carries a repaired 0003.
  - The repair allocates each `AVBufferPool` entry with one granule of slack and places the entry's lifetime at its
    first granule-aligned address.
  - The lifetime's length is the size rounded up to a granule (`capstone/representable.h`), and every child it
    issues has that length: tail padding inside the entry's own allocation only, as in the memcached, postgres and
    wmem repairs.
- At teardown, after every frame, mallocng's `free` refuses a CREVOKE (29). `pool_return_entry` keeps an
  `AVRefStructPool` entry as a pointer derived from its lifetime, a grandchild of the heap, and `pool_free_entry`
  freed that pointer.
  - The repair records the allocator's own pointer in the entry's header (`sublet_raw`) and frees that. The user
    can read the header, but it already holds `sublet_life`, which outlives the user's child the same way, so the
    repair widens nothing.
- Residual: refstruct pool children are still derived exactly. None of this decode's sizes is refused.

**Inputs.** A 10-second clip, made by the port's `build-native.sh` with `FFAPP_CLIP_SECONDS=10`: 320x180, 30 fps,
300 frames, mpeg4 in matroska (`input-10s.mkv`, `3b27ca89df1e49ce…`). The reference is stock FFmpeg's framemd5
(`stock-10s.framemd5`, `f3d81ef3cc56304d…`), from the unpatched tree. The port's own native oracle check MATCHes, and
its flipped-input control FIRES (3 hashes change).

**Twin results (2026-11-15 05:21-05:23)**, image set w4: unprotected `dfd15bb734a6e26c…`, protected
`daf46ea0cca1bdf8…`. One ABBA boot (`nff-a1`), and a control boot (`nff-s4`) in which both flipped-input runs FIRE.
Every command decodes 300 frames equal to the stock reference.

| | u1 | p1 | p2 | u2 | protected vs not (ABBA means) |
|---|---|---|---|---|---|
| in-context (`run_cycles`) | 889,344,936 | 889,077,223 | 889,076,677 | 889,344,727 | **-0.030 %** (-268 k) |
| kernel and other (`between_cycles`) | 12,134,980 | 12,086,302 | 12,067,036 | 12,106,157 | -0.36 % |
| context entries | 690 | 684 | 684 | 690 | -6 |

**Lifetime counts** (qemu-nst, one boot per arm):

| | not protected | protected |
|---|---|---|
| CDERIVE, whole boot | 5,730 | 9,057 |
| CREVOKE / nodes ended | 5,677 / 5,677 | 8,974 / 9,007 |
| revoked subtree size: max | 1 | 2 |
| peak live records | 420 | 474 |

- FFmpeg's pools recycle their entries, so the protection adds about 3,300 lifetimes per 300-frame decode: 11 per
  frame, against 889 M in-context cycles. Its direct cost is below the twin's resolution of a few hundred thousand
  cycles.
- **The arms differ by 0.03 % either way, and the difference is not attributed.** The repeats agree to 0.0001 %, so it
  is real on the twin, but the protected arm differs in more than the lifetime ops:
  - its pool buffers are allocated with a granule of slack and rounded up;
  - its `RefCount` header is 16 bytes wider (`sublet_raw`, on top of 0003's two fields), so every refstruct allocation
    is larger;
  - it enters the kernel 6 times fewer per run.
  - Separating the allocation path from the protection would take a third image with the repair's sizing and no
    derives. It was not built.

### Pre-registration: FFmpeg on silicon, bitstream F, both arms in one boot (written before the boot)

**Configuration.**
- Bitstream F `a948e4134` (`ec675c2b…`), kit bkit75n (hpm8 = event 23), a board the SPEC lane slots, rung budget 900
  (`BAKED_TIMEOUT=3600`).
- Stage `nested/ffmpeg/board/f2-stage` (`f2-stage.SHA256SUMS`): the f3 skeleton, byte-identical, plus `train/`.
  Its `run.sh` is byte-identical to the twin stage `ffmpeg/stages/b1` that the twin ran (`nff-b1`): u1 p1 p2 u2 MATCH,
  in-context 889.344 M / 889.076 M, as in `nff-a1`, and fp FIRES.
- Commands u1 p1 p2 u2 (ABBA), then fp, the protected flipped-input control: it must decode 300 frames and differ.
- Images and inputs are xz-packed and unpacked per command, outside the marks. Image set w4 (unprotected `d6668eed…`,
  protected `eb2d5228…`), the 10-second clip.
- Checked off the board, as rule 8 asks (`nested/ffmpeg/board/f2-recheck`):
  - a real bake: cpio `e213187a3ec857dc`, all 27 manifest files. The baked `.xz` files unpack to the measured image
    hashes and the measured input `3b27ca89…`.
  - The kit's `BOARD_STUB=2` run (relink `b4a719fa8ad9e53b`, smoke PASS, `BAKED_TIMEOUT=3600`).
  - A real firmware build: `69c202367c075729…`, 23,760,392 B (cap 26,000,000), dom_stack gate PASS.
  - The board BusyBox's applet table is the one checked for PostgreSQL (279 applets, with `xz`).

**Predictions.**
- **Answers:** u1 p1 p2 u2 each decode 300 frames equal to stock's framemd5; fp decodes 300 frames with at least one
  different. Anything else stops the comparison.
- **Repeats:** within 1 % on `run_cycles` and 2 % on `between_cycles`, or that difference is not read.
- **In-context cost: -1 % to +1 %.** The twin reads -0.030 %. The protection adds about 3,300 lifetimes per decode, about
  11 per frame, which at even 500 cycles each is about 1.7 M cycles, under 0.1 % of the work.
- **Event 23, protected minus not:** under +10,000 per command, on the same count.
- **Duration:** about 150 s per command (twin x speedtest1's ratios), so about 13 min of commands.
- **VOID** if `b0-stats` or `b0-hello` fails, or the smoke fails outside the kit's accounting waiver.

### Results: FFmpeg on silicon, bitstream F (board 2, 2026-11-15 10:11-10:36)

**Configuration.** As pre-registered above (fe20549f1931):
- Bitstream F, read back resident on board 2 by name; kit bkit75n.
- Stage `nested/ffmpeg/board/f2-stage`, still matching `f2-stage.SHA256SUMS`; image set w4 (`-fno-builtin`, llvm-c9).
- Board firmware `a0ca63c3…`. Transcript `spec/board/out-nat-twcf-hpmnest-ffmpeg-b2/f2-stage/board.txt` (sha256
  `bddbec17bd1419f8…`), read by `ffmpeg/parse-ff.py`.

**Controls.** `b0-stats` and `b0-hello` returned 0, and the smoke passed all 9 steps under the kit's waiver. The kit's
classifier printed MALFORMED for this stage's format; all five commands printed `@@SPEC-END`. Not VOID.

| | u1 | p1 | p2 | u2 | repeat spread, u / p | protected vs not (ABBA means) |
|---|---|---|---|---|---|---|
| frames equal to stock (of 300) | 300 | 300 | 300 | 300 | | |
| `run_cycles` (in-context) | 1,689,065,434 | 1,692,002,837 | 1,692,933,020 | 1,706,619,616 | **1.04 %** / 0.06 % | not read (-0.32 %) |
| `between_cycles` (kernel and other) | 79,386,015 | 80,175,274 | 80,200,335 | 81,537,400 | **2.71 %** / 0.03 % | not read (-0.34 %) |
| uptime span, s | 72.68 | 72.22 | 72.26 | 72.87 | 0.26 % / 0.06 % | -0.74 % |
| hpm8 event 23 | 33,994 | 35,423 | 35,380 | 34,034 | | **+1,388** |

The flipped-input control fp decodes 300 frames with 3 hashes different: FIRES.

**Against the pre-registration.**
- Answers: **met**: u1 p1 p2 u2 match stock's framemd5, and fp fires.
- Repeats: **missed for the unprotected pair** on both counters (`run_cycles` 1.04 %, band 1 %; `between_cycles` 2.71 %,
  band 2 %). So neither difference is read. The protected pair agrees to 0.06 % and 0.03 %.
  - u2 runs 1.0 % longer than u1, while p1 and p2 agree. That is not the linear drift ABBA cancels.
  - The ABBA means, -0.32 % in-context, sit inside the -1 % to +1 % band, but they are not the reading.
- Event 23: **met, +1,388 per command** (band under +10,000).
- Duration: about 72 s per command, against about 150 s predicted. Silicon in-context cycles are 1.9 times the twin's
  here, against the 4.1 assumed from speedtest1.
- The wall-clock span had no band. It reads -0.74 %, with repeats 0.26 % / 0.06 %.

**Reading.** On silicon, as on the twin, FFmpeg's protection is not resolved from noise. The protection adds about
3,300 lifetimes per decode. At the timing boot's 63 + 55 cycles each, that is about 0.4 M cycles, 0.02 % of a run, well
under this boot's 1 % repeat spread.

## tshark 4.6.8: offline dissection (`-r w.pcap -V -n`)

**Patch.** `capstone/ports/wireshark/wmem/patches/wireshark-4.6.8-0001-wmem-sublet-lifetimes.patch`, as on dev (which
carries its Q-15 repair, d75137deed6e: every derive exact, tail padding only). With it, every wmem `block` and
`block_fast` object is a CDERIVE child of its block, revoked by free, by realloc, and by the reset that ends its pool.
The arms' source trees differ in exactly `wsutil/wmem/{wmem_allocator_block.c,wmem_allocator_block_fast.c,wmem_allocator.h}`.
Applying the patch to the unprotected tree reproduces the protected one byte for byte.

**Build.** `nested/tshark/scripts/`: forks of the port's `deps/env.sh`, `deps/capstone-cc`, `host/cross-build.sh` and
`host/build-domain.sh`, with the diffs in `scripts/diffs/`.
- The FRCDKMS SDK and flags are on every compile and link, of tshark and of its seven dependencies (zlib, pcre2,
  c-ares, libgpg-error, libgcrypt, libxml2, glib). The dependencies were built once and are shared by both arms.
- The port's own flags are kept (-O2 -DNDEBUG, single-threaded domain, `-fno-stack-protector`).
- The build's gates pass (no undefined weak symbol, no stray constructor section), and both were negative-tested.
- **`-Wl,--no-relax`**: the SDK's runtime archive carries 236 TPREL relocations; tshark's own objects and the
  dependencies have none. Relinking without it changes both images.
- The build was done by a delegated agent. I re-checked the image hashes and the source diff; the rest is from its
  report.
- Images (m5, unstripped): unprotected `5c2a75c6c79f7154…`, protected `46d254b58b415d1e…`. Stripped they are
  `2061f661…` and `bcc251f3…`, 20.3 MB each.
- tshark ran on the native route with no harness change. Its stderr carries the minimal build's registration notices,
  and the runtime's own report (`TSAPP-HEAP status=0 … unserved=160x5,48,278x4`) of calls the route did not serve:
  uname, faccessat and getrandom. None of them changed the output.

**Workload.** `nested/tshark/wl/w1000.pcap` (`5ba7788d208894d1…`, by `wl/mkpcap.py`): the port's whitelisted
captures `dhcp`, `dns_port` and `http` (link type 1), concatenated 1,000 times over, 9,000 frames. `arp.pcap` is link
type 107 and is left out. The reference is stock native tshark 4.6.8's output on the same file (`wl/ref1000.out`,
`056c1f8ce80b4012…`, 1,140,005 lines), compared by sha256, as the port's oracle compares stdout byte for byte. An
empty capture (`w0.pcap`) measures startup alone.

**Twin results (2026-11-15 05:45-06:45).** One ABBA boot `nts-a1`; every command's output equals the stock reference
(sha256, 1,140,005 lines, 9,000 frames).

| | u1 | p1 | p2 | u2 | repeat spread, u / p | protected vs not (ABBA means) |
|---|---|---|---|---|---|---|
| in-context (`run_cycles`) | 7,556,567,751 | 7,711,471,512 | 7,779,724,440 | 7,528,730,252 | 0.37 % / 0.88 % | **+2.69 %** (+203 M) |
| kernel and other (`between_cycles`) | 516,743,172 | 516,657,412 | 516,762,201 | 516,437,594 | 0.06 % / 0.02 % | +0.02 % |

- Startup is not what is measured: the empty capture costs 10.4 M in-context cycles in both arms (`nts-z1`), 0.14 % of
  a run.
- The cost is 22.5 k in-context cycles per frame.
- The repeats are wider than the other apps' twin repeats (0.37 % and 0.88 % against 0.01 % or less), but within the
  1 % band.

**Lifetime counts** (qemu-nst, one boot per arm, `w1000`; both outputs equal the reference):

| | not protected | protected |
|---|---|---|
| CDERIVE, whole boot | 2,670,583 (mallocng's own) | **7,177,567** |
| CREVOKE / nodes ended | 2,670,335 / 2,670,335 | 2,727,471 / 7,177,318 |
| revoked subtree size: mean / max | 1 / 1 | **2.63 / 29,134** |
| peak live records | 3,218 | **33,998** (544 KB) |

- The subtree histogram shows the packet scope: 9,001 CREVOKEs end 256-1,023 nodes each, one per frame. That is
  the reset of the packet-scope wmem pool, which ends every object a frame's dissection allocated, in one instruction.
- The largest single CREVOKE ends 29,134 nodes.

### Pre-registration: tshark on silicon, bitstream F, one arm per boot (written before the boots)

**Configuration.**
- Bitstream F `a948e4134` (`ec675c2b…`), kit bkit75n (hpm8 = event 23), a board the SPEC lane slots.
- Rung budget 900: `BAKED_TIMEOUT=3600` at the kit's x4, as the stubbed runs print.
- **One arm per boot, four boots U P P U, all on one board.** Each image is 20.3 MB stripped, and both in one boot
  exceed the 26,000,000 B firmware cap.
- Board stages (the f3 native-smoke skeleton, byte-identical, plus `train/`):
  - `nested/tshark/board/t-u-stage` (`t-u-stage.SHA256SUMS`), unprotected image `2061f661…`;
  - `nested/tshark/board/t-p-stage` (`t-p-stage.SHA256SUMS`), protected image `bcc251f3…`.
  - Each `train/` equals the twin stage `tshark/stages/b-{u,p}/stage` file for file, plus the skeleton's
    `in/NO-INPUTS`.
  - The baked `w.pcap.xz` unpacks to the measured workload `5ba7788d…` (w1000, 9,000 frames).
  - `plan.json` names the size `w100`. That label is stale; the file is w1000, by its hash.
- The image and the capture are xz-packed and unpacked per command, outside the marks.
- The twin ran these exact stages, one arm per boot (`nts-bu`, `nts-bp`, 15 Nov):
  - both outputs equal stock tshark's (sha256, 9,000 frames);
  - in-context 7,449,089,370 (u) and 7,777,037,664 (p).
  - Across one-arm boots, the twin reads +4.40 % against the ABBA boot's +2.69 %. Its u1 also differs by 1.4 % from
    the ABBA boot's u1. On the twin, a boot's history moves this program's cycles by about 1.4 %. That is why the
    silicon boots run U P P U and are read as means.
- Checked off the board, as rule 8 asks (`nested/tshark/board/recheck-t-{u,p}.sh`, output in `t-{u,p}-recheck/`):
  - real bakes: cpio `0521309f0c19f77d` (U) and `12bd33d953202c99` (P), each holding all 25 manifest files. Each baked
    `.xz` image unpacks to the measured stripped hash (`2061f661…`, `bcc251f3…`).
  - The kit's `BOARD_STUB=2` runs, both classified as smoke PASS (all 9 steps) at `BAKED_TIMEOUT=3600`.
  - Real firmware builds: `3dc28c9685f99f84…` (U) and `65b4d2acc7a93ea7…` (P), 23,760,392 B each, under the
    26,000,000 B cap (the same size as FFmpeg's f2; the hashes differ). The dom_stack gate PASSES on both
    (25,872 B of 32,768).

**Readings:** per boot: rc, the output's sha256, lines and frames against the stock reference, `run_cycles`,
`between_cycles`, hpm3..8 and the uptime span.

**Predictions.**
- **Answers:** every boot's output equals stock tshark's (sha256 `056c1f8c…`, 1,140,005 lines, 9,000 frames). Anything
  else stops the comparison; a protected-only failure is a silicon finding.
- **Repeats across boots** (U1 against U2, P1 against P2): within 1 % on `run_cycles` and 2 % on `between_cycles`, or
  that difference is not read.
- **In-context cost, protected vs not (means of the two boots each): +1.5 % to +10 %.**
  - The twin reads +2.69 % (ABBA) and +4.40 % (one-arm boots).
  - speedtest1 read 1.7-2.7 times its twin cost on silicon.
  - The 33,998 peak live records against a 256-entry verdict cache can add capacity misses the twin does not model.
- **Kernel: -2 % to +4 %** (twin +0.02 %). **Whole command (uptime span): +1.5 % to +9 %.**
- **Event 23, protected minus not, per command: +1 M to +20 M.** The protected run adds 4.51 M derives. At F's
  speedtest1 rate, 1.65 misses per derive, that is +7.4 M; PostgreSQL's band scales the same way.
- **Derived, not predicted:**
  - the share of the in-context cost that event 23 explains, at 17-20 cycles per miss;
  - the cost of the 9,001 packet-scope CREVOKEs (256-1,023 nodes each) at the per-node cost the timing boot reads.
- **Duration:** about 22 min per command (twin x speedtest1's ratios: in-context x4.1, kernel x2.5, 24 M cycles/s;
  about 31 G + 1.3 G cycles). It is silent while it runs, against a 3,600 s idle limit. Book 35 min per boot.
- **VOID** if `b0-stats` or `b0-hello` fails, or the smoke fails outside the kit's accounting waiver.

### Results: tshark on silicon, bitstream F (board 1, 2026-11-15 10:01-13:40): the protected arm did not return

**Configuration.** As pre-registered above (b12fa3f29eb2):
- Four one-arm boots, U P P U, on board 1: `t-u1` 10:01, `t-p1` 10:39, `t-p2` 11:54, `t-u2` 13:01.
- Bitstream F, read back resident by name (`…-twc16k-merge-hpmtag_a948e4134`; the board does not expose its sha256),
  kit bkit75n, rung timeout 3,600 s (budget 900 at the kit's x4), idle limit 3,600 s.
- The SPEC lane's stage copies (`perf/natstage/board-tshark/t-*`) equal this lane's `t-{u,p}-stage` file for file, and
  each stage still matches its `SHA256SUMS`. Their images unpack to `2061f661…` (unprotected) and `bcc251f3…`
  (protected).
- Transcripts `spec/board/out-nat-twcf-hpmnest-tshark-b1/t-*/board.txt` (sha256 `612be098…` u1, `97b15027…` p1,
  `2fefe8e4…` p2, `dc860ad5…` u2), read by `tshark/parse-ts.py` against `wl/ref1000.out`.

**Controls.** In every boot `b0-stats` and `b0-hello` returned 0, and the smoke passed all 9 steps under the kit's
waiver. Not VOID.

| | u1 | p1 | p2 | u2 |
|---|---|---|---|---|
| command returned | yes, rc 0 | **no** | **no** | yes, rc 0 |
| output equal to stock (sha256 `056c1f8c…`, 1,140,005 lines, 9,000 frames) | yes | - | - | yes |
| `run_cycles` (in-context) | 24,198,529,160 | - | - | 23,874,061,846 |
| `between_cycles` (kernel and other) | 3,831,450,213 | - | - | 3,833,271,849 |
| uptime span, s | 1,124.8 | > 3,327 | > 3,327 | 1,111.9 |
| hpm8 event 23 | 3,093,334 | - | - | 3,031,588 |

**The protected boots.** Both printed `@@SPEC-BEGIN` and `@@NATIVE-CNT 0 begin` at uptime 272.7 s, and then nothing.
The runner's rung timeout ended each (`vbatch: no return within 3600s (ActionTimeout)`), so the command ran at least
3,327 s without returning: 2.98 times the unprotected mean. The driver then read the core's apertures:
- They are identical in the two boots except the rev-node fields (`rev_node_serving_idx_ex` 0x116c and 0xcb1).
- The commit pc, 0x3ff6780f04, is the same in both boots, and `commit_instr[0].valid` is 0, so it is slot-0 residue.
  - The native route maps this static executable with its text at 0x3ff6000000 (vaddr 0x10000), as a twin dump of the
    protected process's `/proc/<pid>/maps` shows (`nts-mp`, the same image).
  - So the pc is vaddr 0x790f04 in `work-sublet/domain/tshark_m5.dom`, whose strip is the image `bcc251f3…`. That is
    in `wmem_block_fast_sublet_alloc`, the patched allocator, and the word there (`0xa283955b`, `.insn r 0x5b, 1,
    0x51`) is the CDERIVE of the inlined `capstone_cap_derive`.
  - The silicon load base is assumed to equal the twin's (the same module and image; both boots read the same pc).
- `sw=225` reads `tbe`, `wrev` and `memwr`, and a CAPSTONE_DYN op is valid at the bypass head (`sw=220` 0x8c). On an
  older bitstream, `wrev` and `memwait` were set in a healthy control too ("resting state", `SILICON-BLOCKER.md` §3).
  The unprotected boots' apertures were read only after their command had ended, so there is no running sample to
  compare. These apertures do not tell a stuck core from a running one.
The allocator is hot, so a residue pc landing in it is not unlikely. A CDERIVE that does not complete would fit these
readings, but this boot does not show it, and it does **not** separate "slower than 3x" from "stops". It is recorded as **no return within 3,600 s, 2 of
2 boots**, not as a wedge.

**Against the pre-registration.**
- Answers: **missed for the protected arm**, which gave no output in either boot. Met for the unprotected arm (u1 and
  u2 equal stock's output). As pre-registered, a protected-only failure stops the comparison and is a silicon finding.
- Repeats: the unprotected pair's `run_cycles` differ by **1.35 %** (band 1 %), and `between_cycles` by 0.05 %.
- In-context cost, kernel, uptime span, event 23: **not read**, as there is no protected reading.
- Duration: 18.6 min per unprotected command against 22 min predicted. Silicon in-context cycles are 3.2 times the twin's.

**What the twin says.** The same images, stages and one-arm boots run on the twin in 7.99 s (u) and 8.32 s (p) of
uptime, and 7,449 M against 7,777 M in-context (+4.4 %). Both outputs equal stock. A silicon-only slowdown of 3x or
more would need a mechanism the twin does not model. A silicon stop would need a defect the twin does not reproduce.
This boot cannot choose between them. The heartbeat boot below is the instrument that can.

### Pre-registration: the protected tshark with a heartbeat, one boot on F (written before the boot)

**Question.** Does the protected tshark on silicon run more than 3x slower, or does it stop, and if it stops, at which
frame and with what still running?

**Configuration.**
- Bitstream F `a948e4134`, kit bkit75n, board 2 behind the SPEC lane's bo-b2 (the board stays on F). One boot, the
  protected arm only.
- Stage `nested/tshark/board/t-ph-stage` (`t-ph-stage.SHA256SUMS`). It is `t-p-stage` plus a heartbeat, and nothing else:
  - `train/run.sh` starts a background loop before the unpack. Every `train/hb-period` seconds (60) it prints
    `@@HB <n> <uptime> out=<bytes of st.out> cnt <the native counters>`. The loop is killed after `mark2`
    (`@@HB-STOP`).
  - The image is the same `bcc251f3…`, the same capture, and the same command line.
- Rung budget 3600, so the kit's x4 gives `BAKED_TIMEOUT=14400`, and the idle limit is the same 14,400 s
  (`board-run.sh:155`). The SPEC lane's list line is `3600 300`. This lane first sent `14400 300`, which the SPEC lane's
  dry run showed would bake `BAKED_TIMEOUT=57600`; the SPEC lane corrected it.
- Read by `tshark/hb-frames.py`, which maps each beat's byte count to the frame of the stock reference that holds that
  byte, and exits 2 when a transcript has no beat.

**Checked off the board** (rule 8; `recheck-t-ph.sh`, output in `t-ph-recheck/`):
- The real bake holds 26 manifest files, `train/hb-period` among them, and the baked `run.sh` carries the heartbeat.
- Two kit `BOARD_STUB=2` runs on synthetic transcripts that carry `@@HB` lines (`t-ph-recheck/synth/`):
  - one that returns classifies as smoke PASS, spec MALFORMED (the nested format, as in every nested boot);
  - one that does not return (t-p1's transcript with beats added) classifies as NO RESULT.
  - Both print `BAKED_TIMEOUT=14400`.
- The real firmware build is 23,760,392 B, under the cap, and the dom_stack gate passes.
- Twin control `nts-hb` (stage `tshark/stages/b-ph`, the same run.sh, `hb-period` 1):
  - the output equals stock's (sha256 `056c1f8c…`, 9,000 frames), rc 0;
  - six beats: two during the unpack, four during the native command, with `out` growing (frames 1,265, 2,987,
    5,145 and 7,834 by `hb-frames.py`), and `@@HB-STOP` printed;
  - during the command the beats came 1.5-2.9 s apart instead of 1 s, and the span rose from 8.32 s to 11.46 s. A 1 s
    beat costs the twin that much; a 60 s beat on silicon costs about 1/60 of it.

**Outcomes, decided by the beats (pre-registered reading):**
- **A, slow:** the beats continue and `out` grows through the run.
  - If the command returns, its output must equal stock's. Its uptime span and `run_cycles` against the u1/u2 means
    give the slowdown. It is one boot, so the repeat rule cannot apply and this is not the cost.
  - If it does not return within 14,400 s, the frames per minute give the rate.
- **B, stops while the kernel runs:** the beats continue but `out` stops growing for 10 or more beats. The counters
  then tell where the time goes: `run_cycles` growing means in-context (a loop in the domain), `between_cycles` alone
  means outside it.
- **C, the core or the kernel stops:** the beats stop. The last beat gives the frame and the time.
- **Not read:** if no beat is printed after `@@SPEC-BEGIN` while beats were printed during the unpack, a native task
  that never yields cannot be told from C at time 0. The twin control shows beats during the native task, but
  silicon's scheduling of it is not shown.
- **Not read** either if no `@@HB` is printed during the unpack, before `@@SPEC-BEGIN` (added before the boot at the
  SPEC lane's request).
- **VOID** as in the four-boot pre-registration (`b0-stats`, `b0-hello`, smoke).

**Prediction.** B or C: the output stops growing at some frame. No mechanism in hand would make the protected arm 3x
slower: the twin cost is +4.4 %, and on silicon the other four applications' protection costs at most +6.7 %.
**Not predicted:** which frame, or whether it is B or C. The u1/u2 rate is 8.05 frames/s, so a stop shows within one
or two beats.

**What follows.** A: the comparison is re-run with this budget. B or C: the frame number reduces the capture for a
folder under `fpga-repros/`, and the hardware side gets one link.

**Not run.** On 15 Nov evening the lead paused the nested work for the final SPEC test run, before this boot started
(its chain had only waited for the board). The stage, its checksums, the stubbed dry runs and the twin control remain,
so the boot can run as written when the work resumes.

## Re-timing on P1 (paged lifetime table)

### Pre-registration: speedtest1, PostgreSQL, CPython and CREVOKE chains on P1 (written 2026-11-17, before the boots)

**Question.** The same two arms as on F (the sublet processor in both; the application's lifetime patch applied or not)
on bitstream P1, whose lifetime table is paged and reached through a 16-entry record TLB. What does each protection cost
there, and how does CREVOKE scale when a subtree's records span more pages than the record TLB holds?

**Which patches, and why not Fig. 10's.** The images are the F images, unchanged: SQLite memsys5 + lookaside, CPython
0014 `6c2c170e…`, PostgreSQL 0003 `657c0e1f…` (Q-15-repaired). Fig. 10's reuse rows (llvm-capstone 5060011844e, tree
trap/eval ed8cc375c2) use later revisions, CPython 0014 `fc63552d…` and PostgreSQL 0003 `65580ed3…`. Those call
`capstone_cap_release`, a CREVOKE with the object alone (`.insn r 0x5b, 0x1, 0x52, x0, obj, x0`; ed8cc375c2
`capstone/runtime/include/capstone/capability.h:67`), the QEMU model's ISA. P1's CREVOKE takes rs1 = parent and
rs2 = child, and raises UNEXPECTED_OPERAND when rs2 is not a capability (capstone-ariane d83d3a633
`core/anvil_build/capstone_dyn_unit.anvil:303-330`). So Fig. 10's patches cannot run on P1. The SPEC lane chose these
variants; the rows say they are not Fig. 10's revisions.

**Configuration.**
- Bitstream P1 `…-tagoverlap6c-pagedtbl_d83d3a633` (sha256 `dc411e36…`), board 2, kit bkit91
  (`bkit91-k2n-native-p1-tree-cscfix`). Flashed by the chain if not resident, the seal's sha256 checked first and the
  resident name read back after.
- The kit's firmware sets the counters for the paged table (`CAPSTONE_HPM_EVENTS_PAGED`, `sbi_hart.c:99-108`):
  hpm3 = event 29 (record-TLB miss, a walk started), hpm4 = 30 (walk cycles), hpm5 = 31 (record accesses in paged
  mode), hpm6 = 33 (cause-30 grow), hpm7 = 23 (verdict-cache miss replays), hpm8 = 2 (L1D miss). F's L1I, DTLB,
  mispredict and issue-stall counters are not on this kit.
- Stages `nested/p1/<name>`, made by bkit91's `make-train-stage.sh` around each F boot's `train/` directory, which is
  byte-identical to F's (`diff -r`); only the kit's smoke differs. Checksums `nested/p1/<name>.SHA256SUMS`: st1
  `8448689f…`, pg `ccdbfbdd…`, crvt `81b6a7f1…`, cpy-u1/u2 `2d27c6b4…`, cpy-p1/p2 `585e2049…`.
- Images: speedtest1 `817b1652…` / `945fb69f…`; PostgreSQL `pg_unprot.xz` `26a9511f…` / `pg_prot.xz` `194d7fec…` (they
  unpack to F's `1ea084ab…` / `4cc5f247…`); CPython `b0b29e89…` / `eabba900…`.
- Boots, in this order, one board:
  1. `st1`: speedtest1 `--size 1`, U P P U in one boot, budget 900 (x4 = 3,600 s);
  2. `pg`: PostgreSQL, U P P U in one boot, budget 2700 (10,800 s; F's boot took 86 min);
  3. `crvt`: CREVOKE chains (below), budget 900;
  4. `cpy-u1`, `cpy-p1`, `cpy-p2`, `cpy-u2`: CPython, one arm per boot, budget 3600 each.
- Read by the F parsers (`stage-gen/parse.py`, `pg/parse-pg.py`, `cpy/parse-cpy.py`), with hpm3..8 relabelled as above.

**CREVOKE chains (`crvt`, new domain `crvtime` `b7b4af35…`, source `crvt/crvtime.c` `6bc727e4…`; stage maker
`crvt/make-stage-crv4.py`).** One process per command, the cycle counter as on F:
- probes: `control`, `readclock` with cycle and mcycle, and `readclock` with `CRVT_HPM=1`. If hpmcounter3 reads from
  the domain, the chain commands read it and hpmcounter4 around the CREVOKE;
- contiguous, as on F: a chain of 1,000, 10,000 and 100,000 nodes, then one CREVOKE of it;
- scattered: 16,384 siblings (64 record pages; 102,400 and 400 pages for the largest) are derived, then revoked one by
  one page-strided, so the LIFO free list hands out records a page apart. Then a chain of 1,000, 10,000 or 100,000
  nodes is built on them, and one CREVOKE of it;
- last, the probe toward 10^6 nodes, with progress lines every 65,536.
- Twin (qemu55, flat table, run `ncrv-c4`): every command returns, and the probe dies after 196,609 nodes, as on F.
  The twin's cycle counts are instruction-based and its table is flat, so it shows only that the commands run.

**Outcomes and bands.**
- Applications:
  - answers equal to the reference in every command, else that boot is VOID for the cost;
  - the repeat rule as on F: within 1 % `run_cycles` and 2 % `between_cycles`, or that difference is not read;
  - in-context cost, protected vs not: speedtest1 +0.5 % to +5 %; PostgreSQL +3 % to +25 %; CPython, the benchmarks
    read, summed, +3 % to +25 %. The bands are wide on purpose: on F the protected arms' peak live records (398, 105,043
    and 76,767) stayed in the flat table, and here every record-TLB miss costs a three-read walk;
  - record-TLB misses (hpm3) per command: more in the protected arm than in the unprotected arm, for each application.
- CREVOKE:
  - contiguous, per node: 15 to 60 cycles for 10,000 and 100,000 nodes (F: 27.9 and 28.2);
  - scattered, per node: more than contiguous at the same N, predicted 40 to 400 cycles at 10,000 and 100,000;
  - the scatter took only if the scattered CREVOKE's record-TLB misses are at least 4 times the contiguous one's at the
    same N. If they are not, or hpmcounter3 cannot be read, the scattered rows are reported as chains on churned
    records, not as scattered;
  - the 10^6 probe: not predicted. The paged table can grow (event 33), so it may reach 10^6. The reading is the count
    reached and whether the process died.
- **VOID**, as on F: `b0-stats` or `b0-hello` fails, or the smoke fails beyond the kit's waiver.

**Deadline.** The SPEC lane uses P1 rows that land by 2026-11-18 14:00 +08. Until then the paper appendix carries the
F rows, marked as the dev-era patches on bitstream F.

### Pre-registration addendum: a second session on boards 1 and 3 (written 2026-11-17 evening, before these boots and before board 2's PostgreSQL result)

The SPEC lane offered two more board slots tonight. Both run on P1 with bkit91, with stages made as above
(`nested/mk-p1s2-stages.sh`, F's byte-identical `train/` inside bkit91's smoke; checksums `nested/p1s2/<name>.SHA256SUMS`).
- **Board 1**, after the SPEC lane's x264 run (wait pid 40534). The chain flashes P1 from the matched-baseline
  bitstream, with the seal check and readback.
  1. `ff`: FFmpeg, U P P U, then `fp` (the protected flipped-input control); F's `f2-stage` train, budget 900. Checksums
     `a5e54049…`; images `ff_unprot.xz` `d92e0113…`, `ff_prot.xz` `1afff123…`, `ff_prot_flip.xz` `8ef6bcc0…`.
  2. `pg2`: PostgreSQL, a second U P P U session; the same stage as `pg` (`ccdbfbdd…`), budget 2700.
- **Board 3**, after the SPEC lane's embedded-suite chain (wait pid 403643); it is already on P1.
  `cpy2-u1`, `cpy2-p1`, `cpy2-p2`, `cpy2-u2`: a second CPython session, one arm per boot, the same stages as
  `cpy-*` (`2d27c6b4…` u, `585e2049…` p), budget 3600 each.

**Reading rule for two sessions.** Each session is read on its own pairs by the 1 % / 2 % rule. A cost is reported for
each session that passes. When both sessions of an application pass, its reading is the mean of the two sessions' ABBA
means, with both shown. A session that fails is reported as not read, and its runs are never pooled with the other
session's. For CPython the rule applies per benchmark, as before.

**Bands.** PostgreSQL and CPython: as for the first P1 session. FFmpeg, as on F: every command decodes 300 frames equal
to stock's framemd5, `fp` decodes 300 frames with at least one different (else the comparison stops); in-context cost
-1 % to +1 %; event 23 (hpm7 on this kit), protected minus not, under +10,000 per command.

### Pre-registration: the protected tshark with a heartbeat on P1, one boot (written 2026-11-17 23:25, before the boot)

**Question.** On P1, does the protected tshark complete, and if it does not, is it slow (A), stopped while the kernel
runs (B), or stopped with the core or kernel (C), and at which frame? This is M-16's open question on a different
lifetime table: F's flat table against P1's paged one (4 KiB pages of 256 records, a 16-entry record TLB, a three-read
walk, and a table that grows on cause 30).

**Configuration.**
- P1 `…-tagoverlap6c-pagedtbl_d83d3a633`, kit bkit91, board 1. The chain `nested/p1ts/p1ts-b1.*` waits for this lane's
  pg2 chain (pid 886911); board 1 is already on P1, read back by name before the boot. One boot, the protected arm only.
- Stage `nested/p1ts/tsph` (`tsph.SHA256SUMS`): bkit91's `make-train-stage.sh` around the heartbeat stage's `train/`
  (`nested/tshark/board/t-ph-stage/train`), copied unchanged (a file-for-file sha256 diff is empty).
  - Image `bcc251f3…`, the same as F's protected boots: wmem patch 0001, the dev-era revision with the Q-15 repair,
    which calls the two-operand CREVOKE. A census of the image finds no CREVOKE word with rs2 = x0 and 11 with rs2 != x0.
    Fig. 10's wmem revision, which calls one-operand `capstone_cap_release`, cannot run on P1.
  - The same capture (9,000 frames), command line and 60 s heartbeat as the F pre-registration above.
  - This is not Fig. 10's workload (tshark over 169 test captures). The appendix row says so.
- Rung budget 1800, which the kit's x4 makes `BAKED_TIMEOUT=7200`; the idle limit is the same. The list line is
  `tsph 1800 130`.
- Read by `tshark/hb-frames.py` against `wl/ref1000.out` (stock, sha256 `056c1f8c…`) and by `tshark/parse-ts.py` if the
  command returns.

**Checked off the board** (rule 8; `nested/p1ts/recheck-ts.sh`, output `nested/p1ts/tsph-recheck/`):
- The real bake on bkit91 returned 0.
- Two kit `BOARD_STUB=2` runs returned 0: F's heartbeat synthetic transcript that returns, and the one that does not
  (both carry `@@HB` lines). Both print `BAKED_TIMEOUT=7200`.
- The real firmware build is 23,760,392 B (sha `b946125b…`), under the cap.
- An `NR_DRY=1` pass of the chain.
- Twin control `nts-p1hb2` (the staged `train/` itself, slot 28, twin kernel `perf/natio/images`, flat-table model):
  returned rc 0 in 8.25 s of command uptime, output equal to stock (`056c1f8c…`, 1,140,005 lines, 9,000 frames).
  With a 60 s period it printed no beat inside the command; F's twin control `nts-hb` (period 1) showed the beats. The twin models neither P1's paged table nor F's, so it shows only that the stage runs.

**Outcomes:** A, B, C, "not read" and VOID exactly as in the F heartbeat pre-registration above. In addition:
- **A with a return inside 7,200 s:** the output must equal stock's. The uptime span against F's unprotected mean
  (1,118 s) and the counters give the slowdown. This is one boot, so it is not the cost. If it does return, a U P P U
  comparison follows when a board is free.

**Prediction.** Not predicted. On F the two protected boots gave no output for at least 3,327 s. The record table is the
part of the core that P1 changed most, so P1 may complete, or it may stop as F did. If it stops, the frame and B
against C are the readings. If it completes, M-16 is narrowed to F's table, not fixed: no root cause is claimed from
this boot.

**What follows.**
- A: the time is reported, and the U P P U comparison is scheduled.
- B or C: the frame reduces the capture for a folder under `fpga-repros/`; the appendix row reads "does not complete
  on P1, M-16".

## Appendix D completeness on P1: mruby, PHP, Perl and memcached

These four were not timed on silicon before. Each runs on P1 (`…-pagedtbl_d83d3a633`, kit bkit91) with the design of
the P1 sections above: stock against the lifetime patch, U P P U, the same counters, the same repeat rule (1 %
`run_cycles`, 2 % `between_cycles`, else not read). **Every patch is the dev-era two-operand revision**, not the one on
native-kernel-new-qemu (5060011844e) that Fig. 10 used: Fig. 10's revisions call the one-operand
`capstone_cap_release` (and memcached's CRENEW), which P1 does not implement. Every image is a virtual-profile domain
compiled and linked through the FRCDKMS SDK (`perf/cclever/arms/FRCDKMS/sdk`, llvm-c9, the F/P1 flags), as the
CPython, PostgreSQL and FFmpeg images were. Stages are made by `nested/appd/make-abba-stage.py` (one boot, the four
commands in order, a fresh tmpfs per command, the arm's image staged as given) and read by `nested/appd/parse-abba.py`
(the normalised output's sha256 against the native reference) and `nested/p1/abba.py`.

### Pre-registration: mruby on P1, both arms in one boot (written 2026-11-17 23:55, before the boot)

**Configuration.**
- mruby 4.0.0-rc2 (pin 9d523e2f), built by the compiler lane (llvm-capstone `appd-native-images` c839ecfb2add, from
  dev 83f60f4660cb; `perf/appd/mruby/build.sh`). Stripped images: stock `9d61ccdd…`, protected `67217aa5…` (full
  `081a8f55…` / `de2f200f…`).
  - Protected: patch 0008 `gc-slots-as-sublet-lifetimes`, dev's revision (blob `1541cee2`) **with this lane's Q-15
    repair** (blob `15c05a92`, `nested/appd/mrb-q15/0008-q15.patch`), live under `-DMRB_CAPSTONE_SUBLET`.
    - Why the repair: the dev revision's protected image faulted on the twin at its first heap page (cause 29 at the
      CDERIVE in `add_heap`, vaddr 0x4ad5e; twin `nmrb-2`). Its `page->life` is 80 KiB at a 16-byte-aligned
      address, and the compact format's granule for 80 KiB is 128 (`capstone/representable.h`), so the virtual
      profile refuses it (Q-15).
    - The repair makes `life` the smallest representable range around `objects`: the base rounded down into
      `issued[]`, the end rounded up into a new 256-byte `life_pad`, and objects' offset kept in `life_skew`. Only
      the slots' 80-byte children (exact below 4 KiB) are handed out.
  - The port applies 0001-0003 and 0008-0010 to both arms. The stock arm's `gc.o` is byte-identical to `gc.c` with
    0008 reversed, for the repaired 0008 too.
  - CREVOKE census (encoding `(w & 0xfe00707f) == 0xa400105b`, 2-byte stride): protected 0 with rs2 = x0 and 8 with
    rs2 != x0 (2 from 0008, 6 in the libc's mallocng); stock 0 and 6 (the same six).
  - mrb_malloc sits on the libc's mallocng in both arms.
- Workload, Fig. 10's: `mruby bm_ao_render.rb 16` (`bm_ao_render.rb` `e44efffd…`, from the same tree). The answer is the
  PPM on stdout; native reference `cfd73bb1…` (781 bytes), the same from both arms' native builds.
- Stage `nested/p1a/mrb` (`mrb.SHA256SUMS`): bkit91's `make-train-stage.sh` around `nested/appd/mrb/stage`, copied unchanged;, commands u1 p1 p2 u2 in one
  boot. Board 2, after the SPEC lane's chain pid 1217768; the chain flashes P1 (board 2 is on the matched bitstream)
  from the seal, sha256 checked, and reads it back by name. Chain `nested/p1a/b2-mrb.*`. Budget 1200 (x4 = `BAKED_TIMEOUT=4800`), booking 70 min.
- Checked off the board: real bake, `BOARD_STUB=2` on a returning and a non-returning transcript (both rc 0, both
  print 4,800), real firmware 19,566,088 B (`c52a7e42…`), and an `NR_DRY=1` pass of the chain.
- Twin control `nmrb-3` (slot 28, `perf/natio/images`, flat-table model): all four commands rc 0 with the native PPM,
  4.4-4.5 s each. In-context u1 / p1 / p2 / u2 = 4,384.1 / 4,455.9 / 4,455.9 / 4,384.1 M, **+1.64 %**, repeats 0.00 %.

**Outcomes and bands.**
- Answers: every command's PPM equals the native reference, else VOID for the cost.
- Repeat rule as above.
- In-context cost, protected vs not: +0.5 % to +10 % (the twin's +1.64 % times the 1.9 to 3.5 silicon-to-twin cost
  ratios of PostgreSQL, CPython and speedtest1 on P1, widened).
- Duration: 6 to 14 min per command (the twin's in-context x 2.3 to 4.5, at 25 M cycles/s).
- Record-TLB misses (hpm3): more in the protected arm.
- **VOID** if `b0-stats`, `b0-hello` or the smoke fails beyond the kit's waiver.

### Pre-registration: memcached on P1, an in-process driver, two boots (written 2026-11-18 00:20, before the boots)

**Why a driver.** The native route refuses memcached's server by design: `native-syscall-table.c` lists `socket`,
`socketpair`, `bind`, `listen`, `connect`, `sendto`, `setsockopt`, `shutdown`, `epoll_create1`/`epoll_ctl`, `eventfd2` and
`timerfd_*` as NS_REFUSE, and there are no threads. So this row runs **memcached 1.6.45's own item and slab code in one
process, not the server**: `items.c` (unchanged, through `nested/mc/driver/items_tu.c`), `slabs.c`, `assoc.c`, the hash
functions and `cache.c`, with libevent's types, the mutexes, the condition variables and the thread starts stubbed
(`nested/mc/driver/`). No socket, thread or event loop is built.

**Configuration.**
- Driver `nested/mc/driver/mcdrv.c`, built by `nested/mc/build-mcdrv.sh` through the FRCDKMS SDK with the F/P1 flags.
  - The command stream is Fig. 10's mc-crusher mix (`experiments/study/reuse-hooks/memcached/mc-crusher-established.conf`
    at 5060011844e): 16 connections, per prefix a:/b:/c:/d: one set and three get connections, values 100 / 500 /
    2,000 / 8,000 bytes, 10,000 keys drawn zipf(0.25), pipelines of 4, 15,000 commands per connection, 240,000 in all,
    a fixed seed. A set follows `do_store_item`'s NREAD_SET path; a get is `item_get` then `item_remove`; one `cache.c`
    read buffer per pipeline; one LRU maintainer pass every 256 commands.
  - Settings are `settings_init`'s with `-o slab_automove=0` and `-m` from argv[1].
  - It prints the stream's hash, a CHECK line (hits, misses, set failures, value-integrity failures, items, slab
    allocations, evictions) and ends `MCDRV-END rc=0`. The answer is the stream hash, `set_fail=0` and `bad=0` (the
    other CHECK fields are layout-dependent).
- Arms: stock = patches 0001-0005; protected = + 0006 `slabs-and-cache-sublet-lifetimes` (dev's revision, blob
  `311a14e7`, two-operand) **with this lane's Q-15 repair**: under `MC_CAPSTONE_SUBLET` the cache's buffer size, the slab
  class sizes, a page's generation length and a page's own length go through `capstone_repr_round`, so every CDERIVE is
  exact (`nested/mc/src-prot` against `src-prot-0006`). Without it the protected driver faulted on the twin at its first
  `cache_alloc` (cause 29: 16,400 bytes is not a multiple of its granule, 32; twin `nmc-1`). `diff -rq src-stock
  src-prot` lists only the four files 0006 touches.
- Images: stock `02cc0e9a…` (full `96680334…`), protected `df5744e2…` (full `0ce78f23…`). CREVOKE census: stock 0 with
  rs2 = x0 and 6 with rs2 != x0 (mallocng); protected 0 and 11 (5 from 0006).
- **Two boots, one chain** (`nested/p1a/b2-mc.*`, stages `nested/p1a/mc128` and `mc32`, bkit91 wraps of `nested/appd/mc128` and `mc32`), board 2 after this lane's mruby chain:
  1. `mc128`: `-m 128`, **the reading**. Nothing is evicted in either arm, so both arms do the same work: on the twin
     both read 88,691 hits, 0 evictions and 30,360 items.
  2. `mc32`: `-m 32`, Fig. 10's setting. 0006 adds `sublet_parent` to the item header, and the Q-15 repair rounds the
     class sizes, so the protected arm holds fewer items and evicts more: on the twin the protected arm reads 53,506 hits and 29,807 evictions against 57,586 and 27,108, and its in-context cycles are 2.41 % fewer. **The arms do different
     work here**, and the row says so; it is reported, not read as the cost.
  - Each: u1 p1 p2 u2 in one boot, budget 900 (x4 = 3,600 s), booking 50 min.
- Twin controls `nmc128-1`, `nmc32-1`: all commands rc 0 with the native normalised output. `mc128` in-context
  3,177.9 / 3,180.9 / 3,180.9 / 3,177.9 M, **+0.10 %**, kernel +6.9 %.
- Natively the protected driver makes 120,065 CDERIVE and 105,188 CREVOKE per run at `-m 32`.

- Checked off the board, for every stage: real bake, `BOARD_STUB=2` on a returning and a non-returning transcript (rc 0, the budgets printed as registered), real firmware (19,566,088 B), and an `NR_DRY=1` pass of the chain (`nested/p1a/b2-mc.*`).

**Outcomes and bands.**
- Answers: stream hash, `set_fail=0` and `bad=0` in every command, else VOID for the cost.
- Repeat rule as above.
- `mc128` in-context cost, protected vs not: -1 % to +3 %.
- `mc128`: the hit, miss, eviction and item counts equal between the arms (read from the CHECK lines).
- Duration: 5 to 10 min per command.
- **VOID** as above.

### Pre-registration: PHP on P1, both arms in one boot (written 2026-11-18 00:15, before the boot)

**Configuration.**
- PHP 5.0.0, built by the compiler lane (llvm-capstone `appd-native-images` 1ab971367ac4, from dev 83f60f4660cb;
  `perf/appd/php/build.sh`, `build-virtual.sh realphp` with the FRCDKMS SDK and the F/P1 flags at -O2). **One image for
  both arms**, `php-nested.strip.dom` `d63abbc5…` (full `1ec91755…`), with `zend-alloc/sublet` patch 0001 (blob
  `a0274b91`) and `zend-cap-mode.c` (blob `2851e112`, dev's revision: `heap_end(block, ptr)`, the two-operand CREVOKE).
- **The arms are run-time modes of that one image** (`PHP_CAP_MODE`, read from the environment; the image aborts if it
  is unset, and prints its mode to stderr):
  - not protected, **mode 4**: blocks returned to mallocng, and the caller handed a SHRINKed copy of the block over the
    bound (`zend-cap-mode.c:150-155`);
  - protected, **mode 6**: the same block geometry and the same bound, the caller handed a CDERIVE child
    (`heap_issue`), ended at efree by `heap_end(block, ptr)`.
  - So the arms differ in the lifetime alone. This is **not Fig. 10's pairing**: Fig. 10's stock arm was a second,
    pristine binary (`php-stock` `13d187b0…` here), which also differs in block geometry. The row says so.
- CREVOKE census: the image has 0 with rs2 = x0 and 7 with rs2 != x0 (1 in `zend_cap_reclaim`, 6 in mallocng).
- Workload: Fig. 10's pinned tenth of bench.php (`bench-tenth.php` `6b9fc8f9…`, Fig. 10's own bench.php hash), run as
  `php -n bench10-check.php`, where `bench10-check.php` (`655fd6d4…`) differs from it in one line (end_test's
  `ob_end_clean()` is `ob_end_flush()`) so that every test's result is printed. The answer is the output with the
  timing column stripped (`sed -E 's/ +[0-9]+\.[0-9]{3}$//'`): native reference `8f692e64…` (two native runs identical).
  The printed results add a little output work to both arms.
- Stage `nested/p1a/php10` (bkit91's wrap of `nested/appd/php10/stage`), u1 p1 p2 u2 in one boot. Board 2, after this
  lane's memcached chain. Budget 2400 (x4 = 9,600 s), booking 120 min.
- Twin control `nphp10-1`: four commands rc 0 with the native normalised output; the domain printed `ZEND_CAP_MODE 4
  … revoke=0` and `6 … revoke=1`. In-context 8,765.5 / 8,779.2 / 8,779.2 / 8,765.5 M, **+0.16 %**, repeats 0.00 %. In
  the smaller bench100g twin (`nphp-1`) the protected arm issued and revoked 364,388 children per run.

- Checked off the board, for every stage: real bake, `BOARD_STUB=2` on a returning and a non-returning transcript (rc 0, the budgets printed as registered), real firmware (19,566,088 B), and an `NR_DRY=1` pass of the chain (`nested/p1a/b2-php.*`).

**Outcomes and bands.**
- Answers: every command's normalised output equals the native reference, else VOID for the cost.
- Repeat rule as above.
- In-context cost, protected vs not: -1 % to +3 % (the twin's +0.16 % is the smallest of these applications; on P1 the
  others ran 1.9 to 3.5 times their twin cost).
- Duration: 13 to 26 min per command (the twin's in-context x 2.3 to 4.5 at 25 M cycles/s).
- **VOID** as above.

### Pre-registration: Perl on P1, both arms in one boot, a 2^20-record table (written 2026-11-18 00:35, before the boot)

**Configuration.**
- Perl 5.36.3 built for the native route by this lane (`nested/perl/build-arms.sh`: the port's `build-perl-domain.sh`
  with its virtual SDK build replaced by the FRCDKMS SDK, the F/P1 flags at -O2; perl-cross c2d8f8b7027e; the release
  tarball `f2a1ad88…`). Stripped images: stock `3012f392…`, protected `ac2507d3…` (full `1a768a52…` / `2ba4dd05…`).
  - Protected: `perl/sublet` patch 0001 `sv-heads-as-sublet-lifetimes` (dev's revision, blob `72db9236`, two-operand
    `capstone_cap_revoke_child`), selected by the port's `PERLD_SUBLET=1` (`-DPERL_CAPSTONE_SUBLET`). The two arms'
    `config.sh` differ only in that define and its include path.
  - CREVOKE census: stock 0 with rs2 = x0 and 6 with rs2 != x0 (mallocng); protected 0 and 9 (3 in `Perl_sv_replace`,
    `Perl_sv_clear` and `Perl_sv_free2`).
- Workload, Fig. 10's: SPEC CPU2017 500.perlbench_r train `suns.pl` (`7545b37d…`, with `WORDS` `f32a0a4b…`), run as
  `PERL_HASH_SEED=0 perl -I. -I./lib suns.pl`. The answer is stdout against SPEC's `suns.out` (`01688ea0…`). suns.pl
  loads no module (a native `%INC` dump), so the stage carries suns.pl and WORDS without the lib tree.
- **The record table.** At bkit91's `node_flat_records=262144` the protected arm cannot run: on the twin it ends with
  cause 30 (node exhaustion) in both protected commands (`nperl-1`). The peak is bracketed on the twin's table model: the protected arm still ends with cause 30 at 327,680 records (`nperl-320k`) and completes at 393,216 (`nperl-384k`), 655,360 (`nperl-640k`) and 1,048,576 (`nperl-1m`), so its peak lies between 327,680 and 393,216 records, counting the floor and the context's other lifetimes. So this boot runs a copy of the kit,
  `nested/kits/bkit91-rec1m`, which differs from bkit91 in one word of `board.env` (`node_flat_records=1048576`; `diff -r`
  shows only that line), and **both arms run under it**. The tree build grows the table up to that ceiling
  (`capstone_vm.c:362-364, 887-905`); 2^20 is the module's own default (`:415`). The row says: the record table is
  configured for 2^20 records for this program, because the default 262,144 is exhausted by its peak of 327,680 to 393,216 live records.
- Stage `nested/p1a/perl` (the rec1m kit's wrap of `nested/appd/perl/stage`), u1 p1 p2 u2 in one boot, images
  xz-compressed in the stage. Board 1, after this lane's tshark heartbeat chain (pid 1526065); board 1 is on P1, read back
  by name. Chain `nested/p1a/b1-perl.*`. Budget 900 (x4 = 3,600 s), booking 45 min.
- Checked off the board: real bake on the rec1m kit, `BOARD_STUB=2` on a returning and a non-returning transcript (both
  rc 0, both print 3,600), real firmware 23,760,392 B (`69c4d8a1…`), the stub's relinked kernel configuration carrying
  `node_flat_records=1048576`, and an `NR_DRY=1` pass of the chain.
- Twin control `nperl-1m` (2^20 records): four commands rc 0 with SPEC's `suns.out`. In-context 1,731.9 / 2,429.1 /
  2,429.1 / 1,731.9 M, **+40.3 %**, repeats 0.00 %; kernel +14.5 %.

**Outcomes and bands.**
- Answers: every command's stdout equals `suns.out`, else VOID for the cost.
- Repeat rule as above.
- In-context cost, protected vs not: +20 % to +150 % (the twin's +40.3 %; the P1 cost ratios of the other applications
  would put it at 75-140 %, widened down because Perl's cost may be less cache-bound).
- Duration: 3 to 8 min per command.
- Record-table growth (hpm6, event 33): more in the protected arm.
- **VOID** as above.

### Pre-registration: tshark on P1, U P P U without a heartbeat, one arm per boot (written 2026-11-18 02:30, before the boots)

**Question.** The heartbeat boot above returned (outcome A), so the U P P U comparison it pre-registered follows. What
does the protection cost tshark on P1? And is the heartbeat boot's long span the beat's own `wc` reading `st.out`
(H1), or a kernel cost per frame inside tshark that grows over the run (H2)?

**Configuration.**
- P1 `…-tagoverlap6c-pagedtbl_d83d3a633`, kit bkit91, board 1. The chain `nested/p1tu/p1tu-b1.*` waits for this lane's
  Perl chain (pid 1793373). Board 1 is on P1; the chain reads it back by name before the first boot.
- Four boots, one arm each, in the order U P P U, as on F: `tsu-u1`, `tsu-p1`, `tsu-p2`, `tsu-u2`.
  - The stages come from `nested/p1tu/mk-stages.sh`: bkit91's `make-train-stage.sh` around F's
    `nested/tshark/board/t-u-stage/train` and `t-p-stage/train`, copied unchanged. A file-for-file sha256 diff is
    empty, and the wrapper equals the heartbeat stage's. Checksums are in `nested/p1tu/<name>.SHA256SUMS`.
  - The images are F's: `ts_unprot.xz` (`2ceb6806…`) unpacks to `2061f661…`, with 6 CREVOKE words, all rs2 != x0.
    `ts_prot.xz` (`c1a1293f…`) unpacks to `bcc251f3…`: wmem 0001, the dev-era two-operand revision with the Q-15
    repair, with 11 CREVOKE words, all rs2 != x0.
  - The capture (9,000 frames) and the command line `-r w.pcap -V -n` are the heartbeat boot's. There is no heartbeat.
  - This is not Fig. 10's workload (tshark over 169 test captures); the appendix row says so.
- List lines `<stage> 900 60`; the kit's x4 makes `BAKED_TIMEOUT=3600`.
- Read by `tshark/parse-ts.py` against `wl/ref1000.out` (`056c1f8c…`) and by `nested/p1/abba.py`.

**Checked off the board** (rule 8; `nested/p1tu/recheck-all.sh` runs `nested/p1/recheck-p1.sh` for each stage, with
output in `nested/p1tu/<name>-recheck/`), for each stage:
  - The real bake on bkit91 returned 0.
  - Kit `BOARD_STUB=2` returned 0 on a real transcript that returns: F's `t-u1` for the U stages, and the P1 heartbeat
    boot's for the P stages. It also returned 0 on that transcript cut after `@@SPEC-BEGIN`, which does not return.
    Both print `BAKED_TIMEOUT=3600`.
  - The real firmware build is 23,760,392 B, under the cap.
- `nested/p1tu/tsu-abba.py` combines the four parses for `abba.py`. On the stubs' real transcripts it reproduces F's U
  readings. On the cut transcript it exits 2 (`NO @@SPEC-END`).
- An `NR_DRY=1` pass of the chain reached `DONE` (`nested/p1tu/dry.log`).

**Bands.**
- Answers: every command returns rc 0 with output equal to stock (`056c1f8c…`, 1,140,005 lines, 9,000 frames).
  Otherwise that arm is not read.
- Repeats: 1 % `run_cycles`, 2 % `between_cycles`, per arm pair. Otherwise the difference is not read.
- In-context cost (ABBA means): +0 % to +6 %. The twin gave +4.4 %. The heartbeat boot gave +1.5 %, but that
  compared across two bitstreams.
- Kernel and other: no band.

**H1 against H2, decided by the protected arm's mean.**
- H1: `between_cycles` at most 8.5 G and an uptime span at most 1,400 s. The heartbeat boot's fit predicts 6.7 G
  (0.741 M per frame) and about 1,250 s.
- H2: `between_cycles` at least 14 G, or a span at least 1,600 s. The heartbeat boot had 17.3 G and 1,669 s.
- Between those, neither is decided.

**Prediction.** H1. In-context +1 % to +5 %. The protected arm's kernel cycles exceed the unprotected arm's: on F
the unprotected arm had 0.43 M per frame, and the fit gives the protected arm 0.74 M on P1.

**What follows.** If read, the cost goes into the appendix-D row together with the H1/H2 outcome. If H2, the frame
profile from the heartbeat boot reduces the capture for a folder under `fpga-repros/`.

### Pre-registration: Perl on B5a (the record cache), U P P U in one boot, a 2^20-record table (written 2026-11-18 04:50, before the boot)

**Why.** On P1, Perl's protection costs +208.1 % in-context, and the counters put the difference in the record table:
49.4 M record-TLB misses, 131 M paged record accesses and 260 M more L1D misses per protected command (results above). B5a
adds the paper's record cache (2 ways, 8 KiB, 64-byte lines of four records, inside a context only) on P1's RTL. This
boot asks whether the record cache removes that cost. The question was proposed by the runtime lane.

**Configuration.**
- Bitstream B5a, `caplifive_…-pagedtbl-reccache_6e82d7bd2`, sealed `806c5f55…` (capstone-ariane 6e82d7bd2, on P1
  d83d3a633). Its grade is in land/rtl-sublet-p2 `capstone/docs/plans/2026-11-16-sublet-phase2-record-cache.md`, with
  key 5 NOT MET as on P1. The runtime lane has run SPEC perl and gcc on it (Addenda 157/158). The flash is ours under the
  reflash grant: the chain flashes it from the seal (sha256 checked) and reads it back by name.
- Kit `nested/kits/bkit94-rec1m`: a `cp -rL` of bkit94 (B5a, the record-cache event set) with `node_flat_records=1048576`.
  `diff` against bkit94 shows that one word plus a comment line. Against `bkit91-rec1m` (Perl's P1 boot), it differs in the
  bitstream, its sha256, and the firmware wrapper and define (`wt-wrap-hpmrc`, `-DCAPSTONE_HPM_EVENTS_RCACHE`); every other
  file but `SHA256SUMS` is identical.
- **Events, bkit94's** (`mhpmevent3..8` = 34, 35, 31, 37, 36, 2):
  - hpm3: record-cache read hit;
  - hpm4: record-cache read miss;
  - hpm5: paged record accesses (walker);
  - hpm6: background fill reads;
  - hpm7: record-cache write hit;
  - hpm8: L1D misses.
  - So the record-TLB miss and walk-cycle counts of P1's set are not read here; hpm5 and hpm8 are common to both.
  - Read with `ABBA_EVENTS=rc python3 nested/p1/abba.py`. The `rc` label set was added today, and the default P1 output
    is byte-identical on `p1a/perl.json`.
- Stage `nested/b5a/perl`, a copy of `nested/p1a/perl`. `diff -r` is empty and it matches `perl.SHA256SUMS`, so the
  images, the workload (`suns.pl`) and the answer (`suns.out`, `01688ea0…`) are the P1 boot's.
- u1 p1 p2 u2 in one boot, budget 900 (x4 = 3,600 s), booking 60 min. Board 2, after the runtime lane's spa3p chain
  (nat-run3 pid 1792511). Chain `nested/b5a/b2-perl.*`. **It leaves board 2 on B5a.**
- Checked off the board:
  - real bake on bkit94-rec1m;
  - `BOARD_STUB=2` on a returning transcript (P1's Perl boot) and a non-returning one: rc 0 and 0, BAKED_TIMEOUT 3600 in
    both, verdicts PASS and NO RESULT;
  - the relinked kernel's configuration and Image carry `node_flat_records=1048576`;
  - real firmware 23,760,392 B (`92dcc03f…`);
  - an `NR_DRY=1` pass of the chain to DONE (`nested/b5a/dry.log`).

**Outcomes and bands.**
- Answers: every command's stdout equals `suns.out`, else VOID for the cost.
- Repeat rule as above.
- Counters fire: hpm3 + hpm4 > 0 in both protected commands, and ev31 − ev35 − ev37 ≥ 0 in every command (the
  identity in the record-cache plan). Else the event readings are void; the cycles still stand.
- **The reading:** B5a's in-context cost, protected vs not, set against P1's +208.1 %. P1's boot was on board 1, so the
  two costs come from different boards.
  - ≤ +104 %: the record cache removes at least half of Perl's cost;
  - +104 % to +170 %: it removes part of it;
  - ≥ +170 %: it does not remove it.
- **Prediction: +120 % to +230 %, read hit rate hpm3 / (hpm3 + hpm4) 15 % to 50 % in the protected arm.**
  - The cache holds 512 records. Perl's protected peak is 327,680-393,216 live records, and P1's 16-entry record TLB,
    which reaches 4,096, already misses on 38 % of paged accesses (49.4 M of 131 M).
  - Perl makerand read a 25.9 % hit rate on B5a.
  - A hit saves at most P1's ≈88 added cycles per added paged access (10.22 G / 115.7 M). At 25 % hits that is about
    3 G of the 10.22 G.
  - The fill's extra L1D reads cost against that: gcc ran +0.4 % slower on B5a.
- Stock arm against P1's stock arm (4.91 G, board 1): within ±1 %, since its record traffic is small (15.5 M paged
  accesses). Reported only.
- Duration: 3 to 12 min per command.
- **VOID** as above.

### Results: Perl on B5a (the record cache), board 2, 2026-11-18 08:20-09:08: +260.2 % in-context, the record cache does not remove the cost

**Configuration.** As pre-registered (7fc3183ce3b2; Addendum 174, `8299e1ed6287`).
- The chain `nested/b5a/b2-perl.*` (nat-run3 pid 2290383) ran after the runtime lane's spa3p chain. It flashed B5a
  from the seal: the local sha256 matched the sealed hash, and the name read back after the power cycle.
- Transcript `spec/board/out-nat-b5a-perl-b2/perl/board.txt`, sha256 `bf795fc04d3a185b6c77df8c355b9471f332215b9d83ad494beca447e9134a0e`,
  firmware `b09d5dd5…`.
- `b0-stats` and `b0-hello` returned 0, and the smoke passed all 9 steps.
- The kernel command line equals that of P1's Perl boot (`out-nat-p1a-perl-b1`), including `node_flat_records=1048576`.
  The bake recompiled the four `CONFIG_CMDLINE` objects because the bake before it, on the shared tree, used 262144.

**Readings** (`parse-abba.py` against `suns.out` `01688ea0c1058063…`, then `ABBA_EVENTS=rc nested/p1/abba.py`; JSON
`nested/b5a/perl.json`):

| | u1 | p1 | p2 | u2 |
|---|---:|---:|---:|---:|
| in-context cycles | 4,958,653,357 | 17,857,967,802 | 17,856,420,707 | 4,956,113,253 |
| kernel + other | 410,576,459 | 597,740,151 | 584,704,256 | 404,386,510 |
| uptime span, s | 219.09 | 743.63 | 743.05 | 218.74 |
| context entries | 4,604 | 7,541 | 7,496 | 4,550 |
| hpm3, rcache read hit (ev 34) | 8,016,399 | 17,180,859 | 17,181,414 | 8,016,662 |
| hpm4, rcache read miss (ev 35) | 1,217,980 | 101,090,815 | 101,089,340 | 1,215,010 |
| hpm5, paged record accesses (ev 31) | 10,494,724 | 330,597,203 | 330,595,706 | 10,485,071 |
| hpm6, fill reads (ev 37) | 3,039,798 | 216,571,136 | 216,571,113 | 3,033,115 |
| hpm7, rcache write hit (ev 36) | 6,083,491 | 12,565,629 | 12,565,642 | 6,083,500 |
| hpm8, L1D miss (ev 2) | 17,561,666 | 513,510,924 | 513,504,436 | 17,534,116 |

**Against the pre-registration.**
- Answers: all four commands rc 0 with `suns.out`'s sha256.
- Repeat rule: in-context 0.05 % (u) and 0.01 % (p), so the in-context cost is read. Kernel + other repeats at 1.53 % and
  2.23 %, above the 2 % rule, so its cost (+45.1 %) is not read; on P1 it was not read either.
- Counters fire: hpm3 + hpm4 > 0 in both protected commands. ev31 − ev35 − ev37 is 6,236,946 / 12,935,252 / 12,935,253 /
  6,236,946, all ≥ 0. The event readings stand.
- **In-context cost: +260.21 %** (+12.90 G per protected command), against P1's +208.08 % (+10.22 G). That is in the
  **≥ +170 % band: the record cache does not remove Perl's cost.** The two costs come from different boards (P1's from
  board 1, this one from board 2) and different bitstreams.
- **The prediction missed on both counts.** The cost, +260.2 %, is above the predicted +120 % to +230 %. The read hit
  rate, hpm3 / (hpm3 + hpm4), is 14.53 % in both protected commands, just below the predicted 15 % to 50 %.
- Stock arm: 4.957 G against P1's 4.914 G, +0.88 %, inside ±1 %. Reported only.
- Duration: the unprotected commands took 3.6 min and the protected ones 12.4 min, just over the 3-12 min stated.

**Note, 2026-11-18.** The cost that this boot set against the record cache is the patch's arena search ("Withdrawn
2026-11-18" under "Results: Perl on P1"): about 100 M loads, each through a different arena. So "the record cache does
not remove the cost" holds for that access pattern, and says nothing about a lifetime per SV head.

**What the counters show, not attributed.**
- B5a's protected command runs 2.72 G cycles (+18.0 %) longer than P1's, from the same images, workload and command line.
- B5a's protected command makes 101.1 M record-cache read misses and 216.6 M fill reads, 2.14 fill reads per miss.
- Protected, B5a's ev31 reads 330.6 M where P1's read 131.2 M, and its L1D misses read 513.5 M where P1's read 274.8 M.
  Stock, the same events read 10.5 M against 15.5 M and 17.5 M against 14.5 M.
- P1's record-TLB miss and walk-cycle events are not in B5a's set, so the walk itself is not compared.
- So this boot does not say which of the fill reads, the extra L1D misses or anything else makes up the +2.72 G. Whether
  ev31 counts the same accesses on both bitstreams is not checked here.

### Results: speedtest1, PostgreSQL and CREVOKE chains on P1 (board 2, 2026-11-17 17:24-19:22)

**Configuration.** As pre-registered (0a57804ea778; SPEC-lane Addendum 160, spec-silicon f7e660db383b). Board 2 was
flashed with P1 at 17:24 from the seal (sha256 checked) and read back resident by name. One chain, `nested/p1/p1n-b2.*`,
nat-run3 pid 537579. Before it: the rule-8 recheck of every stage (`nested/p1/recheck-p1.sh`: real bake, BOARD_STUB=2
on the stage's real F transcript and on a cut that does not return, real firmware under the 26 MB cap), and an
`NR_DRY=1` pass of the chain. Transcripts `spec/board/out-nat-p1-nested-b2/<stage>/board.txt`: st1 `adefc453…`
(firmware `ab7626e5…`), pg `dacb8df4…` (`6a341fd…`), crvt `c905ef59…` (`7d9789f…`). Read by the F parsers and
`nested/p1/abba.py` (the U P P U table and the repeat rule; it reproduces the F section's figures from F's
transcripts) and `nested/p1/crvt-read.py`.

**Controls.** In every boot `b0-stats` and `b0-hello` returned 0 and the smoke passed all 9 steps. The kit's classifier
printed NO RESULT or MALFORMED for these stages' format, as on F; every command printed `@@SPEC-END`. Not VOID.

**speedtest1 `--size 1`.** 32/32 oracle phases equal to the reference in every command.

| | u1 | p1 | p2 | u2 | repeat spread, u / p | protected vs not (ABBA means) |
|---|---|---|---|---|---|---|
| `run_cycles` (in-context) | 660,481,623 | 674,241,669 | 674,930,135 | 659,578,365 | 0.14 % / 0.10 % | **+2.21 %** (+14.6 M) |
| `between_cycles` | 5,051,005,067 | 5,127,715,362 | 5,133,707,324 | 5,051,341,672 | 0.01 % / 0.12 % | **+1.57 %** |
| uptime span, s | 229.32 | 232.94 | 233.20 | 229.28 | 0.02 % / 0.11 % | **+1.64 %** |
| hpm3 record-TLB misses | 15,383 | 17,842 | 17,843 | 15,383 | | +2,460 |
| hpm4 walk cycles | 530,390 | 431,503 | 617,533 | 508,026 | | +5 k (noisy) |
| hpm5 paged record accesses | 75,580 | 571,807 | 571,810 | 75,579 | | +496 k |
| hpm7 event 23 | 69,323 | 115,473 | 117,689 | 69,283 | | +47.3 k |
| hpm8 L1D misses | 3,011,267 | 3,192,803 | 3,217,104 | 2,980,705 | | +6.97 % |

The unprotected arm's record-TLB misses equal its context entries (15,383), one per entry.

**PostgreSQL.** Every command gives the native answers (4 CHECK lines, 2,000 rows).

| | u1 | p1 | p2 | u2 | repeat spread, u / p | protected vs not (ABBA means) |
|---|---|---|---|---|---|---|
| `run_cycles` (in-context) | 11,734,796,918 | 12,652,577,957 | 12,653,535,263 | 11,733,066,120 | 0.01 % / 0.01 % | **+7.83 %** (+919 M) |
| `between_cycles` | 8,720,320,127 | 8,795,808,032 | 8,800,476,915 | 8,737,671,836 | 0.20 % / 0.05 % | **+0.79 %** |
| uptime span, s | 821.21 | 861.29 | 861.51 | 821.82 | 0.07 % / 0.03 % | **+4.86 %** |
| hpm3 record-TLB misses | 201,628 | 1,278,719 | 1,289,702 | 201,627 | | +1.08 M |
| hpm4 walk cycles | 7,392,286 | 33,711,607 | 33,401,230 | 7,336,900 | | +26.2 M |
| hpm5 paged record accesses | 1,849,553 | 30,203,048 | 30,195,876 | 1,849,558 | | +28.3 M |
| hpm7 event 23 | 1,411,111 | 3,470,832 | 3,461,592 | 1,410,790 | | +2.06 M |
| hpm8 L1D misses | 66,619,089 | 75,020,134 | 75,164,650 | 66,513,338 | | +12.8 % |

**CREVOKE chains.** hpmcounter3/4 read from the domain (`p-hpm` rc 0), so the misses below are read around the one
CREVOKE. `mcycle` faults (rc 139), as on F.

| chain | N | CREVOKE cycles | per node | record-TLB misses | walk cycles |
|---|---|---|---|---|---|
| contiguous | 1,000 | 16,926 | 16.9 | 0 | 0 |
| contiguous | 10,000 | 318,824 | 31.9 | 40 | 1,063 |
| contiguous | 100,000 | 3,217,879 | 32.2 | 391 | 10,038 |
| scattered (64 pages) | 1,000 | 38,868 | 38.9 | 1,001 | 6,650 |
| scattered (64 pages) | 10,000 | 382,475 | 38.3 | 10,001 | 61,863 |
| scattered (400 pages) | 100,000 | 3,853,815 | 38.5 | 100,001 | 611,824 |

The 10^6 probe reported 65,537, 131,073 and 196,609 nodes and died (rc 139) before 262,145, with 39 grow events
(hpm6) over the command.

**Against the pre-registration.**
- Answers, repeats, controls: **met** for both applications. Both pairs pass the repeat rule on both counters.
- In-context cost: speedtest1 **+2.21 %**, band +0.5 % to +5 %, **met**; PostgreSQL **+7.83 %**, band +3 % to +25 %,
  **met**.
- Record-TLB misses, more in the protected arm: **met** for both (+16 %, and 6.4 times).
- CREVOKE contiguous, 15 to 60 per node at 10,000 and 100,000: **met** (31.9, 32.2).
- Scattered more than contiguous: **met** (+6.4 cycles per node at 100,000). The scatter took: the scattered CREVOKE
  misses once per node, 256 times the contiguous count (the check needed 4). Scattered 40 to 400 per node: **missed,
  low** (38.3 and 38.5).
- 10^6: the count reached is 196,609; the process died before 262,145. A 10^6-node subtree cannot be built here.

**Reading.**
- PostgreSQL's in-context cost is read on silicon for the first time: +7.83 %, with repeats 0.01 %. On F the same images
  missed the repeat rule (ABBA mean +6.67 %, not read). The walker's own cycles (+26.2 M) are 2.8 % of the +919 M, so
  the record TLB is not where most of PostgreSQL's cost goes on P1; event 23 (+2.06 M replays) and the L1D misses
  (+8.5 M) are larger candidates, and this boot does not attribute the rest.
- speedtest1 costs +2.21 % in-context on P1 against +1.67 % on F.
- A record-TLB miss costs about 6 walk cycles (611,824 over 100,001 misses), so the three dependent reads of a walk hit
  the L1D here. A CREVOKE whose subtree's records sit on a different page at every node costs 20 % more per node than a
  contiguous one. The worst single CREVOKE measured, 100,000 scattered nodes, holds the core for 3.85 M cycles, about
  154 ms at 25 MHz, uninterruptible.

### Results: CPython on P1, first session (board 2, 2026-11-17 19:22-21:15)

**Configuration.** As pre-registered (0a57804ea778), the same chain as the section above (`nested/p1/p1n-b2.*`, nat-run3
pid 537579), after the PostgreSQL stage. Four one-arm boots, U P P U: `cpy-u1` 19:22, `cpy-p1` 19:49, `cpy-p2` 20:18,
`cpy-u2` 20:47. F's images unchanged: unprotected `b0b29e89…`, protected `eabba900…`. Transcripts
`spec/board/out-nat-p1-nested-b2/cpy-*/board.txt` (sha256 `b8f310ce…`, `ef2f8b45…`, `170d24ea…`, `6406ce24…`), read by
`cpy/parse-cpy.py` and `nested/p1/cpy-abba.py`. The analyzer was F's, which read event 23 from hpm8; on P1 hpm8 counts
L1D misses and event 23 is hpm7, and it was corrected before this reading.

**Controls.** In every boot `b0-stats` and `b0-hello` returned 0 and the smoke passed all 9 steps under the kit's
waiver (the two accounting checks); all seven commands printed `@@SPEC-END`. Not VOID.

**Answers.** 28 of 28 commands exit 0 with CHECK equal to native.

| benchmark | in-context, not protected (mean) | `run_cycles` repeats u / p | `between_cycles` repeats u / p | **in-context cost** | kernel cost | hpm3 record-TLB misses, u / p | hpm7 event 23, p − u |
|---|---|---|---|---|---|---|---|
| json_dumps | 1,435.1 M | 0.05 % / 0.02 % | 0.24 % / 0.10 % | **+7.40 %** | +0.58 % | 18,910 / 145,744 | +723,746 |
| json_loads | 1,303.0 M | 0.06 % / 0.03 % | 0.21 % / 0.01 % | **+6.46 %** | +0.40 % | 19,472 / 183,556 | +744,244 |
| pickle | 1,217.1 M | 0.01 % / 0.03 % | 0.27 % / 0.09 % | **+5.01 %** | +0.67 % | 21,908 / 178,226 | +799,672 |
| richards | 7,349.7 M | 0.02 % / 0.04 % | 0.10 % / 0.13 % | **+2.70 %** | +0.68 % | 28,172 / 676,478 | +1,741,838 |
| deltablue | 1,319.9 M | 0.02 % / 0.08 % | 0.06 % / 0.23 % | **+4.54 %** | +0.46 % | 21,188 / 170,412 | +778,012 |
| regex_v8 | 3,325.6 M | 0.15 % / 0.11 % | 0.06 % / 0.36 % | **+5.32 %** | +0.76 % | 42,536 / 275,124 | +930,196 |
| nbody | 7,118.3 M | 0.02 % / 0.15 % | 0.20 % / 0.00 % | **+3.90 %** | +0.76 % | 23,960 / 167,274 | +1,426,855 |

All seven pass the repeat rule, so all seven are read (on F richards was not). Summed over the seven:

| | not protected | protected | protected vs not |
|---|---|---|---|
| in-context (`run_cycles`) | 23,068.8 M | 24,033.3 M | **+4.18 %** (+964.5 M) |
| kernel and other (`between_cycles`) | 2,862.4 M | 2,880.3 M | +0.63 % |
| uptime span | 1,051 s | 1,091 s | +3.78 % |
| hpm3 record-TLB misses | 176,148 | 1,796,816 | x10.2 |
| hpm4 walk cycles | 6.40 M | 40.83 M | +34.4 M |
| hpm5 paged record accesses | 2.60 M | 54.27 M | +51.7 M |
| hpm6 table grows (cause 30) | 1 | 19 | +18 |
| hpm7 event 23 | 823,383 | 7,967,946 | **+7.14 M** |
| hpm8 L1D misses | 58.08 M | 68.57 M | +18.1 % |

Over F's six (without richards), for comparison with F: +4.87 % in-context (+766.0 M), against F's +5.17 % (+877.6 M).

**Against the pre-registration.**
- Answers: **met**, 28/28. Controls: **met**.
- Repeats: **met** for all 14 pairs on both counters (largest 0.15 % `run_cycles`, 0.36 % `between_cycles`).
- In-context cost, the benchmarks read, summed: **+4.18 %**, band +3 % to +25 %, **met**.
- Record-TLB misses, more in the protected arm: **met** for every benchmark (6.5 to 24 times).

**Reading.**
- CPython's in-context cost on P1 is +4.18 % over all seven, each read, +2.7 % (richards) to +7.4 % (json_dumps).
- The paged table's walker is a small part of it: +34.4 M walk cycles are 3.6 % of the +964.5 M. Event 23 at 17-20
  cycles per replay explains 12.6-14.8 %, about the share it had on F (10.5-12.3 % over six). L1D misses rise 18 %. As
  on F and for PostgreSQL here, most of the cost is not attributed by these counters.
- The unprotected arm runs 7.4 % fewer in-context cycles on P1 than on F over the same six (15,719 M against
  16,983 M), so P1's cost sits on a smaller base. The two bitstreams differ in more than the table, and this run does
  not separate which difference moved the base.

### Results: CPython on P1, second session, and the two sessions together (board 3, 2026-11-17 20:28-22:20)

**Configuration.** As in the session-2 addendum above (858ddee10a4f; SPEC-lane Addendum 161, spec-silicon
35a7194b03da). P1 was already resident on board 3 and read back by name, so there was no flash. One chain,
`nested/p1s2/p1s2-b3.*`, nat-run3 pid 860657, after the same rule-8 recheck. Four one-arm boots, U P P U: `cpy2-u1`
20:28, `cpy2-p1` 20:56, `cpy2-p2` 21:24, `cpy2-u2` 21:53. The stages are session 1's, the same images. Transcripts
`spec/board/out-nat-p1s2cpy-nested-b3/cpy2-*/board.txt` (sha256 `b240cdf7…`, `21ec0605…`, `ae544fa9…`, `3f193174…`),
read by `nested/p1/cpy-abba.py` (with the `cpy2` prefix).

**Controls.** In every boot `b0-stats` and `b0-hello` returned 0, the smoke passed all 9 steps under the kit's waiver
(the two accounting checks), and all seven commands printed `@@SPEC-END`. Not VOID. **Answers:** 28 of 28 commands exit
0 with CHECK equal to native.

| benchmark | in-context, not protected (mean) | `run_cycles` repeats u / p | `between_cycles` repeats u / p | session 2 cost | session 1 cost | **two-session mean** | hpm7 event 23, p − u |
|---|---|---|---|---|---|---|---|
| json_dumps | 1,434.8 M | 0.03 % / 0.02 % | 0.09 % / 0.42 % | +7.43 % | +7.40 % | **+7.41 %** | +722,624 |
| json_loads | 1,302.8 M | 0.03 % / 0.01 % | 0.68 % / 0.38 % | +6.49 % | +6.46 % | **+6.48 %** | +744,004 |
| pickle | 1,216.6 M | 0.10 % / 0.05 % | 0.11 % / 0.36 % | +5.03 % | +5.01 % | **+5.02 %** | +799,524 |
| richards | 7,349.8 M | 0.03 % / 0.00 % | 0.34 % / 0.12 % | +2.70 % | +2.70 % | **+2.70 %** | +1,742,515 |
| deltablue | 1,320.2 M | 0.03 % / 0.06 % | 0.12 % / 0.24 % | +4.52 % | +4.54 % | **+4.53 %** | +778,063 |
| regex_v8 | 3,327.1 M | 0.05 % / 0.08 % | 0.08 % / 0.17 % | +5.24 % | +5.32 % | **+5.28 %** | +930,124 |
| nbody | 7,118.9 M | 0.03 % / 0.04 % | 0.18 % / 0.25 % | +3.94 % | +3.90 % | **+3.92 %** | +1,426,605 |

Summed over the seven, session 2: in-context **+4.18 %** (+965.1 M), kernel +0.66 %, wall-clock +3.79 %; record-TLB
misses 176,114 against 1,797,613; walk cycles +34.4 M; event 23 +7.14 M; L1D misses +18.0 %.

**Against the pre-registration.**
- Answers, controls: **met**. Repeats: **met** for all 14 pairs on both counters (largest 0.10 % and 0.68 %), so every
  benchmark is read in both sessions.
- In-context cost summed, band +3 % to +25 %: **met**, +4.18 %.
- Two-session rule: both sessions pass for every benchmark, so the reading is the mean of the two ABBA means. Summed
  over the seven: **+4.18 %** (session 1 +4.181 %, session 2 +4.183 %).

**Reading.**
- On two boards, the two sessions agree to 0.08 points or better per benchmark, and to 0.002 points summed. The
  unprotected arms' in-context cycles agree to 0.05 % per benchmark across boards. The P1 CPython figure, **+4.18 %**,
  is reproducible at the boot-to-boot level these counters resolve.
- The attribution is session 1's: the walker 3.6 %, event 23 12.6-14.8 %, the rest not attributed.


### Results: FFmpeg on P1 (board 1, 2026-11-17 23:52 - 2026-11-18 00:13)

**Configuration.** As in the session-2 addendum (858ddee10a4f; SPEC-lane Addendum 161). Board 1 was flashed with P1 at
23:52 from the seal (sha256 checked) and read back by name; chain `nested/p1s2/p1s2-b1.*`, nat-run3 pid 886911. Stage
`nested/p1s2/ff`, F's `f2-stage` train unchanged (image set w4). Transcript
`spec/board/out-nat-p1s2-nested-b1/ff/board.txt` (sha256 `40ec2aa5…`, firmware `46facaf2…`), read by `ffmpeg/parse-ff.py`
and `nested/p1/abba.py`.

**Controls.** `b0-stats` and `b0-hello` returned 0 and the smoke passed all 9 steps under the kit's waiver; all five
commands printed `@@SPEC-END` with rc 5, as on F: the decode driver's success status, `FFAPP_M5_ALL` "every frame decoded, decoder drained" (`ports/ffmpeg/app/src/shared/ffapp_decode.h:20`, set at `ffapp_decode.c:173-174` after the drain), and every command's output ends `STAGE M5 frames=300 packets=300`. The stage's `train/` equals F's `f2-stage/train` file for file. Not VOID.

| | u1 | p1 | p2 | u2 | repeat spread, u / p | protected vs not (ABBA means) |
|---|---|---|---|---|---|---|
| frames equal to stock (of 300) | 300 | 300 | 300 | 300 | | |
| `run_cycles` (in-context) | 1,663,440,646 | 1,661,793,248 | 1,663,049,013 | 1,664,390,287 | 0.06 % / 0.08 % | **-0.09 %** (-1.5 M) |
| `between_cycles` | 79,112,595 | 79,348,369 | 78,672,314 | 78,738,866 | 0.47 % / 0.86 % | +0.11 % |
| uptime span, s | 70.57 | 70.51 | 70.53 | 70.60 | 0.04 % / 0.03 % | -0.09 % |
| hpm3 record-TLB misses | 1,130 | 1,128 | 1,128 | 1,130 | | -2 |
| hpm5 paged record accesses | 118,526 | 167,252 | 167,248 | 118,527 | | +48.7 k |
| hpm7 event 23 | 34,008 | 35,354 | 35,392 | 34,019 | | +1,360 |
| hpm8 L1D misses | 6,729,147 | 6,729,147 | 6,728,823 | 6,727,175 | | +0.01 % |

The flipped-input control fp decodes 300 frames with 3 hashes different: FIRES.

**Against the pre-registration.**
- Answers: **met**, every command matches stock's framemd5 and fp fires.
- Repeats: **met** on both counters (largest 0.08 % `run_cycles`, 0.86 % `between_cycles`), so the difference is read.
- In-context cost: **-0.09 %**, band -1 % to +1 %, **met**.
- Event 23: **met**, +1,360 per command (band under +10,000).

**Reading.** FFmpeg's protection is read on silicon for the first time (on F the unprotected repeats missed the rule):
it costs nothing measurable, -0.09 % with repeats of 0.06-0.08 %. Its 3,300 lifetimes per decode add 48.7 k paged-record
accesses and no record-TLB misses.

### Results: PostgreSQL on P1, second session, and the two sessions together (board 1, 2026-11-18 00:13-01:35)

**Configuration.** As in the session-2 addendum (858ddee10a4f; SPEC-lane Addendum 161), the same chain and flash as
FFmpeg above, the boot after it. Stage `nested/p1s2/pg2`, the same stage and images as session 1's `pg` (`ccdbfbdd…`).
Transcript `spec/board/out-nat-p1s2-nested-b1/pg2/board.txt` (sha256 `0d192aaf…`, firmware `5e70c31c…`), read by
`pg/parse-pg.py` (`PG_REF=pg/work/nref4-main/ref.txt`) and `nested/p1/abba.py`.

**Controls.** `b0-stats` and `b0-hello` returned 0, the smoke passed all 9 steps under the kit's waiver, and all four
commands printed `@@SPEC-END` with rc 0. Not VOID. **Answers:** every command gives the native answers (4 CHECK lines,
2,000 rows).

| | u1 | p1 | p2 | u2 | repeat spread, u / p | protected vs not (ABBA means) |
|---|---|---|---|---|---|---|
| `run_cycles` (in-context) | 11,731,540,161 | 12,647,014,256 | 12,651,830,481 | 11,731,043,711 | 0.00 % / 0.04 % | **+7.83 %** (+918 M) |
| `between_cycles` | 8,718,865,729 | 8,770,925,841 | 8,838,944,285 | 8,731,608,422 | 0.15 % / 0.78 % | **+0.91 %** |
| uptime span, s | 821.01 | 860.06 | 862.97 | 821.49 | 0.06 % / 0.34 % | +4.90 % |
| hpm3 record-TLB misses | 201,631 | 1,184,331 | 1,313,986 | 201,631 | | +1.05 M |
| hpm4 walk cycles | 7,408,181 | 31,364,659 | 33,898,942 | 7,383,398 | | +25.2 M |
| hpm5 paged record accesses | 1,849,511 | 30,180,731 | 30,197,405 | 1,849,556 | | +28.3 M |
| hpm6 grow | 0 | 25 | 0 | 0 | | |
| hpm7 event 23 | 1,410,942 | 3,447,424 | 3,460,509 | 1,411,749 | | +2.04 M |
| hpm8 L1D misses | 66,598,412 | 74,977,730 | 75,172,420 | 66,679,856 | | +12.7 % |

**Against the pre-registration.**
- Answers, controls: **met**. Repeats: **met** on both counters (0.04 % `run_cycles`, 0.78 % `between_cycles`), so the
  session is read.
- In-context cost, band +3 % to +25 %: **met**, +7.83 %. Record-TLB misses more in the protected arm: **met** (5.9 times).
- Two-session rule: both sessions pass, so the reading is the mean of the two ABBA means: in-context **+7.83 %**
  (session 1 +7.833 %, session 2 +7.826 %); kernel and other **+0.85 %** (+0.79 %, +0.91 %).

**Reading.**
- On two boards, the two sessions agree to 0.007 points in-context, and the unprotected arms' in-context cycles agree
  to 0.03 %. The P1 PostgreSQL figure, **+7.83 %**, is reproducible at the boot-to-boot level these counters resolve.
- The first protected command grew the record table 25 times and the second none, as in session 1 (25, 0). The two protected commands' record-TLB misses differ by 11 % and their walk cycles by 8 %, while
  their in-context cycles agree to 0.04 %: the walker's +25 M is 2.7 % of the +918 M, as in session 1.

### Results: the protected tshark with a heartbeat on P1 (board 1, 2026-11-18 01:35-02:23): A, it returns with stock's output

**Configuration.** As pre-registered (2475d3fb7944). The chain `nested/p1ts/p1ts-b1.*`, nat-run3 pid 1526065, waited for
the pg2 chain. P1 was resident on board 1 and was read back by name, so there was no flash. Transcript
`spec/board/out-nat-p1ts-nested-b1/tsph/board.txt` (sha256 `d4e0052e…`, firmware `955dadb6…`), read by
`tshark/hb-frames.py` and `tshark/parse-ts.py` against `wl/ref1000.out` (`056c1f8c…`).

**Controls.** `b0-stats` and `b0-hello` returned 0, the smoke passed all 9 steps under the kit's waiver, and the command
printed `@@SPEC-END` with rc 0. Not VOID.

**Outcome: A.** The beats continued and the output grew through the run. The command returned rc 0 after 1,669.4 s of
uptime (`@@SPEC-BEGIN` at 267.4 s), and its output equals stock: sha256 `056c1f8c…`, 1,140,005 lines, 9,000 frames.

| beat | uptime, s | since the previous beat, s | `st.out`, MB | frame | `run_cycles` per frame in the beat | `between_cycles` per frame in the beat |
|---|---|---|---|---|---|---|
| 3 | 307.5 | | 1.46 | 276 | | |
| 4 | 377.1 | 69.6 | 4.58 | 863 | 2.68 M | 0.95 M |
| 6 | 565.0 | 102.6 | 11.96 | 2,249 | 2.70 M | 1.47 M |
| 8 | 830.8 | 143.7 | 21.17 | 3,981 | 2.70 M | 1.89 M |
| 10 | 1,194.0 | 195.1 | 32.69 | 6,146 | 2.71 M | 2.22 M |
| 12 | 1,679.0 | 259.5 | 47.06 | 8,847 | 2.72 M | 2.50 M |

Over the whole command: `run_cycles` 24,407 M, `between_cycles` 17,261 M, hpm3 record-TLB misses 7.16 M, hpm4 walk
cycles 90.5 M, hpm5 paged record accesses 98.9 M, hpm6 grow 8, hpm7 event 23 11.27 M, hpm8 L1D misses 129.6 M.

**Where the time went: the overrun of the heartbeat's own beats.**
- The beats are 60 s apart by `sleep`. During the command each beat ran past its period by more, from 9.6 s at beat 4 to
  199.5 s at beat 12. Before the command the overrun was 1.6 s.
- The overrun grows with the size of `st.out`: 4.47 s per MB, minus 10.8 s, a linear fit over beats 4-12. The only
  step of a beat whose work depends on that size is `wc -c < st.out`. In the shared buildroot target `wc` is busybox
  1.36.1, whose `wc` reads every byte with `getc` (`coreutils/wc.c:166`; its header notes that the old `-c` stat
  shortcut was removed); the kit's own rootfs was not checked.
- Over the nine intervals between beats, the two counters fit to within 0.13 %:
  - in-context: 2.683 M cycles per frame, constant through the run;
  - kernel and other: 0.741 M cycles per frame, plus 12.5 M cycles per second of overrun.
- So the in-context cost per frame does not grow as the run proceeds. The 12.5 M cycles per second of overrun over
  4.47 s per MB come to about 56 cycles per byte of `st.out`, a plausible cost for a `getc` loop. That fits the beat
  reading the whole output each time, charged to `between_cycles`. The beat's `$(up)` is taken before its `wc` and
  `$(cnt)` after it, so the counters at each beat are read when its `wc` ends.
- 12.5 M cycles per second is half the core's 25 MHz clock (PostgreSQL's U arms count 24.9 M cycles per second of
  uptime). That is what a single core shared evenly between the beat's `wc` and tshark would show, with tshark
  still running during the overrun. Over the whole command the two counters add to 41.67 G cycles, 1,667 s at 25 MHz,
  so the counters cover the span. Taking out 12.5 M cycles for each second of overrun (about 833 s over beats 3-12,
  about 10.4 G) leaves about 31.3 G, which is about 1,250 s. On the twin a 1 s beat stretched the span from 8.32 s to 11.46 s, which is consistent with this.

**Against the pre-registration.**
- A with a return inside 7,200 s: **met**, and the output equals stock's.
- The pre-registered slowdown, the uptime span against F's unprotected mean of 1,118 s, reads **+49 %**. But this boot
  shows that most of it is the beat's own overrun, so it is not read as the protection's cost.
- In-context `run_cycles`, 24,407 M against F's unprotected mean of 24,036 M: +1.5 %. That compares across two bitstreams
  in one boot, so it is not the cost either.

**Reading.**
- On P1, the protected tshark completes, with stock's output. On F, the same image gave nothing for at least 3,327 s in
  two boots. ~~So M-16 is narrowed to F's flat lifetime table.~~ **Withdrawn 2026-11-18:** without the heartbeat the
  protected arm did not return on P1 either ("Results: tshark on P1, U P P U without a heartbeat"). This boot
  shows only that the protected arm can complete on P1 under a 60 s heartbeat. That is not a root cause, and nothing
  is fixed on F.
- The output's size grows with the frames processed, so this boot cannot by itself exclude a kernel cost per frame
  that grows with the run; the fit and the `getc` cost favour the beat's own reads. The U P P U comparison without
  the heartbeat, which follows as pre-registered, tells them apart and gives the protection's cost on P1. (It
  gave no cost: one of its two protected boots did not return. See "Results: tshark on P1, U P P U without a
  heartbeat".)

### Results: mruby on P1 (board 2, 2026-11-18 01:57-02:37)

**Configuration.** As pre-registered (f7b222e43189). The chain `nested/p1a/b2-mrb.*` (nat-run3 pid 1671139) waited for the
SPEC lane's chain, then flashed P1 at 01:57 from the seal (sha256 checked) and read it back by name. Stage `nested/p1a/mrb`,
commands u1 p1 p2 u2 in one boot. Transcript `spec/board/out-nat-p1a-mrb-b2/mrb/board.txt` (sha256 `40f83990…`, firmware
`68c9437f…`), read by `nested/appd/parse-abba.py` and `nested/p1/abba.py`.

**Controls.** `b0-stats` and `b0-hello` returned 0, the smoke passed all 9 steps under the kit's waiver, and all four
commands printed `@@SPEC-END` with rc 0. Not VOID. **Answers:** every command's PPM equals the native reference
(`cfd73bb1…`).

| | u1 | p1 | p2 | u2 | repeat spread, u / p | protected vs not (ABBA means) |
|---|---|---|---|---|---|---|
| `run_cycles` (in-context) | 10,355,627,967 | 10,785,150,505 | 10,785,737,662 | 10,355,031,358 | 0.01 % / 0.01 % | **+4.15 %** (+430 M) |
| `between_cycles` | 143,994,494 | 145,280,458 | 144,998,140 | 142,419,166 | 1.11 % / 0.19 % | +1.35 % |
| uptime span, s | 420.93 | 438.17 | 438.18 | 420.83 | 0.02 % / 0.00 % | +4.11 % |
| hpm3 record-TLB misses | 415,450 | 2,279,942 | 2,279,563 | 416,178 | | +1.86 M |
| hpm4 walk cycles | 8,768,363 | 28,381,542 | 28,293,493 | 8,748,182 | | +19.6 M |
| hpm5 paged record accesses | 20,467,276 | 39,137,469 | 39,138,102 | 20,467,393 | | +18.7 M |
| hpm6 grow | 1 | 2 | 0 | 0 | | |
| hpm7 event 23 | 1,720,965 | 5,745,913 | 5,746,536 | 1,721,116 | | +4.03 M |
| hpm8 L1D misses | 33,904,175 | 39,646,362 | 39,637,220 | 33,899,227 | | +16.9 % |

**Against the pre-registration.**
- Answers, controls: **met**. Repeats: **met** on both counters (0.01 % `run_cycles`, 1.11 % `between_cycles`), so the
  difference is read.
- In-context cost, band +0.5 % to +10 %: **met**, **+4.15 %**.
- Duration, 6 to 14 min per command: **met**, 7.0-7.3 min.
- Record-TLB misses more in the protected arm: **met** (5.5 times).

**Reading.**
- mruby's protection (patch 0008 with the Q-15 repair, the dev-era two-operand revision) costs **+4.15 %** in-context on
  P1, against +1.64 % on the twin: a silicon-to-twin ratio of 2.5, inside the 1.9-3.5 of the other applications.
- The walker's own cycles (+19.6 M) are 4.6 % of the +430 M. Event 23 (+4.03 M replays) and the L1D misses (+5.7 M)
  are larger candidates; this boot does not attribute the rest.
- This is not Fig. 10's revision of 0008 (that one calls the one-operand `capstone_cap_release`, which P1 does not
  implement); the appendix row says so.

### Results: Perl on P1, a 2^20-record table (board 1, 2026-11-18 02:24-03:04)

**Configuration.** As pre-registered (ff011c64d374). The chain `nested/p1a/b1-perl.*` (nat-run3 pid 1793373) ran on the
rec1m kit copy after the tshark heartbeat chain; P1 was resident and read back by name. The boot's kernel command line
carries `capstone_vm.node_flat_records=1048576`. Stage `nested/p1a/perl`, still matching `perl.SHA256SUMS`; its images
unpack to `3012f392…` (stock) and `ac2507d3…` (protected). Transcript `spec/board/out-nat-p1a-perl-b1/perl/board.txt`
(sha256 `90101a1a…`, firmware `bf734d65…`), read by `nested/appd/parse-abba.py` and `nested/p1/abba.py`.

**Controls.** `b0-stats` and `b0-hello` returned 0, the smoke passed all 9 steps under the kit's waiver, and all four
commands printed `@@SPEC-END` with rc 0. Not VOID. **Answers:** every command's stdout equals SPEC's `suns.out`
(`01688ea0…`).

| | u1 | p1 | p2 | u2 | repeat spread, u / p | protected vs not (ABBA means) |
|---|---|---|---|---|---|---|
| `run_cycles` (in-context) | 4,915,780,479 | 15,139,569,760 | 15,137,911,482 | 4,912,103,328 | 0.07 % / 0.01 % | **+208.1 %** (+10.22 G) |
| `between_cycles` | 409,343,475 | 579,997,839 | 565,192,418 | 401,287,422 | **2.01 %** / **2.62 %** | not read (+41.3 %) |
| uptime span, s | 217.30 | 634.11 | 633.45 | 216.84 | 0.21 % / 0.10 % | +192.0 % |
| context entries | 4,600 | 7,262 | 7,194 | 4,550 | | +58.0 % |
| hpm3 record-TLB misses | 691,108 | 49,445,643 | 49,445,271 | 690,378 | | +48.75 M (71 times) |
| hpm4 walk cycles | 18,497,874 | 1,114,841,431 | 1,114,716,428 | 18,545,254 | | +1.10 G |
| hpm5 paged record accesses | 15,471,124 | 131,204,043 | 131,202,594 | 15,468,357 | | +115.7 M |
| hpm6 grow | 41 | 52 | 0 | 0 | | |
| hpm7 event 23 | 1,360,181 | 101,963,779 | 101,962,474 | 1,357,428 | | +100.6 M |
| hpm8 L1D misses | 14,518,929 | 274,831,168 | 274,799,352 | 14,505,393 | | +260.3 M (19 times) |

**Against the pre-registration.**
- Answers, controls: **met**. Repeats: **met** for `run_cycles` (0.07 % / 0.01 %), so the in-context cost is read;
  **missed** for `between_cycles` (2.01 % and 2.62 %, band 2 %), so the kernel difference is not read.
- In-context cost, band +20 % to +150 %: **missed, high: +208.1 %**.
- Duration, 3 to 8 min per command: met for the stock arm (3.6 min), **missed** for the protected arm (10.6 min).
- Record-table growth more in the protected arm: hpm6 reads 41 / 52 / 0 / 0. The table grows during the boot's first two commands
  and then stays grown, so this counter does not compare the arms; not read.

**Reading.**
- On P1, Perl's protection (`perl/sublet` 0001, dev's two-operand revision, a lifetime per SV head) costs **+208 %**
  in-context: the protected arm's in-context cycles are 3.08 times the stock arm's, 15.14 G against 4.91 G. On the twin it costs +40.3 %.
  The stock arm's silicon-to-twin ratio is 2.84, as for the other applications; the protected arm's is 6.2.
- The counters put the difference in the record table:
  - the protected arm misses the record TLB 49.4 M times per command, against 0.69 M (71 times);
  - it makes 131 M paged record accesses against 15.5 M;
  - it takes 260 M more L1D misses, 19 times the stock arm's 14.5 M;
  - and it adds 100.6 M event-23 replays.
- Perl's peak of 327,680-393,216 live records fills 1,280-1,536 pages of 256 records, against a 16-entry record TLB.
- The walker's own cycles, +1.10 G at 22.5 cycles per miss, are 10.7 % of the +10.22 G. The rest is not attributed by
  this boot. If the extra L1D misses carried it all, each would cost 39 cycles; how much comes from the table's
  pages evicting the program's data is not shown here.
- The protected live set is 80-96 times the record TLB's reach. The counters point at the paged table rather
  than at the instructions the patch adds, which the twin prices at +40.3 %.

**Withdrawn 2026-11-18: the reading above, not the counters.** "The counters point at the paged table rather than at
the instructions the patch adds" is withdrawn, and so is the live-record count as the cause.
- Patch 0001 finds a head's arena by walking the list of all arenas (`S_sv_arena_of`). In the protected image the walk
  is a 6-instruction loop at `Perl_sv_arena_head` 0xde994-0xde9a6 whose first instruction, `ldc a1, 0(a1)`, loads
  through the next arena. Every arena is a heap object with a lifetime of its own, so every step checks another
  record. Stock Perl has no such lookup.
- A native count by another session (x86-64, suns.pl, SPEC's output; not re-run by this lane) reads 97.4 M search
  steps over 2,534 arenas, 71.9 M of them from 80,205 `SvIS_FREED` tests.
- This lane's counters agree with that count: +100.6 M event-23 replays per protected command here, +99.9 M
  record-cache read misses on B5a, and +697 M instructions on the twin, 7.2 per step.
- So +10.22 G is about 105 cycles for each step of a search that the patch adds. It is the cost of this revision of
  the patch, not of a lifetime per SV head, and it is not shown to be the cost of the table's reach.
- **The +208 % row is withheld** from the appendix until the boot is repeated with the lookup repaired
  (`nested/perl/lookup-fix/0002-sv-arena-lookup-by-index.patch`: an index of arenas sorted by address, searched
  without reading any arena). The B5a result above measured the same search.
- The record table is configured for 2^20 records for this program; the default 262,144 is exhausted by its peak of
  327,680 to 393,216 live records.

### Results: memcached on P1, the in-process driver (board 2, 2026-11-18 02:37-03:20)

**Configuration.** As pre-registered (c3d5b844728e; Addendum 170, `5eeb893bb63e`). The chain `nested/p1a/b2-mc.*` (nat-run3
pid 1771874) waited for this lane's mruby chain; P1 was resident and read back by name. Stages `nested/p1a/mc128` and
`mc32`, commands u1 p1 p2 u2 in each boot. Transcripts `spec/board/out-nat-p1a-mc-b2/mc128/board.txt` (sha256
`b030d2e2…`, firmware `7e6335c9…`) and `…/mc32/board.txt` (sha256 `b21d60c1…`, firmware `c0271018…`), read by
`nested/appd/parse-abba.py` and `nested/p1/abba.py`; the CHECK lines are read from each command's `st.out` in the
transcript.

**Controls.** In both boots `b0-stats` and `b0-hello` returned 0, the smoke passed all 9 steps under the kit's waiver,
and all four commands printed `@@SPEC-END` with rc 0. Not VOID. **Answers:** every command's stream hash equals the
native reference (`f7733aa9…` at `-m 128`, `b897c51d…` at `-m 32`), with `set_fail=0`, `bad=0` and `MCDRV-END rc=0`.

`mc128`, the reading:

| | u1 | p1 | p2 | u2 | repeat spread, u / p | protected vs not (ABBA means) |
|---|---|---|---|---|---|---|
| `run_cycles` (in-context) | 5,005,998,913 | 5,102,897,092 | 5,103,534,319 | 5,009,583,660 | 0.07 % / 0.01 % | **+1.91 %** (+95.4 M) |
| `between_cycles` | 414,090,695 | 441,882,640 | 440,203,590 | 413,417,945 | 0.16 % / 0.38 % | +6.60 % (+27.3 M) |
| uptime span, s | 223.49 | 228.98 | 228.95 | 223.62 | 0.06 % / 0.01 % | +2.42 % |
| context entries | 2,912 | 3,075 | 3,070 | 2,912 | | +5.5 % |
| hpm3 record-TLB misses | 4,904 | 322,834 | 322,807 | 4,904 | | +0.318 M (66 times) |
| hpm4 walk cycles | 176,793 | 10,430,353 | 10,402,157 | 170,994 | | +10.2 M |
| hpm5 paged record accesses | 48,620 | 1,761,957 | 1,761,891 | 48,627 | | +1.71 M |
| hpm6 grow | 0 | 7 | 0 | 0 | | |
| hpm7 event 23 | 48,219 | 382,718 | 382,655 | 48,295 | | +0.334 M |
| hpm8 L1D misses | 20,228,321 | 21,086,346 | 21,086,549 | 20,224,523 | | +4.25 % |

CHECK, all four commands: hits 88,691, misses 91,309, items 30,360, slab allocations 60,000, evicted 0, as on the twin.
The one field that differs is the LRU maintainer's `maint_moves`: 49,512 in both stock commands, 49,509 in both
protected ones.

`mc32`, reported and not read as the cost (the arms do different work):

| | u1 | p1 | p2 | u2 | repeat spread, u / p | protected vs not (ABBA means) |
|---|---|---|---|---|---|---|
| `run_cycles` (in-context) | 3,907,830,017 | 3,870,755,500 | 3,871,697,661 | 3,904,145,928 | 0.09 % / 0.02 % | -0.89 % (-34.8 M) |
| `between_cycles` | 171,274,360 | 172,040,263 | 171,502,387 | 170,956,290 | 0.19 % / 0.31 % | +0.38 % |
| uptime span, s | 166.18 | 164.75 | 164.77 | 166.01 | | -0.80 % |
| hpm3 record-TLB misses | 1,542 | 218,816 | 218,620 | 1,542 | | +0.217 M |
| hits / evictions / items | 57,586 / 27,108 / 13,739 | 53,506 / 29,807 / 12,396 | the same as p1 | the same as u1 | | |

The hit, eviction and item counts are the twin's, arm for arm.

**Against the pre-registration.**
- Answers, controls: **met** in both boots. Repeats: **met** on both counters in both boots, so `mc128`'s difference is read.
- `mc128` in-context cost, band -1 % to +3 %: **met**, **+1.91 %**.
- `mc128` hits, misses, evictions and items equal between the arms: **met**. `maint_moves` is not one of the registered
  fields and differs by 3, which the row discloses.
- Duration, 5 to 10 min per command: **missed, short**: 3.7-3.8 min at `-m 128`, 2.7-2.8 min at `-m 32`.

**Reading.**
- At `-m 128`, where both arms do the same work, memcached's protection (0006 with the Q-15 repair, the dev-era
  two-operand revision) costs **+1.91 %** in-context on P1 and +6.6 % in the kernel. On the twin it cost +0.10 % and
  +6.9 %.
- The stock arm runs 1.58 times its twin cycles, against 2.84 for Perl and 2.55 for PHP. The twin's cost (+0.10 %) is too
  small for a silicon-to-twin ratio of the costs (19) to be compared with the 1.9-3.5 of the other applications.
- The walker's own cycles (+10.2 M) are 10.7 % of the +95.4 M. The protected arm misses the record TLB 66 times as
  often, though at 0.32 M misses per command that is still few. This boot does not attribute the rest.
- At `-m 32`, Fig. 10's setting, the protected arm evicts 2,699 more items and serves 4,080 fewer hits, and it runs
  0.89 % fewer in-context cycles (the twin: 2.41 % fewer). That is the different work, not a saving from the
  protection.

### Results: PHP on P1 (board 2, 2026-11-18 03:21-04:31)

**Configuration.** As pre-registered (c3d5b844728e; Addendum 170, `5eeb893bb63e`). The chain `nested/p1a/b2-php.*`
(nat-run3 pid 1771932) waited for the memcached chain; P1 was resident and read back by name. Stage `nested/p1a/php10`,
commands u1 p1 p2 u2 in one boot. Transcript `spec/board/out-nat-p1a-php-b2/php10/board.txt` (sha256 `8ec4e742…`,
firmware `aacdf7ac…`), read by `nested/appd/parse-abba.py` and `nested/p1/abba.py`.

**Controls.** `b0-stats` and `b0-hello` returned 0, the smoke passed all 9 steps under the kit's waiver, and all four
commands printed `@@SPEC-END` with rc 0. Not VOID. **Answers:** every command's normalised output equals the native
reference (`8f692e64…`). **Modes**, from each command's `st.err`: u1 and u2 print `ZEND_CAP_MODE 4 exact=0 revoke=0`,
p1 and p2 print `ZEND_CAP_MODE 6 exact=0 revoke=1`. Every command reports 1,088,437 blocks and 3,442,469 issues; the
protected commands retire and revoke all 3,442,469 (`live=0`), the stock ones none.

| | u1 | p1 | p2 | u2 | repeat spread, u / p | protected vs not (ABBA means) |
|---|---|---|---|---|---|---|
| `run_cycles` (in-context) | 22,369,239,344 | 22,897,759,008 | 22,908,679,228 | 22,389,029,560 | 0.09 % / 0.05 % | **+2.34 %** (+524 M) |
| `between_cycles` | 426,939,061 | 431,539,605 | 429,081,292 | 424,053,296 | 0.68 % / 0.57 % | +1.13 % |
| uptime span, s | 913.35 | 934.71 | 935.04 | 914.02 | 0.07 % / 0.04 % | +2.32 % |
| context entries | 7,503 | 7,570 | 7,563 | 7,496 | | +0.9 % |
| hpm3 record-TLB misses | 402,352 | 1,012,800 | 1,012,584 | 402,363 | | +0.61 M |
| hpm4 walk cycles | 6,045,947 | 15,433,397 | 15,425,147 | 6,024,824 | | +9.4 M |
| hpm5 paged record accesses | 19,810,404 | 74,406,491 | 74,406,720 | 19,810,458 | | +54.6 M |
| hpm6 grow | 10 | 8 | 0 | 0 | | |
| hpm7 event 23 | 2,341,379 | 2,677,368 | 2,677,663 | 2,341,526 | | +0.336 M |
| hpm8 L1D misses | 98,671,519 | 100,323,517 | 100,345,205 | 98,679,914 | | +1.68 % |

**Against the pre-registration.**
- Answers, controls, modes: **met**. Repeats: **met** on both counters, so the difference is read.
- In-context cost, band -1 % to +3 %: **met**, **+2.34 %**.
- Duration, 13 to 26 min per command: **met**, 15.2-15.6 min.
- hpm6 reads 10 / 8 / 0 / 0: the table grows in the boot's first two commands, as for Perl, so it is not read.

**Reading.**
- On P1 the lifetime alone (mode 6 against mode 4, one image, the dev-era two-operand `heap_end`) costs **+2.34 %**
  in-context and +1.1 % in the kernel. That is 152 cycles per child issued and ended (3,442,469 per command). On the
  twin it cost +0.16 %, +13.7 M cycles against P1's +524 M.
- The stock arm runs 2.55 times its twin cycles (Perl: 2.84). The added cycles are 38 times the twin's, and the cost ratio, 14.6,
  lies far outside the 1.9-3.5 the band was drawn from; the band held only because the twin's cost was so small.
- The protected arm makes 54.6 M more paged record accesses, 15.9 per child, and misses the record TLB 2.5 times as
  often. The walker's own cycles (+9.4 M) are 1.8 % of the +524 M; event 23 and the L1D misses add 0.34 M and 1.67 M.
  This boot does not attribute the rest.
- This is not Fig. 10's pairing (a pristine second binary); the appendix row says so.

### Results: tshark on P1, U P P U without a heartbeat (board 1, 2026-11-18 03:04-06:07): p1 did not return, p2 did; no cost is read

**Configuration.** As pre-registered (e42113e2b85d; Addendum 173, `0fe1da3b0398`). The chain `nested/p1tu/p1tu-b1.*`
(nat-run3 pid 1980951) waited for this lane's Perl chain; P1 was resident and read back by name. One arm per boot.
- **Disclosure:** the u2 line of `p1tu-b1.list` was commented out in place from about 04:53 to 05:00, while p2 ran, and
  then restored byte-identical (`cmp` against the original). The chain had not read that line in either state, so the
  run follows the list as registered.

**tsu-u1** (transcript `…/out-nat-p1tu-nested-b1/tsu-u1/board.txt`, sha256 `ff973eb4…`, firmware `f327637d…`):
- `b0-stats` and `b0-hello` returned 0, and the smoke passed all 9 steps.
- rc 0, with stock's output: `056c1f8c…`, 1,140,005 lines, 9,000 frames (`parse-ts.py` against `wl/ref1000.out`).
- Uptime span 1,064.75 s. In-context 22,743,759,258 cycles, `between_cycles` 3,821,114,782, 25,853 context entries.
- hpm3-8: 279,902 / 11,382,678 / 45,386,001 / 1 / 3,041,091 / 108,891,108.

**tsu-p1** (transcript `…/tsu-p1/board.txt`, sha256 `d179bf9a…`, firmware `7128b53b…`):
- `b0-stats` and `b0-hello` returned 0, and the smoke passed all 9 steps.
- `@@SPEC-BEGIN 0 run p1` at about 266 s uptime, then nothing: `vbatch: no return within 3600s (ActionTimeout)`. So it
  ran at least about 3,330 s, 3.1 times u1's span, with no `@@SPEC-END`.
- The runner's apertures after the timeout (`BAKED_WEDGE_APERTURES=1`), verbatim:
  - trap log `0x89`; sw=224 `0x05`; sw=225 `0x96`; sw=226 `0x00`; sw=227 `0x07`; sw=228 `0x9f`; sw=229 `0x00`;
  - sw=192-195 all `0x00`, so `commit_instr[0].valid` = 0; sw=219 `0xd7`; sw=220 `0x8c`; sw=221 `0x02`; sw=222 `0x00`;
  - commit pc `0x0000003ff6780f04`; trap-log mepc `0xffffffff80007376`, tval `0x73`;
  - rev_node_debug_ex `0x000010ff40001134`, head `0x0071`, serving index `0x000011de`.
- These are M-16's F readings: the same commit pc, `0x3ff6780f04`, which is slot-0 residue at the CDERIVE of
  `wmem_block_fast_sublet_alloc`, and the same sw=225 and sw=220. Only the serving index differs: F read `0x116c` and
  `0xcb1`.

**tsu-p2** (transcript `…/tsu-p2/board.txt`, sha256 `f464e437…`, firmware `6c2ffae9…`): the same stage as p1 (`diff -r`
of the two stage directories is empty) and the same kit, no heartbeat, the next boot on the same board.
- The kernel differs by more than the bake's usual relink: its bake recompiled `arch/riscv/kernel/setup.o`, `drivers/of/fdt.o`, `arch/riscv/kernel/pi/cmdline_early.o` and
  `drivers/firmware/efi/libstub/efi-stub.o` in the shared kernel tree `perf/k2n/linux-board-k2n-native-hpm2tree`;
  the other three bakes recompiled only `init/version-timestamp.o`. The cause is this lane's B5a kit recheck
  (`nested/b5a/recheck-kit.sh`, two relinks, 04:47:01-04:47:11), which relinked the same tree under its `.native-kernel.lock` with
  bkit94-rec1m's command line (`node_flat_records=1048576`); p2's bake put 262144 back, and kbuild recompiled the
  objects that depend on `CONFIG_CMDLINE`. p2's `.config` `CONFIG_CMDLINE` and its dmesg command line equal p1's
  (runtime, Addendum 176, and re-read here). The objects come from the same source and configuration, but they are not
  archived, so their identity with p1's is not shown, and p1 to p2 is not shown to be one variable.
- `b0-stats` and `b0-hello` returned 0, and the smoke passed all 9 steps.
- rc 0, with stock's output: `056c1f8c…`, 1,140,005 lines, 9,000 frames.
- Uptime span 1,118.38 s. In-context 24,073,705,090 cycles, `between_cycles` 3,830,848,086, 25,862 context entries.
- hpm3-8: 8,600,200 / 99,813,106 / 98,841,736 / 8 / 11,238,778 / 126,754,707 (u1 and u2 read 0.28 M record-TLB
  misses and 1 grow).

**tsu-u2** (transcript `…/tsu-u2/board.txt`, sha256 `bda62b6d…`, firmware `07ab36a2…`):
- `b0-stats` and `b0-hello` returned 0, and the smoke passed all 9 steps.
- rc 0, with stock's output: `056c1f8c…`, 1,140,005 lines, 9,000 frames.
- Uptime span 1,073.60 s. In-context 22,942,461,402 cycles, `between_cycles` 3,843,747,930, 25,853 context entries.
- hpm3-8: 282,369 / 11,684,611 / 45,317,736 / 1 / 3,075,009 / 109,124,212.
- u1 and u2 agree within the repeat rule: `run_cycles` 0.87 % apart, `between_cycles` 0.59 %.

**Against the pre-registration.**
- **The pre-registration listed no outcome in which the protected arm does not return.** It said only that an arm
  without stock's output "is not read".
- So p1 is not read. H1 against H2 is **undecided**. The span is undefined, and a timeout is not read as "at least
  1,600 s".
- Runtime's Addendum 175 (`9eb0bcf50990`) reads a cost only if both protected boots return. p1 did not, so **no cost is
  read** from p2 either.
- Reported and not read: p2 against the mean of u1 and u2 is +5.39 % in-context and +4.60 % in uptime span.
- The prediction (H1, in-context +1 % to +5 %) **missed**: no cost is read.

**Reading, and a retraction.**
- On P1 the protected tshark has now returned in 2 of 3 boots: the heartbeat boot and p2 returned, p1 did not. p1 and
  p2 are the same stage and kit, both without the heartbeat, in consecutive boots on one board; p2's kernel also
  carries four recompiled objects (above). So the heartbeat is not needed for a return, and the non-return is
  intermittent across P1 boots, which are not shown to differ in one variable only.
- **Withdrawn:** "M-16 is narrowed to F's flat lifetime table" (497d4d6a189e, in the heartbeat result above and in
  ISSUES M-16). One return under a 60 s heartbeat does not place the non-return on F's table.
- No mechanism is claimed. On F the protected arm did not return in 2 of 2 boots, on P1 in 1 of 3; counts this small
  do not tell the two bitstreams apart.
- The sleep-and-echo beat proposed after p1 is no longer needed: a boot without any beat returned.

## CREVOKE at the table's capacity on P1

### Pre-registration: CREVOKE subtrees of 262,000, 450,000 and 10^6 nodes on P1 with a 2^20-record table, one boot (written 2026-11-18 10:40, before the boot)

**Why.** Appendix F states 0.4 s for 262,144 records and 1.6 s for 2^20 as extrapolations at 39 cycles per node. The
P1 CREVOKE chains (above) measured at most 100,000 nodes, and their 10^6 probe was refused after 196,609 because
bkit91's table holds 262,144 records. This boot measures subtrees near the capacity of a 2^20-record table. The SPEC
lane asked for it, and the lead approved it.

**Configuration.**
- Bitstream P1 `…-pagedtbl_d83d3a633`, sealed. The chain flashes it from the seal if it is not resident (sha256 checked)
  and reads it back by name.
- Kit `nested/kits/bkit91-rec1m` (`node_flat_records=1048576`), the kit of P1's Perl boot.
- Stage `nested/p1cap/crvt` (`crvt.SHA256SUMS`). It is the kit's `make-train-stage.sh` around `nested/crvt/stages-c5`,
  written by `crvt/make-stage-crv5.py`.
  - The domain is the P1 boot's `crvtime`, byte-identical: stripped `b7b4af35…`, source `6bc727e4…`.
  - The smoke skeleton is byte-identical to `nested/p1/crvt`.
- Commands, one process each:
  - `control`, and the `readclock` probes for `cycle`, `mcycle` and the HPM counters, as on P1;
  - the controls: `chain 100000` and `scatter 100000 102400`, which repeat the P1 boot's rows;
  - `scatter 262000 262144`: 1,024 record pages;
  - `chain 450000`, and `scatter 450000 450048` over 1,758 pages;
  - `chain 1000000`, and `scatter 1000000 1000192` over 3,907 pages;
  - last, `probe 2000000`, with a progress line every 65,536 nodes.
  - With `CRVT_HPM=1`, hpmcounter3/4 (record-TLB misses, walk cycles) are read around each CREVOKE.
- Board 2, after the SPEC lane's spa3x chain (nat-run3 pid 2534574), which leaves board 2 on P1. Chain
  `nested/p1cap/b2-crvt.*`. Budget 900 (x4 = `BAKED_TIMEOUT` 3600), booking 45 min.
- Checked on the twin: run `ncrv-c5`, qemu55, `--records 1048576`.
  - Every command up to `scatter 450000` returns.
  - `chain 1000000`, `scatter 1000000` and the probe end with cause 30 (node exhaustion) and rc 139. The probe was
    refused after 458,753 nodes.
  - The twin's table is flat, not P1's paged tree, so it does not predict the board's capacity.
- Checked off the board (`nested/p1cap/crvt-recheck`, via `b5a/recheck-kit.sh`):
  - real bake;
  - `BOARD_STUB=2` on the P1 crvt transcript and on its cut after the first `@@SPEC-BEGIN`: rc 0 and 0,
    `BAKED_TIMEOUT=3600` in both;
  - the relinked kernel's configuration carries `node_flat_records=1048576`;
  - real firmware 19,566,088 B (`f40d2586…`);
  - an `NR_DRY=1` pass of the chain to DONE (`nested/p1cap/dry.log`).
  - The recheck relinked the shared kernel tree with `node_flat_records=1048576`. The next bake from that tree with
    262,144 recompiles the four `CONFIG_CMDLINE` objects (see tsu-p2 above).

**Outcomes and bands.**
- **VOID**, as on P1: `b0-stats` or `b0-hello` fails, or the smoke fails beyond the kit's waiver.
- **Controls.** `chain 100000` per node within ±10 % of P1's 32.2, and `scatter 100000` within ±10 % of P1's 38.5.
  Otherwise the boot's figures are reported as not reproducing P1's.
- **The scatter took** if each scattered CREVOKE's record-TLB misses are at least N, one per node, as at 100,000 on P1.
  Otherwise those rows are reported as chains on churned records.
- **Prediction.**
  - Scattered, per node, at 262,000 and 450,000: 35 to 50 cycles.
  - Contiguous at 450,000: 28 to 40 cycles per node.
  - So 262,000 scattered nodes revoke in about 9-13 M cycles, 0.37-0.52 s at P1's 25 MHz, in one uninterruptible
    instruction.
- **Linearity.** Linear if the per-node figure at 262,000 and 450,000 lies within ±15 % of the same boot's at
  100,000, for each kind. If so, the extrapolation to 2^20 (39 per node) stands on a measured trend. If not, it is
  replaced by the measured figures.
- **10^6, not predicted.**
  - If a command returns, its figure is measured.
  - If the table refuses it (cause 30, rc 139), the 2^20 figure stays an extrapolation, and the largest N measured
    is stated with it.
- **Probe:** the count reached is the effective capacity of this kit for one domain. Not predicted.

### Results: CREVOKE at the table's capacity on P1 (board 2, 2026-11-18 10:48-10:58): linear to 450,000 nodes, 262,000 scattered nodes in 0.40 s

**Configuration.** As pre-registered (5d5210277fdf; Addendum 180, `1b12d2463fea`). Chain `nested/p1cap/b2-crvt.*`
(nat-run3 pid 2714880), after the SPEC lane's spa3x chain; P1 was resident and read back by name. Transcript
`spec/board/out-nat-p1cap-crvt-b2/crvt/board.txt`, sha256 `bbb3c7726f0a5f26fe82c4228dd10aa1b6e8307bbd6da75f6ffe724548a5f915`,
firmware `178e8197…`.

**Controls.** `b0-stats` and `b0-hello` returned 0 and the smoke passed all 9 steps. `cycle` reads from the domain,
`mcycle` faults (rc 139) and hpmcounter3/4 read (`p-hpm` rc 0), as in the first P1 boot. Not VOID.

| subtree | N | CREVOKE cycles | per node | at 25 MHz | record-TLB misses | walk cycles |
|---|---:|---:|---:|---:|---:|---:|
| contiguous | 100,000 | 3,209,994 | 32.10 | 0.128 s | 391 | 9,501 |
| scattered (400 pages) | 100,000 | 3,894,368 | 38.94 | 0.156 s | 100,001 | 611,896 |
| scattered (1,024 pages) | 262,000 | 10,034,529 | 38.30 | 0.401 s | 262,002 | 1,604,791 |
| contiguous | 450,000 | 14,508,643 | 32.24 | 0.580 s | 1,759 | 43,998 |
| scattered (1,758 pages) | 450,000 | 17,261,371 | 38.36 | 0.690 s | 450,002 | 2,758,329 |
| contiguous | 1,000,000 | not built: rc 139 | | | | |
| scattered | 1,000,000 | not built: rc 139 | | | | |

The probe toward 2·10^6 reported 458,753 nodes and died (rc 139) before 524,289.

**Against the pre-registration.**
- Controls: **met**. `chain 100000` reads 32.10 against P1's 32.2, and `scatter 100000` 38.94 against 38.5.
- The scatter took: **met**. Each scattered CREVOKE misses the record TLB once per node (N + 1 or N + 2 misses).
- Prediction: **met**. Scattered 38.30 and 38.36 per node (35 to 50); contiguous 32.24 at 450,000 (28 to 40);
  262,000 scattered nodes in 10.03 M cycles (9 to 13 M).
- Linearity: **met**. Against the same boot's 100,000: scattered −1.6 % and −1.5 %, contiguous +0.4 %.
- 10^6: **refused** in both commands, so nothing at 10^6 is measured.

**Reading.**
- **Measured:** one CREVOKE of 262,000 scattered nodes takes 10.03 M cycles, 0.40 s at 25 MHz, in one
  uninterruptible instruction. Appendix F's 0.4 s for 262,144 records is now a measurement to within 144 nodes.
- **The largest subtree measured is 450,000 nodes:** 0.69 s scattered, 0.58 s contiguous.
- **2^20 stays an extrapolation.** The cost is linear from 100,000 to 450,000 nodes at 38.3 to 38.9 cycles per scattered
  node, which gives 40.2 M cycles, 1.61 s, for 2^20 nodes.
- **A subtree of 2^20 nodes cannot be built on this kit.** With `node_flat_records=1048576` one domain's chain was
  refused between 458,753 and 524,288 nodes. The board console does not show the fault's cause; on the twin the same
  commands end with cause 30 after the same count. What limits a domain to under 2^19 nodes of a 2^20-record table is
  not established here.

## Perl with the arena lookup repaired

### Pre-registration: Perl suns.pl on P1 with the arena-lookup repair, U P P U in one boot (written 2026-11-18 11:25, before the boot)

**Why.** The +208 % of "Results: Perl on P1" is withheld: patch 0001 finds a head's arena by walking the list of all
arenas, about 100 M loads through cold arenas per command ("Withdrawn 2026-11-18" there). This boot repeats that boot
with the lookup repaired and nothing else changed. It asks what a lifetime per SV head costs once the search is gone.

**Configuration.**
- Bitstream P1, kit `nested/kits/bkit91-rec1m` (`node_flat_records=1048576`), workload suns.pl with `PERL_HASH_SEED=0`,
  answer `suns.out` (`01688ea0…`): all as in the P1 Perl boot. The lifetimes are the same, so the live-record peak is
  the same and the 2^20-record table is still needed.
- **The repair**, this lane's, on top of patch 0001 (blob `72db9236`):
  `nested/perl/lookup-fix/0002-sv-arena-lookup-by-index.patch` (`04540913…`; written by `fix-lookup.py`,
  `f5ccf26e…`). It keeps every arena in an array sorted by address. `Perl_more_sv`, the only place that adds an arena, enters
  it. A lookup tries the arena found last, then binary-searches the array, and reads no arena. The array is freed in
  `Perl_sv_free_arenas`. CDERIVE and CREVOKE sites, the arena geometry and the free list are unchanged.
  - Only `sv.c` differs from the protected source tree of the P1 boot (`cmp` of `sv.h`, `sv_inline.h`, `gv.h`, `pp.c`,
    `pp_ctl.c`, `perl.h`).
  - Built by the same `build-arms.sh` (arm `protfix`, `PERLD_EXTRA_PATCH`). Stripped image `693568f6…` (full
    `9f1fad49…`). CREVOKE census as the P1 protected image: 0 with rs2 = x0, 9 with rs2 != x0.
  - The stock image is the P1 boot's, `3012f392…`.
- Stage `nested/p1g/perlfix` (`perlfix.SHA256SUMS`): the kit's wrap of `nested/appd/perl-fix`. Against the P1 boot's
  stage `nested/p1a/perl`, only `perl_prot.xz` and `plan.json` differ.
- u1 p1 p2 u2 in one boot. Board 2, after the SPEC lane's filler chain (nat-run3 pid 2941955); P1 is resident. Chain
  `nested/p1g/b2-perlfix.*`. Budget 900 (x4 = `BAKED_TIMEOUT` 3600), booking 45 min.
- Expected wall time: 3.6 min per stock command (the P1 boot), 4 to 6 min per protected command; about 20 min of
  commands.
- **Twin** (`nperl-fix`, 2^20 records): four commands rc 0 with `suns.out`. In-context 1,731.9 / 1,905.1 / 1,905.1 /
  1,731.9 M, **+10.00 %**, repeats 0.00 %. With the walk (`nperl-1m`) it was +40.26 %: the repair removes 524 M of
  the 697 M added instructions.
- Checked off the board (`nested/p1g/perlfix-recheck`): real bake; `BOARD_STUB=2` on the P1 Perl transcript and on its
  cut, rc 0 and 0, `BAKED_TIMEOUT=3600` in both; the relinked kernel carries `node_flat_records=1048576`; real firmware
  23,760,392 B (`f7748f4f…`); an `NR_DRY=1` pass of the chain to DONE (`nested/p1g/dry-perlfix.log`).

**Outcomes and bands.**
- Answers: every command's stdout equals `suns.out`, else VOID for the cost. Repeat rule as for every P1 boot (1 %
  in-context, 2 % kernel + other, else not read). VOID as before.
- **Stock control:** the stock arm's in-context cycles within ±1 % of the P1 boot's 4.914 G (same image, same kit,
  another board). Outside it, the two boots are not compared.
- **The reading:** the in-context cost, protected against stock.
  - ≤ +30 %: the arena search carried at least 85 % of the +208 %;
  - +30 % to +100 %: the search carried part of it, and the rest is still unexplained;
  - ≥ +100 %: the search was not the main cost, and the new attribution is wrong too.
- **Prediction: +10 % to +30 %.**
  - The twin's +173 M instructions at the stock arm's 2.84 cycles per twin instruction: about 0.5 G, +10 %.
  - CDERIVE and CREVOKE on 1.94 M allocations and frees, at 63 to 100 cycles each: 0.1 to 0.2 G.
  - Checks on cold records that are not the walk's: the P1 boot read 100.6 M added event-23 replays against 97.4 M
    counted steps, so about 3 M, at about 105 cycles: 0.3 G.
- **Counters** (hpm3-8 as in the P1 boot), protected minus stock, per command:
  - event-23 replays (hpm7): under +10 M, against +100.6 M;
  - record-TLB misses (hpm3): under +10 M, against +48.8 M;
  - L1D misses (hpm8): under +30 M, against +260 M.
  - A counter above its bound with the cost in the first band is reported, not explained.
- **What this boot cannot show:** it changes the lookup only. It does not separate the cost of CDERIVE and CREVOKE from
  the cost of checking each head's own record.

## Perl on SPEC's own perlbench test, with SV heads protected

### Pre-registration: 500.perlbench_r test on P1 with the sv-heads protection, one boot (written 2026-11-18 11:35, before the boot)

**Why.** The lane owner chose SPEC's perlbench test as the Perl row of the appendix: the workload of the paper's SPEC
table, run on SPEC's own Perl 5.22.1 sources and checked against SPEC's reference outputs. suns.pl on stock Perl
5.36.3 is no longer the row.

**Images.** SPEC CPU2017 500.perlbench_r, the table's Full build (FRCDKMSa3, the SPEC lane's).
- Stock: the table's image itself, stripped `c1744e87…`.
- Protected: the same build with the sv-heads patch ported to 5.22.1 (`nested/perlspec/port-sv-heads.py`, `0020f23c…`;
  from patch 0001, blob `72db9236`) and the arena-lookup repair (`nested/perl/lookup-fix/fix-lookup.py`,
  `f5ccf26e…`), under `-DPERL_CAPSTONE_SUBLET`. Stripped `83703c92…` (full `1cdc258b…`).
  - Port differences from 5.36.3: `uproot_SV` and `S_more_sv` live in `sv.c`, and the eval compiler is `S_doeval`.
  - **Control:** the ported sources compiled without the define give 69 of 69 objects byte-identical to the table's
    build, and the relink reproduces its image (`51bfbd41…`).
  - CREVOKE census: stock 0 with rs2 = x0 and 6 with rs2 != x0; protected 0 and 9.

**Stage** `nested/p1g/ptest` (`ptest.SHA256SUMS`, 574 files): kit bkit91's wrap of `nested/perlspec/stage-test5`, made
by `make-perltest-stage.py` (`4b536a30…`) from the table's stage `stages-frcdkmsa3-full/500.perlbench_r-native`. `in/`
and `natperl` are byte-identical to the table's board stage. One boot, cheap commands first:
- commands 0-3: `makerand.pl` as u1 p1 p2 u2;
- commands 4-5 (`pt`): `makerand.pl` and `test.pl` with the protected image. test.pl runs each of its test scripts
  with the arm's image as `../sw/perl`, as in the table's run.
- **test.pl has no stock arm in this boot.** A stock test.pl takes 12,799 s, and the deadline leaves room for one such
  run. Its reference is the SPEC lane's two Full runs on board 3:
  - `out-nat-p1-final-a3-perl-b3` (`bc778b6f…`): makerand 276.01 s, 6,837,496,467 in-context cycles; test.pl
    12,798.66 s, 267,212,372,330 in-context cycles, 44,357,708,022 kernel + other;
  - `out-nat-p1-a3rep-b3` (`6f4d9f8e…`): 276.11 s and 6,840,289,021; 12,799.00 s, 267,230,625,762 and 44,358,613,410.
  - The two differ by 0.04 % (makerand) and 0.007 % (test.pl) in in-context cycles. The SPEC lane's measured spread
    between boards for one build is at most 0.3 %.
- Board 2, after this lane's Perl suns.pl chain (nat-run3 pid 2985665); P1 is resident. Chain `nested/p1g/b2-ptest.*`.
  Budget 86400 as the table's run (`BAKED_TIMEOUT` 345600), booking 1,440 min.
- Expected wall time: makerand 4.6 min per command; test.pl 3.6 h plus its cost. The row is usable for the deadline
  if the boot ends by about 18:45, which holds if test.pl's cost is under about +70 %.
- Checked off the board (`nested/p1g/ptest-recheck`): real bake; `BOARD_STUB=2` on the table's transcript and on its
  cut, rc 0 and 0, `BAKED_TIMEOUT=345600` in both; real firmware 23,760,392 B (`89467f76…`); an `NR_DRY=1` pass of
  the chain to DONE (`nested/p1g/dry-ptest.log`).
- **Twin, and a deviation.** The linear-search image passed makerand as U P P U on the twin with SPEC's
  `makerand.out` in all four commands (`npspec-t1`, `npspec-mrx`; in-context +2.6 %). The twin of this stage
  (`npspec-t5`, the repaired image) is queued behind the gcc twin and will not have finished test.pl when the boot
  starts. It serves as an early warning: if it shows the protected test.pl failing, the boot is stopped.

**Outcomes and bands.**
- **VOID** as for every P1 boot.
- **Answers.** `makerand.out` equals SPEC's reference (`539d02dc…`) in all five makerand commands, and `test.out`
  passes specdiff against SPEC's reference with the tool the table used. A makerand mismatch voids that command's
  cost. **If test.out fails specdiff, the protected Perl is reported as failing its tests, with the failing tests
  listed, and no test.pl cost is read.**
- **makerand, interleaved.** Repeat rule as for every P1 boot. In-context cost, protected against stock: predicted
  +1 % to +8 %.
- **makerand, cross-board check.** This boot's stock makerand against the reference's 6,837.5 M in-context cycles:
  within ±1 %. If it is outside, the test.pl comparison below is reported with that offset stated, not corrected.
- **test.pl, not interleaved, another board.** Cost = this boot's protected test.pl against the mean of the two
  reference runs, for in-context cycles and for the uptime span.
  - A difference under 1 % is reported as "not distinguishable from the cross-board spread".
  - Predicted: +5 % to +30 % in-context.
  - ≥ +100 %: the protection has a cost in test.pl that the lookup repair did not remove; reported as such.
- The row states that test.pl's stock run is the table's, from another board and day.

**Addendum 1 (2026-11-18 11:57, before either boot): the same stage also runs on board 1.** Board 1 was freed at
11:36 (the SPEC lane's x264 repeat was stopped on the lane owner's order), and this lane's gcc run, planned for it, is
cancelled (below). So the stage `nested/p1g/ptest` runs unchanged on board 1 as well, chain `nested/p1g/b1-ptest.*`
(an `NR_DRY=1` pass reached DONE, `dry-ptest-b1.log`), starting at once; board 2's boot follows its suns.pl chain as
registered.
- Both boots are read by the rules above, each on its own.
- test.pl's cost is reported per board and as the mean of the two.
- The two protected test.pl readings are a cross-board repeat of one image. If their in-context cycles agree within
  0.5 %, that supports setting them against board 3's stock runs. If they do not, both are reported and their
  difference is stated as the uncertainty of the row.
- If one boot fails for a reason of its own, the other stands alone, as registered above.

**Addendum 2 (2026-11-18 12:55): board 2's boot is stopped by this lane before its test.pl begins.** Board 1's
protected test.pl was running without a fault ten minutes after its start, and board 2 is given to the gcc boot
("gcc ... Addendum 2"). Board 2's makerand commands are not read: the boot did not reach its end. The row rests on
board 1 alone, as first registered.

**Addendum 3 (2026-11-18 13:17): the twin shows the protected test.pl failing 10 of 314 tests, so board 1's boot is
stopped and makerand is measured by a boot of its own.**
- Twin `npspec-t5h` (this stage, with `capstone_vm.dom_grant_root=/tmp/nt/sr` as on the boards; the earlier twin
  `npspec-t5` lacked it and failed on that alone): test.pl ran 55.2 G in-context instructions and reports "Failed 10
  tests out of 314": io/open.t, io/perlio.t, op/bless.t, op/goto.t, op/gv.t, op/inccode-tie.t, op/inccode.t,
  op/reset.t, op/sort.t and re/pat_psycho.t. `test.out` fails specdiff.
  - Six processes end on a stale-lifetime fault (cause 25), in `S_rv2gv`, `Perl_pp_bless`, `Perl_pad_free`,
    `S_glob_assign_glob`, `Perl_sv_resetpvn` and `Perl_pp_aassign` (pcs mapped at 0x3ff7c00000 = vaddr 0x10000).
    Perl 5.22.1 touches a freed SV head at each, and the 5.36.3 patch, which this port follows, does not cover them.
  - One process ends on cause 30 in `malloc`: the table's 262,144 records are exhausted.
- By the rule registered above, a failing `test.out` means no test.pl cost is read. Board 1's protected test.pl would
  end the same way after 3.6 h, so the boot is stopped now (started 12:42:12, running at 13:17). Its makerand commands
  are not read: the boot does not reach its end.
- **makerand gets its own boot on board 1 at once:** stage `nested/p1g/pmrf` (`pmrf.SHA256SUMS`), makerand.pl as
  u1 p1 p2 u2 with the same two images, made by `make-uppu-stage.py` from the table's stage without test.pl, `t/` and
  `lib/`. Chain `nested/p1g/b1-pmrf.*`; budget 900 (`BAKED_TIMEOUT` 3600), booking 60 min; about 25 min. Rule 8:
  real bake, both stubs rc 0, firmware 21,663,240 B (`a58bb54b…`), `NR_DRY=1` to DONE (`dry-pmrf.log`). The makerand
  rules above apply unchanged: answers against SPEC's `makerand.out`, the repeat rule, the predicted +1 % to +8 %,
  and the cross-board check against board 3's 6,837.5 M.
- A boot of test.pl with the port extended to those six sites, on a 2^20-record kit, would be registered separately.

### Results: Perl suns.pl on P1 with the arena lookup repaired (board 2, 2026-11-18 11:54-12:23): +16.05 % in-context, against +208.08 % with the walk

**Configuration.** As pre-registered (f01c2ec2a683; Addendum 182, `a65d0b390a21`). Chain `nested/p1g/b2-perlfix.*`
(nat-run3 pid 2985665), after the SPEC lane's filler; P1 was resident. Transcript
`spec/board/out-nat-p1fix-perl-b2/perlfix/board.txt`, sha256
`865c62d59a4b529a9d3a4f6f2e07b151444f5a992d89401e463b2207a6bdf0b5`, firmware `8c12985b…`. `b0-stats` and `b0-hello`
returned 0 and the smoke passed all 9 steps. Read by `appd/parse-abba.py` and `p1/abba.py` (`nested/p1g/perlfix.json`).

| | u1 | p1 | p2 | u2 | repeat, u / p | protected vs stock | with the walk (P1 boot) |
|---|---:|---:|---:|---:|---|---|---|
| in-context cycles | 4,916,505,883 | 5,704,881,088 | 5,701,428,730 | 4,912,214,234 | 0.09 % / 0.06 % | **+16.05 %** (+788.8 M) | +208.08 % (+10.22 G) |
| kernel + other | 409,309,728 | 517,288,210 | 502,512,229 | 400,859,680 | 2.11 % / 2.94 % | not read (+25.9 %) | not read |
| uptime span, s | 217.32 | 254.21 | 253.50 | 216.82 | 0.23 % / 0.28 % | +16.95 % | +191.97 % |
| record-TLB misses (hpm3) | 691,246 | 1,146,427 | 1,145,881 | 690,486 | | +0.455 M | +48.8 M |
| walk cycles (hpm4) | 18.5 M | 30.9 M | 30.8 M | 18.5 M | | +12.3 M | +1,096 M |
| paged record accesses (hpm5) | 15.47 M | 32.05 M | 32.05 M | 15.47 M | | +16.6 M | +115.7 M |
| event-23 replays (hpm7) | 1,360,473 | 2,813,325 | 2,809,806 | 1,358,065 | | +1.45 M | +100.6 M |
| L1D misses (hpm8) | 14.53 M | 21.99 M | 21.96 M | 14.51 M | | +7.46 M | +260.3 M |

**Against the pre-registration.**
- Answers: all four commands return `suns.out`. Repeat rule: in-context 0.09 % and 0.06 %, read; kernel + other
  2.11 % and 2.94 %, not read.
- Stock control: **met**. The stock arm reads 4,914.4 M in-context cycles against the P1 boot's 4,913.9 M on board 1,
  +0.01 %.
- **The reading: +16.05 %, in the first band (≤ +30 %).** The arena search carried 92 % of the +208 %.
- Prediction (+10 % to +30 %): **met**.
- Counter bounds: **all met.** Event 23 +1.45 M (bound 10 M), record-TLB misses +0.455 M (bound 10 M), L1D misses
  +7.46 M (bound 30 M).

**Reading.**
- With the lookup repaired, a lifetime per SV head costs Perl **+16 %** of its in-context cycles on suns.pl, with
  the same lifetimes, the same peak of live records and the same 2^20-record table as the +208 % boot.
- So the +208 % was the cost of the patch's linear arena search, as the withdrawal said. The record TLB was not the
  cause: with the same number of live records, the protected arm now misses it 0.455 M times more than stock, not
  48.8 M.
- What makes up the +789 M is not separated by this boot. The twin prices the added instructions at +10.00 %; the
  boot adds 1.94 M CDERIVEs and CREVOKEs, 16.6 M paged record accesses and 7.5 M L1D misses.

**Cycle attribution (added 2026-11-22, the method of the test.pl attribution in the prot6 section), protected minus
stock, +788.8 M in-context cycles** (the means of p1/p2 and u1/u2):

| part | cycles | share | price and its source |
|---|---:|---:|---|
| the port's instructions | 491.7 M | 62.3 % | +173,265,263 twin instructions (`nested/out/nperl-fix`, the in-context `run_cycles` of p1/p2 minus u1/u2: 1,905,131,312 against 1,731,866,049) at 2.838 cycles each, the stock arm's own cycles per twin instruction (a model: the mean) |
| record-table walks | 12.3 M | 1.6 % | hpm4, read as cycles (measured; the table's rounded values) |
| event-23 replays | 24.7-29.0 M | 3.1-3.7 % | +1.45 M at 17-20 cycles each (Addendum 1's RTL estimate, from source) |
| **not attributed** | **255.7-260.1 M** | **32.4-33.0 %** | |

- The remainder's share is close to test.pl's (34.4-35.4 % at the same mean price). The L1D misses (+7.46 M) are
  not priced; if they carried the whole remainder, each would cost 34-35 cycles.
- As on test.pl, the instruction price is the softest number here. No board run has priced the CDERIVE/CREVOKE pairs
  and the head checks on their own. At a lower price the instructions take less and the remainder grows.

## gcc (SPEC 502.gcc_r) with its pool allocator protected

**Scope.** SPEC's Full build protects only `malloc`. gcc carves its own objects from three allocators. This lane
counted them on x86 (SPEC's flags plus counters; outputs byte-identical to SPEC's references):

| input | ggc-page allocations | `ggc_free` | ggc peak live | alloc-pool allocations | alloc-pool peak live |
|---|---:|---:|---:|---:|---:|
| test `t1.c` | 12,129 | 157 | 11,980 | 1,247 | 359 |
| train01 | 1,415,393 | 47,290 | 1,369,103 | 3,780,544 | 132,149 |

- **ggc-page is not protected, and cannot be at train01.** SPEC's build disables the collector (`ggc-page.c:1941`:
  `ggc_collect` returns at once under `SPEC`), so a GC object is freed only by an explicit `ggc_free`. 1.37 M
  lifetimes would be live at once, more than the 2^20-record table.
- obstack is not counted or protected.
- **alloc-pool is protected** (`nested/gcc/pool/port-alloc-pool.py`, `ac09f51c…`, under `-DGCC_CAPSTONE_POOL`): every
  element is a CDERIVE child of its block, with a 16-byte header holding the block's own pointer; `pool_free`
  CREVOKEs it; `empty_alloc_pool` and `free_alloc_pool` free the blocks, which ends every element still live in them.

**A use after `pool_free` in gcc itself.** With the pool protected, gcc stops on SPEC's own test input: the twin
faults (cause 25) in `forward_propagate_into` (fwprop.c), loading through a df ref's `insn_info`.
- Found natively with AddressSanitizer and a `pool_free` that poisons and never reuses the element
  (`nested/gcc/asan`): `use-after-poison` at `fwprop.c:1350`. The element is a `df_scan insn` record freed by
  `df_insn_delete`, called from `df_insn_rescan`.
- Read in the source: `df_insn_refs_verify` collects the insn's new refs with the insn's current record as their
  `insn_info`. `df_insn_rescan` then frees that record, allocates a new one and installs the collected refs, which
  still name the freed record. Stock gcc works only because the pool's free list is LIFO, so the new record is the
  element just freed.
- **One fix** (`nested/gcc/pool/fix-stale.py`, `108b7a3b…`): after the new record is created, the collected refs are
  pointed at it. No CREVOKE is skipped and no pool is exempted. With it, ASan with quarantine is clean on t1.c,
  train01.c, 200.c and scilab.c, and all four outputs are byte-identical to SPEC's references.
- Not checked: other inputs, and whether later gcc releases changed this code.

### Pre-registration: 502.gcc_r train01 on P1 with alloc-pool protected, U P P U in one boot (written 2026-11-18 12:30, before the boot)

**Images**, both built from the table's Full build (FRCDKMSa3) with its own compile and link lines
(`nested/gcc/pool/build-fx.sh`):
- stock arm, **the fix only**: stripped `e9b0f80c…`. Its `alloc-pool.o` is byte-identical to the Full build's; only
  `df-scan.o` differs from the Full build.
- protected arm, the fix and `-DGCC_CAPSTONE_POOL`: stripped `54291a9f…`. The arms differ in `alloc-pool.o` only.
- So the stock arm is not the table's image: it carries the fix. The row says so.

**Stage** `nested/p1g/gfx` (`gfx.SHA256SUMS`): kit bkit91's wrap of `nested/gcc/pool/fx-stage-train01`, made by
`make-uppu-stage.py` from the table's untuned train01 stage (`board-frcdkmsa3-notune`). u1 p1 p2 u2 in one boot.
- Board 3, after the SPEC lane's 250 Hz chain (pid 2657209); the chain flashes P1 from the seal (sha256 checked)
  and reads it back by name. Chain `nested/p1g/b3-gfx.*`. Budget 3600 (`BAKED_TIMEOUT` 14400), booking 180 min.
- Expected wall time: 1,772 s per stock command (the table's run) and about the same per protected one; about 2 h.
- Twin: `ngpool-fxt` (test) and `ngpool-fx1` (train01), four commands each, all rc 0 with SPEC's reference output
  (specdiff 1/1). In-context on train01: 12,230.7 / 12,330.2 / 12,330.2 / 12,230.7 M, **+0.81 %**; on test +0.14 %.
- Checked off the board (`nested/p1g/gfx-recheck`): real bake; `BOARD_STUB=2` on the table's train01 transcript and
  its cut, rc 0 and 0, `BAKED_TIMEOUT=14400` in both; real firmware 25,857,544 B (`a8ee5dec…`), under the 26 MB cap; an
  `NR_DRY=1` pass of the chain to DONE (`nested/p1g/dry-gfx.log`).

**Outcomes and bands.**
- **VOID** as for every P1 boot.
- **Answers:** all four commands' `.s` output passes specdiff against SPEC's reference, else no cost is read.
- Repeat rule as for every P1 boot (1 % in-context, 2 % kernel + other).
- **Stock control:** the stock arm's in-context cycles within ±1 % of the table's Full runs (41,713.7 M on board 1,
  41,694.2 M on board 3). Reported only; it also shows what the fix costs.
- **In-context cost, protected against stock: predicted +0.5 % to +10 %.**
  - The twin's added instructions give +0.8 %.
  - 3.78 M CDERIVEs and 1.85 M CREVOKEs at 63 to 100 cycles: about 0.45 G of 41.7 G, +1 %.
  - Up to 132,149 pool elements are live at once, each with its own record, against a 16-entry record TLB. PostgreSQL,
    with 105,043 live records, costs +7.8 %. So the upper end is set by record checks on element accesses.
  - ≥ +30 %: reported as unexplained, and the port's own code is read for a search or a loop before any other cause
    is named.
- The row states its coverage: gcc's pool allocator; the GC heap and obstacks are not protected, and both arms carry
  one stale-use fix.

**Addendum 1 (2026-11-18 12:36, before the boot): board 1, not board 3.** Board 3 is taken by the SPEC lane after
15:00 (its Addendum 185). The boot runs on board 1 behind this lane's perlbench test boot there, with the same stage.
- Chain `nested/p1g/b1-gfx.*` (an `NR_DRY=1` pass reached DONE, `dry-gfx-b1.log`); P1 is resident, so no flash.
  Transcript `spec/board/out-nat-p1-gccpool-b1/gfx/board.txt`.
- It starts when the perlbench test boot ends, expected about 16:15 to 16:55, and ends about two hours later.
- Everything else is as registered above.

**Addendum 2 (2026-11-18 12:55, before the boot): board 2, at once.** On board 1 the protected test.pl has been
running for ten minutes without a fault, so the perlbench test boot on board 2, a repeat of it, is stopped before its
test.pl begins, and this boot takes board 2 now. It would otherwise end near 18:45.
- Chain `nested/p1g/b2-gfx.*` (an `NR_DRY=1` pass reached DONE, `dry-gfx-b2.log`). Transcript
  `spec/board/out-nat-p1-gccpool-b2/gfx/board.txt`. The chain armed on board 1 (pid 3213632) is cancelled; it had
  booked nothing.
- Everything else is as registered above.

### Pre-registration: SPEC perlbench test.pl with the port extended for six stale-head sites, one protected and one stock run on a 2^20-record kit (written 2026-11-18 13:40, before the boots)

**Why.** With the first port, the protected test.pl fails 10 of 314 tests on the twin (Addendum 3 above). The lead asks
for the full Perl row on SPEC's test.pl if it can be had today, with suns.pl as the fallback.

**The extended port** (`nested/perlspec/fx/port-sv-heads.py`, `f55e5096…`; all of it under `PERL_CAPSTONE_SUBLET`,
and without the define 69 of 69 objects are still byte-identical to the table's build):
- `S_rv2gv` (pp.c:245), `pp_bless` (pp.c:614) and `sv_resetpvn` (sv.c:9607): a type test of a stash that may be
  freed; read through the arena's own pointer (`SvTYPE_MAYBE_FREED`), as the 5.36.3 patch does for its sites.
- `S_glob_assign_glob` (sv.c:3991, 4000): `SvTAINTED(sstr)` and `GvAV(sstr)` are read after a `LEAVE` that may have
  freed `sstr`; both are read before the `LEAVE`.
- `pad_free` (pad.c:1820): tests and clears `SVs_PADTMP` in a pad entry that can be a freed head; done through the
  arena's own pointer. **This is the one write through that pointer.** Why the pad entry is stale was not found.
- `pp_aassign` (pp_hot.c:1094): `SvGETMAGIC` on an element that may be a freed scalar; skipped when `SvIS_FREED`.
- No CREVOKE is skipped and no SV is exempt. Protected image: stripped `f5449ace…` (full `781af4cc…`).

**Twin, and a deviation.** Run by a background fork of this lane; this session read its port diff and its build log.
- With the new image, 7 of the 10 failing tests pass or run to their end as stock does (`npfx-tt2b`): io/open.t,
  op/bless.t, op/gv.t, op/reset.t pass in full; op/goto.t and op/sort.t end rc 0 with only the TODO test that stock
  also fails; re/pat_psycho.t passes with 1,048,576 records (`npfx-tt2p`), so it needs the 2^20-record kit.
- io/perlio.t (tests 23-29), op/inccode.t and op/inccode-tie.t (test 61) still fail on the twin, **and the stock
  image fails them identically there** (`npfx-tt2s`). They need a temp file and `/dev/null`; the boards' command line
  has `capstone_vm.dom_local_tmp` and `dom_null`, and the twin's cannot hold them beside `dom_grant_root` (512 bytes).
  That they pass on a board with the protected image is **not verified**.
- **Deviation:** the twin has not run all of test.pl with the new image, and cannot show 314 of 314 for the reason
  above. A full run (`npspec-pt2`, 2^20 records) started at 13:35 and serves as an early warning: it should fail
  those three tests and no other. If it shows another failure or a fault, the protected boot is stopped.

**Boots**, both on kit `nested/kits/bkit91-rec1m` (`node_flat_records=1048576`), each the table's stage
(makerand.pl, then test.pl), budget 86400 (`BAKED_TIMEOUT` 345600), booking 1,440 min:
- **protected, board 1**, after the makerand boot there: stage `nested/p1g/pt2` (`pt2.SHA256SUMS`), the table's stage
  with the protected image (xz-packed; `run.sh` differs from the table's in that one line). Chain
  `nested/p1g/b1-pt2.*`. Transcript `spec/board/out-nat-p1-pt2-b1/pt2/board.txt`. Rule 8: real bake, both stubs rc 0,
  firmware 23,760,392 B (`8a0a857a…`), the kernel at 1048576, `NR_DRY=1` to DONE.
- **stock, board 2**, after the gcc boot there (about 15:05): stage `nested/p1g/st2` (`st2.SHA256SUMS`), whose
  `train/` is file-for-file the table's board stage. Chain `nested/p1g/b2-st2.*`. Transcript
  `spec/board/out-nat-p1-st2-b2/st2/board.txt`. Rule 8: the same, firmware `cc1c0bac…`.
- Expected: 4.6 min for makerand and 3.6 h for a stock test.pl; the protected one 3.6 h plus its cost. A boot must
  end by about 19:00 to reach the paper today.

**Outcomes and bands.**
- **VOID** as for every P1 boot.
- **Answers.** Each boot's `makerand.out` and `test.out` pass specdiff against SPEC's references, read with the tool
  the table used. **If the protected `test.out` fails, the failing tests are listed and no test.pl cost is read.**
- **Cost of test.pl:** protected (board 1) against stock (board 2), in-context cycles and uptime span. The arms are
  on different boards and not interleaved; the SPEC lane's spread for one image across boards is at most 0.3 %, so a
  difference under 1 % is reported as not distinguishable. Predicted: +5 % to +30 % in-context.
- **Checks on the comparison:**
  - the stock test.pl here against the table's two runs on board 3 (267,212.4 M and 267,230.6 M in-context, on the
    262,144-record kit): reported, expected within 1 %;
  - makerand in both boots against each other and against board 1's interleaved makerand boot.
- **If only one boot ends in time,** the protected run is set against the table's two stock runs (another board,
  the smaller table), and the row says so.

### Results: SPEC perlbench makerand.pl on P1 with SV heads protected (board 1, 2026-11-18 13:17-13:47): −2.54 % in-context, the prediction missed low

**Configuration.** As registered in Addendum 3 of the perlbench test pre-registration (cbe092a39bfc; Addendum 187,
`57d7efd2eace`). Chain `nested/p1g/b1-pmrf.*`; P1 resident. Transcript `spec/board/out-nat-p1-pmr-b1/pmrf/board.txt`,
sha256 `ea9177544253a5fb194018a9ae542280d48123c0f58fc1522b7bbb408b22e618`, firmware `dac2141e…`. `b0-stats` and
`b0-hello` returned 0; the smoke reads 8 PASS with `task` waived, as in every P1 boot. Images: stock `c1744e87…` (the
table's), protected `83703c92…` (the first port with the lookup repair).

| | u1 | p1 | p2 | u2 | repeat, u / p | protected vs stock |
|---|---:|---:|---:|---:|---|---|
| in-context cycles | 6,835,370,958 | 6,660,829,111 | 6,659,959,955 | 6,832,475,210 | 0.04 % / 0.01 % | **−2.54 %** (−173.5 M) |
| kernel + other | 46,418,036 | 46,959,596 | 46,922,751 | 46,502,802 | 0.18 % / 0.08 % | +1.03 % |
| uptime span, s | 275.92 | 268.95 | 268.91 | 275.78 | | −2.51 % |
| record-TLB misses (hpm3) | 2,198 | 3,069 | 3,069 | 2,196 | | +872 |
| event-23 replays (hpm7) | 30,784 | 49,591 | 49,731 | 30,736 | | +18.9 k |
| L1D misses (hpm8) | 96,832 | 91,087 | 93,580 | 95,927 | | −4.0 k |

**Against the registration.**
- Answers: all four `makerand.out` equal SPEC's reference. Repeat rule: met on both counters, so both are read.
- Cross-board check: the stock arm reads 6,833.9 M against board 3's 6,837.5 M, −0.05 %. Met.
- **In-context cost −2.54 %; predicted +1 % to +8 %: missed, low.** The protected image runs makerand in fewer cycles
  than the table's stock image.
- On the twin the same pair costs +2.5 % in instructions. So the protected arm executes more instructions and takes
  fewer cycles. **Not explained.** The two images differ in `sv.o` and in every object that inlines the patched
  macros, so a difference in code layout is the obvious candidate; this boot does not show it.
- The stopped boot on board 1 read the same from its console (6,795.7 / 6,655.1 / 6,665.0 / 6,838.9 M; not a
  registered reading).

**Reading.** makerand.pl allocates few scalars (872 more record-TLB misses, 18.9 k more event-23 replays in 6.8 G
cycles), so the protection has almost nothing to do in it, and what this boot measures is the difference between two
builds of the interpreter. It bounds the protection's cost on this input at about zero; it is not evidence that the
protection speeds Perl up.

### Results: 502.gcc_r train01 on P1 with alloc-pool protected (board 2, 2026-11-18 12:53-15:11): +5.26 % in-context

**Configuration.** As pre-registered (d3cd2c484bab with Addenda 1 and 2, 4aec438c4c55 and 117102bcd3f6; the SPEC
lane's Addendum 186, `356949d451d4`). Chain `nested/p1g/b2-gfx.*` (nat-run3 pid 3227005); P1 resident on board 2.
Transcript `spec/board/out-nat-p1-gccpool-b2/gfx/board.txt`, sha256
`4dc936d1e4e233fc014579c39348e9a3bb4729b70b7a667c946400e8b232e89f`, firmware `a27a0b77…`. `b0-stats` and `b0-hello`
returned 0; the smoke reads as in every P1 boot. Images: stock with the fix `e9b0f80c…`, protected `54291a9f…`.

| | u1 | p1 | p2 | u2 | repeat, u / p | protected vs stock |
|---|---:|---:|---:|---:|---|---|
| in-context cycles | 41,800,647,681 | 44,001,429,869 | 43,996,027,573 | 41,801,673,609 | 0.00 % / 0.01 % | **+5.26 %** (+2,197.6 M) |
| kernel + other | 2,015,039,808 | 2,073,001,269 | 2,069,141,818 | 2,016,547,420 | 0.07 % / 0.19 % | **+2.74 %** (+55.3 M) |
| uptime span, s | 1,775.13 | 1,865.70 | 1,865.35 | 1,775.26 | 0.01 % / 0.02 % | **+5.09 %** |
| record-TLB misses (hpm3) | 3,468,420 | 27,863,223 | 27,862,542 | 3,466,983 | | +24.4 M |
| walk cycles (hpm4) | 106.8 M | 461.9 M | 462.0 M | 106.6 M | | +355.2 M |
| paged record accesses (hpm5) | 23.06 M | 96.45 M | 96.45 M | 23.06 M | | +73.4 M |
| event-23 replays (hpm7) | 6.64 M | 31.85 M | 31.85 M | 6.64 M | | +25.2 M |
| L1D misses (hpm8) | 371.2 M | 413.4 M | 413.4 M | 371.2 M | | +42.2 M (+11.4 %) |

**Against the pre-registration.**
- Answers: **met.** All four commands' `train01…s` have SPEC's reference hash (`c786c07f…`); the SPEC lane recomputes
  specdiff from the transcript.
- Repeat rule: **met** on both counters, so both differences are read.
- Stock control: **met.** The stock arm with the fix reads 41,801.2 M against the table's 41,713.7 M (board 1) and
  41,694.2 M (board 3), +0.21 % and +0.26 %.
- In-context cost: **+5.26 %**, inside the predicted +0.5 % to +10 %.

**Reading.**
- Protecting gcc's pool allocator costs **+5.3 %** of its in-context cycles and +5.1 % of its run time on train01,
  with 3.78 M element lifetimes created and at most 132,149 live at once.
- The twin prices the added instructions at +0.8 %. The rest follows the records: the protected arm misses the record
  TLB 24.4 M times more and spends 355 M more cycles in walks (16 % of the +2,198 M); how the remainder divides between
  the CDERIVEs and CREVOKEs and the checks on element accesses is not separated by this boot.
- Coverage, to be stated with the number: gcc's pool allocator only. The GC heap is not protected (SPEC's build never
  collects, so 1.37 M lifetimes would be live at once), obstacks are not, and both arms carry one fix for a use after
  `pool_free` in `df_insn_rescan`.

### Results: SPEC perlbench test.pl on P1 with SV heads protected (boards 1 and 2, 2026-11-18 13:48-19:02): +5.84 % in-context, output correct

**Configuration.** As pre-registered (cbd14ff6b5c4; the SPEC lane's Addendum 188, `eab935564a65`). Both boots on kit
`nested/kits/bkit91-rec1m` (the transcripts read `node_flat_records=1048576`), P1 resident, the table's stage
(makerand.pl, then test.pl), one arm per boot. `b0-stats` returned 0 in both; the smoke reads 8 PASS with `task`
waived, as in every P1 boot.
- **Protected, board 1** (chain `nested/p1g/b1-pt2.*`, 13:48-17:51): image `f5449ace…` (the stage's packed image
  unpacks to it). Transcript `spec/board/out-nat-p1-pt2-b1/pt2/board.txt`, sha256
  `f04a17a30fadafe917ab3da066474a67683f43066652f1f9f7c01156c295ffa6`, firmware `78bf5afe…`.
- **Stock, board 2** (chain `nested/p1g/b2-st2.*`, 15:11-19:02): the table's image `c1744e87…`. Transcript
  `spec/board/out-nat-p1-st2-b2/st2/board.txt`, sha256
  `9b9feb343f566493ab7c04caa2a1c8864a752f48f32b3ce23a905bfcf6bb0d9e`, firmware `527ef774…`.
- The firmware hashes are not the dry runs' (`8a0a857a…`, `cc1c0bac…`): each bake relinks the kernel (builds #250 and
  #251 in the transcripts).

| test.pl (command 1) | stock, board 2 | protected, board 1 | protected vs stock |
|---|---:|---:|---|
| in-context cycles | 267,194,043,894 | 282,785,227,858 | **+5.84 %** (+15,591.2 M) |
| kernel + other | 44,320,834,142 | 45,788,624,164 | +3.31 % (+1,467.8 M) |
| uptime span, s | 12,796.15 | 13,488.91 | **+5.41 %** |
| record-TLB misses (hpm3) | 24,086,975 | 45,827,521 | +21.7 M (+90 %) |
| walk cycles (hpm4) | 511.2 M | 854.6 M | +343.3 M |
| paged record accesses (hpm5) | 361.3 M | 609.0 M | +247.7 M |
| event-23 replays (hpm7) | 52,723,317 | 96,408,702 | +43.7 M (+83 %) |
| L1D misses (hpm8) | 1,305.6 M | 1,427.0 M | +121.4 M (+9.3 %) |

| makerand.pl (command 0) | stock, board 2 | protected, board 1 | protected vs stock |
|---|---:|---:|---|
| in-context cycles | 6,839,873,599 | 6,766,079,003 | −1.08 % |
| uptime span, s | 276.09 | 273.17 | −1.06 % |

Both commands together (the unit of the SPEC table's perlbench bar): in-context 274,033,917,493 against
289,551,306,861, **+5.66 %**; span 13,072.24 s against 13,762.08 s, **+5.28 %**.

**Against the pre-registration.**
- Answers: **met in both boots.** All commands return 0; `makerand.out` (43,841 bytes) and `test.out` (15,555 bytes)
  read from each transcript are byte-equal to SPEC's `data/test/output` files, and both `.err` files are empty. The
  protected `test.out` ends "All tests successful". Neither console shows a fault line during a command.
- So io/perlio.t, op/inccode.t and op/inccode-tie.t, which the twin could not show, **pass under protection on the
  board.** The twin's full run (`npspec-pt2`) failed exactly those three and no other, as registered.
- Cost of test.pl: **+5.84 % in-context, +5.41 % span.** Predicted +5 % to +30 % in-context: met, at the low edge.
- Stock check: **met.** Stock test.pl here reads −0.01 % in-context and −0.02 % in span against the table's two runs
  on board 3 (267,212.4 M and 267,230.6 M; 12,798.66 s and 12,799.00 s), on a table four times the size.
- makerand: stock here reads +0.09 % against the stock arm of board 1's interleaved boot (6,833.9 M). The protected
  image of this boot (the extended port) reads −1.08 % against stock; the first port read −2.54 %.

**Limits.**
- One run per arm, the arms on different boards and not interleaved. The stock arm repeats the table's runs on
  another board to 0.02 %, which is the evidence that the board is not the difference.
- The kit has 2^20 records, not the table's 262,144: re/pat_psycho.t needs more than 262,144 records under
  protection. Stock is unchanged by the larger table (the stock check above).
- The cost is for the SV-head patch with the lookup repair and the six stale-head sites of the extended port; nothing
  else in the interpreter is protected.

**Reading.**
- Protecting every scalar head of SPEC's Perl costs **+5.8 %** of test.pl's in-context cycles and +5.4 % of its run
  time, with all 314 test scripts giving SPEC's reference output.
- The protected arm misses the record TLB 21.7 M times more and spends 343 M more cycles in walks, 2.2 % of the
  +15,591 M. How the remainder divides between the instructions the patch adds (the arena lookup, CDERIVE, CREVOKE),
  the checks and cache effects is not separated by this pair.
- This replaces suns.pl (+16.05 %) as the Perl cost with the widest coverage: suns.pl is one script on Perl 5.36,
  test.pl is SPEC's own test workload on SPEC's Perl 5.22.

### Record-access traces for the record-cache model (QEMU twin, 2026-11-20)

**Purpose.** The post-submission programme (spec-silicon `7b0e8cd4b52b`, brief 7.4 item 1) decides whether a record
cache goes to RTL by replaying traces of record accesses through cache models (the rtl lane). This lane provides the
traces. Instrument: `capstone/tests/record-trace/` (a patch to the twin QEMU a0501ebc39, a reader, the format).

**Traces**, all on slot 28 of the twin with the 2^20-record table unless stated, each named by the sha256 of the image
its stage holds (`nested/rectrace/traces/INDEX.md` on the capstone work volume, with a `.meta` and `.stats` per trace):

| trace | image | input | words | protected instructions | validation |
|---|---|---|---:|---:|---|
| full-gcc-train01 | 58124bcc… (the paper's Full) | train01 | 1,072,847,462 | 12,221,817,514 | specdiff 1/1 |
| full-mcf-test | fa650b98… (the paper's Full) | test | 3,161,718,784 | >= 26,637,508,524 | specdiff 2/2; tail cut, see below |
| full-perl-test | c1744e87… (the paper's Full) | makerand.pl, test.pl | 9,432,356,686 | 84,333,526,628 | test.out: the twin's 3 known scripts |
| pool-gcc-train01 | 54291a9f… (pool protected, two-operand) | train01 | 1,229,181,700 | 12,321,961,369 | specdiff 1/1 |
| three-gcc-train01 | 5920b9f9… (pools, GC heap, obstacks; 2^22 records) | train01 | 1,383,034,105 | 12,409,649,724 | specdiff 1/1 |
| heads-perl-test | f5449ace… (scalar heads protected) | makerand.pl, test.pl | 10,052,744,671 | 87,962,584,323 | as full-perl-test |

- The mcf file ends at the writer's last full buffer: the runner killed QEMU before its exit handler ran, so at most
  131,071 runs at the end are missing and its `.stats` is a snapshot 1,179,648 words short. The writer now also
  flushes every 2 s (control: the gcc test trace repeated with the new writer equals the old one word for word except
  the all-instruction markers, which include the kernel's wall-clock timer).
- Corrections made on the way, all before any replay used them: a first gcc trace was taken on the nat-fair stage,
  whose binary (a73396ab…) is not the paper's (kept, labelled for sizing only); two queued Full traces were refused by
  the wrapper's own no-overwrite guard and were not running when reported so; the gcc test control runs two images
  (U P P U), not one.
- The rtl lane reports that its reader reproduces every code's events and runs of the `.stats` on the control, perl and gcc traces.
  It reports that its model calibrates on mcf against B5a's silicon and did not on gcc (78.6 % against 62.26 % read hits); the board ran
  FRCDKMS images, not FRCDKMSa3, so two more traces on those images (gcc 3ed5c0be…, mcf 3d7e6211…) are being taken,
  with context-edge markers, which B5a needs to flush its caches.

### Line B record-access traces (format v2) for rcm-lb: B5TL1's mcf, gcc and perlbench (trap-route twin, 2026-11-26)

**Purpose.** The rtl lane's record-cache model for Line B (`rcm-lb`, rtl land/rtl-crevoke4b `b35fb24d8e97`) needs
record-access traces of the campaign's own SPEC test stages (brief 7.4). This lane provides them.

**Tracer.**
- capstone-qemu branch `nested/record-trace-lb` (`a248037c2a..702f24732b`, pushed), binary `a1564de7…`. It writes
  format v2. On Line B it writes no csupctl, because this QEMU has no csupctl CSR.
- Each trace is one twin boot through `<hdd>/nested/rectrace-lb/trace-spec.sh` (`4d669078…`):
  - one B5TL1 stage, copied whole into the overlay;
  - Image `749fed26…`, the keep-tags fw_jump.elf, `sublet.dom_wide_grants=1`, 963M;
  - the global QEMU lock taken shared, with the domain lock.
- rcm-lb is built here from `b35fb24d8e97` (`rcm-lb.c` `f589001a…`, `cc -O2 -Wall`). Its synthetic control PASSes,
  and a one-event perturbation of it FAILs.
- rtl's ruling: LB is read under both walks, `asis` and `--walk dedup`. A threshold inside the bracket is UNDECIDED.

| stage (B5TL1) | image | answer | trace | bytes | sha256 | instructions all / protected | LB check misses asis / dedup | checks | self_evict asis / dedup |
|---|---|---|---|---:|---|---:|---:|---:|---:|
| smoke only (background) | `smoke.dom` | rc 7 | `spec-smoke.2198060` | 4,528 | | 574.6 M / 1,396 | 15 / 15 | 383 | 0 / 0 |
| 505.mcf_r test (board-stages = SP g1b's) | `a1daec395a9e7260` | specdiff PASS | `spec-mcf.2201746` | 65,420,275,952 | `633ca24aabba3062…` | 28.76 G / 26.64 G | 181,127,530 / 181,127,654 | 10,044,639,808 | 124 / 124 |
| 502.gcc_r t1.c (board-stages = SP g3's) | `f800e015d161c911` | specdiff PASS | `spec-gcc.2262833` | 29,076,816 | `0cfc265bfa31dbb2…` | 930.7 M / 21.7 M | 8,921 / 8,921 | 6,525,939 | 0 / 0 |
| 500.perlbench_r makerand.pl (SP-B5TL1-g4mk's stage) | `f3c8c80d71844564` | specdiff PASS | `spec-perl3.2883687` | 10,130,629,600 | `1af7b0c90ab7cb56…` | 5.78 G / 3.09 G | 277,290 / 277,290 | 1,216,156,215 | 0 / 0 |

- Each traced stage directory is `diff -rq`-identical to its source.
- rtl recomputed gcc's trace sha256 and its rcm-lb numbers on dev's build and got the same.
- **Disclosed: a void first perl boot.** The stage path passed was a symlink to an absolute path, and the first
  `trace-spec.sh` copied the link itself. The guest ran a dangling stage (rc 2), the grader read it as MISS, and the trace
  is the smoke background only (`spec-perl.2276518`, 4,736 B).
  - The script now resolves the path and refuses a copy without `stage/run.sh` or with absolute symlinks. That refusal
    is tested on a negative case.
  - The re-run traces the campaign's own perl stage (g4mk, makerand.pl), which is what the Full row pairs, not test.pl.
- The traces were taken on the twin's kernel. So a board row matches them by the stage image, not by the kernel.

### gcc's three allocators with one-operand CREVOKE and CRENEW (QEMU, 2026-11-20): +0.78 % instructions in gcc's code

**Question.** What does the collaborator's port of gcc's pools, GC heap and obstacks (one-operand CREVOKE and CRENEW,
`origin/native-kernel-new-qemu:capstone/ports/spec-cpu2017/sublet/502.gcc_r`) cost on a real input, per allocator?

**Environment.** No kit for the reference model c1985bb529 exists (the SPEC lane confirmed: no revision pins it).
The environment of revision 71ce199d8fea, where the port was verified, was rebuilt on this host (QEMU 9e69171f1c,
the native task route; record of the build and every deviation on the capstone work volume, `nested/oneop/STATUS.md`).
Its control on SPEC's test input reproduces the recorded result in kind: 30 of 30 outputs match, the port costs
+0.63 % (recorded +0.58 %), and the port without the gcc fix faults with cause 25 at the recorded address; absolute
counts are 1.8-1.9 % below the record's, whose executables were built by another compiler or flags.

**Arms**, each rebuilt from the stock build directory with the gcc fix, differing only in which allocator is ported
(sha256 of the executables): fix f9327fbd…, pool 79542acf…, ggc a2a748bd…, obst 92fff9aa…, all dda1118b… (= the port's
own sublet executable), and stock 3ec35baa….

**First reading, withdrawn before use.** Whole-job instruction counts read pool +15.3 %, ggc -6.4 %, obst +4.6 %,
all -0.5 %: not additive, and about 590 instructions per lifetime instruction for the pools. They are kernel work:
user-mode ticks are equal in every arm (1,402-1,413), and system ticks follow the pages the kit's collector sweeps
(pool 3.1 times fix's, ggc 0.09 times). On this kit's table (paged, 30-bit ids) ids carry no generation, so a released
record returns to the free list only after a full stale-tag sweep, and CRENEW cannot renew in place (QEMU 9e69171f1c
cap_rev_table.c:9-12, cap_rev_tree.c:361-371). The GC arm's lower sweep count is consistent with node_ensure's policy
at 71ce199d8fea (capstone_vm.c ~1355-1395: collect only once retired records reach a quarter of the capacity), and is
not traced. A record with generations removes this kernel share; it belongs to the kit, not to the port.

**Result.** Exact counts of gcc's own code (the native image's address window, user mode), by `zones.c`, one boot
per arm, train01, -icount shift=0; every output matches SPEC's reference:

| arm | gcc's code, instructions | against fix | lifetime instructions (counting build) |
|---|---:|---:|---|
| fix | 13,999,404,828 | | |
| pool | 14,013,217,259 | +0.099 % | 2,488,995 CDERIVE, 1,832,637 CRENEW |
| ggc | 14,003,801,664 | +0.031 % | 1,414,198 CDERIVE, 46,579 CREVOKE |
| obst | 14,090,979,945 | +0.654 % | 1,289,643 CDERIVE, 105,684 CREVOKE |
| all | 14,109,183,145 | **+0.784 %** | the sum |
| stock | 14,003,000,543 | +0.026 % | none (no gcc fix) |

- Additive: pool + ggc + obst = +109.8 M, exactly the full port's difference.
- About 3 instructions per lifetime instruction for the pools and the GC heap, about 66 for the obstacks, whose port
  keeps one reference per object and walks them on every free.
- These are instructions on QEMU. The two-operand pool port alone cost +0.8 % instructions on the twin and +5.3 %
  cycles on P1; the cycles of this port need a bitstream with one-operand CREVOKE and CRENEW.

**Next.** The obstack port first (a lifetime per chunk or per free-to-mark region, so that a free ends a subtree in
one CREVOKE), with a fixture that a freed object's reference still faults; then the typed free lists, objects of 4 KiB
and more and growing obstack objects, each priced alone the same way.

### gcc's obstack port refined: +0.654 % to +0.348 % (QEMU, 2026-11-20)

Profiled first, with the protection held fixed. Exact counts of every function of gcc's code (TCG plugin
`capstone/tests/record-trace/zones.c`, one boot per arm on the 71ce199d8fea environment, train01) put the obstack
port's +91.6 M in the out-of-line issue call (+34.9 M, 1.29 M calls), the callers of the obstack macros (about +28 M),
the walk that pops a dropped chunk's references one by one (+11.0 M), and memcpy (+9.9 M). A counting build shows
memcpy's excess to be the reference stack's growth: 2,043 reallocs copied 26,327,552 bytes, 9,872,832 instructions
predicted against +9,944,403 measured. The CREVOKEs themselves are 105,684 in all.

A lifetime per chunk or per free-to-mark region was considered and rejected: an object freed to a mark inside a chunk
that stays live would keep a live reference, and the cost it would remove (the CREVOKEs) is not where the cost is.

Three refinements (`capstone/ports/spec-cpu2017/own-allocators/502.gcc_r/obstack-variant.py`), priced alone and
together against the arm with only the gcc fix:

| arm | gcc's code against fix |
|---|---:|
| obstack port as the collaborator wrote it | +0.654 % |
| a: issue fast path inline | +0.475 % |
| b: O(1) forgetting of a dropped chunk's references | +0.592 % |
| a and b | +0.413 % |
| a, b and c: the reference stack in segments, never copied | **+0.348 %** |
| all three allocators, the collaborator's | +0.784 % |
| all three allocators, obstacks refined | **+0.478 %** |

- Every output matches SPEC's reference. A fixture of six modes (`obstack-fixture.c`) answers the same for the
  collaborator's obstack and every refinement: the controls read, every freed object faults with cause 25, the
  survivors of a partial free read; stock obstack reads the freed objects of an in-chunk free.
- Two slips caught before any reading: b's new chunk field first moved every chunk's first object 4 bytes off its
  alignment (a pad keeps it at 16); the fixture first had fewer objects than one stack segment, so c's segment
  boundary was not exercised (a sixth mode crosses it).
- What remains of the obstacks' +48.7 M is mostly in the callers that now carry the inline path, above all
  bitmap_element_allocate (+23.6 M before c), which serves most calls from its own free list: the typed free lists
  are next.

### Opcode census of the +Spatial images for the spatial-only core (QEMU twin, 2026-11-20)

For the rtl lane: every executed 0x5B instruction counted by code region (plugin `capstone/tests/record-trace/opcount.c`
on the twin a0501ebc39), on the FRCDKMSspa3 images of gcc train01 (3d8bfb5a…), mcf test (17b2fa0b…) and perlbench test
(cdd368de…). The images execute CDERIVE (26,065 / 318 / 163,069) and CREVOKE (22,441 / 276 / 46,242), from the SDK
runtime's `__heap_mapping` and `__heap_unmapping`, although the SDK's heap lifetimes are off; a spatial-only core
must not make them illegal. REVOKE, TIGHTEN, DELIN, SPLIT, MREV, DROP, CALL, RETURN, CSSUPERVISE, CSRETIRE, CSNODE and
LCC with zimm 0 execute 0 times in every image (LDC, the positive control, 1.7-8.1 G). The twin's kernel and firmware
are not the board's (Linux 6.1 against the board's 6.4.14, kit-unq's firmware against the board monitor), so their
counts are a proxy only; the rtl lane reads the board's kernel and monitor statically.

### gcc's own free lists as lifetimes, and a second use after free in gcc (QEMU, 2026-11-20): the port at +0.750 %

On top of the three allocators, the objects gcc keeps on private free lists are renewed (CRENEW) when they go on the
list: bitmap elements and heads, EXPR_LIST and INSN_LIST nodes, C front-end bindings and scopes, sequence stacks, gimple
sequences and phi nodes (scripts in `capstone/ports/spec-cpu2017/own-allocators/502.gcc_r/`). SSA names are left out:
gcc reads released names through old references by design (`SSA_NAME_IN_FREE_LIST`).

| part (train01, gcc's code, against the arm with both gcc fixes) | cost | renewals |
|---|---:|---:|
| pools, GC heap, obstacks (refined) | +0.478 % | |
| bitmap elements and heads | +0.236 % | 4,043,806 |
| EXPR_LIST, INSN_LIST | +0.034 % | 686,221 |
| bindings, scopes, sequence stacks, gimple sequences, phi nodes | +0.001 % | 97,371 |
| **all** | **+0.750 %** | |

- Every output matches. A bitmap fixture shows what the list port adds: an element freed by bitmap_clear_bit or
  bitmap_clear faults with it and reads without it, also with the obstack port in place.
- **A second use after free in gcc 4.5** (the first was df_insn_rescan): `remove_phi_node` calls
  `release_phi_node (phi)`, which puts the node on gcc's free list, then `release_ssa_name (gimple_phi_result (phi))`,
  which reads it through the same reference. Stock survives because nothing reuses the node in between; with the phi
  list ported, train01 faults there (cause 25, after 1.47 G instructions). The fix swaps the two calls; output unchanged,
  -543 instructions; it is in every arm, with a new baseline. Whether either gcc fix is mentioned anywhere outside this
  record is the lead's decision.
- **Objects of 4 KiB and more** (2026-11-20): the GC heap's 1,136 large objects on train01 are all exactly
  representable (powers of two on page boundaries), so they now get lifetimes too (`large-port.py`); obstacks finish
  none on train01. +0.33 M instructions; **the whole gcc port +0.752 %** of gcc's code against the arm with both gcc
  fixes. Growing obstack objects stay without a lifetime of their own until obstack_finish: gcc reads a growing
  object's base in four places (pretty-print.c, reorg.c), none of which runs on train01.

### perlbench test.pl: where the scalar-heads port's instructions go, and an O(1) arena lookup (QEMU twin, 2026-11-21)

**Correction to the post-submission brief, sent to the SPEC lane:** the "+789 M unattributed" were in-context cycles of
Perl 5.36 on suns.pl on P1, not perlbench instructions. On SPEC's test.pl, the paper's footing, the port adds +3,629 M
instructions to perlbench's own code (+4.303 %, every function counted exactly by `capstone/tests/record-trace/
zones-twin.c` on the twin a0501ebc39; stock c1744e87…, protected f5449ace…, both with SPEC's output but the twin's three
known scripts).

- About three quarters is the arena lookup, not lifetime work: the lookup repair's binary search over an address-sorted
  index runs about 15 halvings over test.pl's 26,678 arenas, inside sv_setsv_flags (upstream's two SvIS_FREED tests per
  call, 106 instructions a call), the head release (72), Perl_sv_arena_head (80) and the head issue in the
  constructors (66-80).
- An O(1) lookup with the protection unchanged (`own-allocators/500.perlbench_r/fix-lookup-pages.py`): **+2.894 %**,
  -1,188 M instructions, at +0.86 MB of peak memory on test.pl. A first line that aligned arenas to 4 KiB reached
  +2.859 % but cost +8.56 MB (mallocng's aligned blocks), which the memory runs caught; it was replaced.
- The fixture (`make-heads-fixture.py`, from the build's own perlmain.c, which stays outside the repository; an interpreter embedded from the build's own objects): freed heads fault (cause
  25), live heads read, a second interpreter's heads read while the first one's fault, with the normal table and with
  a 1-slot table; a build without the verify-on-hit faults with cause 24 on live heads.
- Not yet read: cycles. The board's +5.84 % on test.pl (prot2) is the paper's footing; the new image needs its own P1
  pair before any cycle figure is attributed or quoted. The paper is unchanged.
- An instrument defect found on the way: the twin runner's per-process memory sampler matched processes by their
  15-character comm, so it read 0 for perlbench ("perlbench_r_bas"); a fixed copy matches argv[0] and prints NONE when
  nothing matched. No reading in this document used that column.

### Pre-registration: perlbench test.pl on P1, stock and the O(1)-lookup port in one boot (written 2026-11-21 08:10, before the boot)

**Question.** What does the scalar-heads port with the page-indexed arena lookup (prot6) cost perlbench in cycles on
SPEC's test.pl, and where do those cycles go? The twin prices its instructions at +2.894 % against +4.303 % for the
paper's protected image (prot2), whose board cost was +5.84 % in-context (st2/pt2 above).

**Boot.** Board 1 (granted by the SPEC lane after its spatial-only chain, pid 2687886), bitstream P1 (`…-pagedtbl_
d83d3a633`, seal `dc411e36…`; the chain flashes it with the seal check if another bitstream is resident, and reads it
back), kit `nested/kits/bkit91-rec1m` (2^20 records), one booking for the whole boot. Stage `nested/p1g/pair6`
(`pair6.SHA256SUMS`, 574 files): the table's perlbench test stage with its arm section twice, **u** = stock
(c1744e87…, the paper's Full image) then **p** = prot6 (37381b03…), each makerand.pl then test.pl. Both images xz-packed.
Budget 86400 (`BAKED_TIMEOUT` 345600), booking 660 min. Expected about 7.5 h of commands.
- Rule 8 (`nested/p1g/pair6-recheck`): real bake rc 0, `BOARD_STUB=2` on st2's transcript and on its cut rc 0 and 0,
  real firmware 23,760,392 B (`d0ea5bf0…`), the relinked kernel at `node_flat_records=1048576`.
- Twin check of the same stage (`census/out/twin-pair6`) must show both arms' makerand.out equal to SPEC's and test.out
  as stock's (the twin's three known scripts) before the chain is armed.
- When the boot ends, the chain touches `perf/natstage/perlpair-b1.DONE` (the SPEC lane's gate).

**Outcomes.**
- VOID as for every P1 boot. Answers: every command rc 0; makerand.out and test.out of both arms byte-equal to SPEC's
  references, else no cost is read.
- Stock check: u's test.pl in-context cycles against st2's 267,194.0 M, expected within 1 %. Outside it, the cost is
  still read within this boot but the cross-boot comparisons are not.
- **Cost of test.pl, p against u, in-context cycles. Predicted +3.5 % to +5.5 %**: the twin's 1,188 M fewer
  instructions than prot2, at the stock arm's 3.17 cycles per twin instruction, are 3.8 G cycles, -1.4 points from
  prot2's +5.84 %; the record traffic is unchanged.
- Readings for the cycle attribution (step 2): hpm3 record-TLB misses, hpm4 walk cycles (taken as cycles directly),
  hpm5 paged accesses, hpm7 event-23 replays, hpm8 L1D misses, each p minus u; the twin's instructions p minus u. The
  per-event prices used will each name their source (a measured pair or a model), and the remainder is stated as a
  number.
- Limits stated in advance: one run per arm, the arms in a fixed order in one boot (u first), not interleaved.

### Results: perlbench test.pl on P1, stock and the O(1)-lookup port in one boot (board 1, 2026-11-22 00:47-08:32): +5.08 % in-context, output correct

**Configuration.** As pre-registered (b50563709f73). Chain `nested/p1g/b1-pair6.*` on board 1; the chain flashed P1
with the seal check and read it back resident (`…-pagedtbl_d83d3a633.bit`). Kit `nested/kits/bkit91-rec1m`; the
transcript reads `node_flat_records=1048576`. Transcript `spec/board/out-nat-p1-pair6-b1/pair6/board.txt`, sha256
`79b7b65f49e46d694b35f8655bbb8b2a063e3b5897f4a7450aef3d13e4300e87`, firmware `0edfee07…`, kernel build #280.
`b0-stats` returned 0. The smoke reads PASS with `task` waived, as in every P1 boot. The watchdog sampled throughout
(1,449 lines).
- The kit's `spec-classify.py` printed "NO RESULT: no @@SPEC-DONE" for this boot, as for st2 and pt2. It expects
  `@@SPEC-DONE <count>`, and the table's stage prints `@@SPEC-DONE 500.perlbench_r test`. All four commands began and
  ended. The readings below come from the transcript itself (`read-pair.py`, a scratch reader). It reproduces st2's
  267,194,043,894 and pt2's 282,785,227,858 to the cycle.

| test.pl (command 1) | u = stock c1744e87 | p = prot6 37381b03 | p vs u |
|---|---:|---:|---|
| in-context cycles | 267,219,156,375 | 280,802,971,216 | **+5.08 %** (+13,583.8 M) |
| kernel + other | 44,339,795,047 | 45,829,237,769 | +3.36 % (+1,489.4 M) |
| uptime span, s | 12,798.09 | 13,410.64 | **+4.79 %** |
| record-TLB misses (hpm3) | 24,091,058 | 45,536,269 | +21.4 M (+89 %) |
| walk cycles (hpm4) | 508.8 M | 817.9 M | +309.1 M |
| paged record accesses (hpm5) | 361.3 M | 608.6 M | +247.3 M |
| event-23 replays (hpm7) | 52,729,351 | 96,018,225 | +43.3 M (+82 %) |
| L1D misses (hpm8) | 1,305.7 M | 1,416.2 M | +110.5 M (+8.5 %) |

makerand.pl (command 0): in-context 6,820,045,701 against 6,598,343,477, −3.25 %; span 275.29 s against 266.40 s.
Both commands together: in-context +4.88 %, span +4.62 %.

**Against the pre-registration.**
- Answers: **met.** All four commands return 0. Both arms' `makerand.out` (43,841 bytes) and `test.out` (15,555
  bytes), decoded from the transcript, are byte-equal to SPEC's `data/test/output` files; all four `.err` files are
  empty. The console shows no fault line.
- Stock check: **met.** u's test.pl reads +0.009 % against st2's 267,194.0 M, so the cross-boot comparisons stand.
- Cost of test.pl, p against u: **+5.08 % in-context. Predicted +3.5 % to +5.5 %: met**, in the upper half.
- Against prot2 (pt2, board 1, 2026-11-18): pt2 reads +5.83 % against this boot's u. The O(1) lookup therefore saves
  1,982.3 M cycles. The prediction priced its 1,188,489,807 fewer twin instructions at 3.8 G. The pair measures
  **1.67 cycles per removed instruction** (cross-boot; both stock checks met), not the stock arm's average of 3.17.
  The removed instructions are the arena index's binary search; that they cost about half the program's mean is read,
  why is not (cache hits and a predictable loop would explain it, and this boot does not show either).
- The record counters are unchanged from prot2, as predicted: hpm3 +21.4 M against +21.7 M, hpm5 +247.3 M against
  +247.7 M, hpm7 +43.3 M against +43.7 M.

**Cycle attribution (step 2), p minus u, +13,583.8 M in-context cycles.**

| part | cycles | share | price and its source |
|---|---:|---:|---|
| the port's instructions | 7,733.2 M | 56.9 % | +2,440,563,172 twin instructions (zone, prot6 minus stock) at 3.169 cycles each, the u arm's own cycles per twin instruction (a model: the mean) |
| record-table walks | 309.1 M | 2.3 % | hpm4, read as cycles (measured) |
| event-23 replays | 735.9-865.8 M | 5.4-6.4 % | +43.3 M at 17-20 cycles each (Addendum 1's RTL estimate, from source) |
| **not attributed** | **4,675.8-4,805.6 M** | **34.4-35.4 %** | |

- The L1D misses (+110.5 M) are not priced: no measured price exists. If they carried the whole remainder, each
  would cost 42-43 cycles.
- **prot2's +15,591.2 M (pt2 against st2, the paper's image), the brief's attribution item.** The same reading of
  that pair puts 11,497.9 M (73.7 %) in its +3,629,052,979 twin instructions at st2's mean (3.168), 343.3 M (2.2 %)
  in walks and 742.7-873.7 M (4.8-5.6 %) in event-23 replays. That leaves 2,876-3,007 M (18.4-19.3 %) unattributed.
  Both images make the same record traffic (the protected arms' hpm3, hpm5 and hpm7 counts within 1 % of each other), yet the remainders differ by 1.8 G. So
  the mean price is wrong for at least the lookup's instructions. The decomposition this boot supports is the
  difference itself: 1,982.3 M (12.7 % of prot2's cost) is the binary-search lookup prot6 removes, measured as a pair.
  The other 13,583.8 M is prot6's cost, divided as in the table above with its 34-35 % remainder.
- **The instruction price decides the remainder, and it is the softest number here.** At the mean (3.17) the
  instructions take 57 %. At the price this boot's pair measured for the lookup's instructions (1.67), they take
  4,070.6 M (30.0 %), and the remainder grows to 8,338-8,468 M (61.4-62.3 %). Neither price is measured for the
  instructions that stay: the CDERIVE/CREVOKE pairs and the head checks. No board run has priced those
  instructions by themselves.
- The paged table's cost beyond its walks is not separated by this boot. That cost is its effect on the program's L1D
  and on the record TLB's reach. A run that would separate it is the same image on a bitstream with a larger record
  TLB, or the same image with the checks compiled out and the lifetimes kept.
- Limits, as registered: one run per arm, u first, not interleaved. makerand reads −3.25 % (p faster), as earlier
  protected images read −1.08 % and −2.54 %; it is not explained, and an order effect on a 270 s command is not
  excluded.
- The paper is unchanged. The paper's perlbench figure is prot2's +5.84 %; a change to prot6's +5.08 % is the lead's
  decision.

### perlbench test.pl: SV bodies as lifetimes on top of the heads port (QEMU twin, 2026-11-22): +0.733 % of perlbench's code

The second of the three perlbench allocators (heads, then bodies, then hash entries and op slabs; each one priced on
its own). `500.perlbench_r/port-sv-bodies.py` runs after the heads port and both lookup fixes, under
`PERL_CAPSTONE_SUBLET_BODIES`:
- Every body that an SV type keeps in an arena becomes a child lifetime bounded to the body (CDERIVE when
  `new_body_inline` pops it), and it is revoked when the body is freed (CREVOKE in `del_body`, from `sv_clear` and
  `sv_upgrade`).
- The arenas are carved from 32 MiB segments, each one Newx block with an aligned interior as the segment's lifetime.
  A body's parent is found by comparing its address with the few segments, so there is no arena lookup.
- The free lists hold the segment's own pointers, and SvANY keeps upstream's offset. For the PV types it points 32 B
  below the body, inside the representable window.
- Hash entries keep upstream's arenas, so this arm prices bodies alone.

Checks:
- **Control.** `sv.c` compiled without the new define is byte-identical to prot6's `sv.o`. The script, run on prot6's
  `sv.c`, reproduces the built file, and `sv.c` is the only source that differs.
- **Fixture** (`make-bodies-fixture.py`, generated from the build's own perlmain.c, which stays outside the
  repository), run on the twin (`census/out/perlfxb-2d7a7d4b`):
  - in the protected build a live PV body reads (mode 0). A freed body (1), a body freed by an upgrade to PVMG (2) and
    a freed body among 5,000 PVs and 500 AVs (3) fault with cause 25, after the 4,999 other PVs read their own lengths
    and the 500 AVs their sizes. A body of a destroyed interpreter faults after a second interpreter's 3,000 bodies read (4);
  - every fault is at the fixture's own `ld a0, 0x20(a0)` through the kept pointer, which is the read of `xpv_cur`;
  - the control built with `PERL_CAPSTONE_SUBLET_BODIES_NOREVOKE` reads in modes 0-3, returning whatever the freed
    body holds.
    It faults in mode 4 too: freeing a segment ends every lifetime carved from it, so mode 4 has no control that
    separates the per-body revocation. The fixture's docstring first said the control reads in every mode. That was
    corrected before this commit.
- **test.pl** (`census/out/zperl-bodies-279647af`, image `279647af…`): makerand.out and test.out are byte-equal to the
  stock twin run's.
  - Image zone: 87,410,533,001 instructions against prot6's 86,774,090,874, **+636,442,127 (+0.733 %)**. Against stock's
    84,333,527,702 that makes +3.649 % for heads and bodies together, against +2.894 % for heads alone.
  - Where it goes: `Perl_sv_upgrade` +286.5 M and `Perl_sv_clear` +227.4 M, the inlined issue and retire. Then
    `__heap_mapping` +90.5 M, which is one 32 MiB segment per process with an index leaf provisioned for each of its
    pages, about 33 instructions a page over 318 processes. Then `memset` +12.4 M, `Perl_more_bodies` +11.2 M,
    `nontrivial_free` +10.0 M and `__heap_unmapping` +7.2 M.
- **Calls to `Perl_more_bodies`: 44,019, against prot6's 5,077.** A diagnostic build counted them per type
  (`census/out/zperl-bodcnt-3a458858`, not committed). All 5,077 are hash-entry arenas (type 0) in both images, and
  the other 38,942 are SV body arenas. Most likely clang inlined prot6's 246-byte `Perl_more_bodies` into its callers
  in `sv.c`, and the carve, at 524 bytes, stopped that; this is inferred from the sizes and was not read from the
  disassembly. The same build saw one segment per process and no body issued or retired outside a segment.
- **Memory** (`census/out/rss-bodies-279647af`): peak used 102,464 kB: -156 kB against prot6's 102,620 kB, and
  +4.6 % against stock's 97,968 kB. With the peak unchanged, the segments' untouched pages are evidently not
  resident; that is inferred, not counted.
- Not yet read: cycles. A board pair is needed first, as for prot6.

### perlbench test.pl: hash entries as lifetimes on top of the heads port (QEMU twin, 2026-11-22): +0.112 % of perlbench's code

The third perlbench allocator, priced alone like the bodies. `500.perlbench_r/port-he.py` runs after the bodies port
and is built with `PERL_CAPSTONE_SUBLET_HE` and without `PERL_CAPSTONE_SUBLET_BODIES`:
- The hash-entry arenas (`PL_body_roots[HE_SVSLOT]`) are carved from the bodies port's 32 MiB segments; SV bodies keep
  upstream's arenas.
- `hv.c`'s `new_he` hands each entry out as a child lifetime bounded to the `HE` (CDERIVE), and `del_HE` revokes it
  (CREVOKE) before it links the entry through the segment's own pointer. Shared string table entries are `HE`s too and
  are covered the same way.
- `hv.c` reaches `sv.c`'s segment table through `capstone_sv_segs.h`, which the script writes.

Checks:
- **Control.** In the HE tree, `hv.c` and `sv.c` compiled without the new define are byte-identical to prot6's `hv.o`
  and `sv.o`.
- **Fixture** (`make-he-fixture.py`, from the build's own perlmain.c; `census/out/perlfxh-b7bb6577`):
  - in the protected build a live entry reads its value (mode 0). The old entry faults with cause 25 after its key is
    deleted (1), after the hash grew from 11 to 5,000 keys and the key was deleted (2: every key fetched its own value
    first and the kept entry still read), after `hv_clear` (3), and after the hash itself was freed (4);
  - every fault is at the fixture's own `ldc a0, 0x20(a0)` in `main`, the read of `he_valu` through the kept pointer;
  - the control built with `PERL_CAPSTONE_SUBLET_BODIES_NOREVOKE` reads in all five modes, mode 4 included: freeing a
    hash frees its entries one by one, while the segment stays, so here every mode has a control.
- **test.pl** (`census/out/zperl-he-e9994b2e`, image `e9994b2e…`): makerand.out and test.out are byte-equal to the
  stock twin run's.
  - Image zone: 86,871,309,097 instructions against prot6's 86,774,090,874, **+97,218,223 (+0.112 %)**. Against stock's
    84,333,527,702 that makes +3.009 % for heads and hash entries together.
  - Where it goes: `__heap_mapping` +86.2 M, one segment per process as in the bodies arm. Then the issue and retire:
    `Perl_hv_common` +33.5 M, `S_hv_free_ent_ret` +22.4 M, `Perl_refcounted_he_chain_2hv` +7.3 M. Then
    `Perl_more_bodies` +10.0 M, `Perl_sv_force_normal_flags` +8.4 M.
  - Against those, `Perl_sv_upgrade` -53.5 M, `Perl_sv_setsv_flags` -12.4 M and `Perl_sv_arena_head` -5.6 M. As in the
    bodies arm, `Perl_more_bodies` is now called 43,701 times instead of 5,077, and `Perl_sv_upgrade` shrank from
    1,604 to 1,306 bytes, so the arena code most likely left it; that explains about 10 M of the 53.5 M (inferred from
    the sizes and the calls, not read from the disassembly). The rest of these decreases is not attributed.
  - The other functions add +0.9 M (858,740), and 15,681 instructions fall outside every named function.
- **Memory** (`census/out/rss-he-e9994b2e`, test.out byte-equal to stock's): peak used 102,504 kB: -116 kB against
  prot6's 102,620 kB, and +4.6 % against stock's 97,968 kB.
- Not read: cycles.

### perlbench test.pl: ops as lifetimes of their op slabs, on top of the heads port (QEMU twin, 2026-11-22)

The last of the perlbench allocators, priced alone. `500.perlbench_r/port-op-slabs.py` runs after the heads port and
both lookup fixes, under `PERL_CAPSTONE_SUBLET_OPS`:
- Each op slab carries a lifetime of its own memory (`opslab_life`, a new field after `opslab_refcnt`, so no other
  field moves and no slot grows).
- `Perl_Slab_Alloc` hands each op out as a child of the slab that holds it (CDERIVE), bounded to its whole slot,
  header included, up to the next slot: Perl reads the slot header through the op, and grows a LISTOP into a LOOP in
  place when the slot has room.
- `Perl_Slab_Free` revokes the child (CREVOKE) and links the slot through the slab's own pointer, so the free chain
  holds only own pointers and a reused slot is handed out as a new child.
- The holding slab is found by walking the CV's slab chain and comparing the op's address with each slab's lifetime.
- `opslab_force_free` frees the live ops of an unfinished CV through the slabs' own pointers. Those are recognised by
  their bounds (the whole slab) and not revoked: their lifetimes end when the slab is freed.

Checks:
- **Control.** Of the build's 69 objects, 68 are byte-identical to prot6's, and `op.c` compiled without the new
  define is byte-identical to prot6's `op.o`.
- **Fixture** (`make-ops-fixture.py`, from the build's own perlmain.c; `census/out/perlfxo-a3d4f121`). The ops are
  made with `newOP` while `PL_compcv` is a fresh CV, so they come from that CV's slabs, and are read through a kept
  `OP*`:
  - in the protected build a live op reads its pp function (mode 0). A freed op faults with cause 25 (1), as does a
    freed op whose slot was reused at the same address, after the new op and a neighbour read (2), and a freed op
    among 3,000 over the slabs' growth, after the other 2,999 read (3). An op of a CV freed with its ops still live
    faults too (4);
  - every fault is at the fixture's own `ldc a0, 0x20(a0)` in `main`, the read of `op_ppaddr` through the kept
    pointer. None is in `Perl_Slab_Free`, so in mode 4 the own-pointer path took the slabs' own pointers and not a
    child;
  - the control built with `PERL_CAPSTONE_SUBLET_OPS_NOREVOKE` reads in modes 0-3 and faults in mode 4: freeing a
    slab ends every op lifetime carved from it, revoked or not, so mode 4 has no control that separates the per-op
    revocation, as in the bodies arm.

**Result, revision 1 on test.pl (twin, `census/out/zperl-ops-b296436b`, image `b296436b…`): the outputs differ from
stock's, so it is not priced.**
- makerand.out equals the twin stock arm's. test.out does not: three tests the stock arm passes print "FAILED--no
  leader found" (comp/package_block, uni/opcroak, uni/package). The twin's stock arm already fails io/perlio and
  op/inccode(-tie), as every twin arm does (prot6, HE and bodies each equal stock's test.out byte for byte), so the
  reference for this check is the twin stock arm, not SPEC's file.
- The console shows three faults with cause 25 during test.pl. With the image mapped at 0x3ff7c00000 = vaddr 0x10000,
  two are at `Perl_op_free`+0x60 (`lhu a0, 0x38(s1)`, the `o->op_type == OP_FREED` test at op_free's head) and one at
  +0xe6 (`lhu a0, 0x38(s7)`, the same test on a kid in the kid loop).
- Perl reads those ops after freeing them by design. op.c says so above both reads: "During the forced freeing of ops
  after compilation failure, kidops may be freed before their parents." All three tests compile string evals that
  fail (`eval q[package Foo {]`, a prototype croak). So this is a defect of the port: a freed op's type and sibling are
  read through its revoked child. It is not a fault in Perl. The fixture did not catch it, because no mode let a
  compilation fail.
- Revision 2 (`port-op-slabs.py`, this commit) reads those two fields through the slab's own pointer, and only while
  `opslab_force_free` frees the chain that holds the op. Every other read of a freed op still goes through its child.
  The fixture gains mode 5: three string evals that fail to compile, which must return with no fault. Revision 1 must
  fault in mode 5, which is the positive control.
- Revision 2's image is `b75cbd06…` (unstripped `06ab5518…`). Its op.h equals revision 1's, and its op.c differs from
  revision 1's only in the hunks above (46 diff lines). The other 68 objects are revision 1's own files.
- Revision 1's zone is not used, because three of test.pl's tests ended early. Its memory run was stopped while it
  waited for the slot (`rss-ops-b296436b-void-stopped`).

**Result, revision 2's fixture (`census/out/perlfxo2-5a1b1473`; revision 1, revision 2 and revision 2's no-revoke
control, each in modes 0-5): revision 2 is not a fix, and mode 5 had no working control.**
- Modes 0-4 read as in revision 1's fixture in all three builds: mode 0 reads; modes 1-4 fault with cause 25 at
  `main`+0x206, +0x370, +0x406 and +0x74e, the fixture's own `ldc a0, 0x20(a0)`. The control reads in modes 1-3, with
  `read_after_free` printed, and faults in mode 4 at `main`+0x74e.
- Mode 5, revision 1: cause 25 at `Perl_op_free`+0x60 (`lhu a0, 0x38(s1)`). That is test.pl's fault, so the positive
  control fires.
- Mode 5, revision 2: cause 25 at `Perl_op_free`+0x98 (`lhu a0, 0x38(a1)`). This is op_free's type test, reached with
  a1 equal to the op's own child. Either no `opslab_force_free` was running, or the chain it was freeing did not hold
  the op. Perl frees ops after a failed compilation outside that window too: op_free's comment also says "ops may be
  freed twice". So the guard had the wrong scope.
- Mode 5, no-revoke control: cause 24 at `Perl_sv_setsv_flags`+0xc2 (`lw a0, 0x14(a1)`). That is the heads port's head
  read when the destination SV is in no arena, on an untagged pointer. None of the three builds printed mode 5's
  `returned` line. A control that faults is not a control, so until mode 5 returns in a build without the ops port, the
  two mode-5 faults above are evidence about op_free's pc and nothing more.
- The memory run of revision 2 was refused before it started (`census/out/rss-ops2-b75cbd06` holds a placeholder).
  Its test.pl run, `zperl-ops2-b75cbd06`, was already running and is kept as data for revision 3.
- **Revision 2 on test.pl** (`zperl-ops2-b75cbd06`): makerand.out equals the twin stock arm's. test.out does not:
  - uni/opcroak now passes;
  - comp/package_block and uni/package still print "FAILED--no leader found";
  - the console shows two faults with cause 25, both at `Perl_op_free`+0x98 (`lhu a0, 0x38(a1)`, vaddr 0x5572a in
    revision 2's image). That is mode 5's pc in revision 2's fixture.
  So the fixture's mode 5 reproduces what test.pl hits, though its own control is still to be shown working. The zone
  (87,136,450,735) is not used.

**Revision 3 (`port-op-slabs.py`, this commit) and its fixture (`perlfxo3-3d40f11b`, queued).**
- Every op slab enters an index sorted by address: in `S_new_slab`, with the slab's own address range (not its
  lifetime's bounds, which may be rounded outward); in `opslab_free`, it leaves before the free.
- op_free's two reads of a possibly freed op, the type test and a kid's type and sibling, find the slab by binary
  search. The search reads no slab and not the op. The two reads then go through the slab's own pointer, in every
  context. Revision 2's `opslab_force_free` window is gone.
- Every other read of an op still goes through its child:
  - a stale op whose slot was reused passes the type test, then faults at op_free's next read. That is the protection
    firing;
  - a slab already freed is not in the index, so the read goes through the child and faults.
- Image `aba18d33…` (unstripped `74d8c86a…`). op.h equals revision 1's. op.c differs from revision 1's by 89 diff
  lines, and its warnings equal revision 2's (24).
- The fixture gains mode 6: three string evals that compile and run, after the same steps as mode 5. Every build must
  return in mode 6, which shows the fixture can run an eval at all.
- A fifth build links the fixture against prot6's objects, with no ops port and a main compiled without the ops
  define. It is mode 5's reference: it must return in modes 5 and 6. A fault there means mode 5 itself is broken.
- The fixture's commands: prot6 in modes 1, 5 and 6; revisions 1 and 2 in modes 5 and 6; revision 3 and its no-revoke
  control in modes 0-6.
- Then test.pl priced (`zperl-ops3-aba18d33`). The memory run (`rss-ops3-aba18d33`) runs only if test.out and
  makerand.out equal the twin stock arm's (`census/chain-ops3.sh`). Before launch, that gate was run on recorded
  pairs: prot6 reads equal, revision 1 differs, a missing run differs.

**Result, revision 3's fixture (`census/out/perlfxo3-3d40f11b`, 21 commands): modes 5 and 6 are broken in the fixture
itself, as the pre-registered branch above allowed ("a fault there means mode 5 itself is broken").**
- prot6, with no ops port, faults in mode 5 and in mode 6, with cause 24 at `Perl_sv_setsv_flags`+0xc2
  (`lw a0, 0x14(a1)`, the head read when the destination SV is in no arena). So does every other build in mode 6
  (revisions 1, 2 and 3 and the control), and revision 3 and the control in mode 5, all at that same instruction.
- Cause: the fixture booted the interpreter from `-e 0`. For `-e`, perl opens /dev/null as the script
  (`S_open_script`, perl.c:3800). The twin's sandbox refuses it, and every command's stderr reads
  "Can't open /dev/null: Permission denied". That croak ends `perl_parse` inside `open_script` (perl.c:2162), before
  `init_predump_symbols` (:2229) and `init_postdump_symbols` (:2234). The fixture never read `perl_parse`'s status. So
  every mode ran on a half-built interpreter. Modes 0-4 only make and free ops, and they read as designed.
- The other readings, for the record:
  - prot6 mode 1 reads the freed op (`read_after_free` printed, rc 0): no protection without the port, as expected;
  - revision 3, modes 1-4: cause 25 in `main` at +0x210, +0x424, +0x37c and +0x48a, each the fixture's own
    `ldc a0, 0x20(a0)`. Mode 2 printed `same=1 o3_ok=1 o2_ok=1` and mode 3 `live_ok=2999 bad=0` before the fault;
  - the control reads in modes 1-3 and faults in mode 4 at `main`+0x48a, as before;
  - in mode 5, revisions 1 and 2 still fault in `Perl_op_free`, at +0x60 and +0x98 with cause 25, before the eval
    reaches `sv_setsv_flags`. Revision 3 no longer faults in op_free; it reaches the fault that prot6 hits.
  That last point is a reading on a broken interpreter, not a result.

**Fixture revision 2 (`make-ops-fixture.py`, this commit) and its run, pre-registered (20:27).**
- The fixture writes `fx-boot.pl` (`1;`) in its working directory through PerlIO, then runs `perl_parse` on that file
  and `perl_run`. If either returns non-zero it prints `FIXTURE boot failed` and exits 3. Mode 7 boots from `-e 0`
  instead, as revision 1 did: that is the boot check's positive control.
- Five builds, objects unchanged (`perlspec/fx2/fixture-ops3b`, built by the same link lines as revision 1's
  builds): prot6 `8d1ba63c…`, revision 1 `19d3f29d…`, revision 2 `45cd5565…`, revision 3 `c81dca75…`, the control
  `7d89f2f1…`. 24 commands: prot6 in modes 0, 1, 5, 6 and 7; revisions 1 and 2 in modes 5 and 6; revision 3 in modes
  0-7; the control in modes 0-6. The run is `perlfxo3b-c81dca75`, after the queued runs on slot 28.
- Predictions:
  - mode 7 prints `FIXTURE boot failed` and exits 3 in prot6 and revision 3. No other command prints it;
  - prot6: mode 0 reads, mode 1 reads after the free, mode 5 returns with `errors=3`, mode 6 with `errors=0`;
  - revision 1: mode 5 faults in `Perl_op_free` with cause 25 (mode 5's positive control); mode 6 returns `errors=0`;
  - revision 2: mode 5 faults in `Perl_op_free` with cause 25; mode 6 returns `errors=0`;
  - revision 3: mode 0 reads; modes 1-4 fault with cause 25 in `main`; mode 5 returns `errors=3`; mode 6 `errors=0`;
  - the control: modes 0-3 read; mode 4 faults in `main`; mode 5 returns `errors=3`; mode 6 `errors=0`.
- Branches:
  - mode 7 does not fail the boot: the check cannot fire, and the boot lines of the other commands are not evidence;
  - prot6 faults in mode 5 or 6: the fixture is still broken there, and modes 5 and 6 decide nothing;
  - prot6 returns in both and revision 3 faults in mode 5: revision 3 is insufficient, whatever test.pl reads;
  - revision 1 returns in mode 5: mode 5 does not reproduce test.pl's fault, and a return from revision 3 there
    proves nothing;
  - any of modes 0-4 reads differently from revision 3's fixture above: the half-built interpreter had changed them,
    and those modes are re-read from this run.

**Result (`census/out/perlfxo3b-c81dca75`, 24 commands): every prediction held. On the fixture, revision 3 fixes
the failed-compilation shape and keeps the protection.**
- The boot check fires: mode 7 prints `FIXTURE boot failed parse=13 run=0` and exits 3 in prot6 and in revision 3.
  No other command prints it.
- prot6: mode 0 reads; mode 1 reads after the free (`read_after_free` printed, rc 0); mode 5 returns `errors=3`,
  mode 6 `errors=0`. So mode 5 has a working reference.
- Revision 1 faults in mode 5 with cause 25 at `Perl_op_free`+0x60, and revision 2 at `Perl_op_free`+0x98, the pcs
  test.pl hit. Both return `errors=0` in mode 6. So mode 5's positive control fires, and the fault needs a failed
  compilation.
- Revision 3: mode 0 reads; modes 1-4 fault with cause 25 in `main` at +0x328, +0x53c, +0x494 and +0x5a2, each the
  fixture's `ldc a0, 0x20(a0)` (mode 2 printed `same=1 o3_ok=1 o2_ok=1` first, mode 3 `live_ok=2999 bad=0`, mode 4
  `cv_freed`); mode 5 returns `errors=3`; mode 6 `errors=0`.
- The control reads in modes 1-3 (`read_after_free` each time), faults in mode 4 at `main`+0x5a2, and returns in
  modes 5 and 6.
- Modes 0-4 read as on the half-built interpreter, so the earlier readings of those modes stand.
- What this does not show: that test.pl passes. That is `zperl-ops3-aba18d33`.

**Revision 3 on test.pl (`census/out/zperl-ops3-aba18d33`, image `aba18d33…`): outputs equal stock's, at +4.775 %.**
- test.out and makerand.out are byte-equal to the stock twin run's (`cmp`; specdiff 1/2, as for stock, prot6 and
  bodies). The console has no `capstone_task: fault` line. comp/package_block, uni/package and uni/opcroak pass.
- Image zone 90,917,822,656: **+4,143,731,782 (+4.775 %) against prot6**, +7.807 % against stock for heads and ops.
  Per function against prot6 (the named functions sum to +4,143,731,347):
  - `memmove` +2,887.3 M, at 709,488 more calls (20,478,129 to 21,187,617), about 4,070 instructions per added call;
  - `Perl_op_free` +792.3 M at the same 2,683,185 entries, about 295 instructions per call;
  - `Perl_Slab_Free` +291.7 M (4,022,039 entries, 0 in prot6), `Perl_Slab_Alloc` +133.6 M at the same entries,
    `S_new_slab` +57.0 M (413,912 entries, absent from prot6's counts).
- Reading. The index is a sorted array: `S_op_slab_register` and `S_op_slab_unregister` shift it with `Move` (a
  `memmove`) on every new and freed slab, and op_free's two lookups are a binary search. 413,912 slabs registered
  and unregistered allow up to 827,824 such moves, against 709,488 added calls. So the `memmove` growth is most likely
  the index's shifts, over arrays of some thousands of live slabs. That is inferred from the counts: no caller of
  `memmove` was read. `Perl_Slab_Free` and `S_new_slab` reporting entries only in revision 3 most likely means clang
  stopped inlining them, which was not read from the disassembly either.
- So revision 3 is correct on test.pl and the fixture, and its index costs most of its price. The memory run
  (`rss-ops3-aba18d33`) was started by the chain's gate. The next revision replaces the sorted array.
- **Memory** (`census/out/rss-ops3-aba18d33`, test.out and makerand.out byte-equal to stock's, END 01:02): peak used
  102,464 kB, -156 kB against prot6's 102,620 kB and +4.6 % against stock's 97,968 kB (2 samples matched). It is the
  bodies arm's peak to the kilobyte, which at this granularity says only that neither port moves the peak.

**Revision 4 (`port-op-slabs.py`, this commit): the index by page, pre-registered (22:27).**
- The index has an entry for each 4 KiB page a slab's memory touches: `{slab, page, end}` in an open-addressing table
  (linear probing; a deletion shifts the cluster back, so there are no tombstones; the table doubles at half load).
  A slab enters at S_new_slab and leaves at opslab_free, one entry per page, at most 9 for the largest slab
  (2,048 pointers of 16 bytes). op_free's two lookups hash the op's page and scan its cluster. As in revision 3, a
  lookup reads no slab and not the op, and the two reads go through the slab's own pointer.
- Image `26dfe710…` (unstripped `d2db8e9c…`), symbols `census/perl-ops4.syms`. op.h equals revision 1's. op.c
  differs from revision 3's by 142 diff lines, the index only. Warnings 24, as in revision 3. op.o has one CREVOKE and
  six CDERIVE, as revision 3's. The no-revoke control's op.o has no CREVOKE and six CDERIVE.
- Fixture (`perlfxo4-26dfe710`, `perlspec/fx2/fixture-ops4`; fixture revision 2's main): prot6 in modes 5 and 7,
  revision 1 in mode 5, revision 4 in modes 0-7 (`ad5f528f…`), its control in modes 0-6 (`7814d89f…`). 18 commands.
- Then test.pl priced (`zperl-ops4-26dfe710`). Then the memory run (`rss-ops4-26dfe710`), only if test.out and
  makerand.out equal the stock twin run's (`census/chain-ops4.sh`). Before launch, a copy with the runs stubbed
  took each gate branch on recorded results: revision 3's (equal: memory run), revision 1's and a missing directory
  (no memory run).
- Predictions:
  - fixture: as revision 3 on fixture revision 2. prot6 returns `errors=3` in mode 5 and fails the boot in mode 7.
    Revision 1 faults in `Perl_op_free` in mode 5. Revision 4 reads in mode 0, faults with cause 25 in `main` in
    modes 1-4, returns `errors=3` in mode 5 and `errors=0` in mode 6, and fails the boot in mode 7. The control reads
    in modes 1-3, faults in mode 4, and returns in modes 5 and 6;
  - test.pl: outputs byte-equal to stock's, and no capability fault on the console;
  - zone: from +0.8 % to +2.0 % over prot6's 86,774,090,874. That is revision 3's +4.775 % less its `memmove` growth
    and most of op_free's search;
  - `memmove` within 5 % of prot6's 1,037,644,188 instructions.
- Branches:
  - `memmove` still more than 5 % above prot6's: revision 3's growth was not the sorted array's shifts, and that
    reading is withdrawn;
  - the zone above +2.0 % with `memmove` back: the remaining price is the carve and the revocation, and it is
    reported per function as it stands;
  - any fixture reading other than predicted, or test.pl's outputs not equal to stock's: revision 4 is not used, and
    the hash table is the suspect, since it is the only change from revision 3.

**Result, revision 4's fixture (`perlfxo4-26dfe710`, END 01:03): every one of the 18 readings as predicted.**
- prot6: mode 5 returns `errors=3`. Mode 7 fails the boot (`parse=13`, rc 3), so the boot check fires.
- Revision 1: mode 5 faults with cause 25 at `Perl_op_free+0x60`, the positive control.
- Revision 4:
  - mode 0 reads (`live ok=1 slabbed=1`);
  - modes 1-4 fault with cause 25 in `main`, at `main+0x328`, `+0x53c`, `+0x494` and `+0x5a2`. Each is the fixture's
    own read through a freed op's pointer;
  - mode 5 returns `errors=3`, mode 6 `errors=0`, and mode 7 fails the boot.
- The no-revoke control reads freed ops in modes 1-3 (`read_after_free`, rc 0), faults in mode 4 at `main+0x5a2`,
  and returns in modes 5 and 6.
- So the page index refuses what revision 3's sorted array refused, on the same fixture. test.pl
  (`zperl-ops4-26dfe710`) is running.

**Revision 4 on test.pl (`census/out/zperl-ops4-26dfe710`, 01:03-05:37): outputs equal stock's at +0.724 % against
prot6. That is below the band of +0.8 % to +2.0 %, so the prediction missed on the cheap side.**
- test.out and makerand.out are byte-equal to the stock twin run's (`cmp`; specdiff 1/2, as for every arm). The
  console has no `capstone_task: fault` line.
- **`memmove`:** 1,037,615,195 instructions against prot6's 1,037,644,188 (−0.003 %), at the same 20,478,129 calls.
  That is within 5 %, so the reading of revision 3's growth stands: the sorted array's shifts.
- Image zone 87,402,132,122: **+628,041,248 (+0.724 %) against prot6**, and +3.639 % against stock's
  84,333,527,702 for heads and ops. Per function against prot6 (`fndelta.py`; the named functions sum to
  +628,039,940):
  - `Perl_op_free` +208.7 M at the same 2,683,185 entries, about 78 instructions per call. In revision 3 it was
    +792.3 M, about 295 per call;
  - `Perl_Slab_Free` +205.8 M at 4,022,039 entries, 0 in prot6;
  - `Perl_Slab_Alloc` +133.6 M at the same entries;
  - `S_new_slab` +51.7 M at 413,912 entries, and `Perl_opslab_free` +23.4 M at 143,772 entries, both absent from
    prot6's counts;
  - everything else lies within ±5 M per function.
- Why the band missed. The band assumed that "most of" op_free's search would go. The page hash cut op_free's added
  price by 74 %, which is more than that. `Perl_Slab_Free` also fell, from +291.7 M to +205.8 M. `Perl_Slab_Alloc`
  (+133.6 M in both) and `S_new_slab` (+57.0 M, now +51.7 M) stayed near revision 3's. These counts are the only
  evidence; no disassembly was read.
- **Memory** (`census/out/rss-ops4-26dfe710`, both outputs byte-equal to stock's): peak used **102,384 kB**.
  - That is −236 kB against prot6 (102,620), −80 kB against revision 3 (102,464), and **+4.5 % against stock's
    97,968 kB**.
  - The page index does not move the peak at this granularity.
- So revision 4 is correct on the fixture (18/18) and on test.pl, keeps the protection, and is the op-slab revision
  to price. Its price on test.pl alone against prot6 is +0.724 %.

## perlbench train on the twin: the image zone per allocator (2026-11-22)

The same twin configuration as test.pl (slot 28, `perf/natio/images`, `dom_grant_root=/tmp/nt/sr`, zones4 plugin at
shift 2, 2^20 flat records), on the FRCDKMSa3 train stage (`stages-frcdkmsa3-train/500.perlbench_r-train-native`,
the protected arms' images replaced). Each zone is the image zone (`ZONE image`) summed over the five commands.

| arm | run | specdiff | image zone | against stock | against prot6 |
|---|---|---|---:|---:|---:|
| stock | `ztrain-stock-c1744e87` | 6/6 | 205,262,327,464 | | |
| prot6 (heads) | `ztrain-prot6-37381b03` | 6/6 | 216,796,670,174 | +5.619 % | |
| hash entries | `ztrain-he-e9994b2e` | 6/6 | 216,596,614,164 | +5.522 % | -200,056,010 (-0.092 %) |
| bodies | `ztrain-bodies-279647af` | **5/6** | not priced | | |
| bodies, 2^22 records | `ztrain-bodies-r22-279647af` | 6/6 | 218,898,234,466 | +6.643 % | +2,101,564,292 (+0.969 %) |
| prot6, 2^22 records | `ztrain-prot6-r22-37381b03` | 6/6 | 216,796,670,174 | +5.619 % | 0 |

- **Hash entries cost less than prot6 on train**, where they cost +0.112 % on test.pl. Per function (fndelta, the
  named functions sum to -200,055,869):
  - decreases: `Perl_sv_upgrade` -191.2 M at the same 46,036,356 entries; `Perl_sv_setsv_flags` -37.7 M; then
    `Perl_sv_free2` -11.9 M and `Perl_sv_arena_head` -11.3 M;
  - the port's own costs: `Perl_hv_common` +28.9 M, `Perl_sv_force_normal_flags` +24.6 M, `S_hv_free_ent_ret`
    +15.7 M, `Perl_more_bodies` +8.7 M (3,133 to 46,002 calls), `__heap_mapping` +1.1 M.
  - These are test.pl's shapes: `Perl_sv_upgrade` and `Perl_sv_setsv_flags` fell there too (-53.5 M, -12.4 M) and are
    mostly not attributed. On train, the segment costs `__heap_mapping` only +1.1 M against test.pl's +86.2 M, because
    train runs 5 processes and test.pl 318. So the unattributed decrease outweighs the port's costs. The size of that
    decrease is still not explained: it is a code-layout change in `sv.c`, read from per-function counts, not from the
    disassembly.
- **Bodies: splitmail (command 3) died with SIGSEGV, rc 139.** The console line is `capstone_task: fault rc=-14
  kind=2 cause=30 pc=0x3ff7c0f2fc`. With the image at 0x3ff7c00000 = vaddr 0x10000, that pc is 0x1f2fc in
  `__heap_mapping`. The word there is 0xa30c15db: opcode 0x5b, funct3 1, funct7 0x51. That is CDERIVE (capability.h's
  `capstone_cap_derive`: `.insn r 0x5b, 0x1, 0x51`), which the debug objdump prints as `<unknown>`. Commands 0-2 and 4
  returned rc 0.
  - Cause 30 is INSUF_RESOURCES. In the twin's own QEMU (`native01/qemu55`, a0501ebc39) it is raised only when the
    lifetime table has too few free records:
    - `capstone_require_node` (op_helper.c:1240-1253);
    - `CAP_REV_EXHAUSTED` (:1265);
    - fewer than 2 free ids (capstone_table.c:476-477);
    - the supervisor's node-pressure paths (capstone_supervisor.c:707, :920).
  - So the table of 2^20 records filled with live lifetimes. The ids55 table recycles records: in a perlbench train arm
    (T3), scrabbl created 14.1 times the table's records and completed (`tests/rtl-smoke/ids55-flat-module/RESULTS-twin.md`).
    What filled it is therefore the live count: heads plus bodies at once, on the largest command. That is inferred
    from the raise sites and the arms that passed. No live count was read.
  - A twin kit at 2^20 records cannot run the bodies port on splitmail train. The board's ids55 default is also 2^20
    (`paper-nested-allocators/notes/section5-rtl-encoding-evidence.txt`, from capstone_vm.c's `node_flat_records`).
  - **Rerun at 2^22 records** (`ztrain-bodies-r22-279647af`, 64 MiB of the 256 MiB CMA; the same image and stage, only
    the table size differs): rc 0, specdiff 6/6, splitmail included. So the 2^20 failure was the table's size, as the
    raise sites predicted; the live count itself was still not read.
  - Image zone 218,898,234,466: **+2,101,564,292 (+0.969 %) against prot6**, +6.643 % against stock for heads and
    bodies together. On test.pl bodies cost +0.733 %. Per function against prot6 (fndelta, the named functions sum to
    +2,101,568,306):
    - the issue and retire: `Perl_sv_upgrade` +1,200.6 M at the same 46,036,356 entries, `Perl_sv_clear` +1,026.2 M at
      35,883,962 entries, `Perl_more_bodies` +10.0 M (3,133 to 46,007 calls, the same rise as on test.pl);
    - decreases: `Perl_sv_setsv_flags` -72.9 M, `Perl_sv_arena_head` -14.7 M, `Perl_sv_free2` -11.9 M,
      `Perl_newSV` -10.5 M, `Perl_sv_mortalcopy_flags` -9.2 M, `__libc_malloc_impl` -5.8 M. These are not attributed.
  - Comparability. The zone was taken at 2^22, the other arms' at 2^20. The failed 2^20 run wrote no zone, so its
    four completed commands cannot be paired with this run. The evidence that the geometry does not move the image's
    count is still T3's F0-vs-W agreement within 0.001 % per command (`tests/rtl-smoke/ids55-flat-module/RESULTS-twin.md`);
    at the time of this run 2^22 was unpaired (the pair below now pairs it). The table's work is done in the emulator's helpers, outside the image range the zone counts.
  - Consequence for the board: an ids55 kit at its 2^20 default cannot run this arm on splitmail train. A bodies board
    cycle needs a larger table. The SPEC lane offers a kit cut from bkit91 with only `node_flat_records` changed to
    4194304 (bkit91-rec1m is the 2^20 precedent). The same cycle must pair stock or prot6 at 2^22 against 2^20 on one
    image, so that the bodies delta in cycles is not confounded by the table's size.
  - **The twin's pair, pre-registered (20:30):** prot6's train stage again at 2^22 (`ztrain-prot6-r22-37381b03`), with
    everything else as `ztrain-prot6-37381b03`. Prediction: specdiff 6/6, and an image zone within 0.001 % of
    216,796,670,174 (T3's F0-vs-W band). Branches:
    - within 0.001 %: the geometry does not move the image's count on this stage, and the bodies delta above stands;
    - beyond it: the bodies delta is restated against prot6 at 2^22, and the difference is reported as the table's
      cost on the image.
  - **Result** (`ztrain-prot6-r22-37381b03`, END 00:07): rc 0, specdiff 6/6, image zone **216,796,670,174, equal to
    the 2^20 run's to the instruction**. Per function, all 2,788 `FN` lines of the two capmix files have equal counts
    and entries (read over every line, not fndelta's top rows). So the first branch holds: on this stage the table's size does not move the image's count, and bodies'
    +2,101,564,292 (+0.969 %) against prot6 stands as measured.
    - Outside the image the two runs differ, and neither zone is a measure here. The user zone outside the image is
      4,332,830,738 at 2^22 against 4,299,404,280 at 2^20 (bodies at 2^22: 4,331,995,799); not attributed. The kernel
      zone varies across this table's arms from 4,048,869,547 (stock) to 201,478,463,827 (prot6 at 2^20), and is
      4,169,137,669 here; it is not read as a cost of any arm.
    - A first launch of this run was stopped before it took the lock, because it was given bodies' symbol file
      (`perl-bodies.syms`) instead of prot6's; its directory is kept as `ztrain-prot6-r22-37381b03-void-stopped` and
      wrote no zone. The run above used `perl-prot6.syms`, checked in its environment before launch.

### The CRENEW and liveness-test variant: no environment measures it yet (2026-11-23)

The programme asks for a perlbench variant with CRENEW and the liveness test. Where each can run, enumerated:
- **The liveness test.** No QEMU decodes it. The programme lists it as a stage-2 proposal, "first in QEMU and the
  compiler" (plan lines 67 and 127). Its performance case is already measured: today a liveness test is mallocng's
  load probe in `free`, `realloc` and `calloc`, and a dedicated instruction replaces one instruction with one. On perl
  that is 26.7 M probes, 0.032 % of instructions and 0.162 % of loads (`compiler/liveness-census` 77bf93aef483,
  `docs/history/21-11-2026_10-00-00_liveness-test-census.md`). So the variant's liveness half has no instruction
  saving to measure. Its case is semantic: no read permission or bytes in bounds are needed.
- **CRENEW**:
  - The twin's QEMU (`native01/qemu55`, a0501ebc39) does not decode it. `insn32.decode` has `csderive` (0x51),
    `cskill` (0x52) and `csretire` (0x53), and no entry at 0x54; `git grep -i crenew` in `target/riscv` finds nothing.
  - P1's RTL decodes 0x54 as CSNODE (plan, section 1.1).
  - The one-operand environment (`nested/oneop`, QEMU 9e69171f1c) has it. Its table carries no generations, so CRENEW
    cannot renew in place, and the table's cost there is the kit's sweeps (the gcc section above).
  - The reference model c1985bb529 has it, with generations, and no kit (plan, section 1.1).
- **What a CRENEW arm could read.** The bodies port's image cost is its code, around the lifetime instructions:
  `Perl_sv_upgrade` +1,200.6 M and `Perl_sv_clear` +1,026.2 M on train. Renewing a body when it goes on the free list
  would replace the release's CREVOKE with CRENEW. A reused body would then need no CDERIVE when it is taken again. The
  image zone of an arm in the one-operand environment would see that change in the port's code. It would not see the
  table's cost, which that kit charges to its sweeps. Cycles need a bitstream with CRENEW, and none is written
  (B5b-1 step (c)). How much of `Perl_sv_upgrade`'s +1,200.6 M is the derive path a renew would skip is not read
  from the port's code here, so no saving is predicted.
- **Decision.** The arm needs perl rebuilt in the one-operand environment, on that environment's compiler and SDK
  (application ABI 2, mallocng, virtual profile), and every bodies-port control repeated there. That build is
  deferred behind item 5 (the bug-corpus images), which has a consumer on the board.

## mcf train on the twin: what separates the Full image's OOM from V1f's pass (2026-11-22)

Asked by the SPEC lane. On bkit91 the Full image of 505.mcf_r train was killed by the OOM killer, while V1f's image
completed. Both twin runs below use the tree kernel (`perf/tree/images-tree`, capstone_vm.c byte-identical to the
board's) with the kit's module flags (`node_table_pool=4 dom_grant_root=/tmp/nt/sr dom_null=1 dom_local_tmp=1`), slot
28, and a 936 MiB guest (MemTotal 921,488 kB).

| image | run | result | memory |
|---|---|---|---|
| Full `fa650b98…` | `census/out/mem2-mcftrain-fa650b98` | OOM-killed at 99 s guest, specdiff 0/2 | anon-rss 892,748 kB, total-vm 1,108,624 kB at the kill (kernel line) |
| V1f `d5e5dc49…` | `census/out/mem2-mcftrain-v1f-d5e5dc49` | rc 0, specdiff 2/2 | VmHWM 887,188 kB |

So V1f fits with a few MB to spare, and Full needs more than 892,748 kB. The OOM is the task's own memory, as the
earlier reading of the arc arrays found. Each image's unstripped build, stripped with the kit's recipe, reproduces the
staged hash, so the provenance below is read from the images' own builds.
- **V1f.** SDK `native01/sdk-native`, with the runtime at a486f0ccbbd1, which has no mapping cache. musl's realloc grows a
  large chunk through `__heap_remapping` (map, copy, unmap). Compiler cc-prmerged.
- **Full.** SDK FRCDKMSa3, whose runtime and libc are `perf/alean/sdkm13b`: the runtime at 3e70344d8c3f built with
  `CAPSTONE_HEAP_MAPPING_REUSE=ON`, musl with the alloc-lean patches. Two allocator differences:
  - (a) the mapping cache keeps freed mappings for reuse, at most 512 of them and 64 MiB in all, and they stay
    resident. It gives them back only when an acquire fails, and a lazy grant does not fail before the OOM killer
    acts;
  - (b) realloc grows a large chunk in place when its mapping has room. Otherwise it moves the chunk to a mapping with
    room for twice the size, up to 1 GiB.

  Also different: the compiler (llvm-c9 with the cclever flags), the rest of the runtime between those two
  revisions, and Full's `CAPSTONE_MEMSET_CAP_ZERO=ON` (not checked in V1f's build).
- **Withdrawn.** My message to the SPEC lane (2026-11-22, not committed anywhere) read Full's 200 MB of total-vm over
  its RSS as the old and new arc arrays mapped together while `__heap_remapping` copies. Full's realloc does not take
  that path for a growing large chunk. The gap fits (b)'s untouched room.

### Pre-registration: two one-object variants of Full on the twin (written 2026-11-22 03:20, before the runs)

Each image is Full's own objects relinked through a copy of its SDK, with one allocator object replaced. A control
relink through the unmodified SDK reproduces `fa650b98` byte for byte. Each replaced object, rebuilt from the unmodified
source with the recorded flags, reproduces the SDK's object byte for byte.
- **R0 `24a4cc5e…`.** `malloc-mapping.c` built with `CAPSTONE_HEAP_MAPPING_REUSE=0`. The symbols that differ are
  `__heap_mapping`, `__heap_unmapping`, `__heap_protected`, the cache's tables and counters (`mr_live` is 768 KiB of
  bss), and `memmove`, which nothing references any more.
- **W0 `8b782488…`.** musl's mallocng `realloc.c` without (b); the result is byte-identical to V1f's `realloc.c`. Only
  `__libc_realloc` differs.

Runs: the configuration of the two runs above, R0 then W0, one each. Readings for a completed run: rc, specdiff and
`@@MEM-PEAK` VmHWM. For a killed run: the kernel's OOM line (anon-rss, total-vm). The sampler's last sample is not an
end reading.

Outcomes:
- R0 completes (2/2) and W0 is killed: the mapping cache is the variable that separates the two images.
- W0 completes and R0 is killed: the moved-with-room realloc is.
- Both complete: either change alone gives back enough, at a margin of a few MB. Both are levers, and neither is "the"
  cause.
- Both are killed: neither allocator difference alone; what is left is the compiler and codegen and the other runtime
  changes. The row stays "killed by OOM on bkit91", with no cause.
- **Predicted: R0 completes and W0 is killed.**

This is a diagnostic, not a measured arm. The paper's Full row does not change. Any board run goes through the SPEC
lane, under a new addendum.

### Results: R0 completes and W0 is killed, as predicted (QEMU twin, 2026-11-22 03:14-05:47)

Both runs used the configuration above, one each, R0 then W0.

| image | run | result | memory |
|---|---|---|---|
| R0 `24a4cc5e…` (no mapping cache) | `census/out/mem2-mcftrain-r0-24a4cc5e` | rc 0, specdiff 2/2 | VmHWM 885,680 kB |
| W0 `8b782488…` (V1f's realloc) | `census/out/mem2-mcftrain-w0-8b782488` | OOM-killed at 100 s guest, specdiff 0/2 | anon-rss 892,728 kB, total-vm 1,065,808 kB at the kill (kernel line) |

This is the first outcome: on this input, the mapping cache is the variable that separates Full from V1f.
- **R0.** Without the cache, Full's image peaks 1,508 kB below V1f's (885,680 against 887,188 kB). It also peaks about 7 MB
  below the anon-rss of both killed runs (892,748 and 892,728 kB).
- **W0.** V1f's realloc alone leaves anon-rss at the kill where Full's was, 20 kB lower. Total-vm at the kill is
  42,816 kB lower (1,065,808 against 1,108,624 kB). So, at the kill, (b)'s room shows in address space and not in
  resident memory.

Limits: one run per image, and R0's margin is about 7 MB in a 921,488 kB guest. The readings come from the twin, not the
board. The paper's Full row does not change. Bounding the cache, or giving its mappings back under memory pressure, is
a runtime change and goes to the runtime side on its own.

## speedtest1 `--size 100` on the twin (2026-11-22)

The programme's item 4 asks for speedtest1 at `--size 100`, which the SPEC lane deferred on 14 Nov at about 6.5 h per
command on silicon. The twin comes first: it gives the answers, the lifetime counts and the instructions.

**The heap.** At `--size 100` the `--size 1` configuration's 8 MiB memsys5 heap is too small. A host build of the same
sources (`7fc92e19…` sqlite3.c, `16750ede…` speedtest1.c, gcc 13.3.0, the oracle build) stops at phase 150 with
`SQL error: out of memory` after 7 phases. It exits 0 with 32 phases at 16, 32, 64, 128 and 256 MiB, and the phase
lines are identical at all five sizes. Capstone's pointers are twice the host's, so the run uses 32 MiB, twice the
smallest heap that completed on the host.
- Reference: `capstone/ports/sqlite/app/results/2026-11-22-size100-host/phases.json`, the 32 phases of the host run at
  `--heap 33554432 64 --size 100`.
- Control: the same host build at `--heap 8388608 64 --size 1` reproduces the native phases of
  `2026-10-11-same-answers/phases.json`, 32/32.

### Pre-registration: QEMU twin, speedtest1 `--size 100` (written 2026-11-22 06:05, before the run; the time first read 06:30, which was wrong)

- **Run.** One twin boot on slot 28 (`nested/st1/stages/s100`, made by `make-stage.py`), the file-database commands u,
  p, p, u, each `--heap 33554432 64 --size 100 st.db`. It is run under `capmixn` (PROF=1), on the kernel and module flags of
  `nst1-q2`, the `--size 1` pair. The stage's `run.sh` differs from `nst1-q2`'s only in the arguments.
- **Images.** The `--size 1` images, unchanged: not protected `817b1652…`, protected `945fb69f…`.
- **Readings.** Per command:
  - rc and the 32 oracle lines against the reference;
  - CDERIVE and CREVOKE in the program window;
  - program instructions (capmix3-read.py's rule, as at `--size 1`);
  - kernel instructions in the window.

Predictions:
- **Answers.** 32/32 equal to the reference in all four commands.
- **Lifetimes.** At `--size 1` the protected arm minus the unprotected one was 28,216 CREVOKEs. The work scales with
  the size, so I predict 1 M to 6 M at `--size 100`, with the unprotected arm below 100.
- **Program instructions, protected vs not.** +0.3 % to +3 % (`--size 1`: +0.63 %). Repeats of an arm agree within
  0.01 %.
- **Control.** The protected arm must show at least 1,000 times the unprotected arm's CREVOKEs, or the patch did not take
  effect and the run says nothing about protection.

Failure branches:
- Both arms out of memory: the heap is too small for Capstone at 32 MiB. That is a configuration result, and the next
  run doubles it.
- Only the protected arm fails:
  - with QEMU cause 30, the flat table's 262,144 records per context are exhausted. That is a table-size result: the
    next run raises `node_flat_records`, and the live-record peak becomes the reading;
  - any other fault in the protected arm alone is a finding, to localise before anything else.
- No line from a command for the whole run: a stall, classified from the serial log.

### Result: every command stops at phase 142 with a disk I/O error, unprotected ones too (`nested/out/nst1-s100`, 06:11-06:40)

Not one of the pre-registered branches.
- All four commands (u1, p1, p2, u2) print the five phases 100-140 and stop at phase 142 ("10 SELECTS w/ORDER BY,
  unindexed") with `step error: disk I/O error`, rc 1. The five phases equal the reference in every command: rows and
  hash, 5/5.
- No fault and no OOM in the serial log. Each command ran about 6.5 s.
- The unprotected arms fail identically, so this is the environment, not the protection. No answer, lifetime count or
  instruction count of this run is read against the predictions.
- **Cause, inferred.** In the reference, phase 142 returns 153,747 rows, enough for SQLite's sorter to spill to a
  temporary file. The images search `SQLITE_TMPDIR`, `TMPDIR`, `/var/tmp`, `/usr/tmp`, `/tmp` and `.` for one. Without
  `capstone_vm.dom_local_tmp` the native exec's grants refuse `/tmp` (`runtime/virtual/module/resource-dom.h`: the
  flag aliases `/tmp` to the cwd's `.capstone-tmp`, which the stage's `run.sh` already creates with mode 0700).
  `nst1-q2`'s flags, which this run copied, have no `dom_local_tmp`; at `--size 1` phase 142 returns 1,050 rows, and
  that run completed (that its sort stayed in memory is inferred from its completion). The boards'
  module flags have it, and so did the twin's mcf train runs. Which call returns the I/O error was not read.

### Amendment 1 (written 2026-11-22 07:27, before the rerun; committed 07:27:45, and the time first read 07:30, which was wrong)

- **One change.** The rerun `nst1-s100b` adds `capstone_vm.dom_local_tmp=1` to the module flags. Everything else
  stays as pre-registered above: the stage `nested/st1/stages/s100` unchanged, the same images, kernel, records
  (262,144), slot and plugin. The predictions and failure branches above apply unchanged.
- **What it tests.** If phase 142 now completes in both arms, the temporary file was the cause. If both arms stop at
  phase 142 again, the cause is something else, and the next step reads the failing call before any further run.

### Result of the rerun: every command stops at phase 142 again; the inferred cause is withdrawn (`nested/out/nst1-s100b`, 07:28-10:20)

- The flag reached the module: the guest's command line carries `capstone_vm.dom_local_tmp=1`, and the stage
  creates `.capstone-tmp` with mode 0700 before each command, as the module requires.
- All four commands print phases 100-140 equal to the reference, 5/5, and stop at phase 142 with
  `step error: disk I/O error`, rc 1, as in `nst1-s100`. There is no fault, OOM or module line in the serial log.
- **Withdrawn: "the temporary file was refused /tmp" as the cause** (the Result section above, "Cause, inferred").
  Amendment 1's second branch holds: the cause is something else. The inference was weak on its own terms. SQLite
  (`unixTempFileDir`, sqlite3.c `7fc92e19…`) falls back to `.`, the grant root, so a refused `/tmp` alone would
  not have stopped it.
- **Candidate, from source, not shown on the twin.** SQLite accepts a temporary directory only if `access(dir, 3)`
  succeeds. musl on riscv64 makes that `faccessat` (48), and the native boundary's `native_supported`
  (`runtime/virtual/module/native-syscall-dom.h` in the natio tree) does not list it. A refused call returns
  -ENOSYS (`native_refused`). Every directory, `.` included, would then be rejected. The sorter's first temporary
  file then fails with `SQLITE_IOERR_GETTEMPPATH`, which reads "disk I/O error".
- **The candidate does not predict the phase.** A host build of the same sources with `access` and `faccessat` made
  to fail with ENOSYS (an LD_PRELOAD shim) also stops with "disk I/O error", but at phase 150 (`CREATE INDEX`),
  after passing 142. This holds at the 32, 16 and 12 MiB heaps; without the shim the host completes 32/32, and at
  `--size 1` with the shim it completes too. On the host, phase 142's sort needs no temporary file. Why the twin's
  does, or whether its error has another source, is not known.
- **Next, before any further measured run:** one diagnostic twin boot (`nested/out/nst1-s100diag`, the flags of
  `nst1-s100b`) running command u1 alone. Stage `nested/st1/stages/s100diag` is the s100 stage cut after u1. It
  turns `native_route_trace` on through sysfs for the command only, since the kernel command line has no room for a
  further flag, and at the end prints `native_refused_calls`, the routes by call and result, and the last failing
  routes. No reading of it enters the predictions.

### Result of the diagnostic boot: the last failing calls before the error are the temporary-directory probe (`nested/out/nst1-s100diag`, 10:26-14:31)

- u1 prints phases 100-140 and stops at phase 142 with `step error: disk I/O error`, rc 1, as in every earlier
  boot. The route trace was on for the command (`@@DIAG route_trace=Y`). `native_refused_calls` reads 3.
- The last four routed calls that fail, which are the last lines of the trace, are: `newfstatat` (79) returning -2
  twice, then `faccessat` (48) returning -38 (ENOSYS) twice. The successful routes include `newfstatat` returning 0
  twice. That is what `unixTempFileDir` does on this guest: stat `/var/tmp` and `/usr/tmp` (absent, -2), then stat
  `/tmp` and `.` (present), and `access(dir, 3)` on each, which the boundary refuses. The order within each pair is
  not shown, because the summary lines are sorted and the failing tail holds only failures.
- This is read from the trace, not shown by a matched pair. The command made 53,499 register routes, and the kernel
  log ring cannot hold a line for each. The summary also stops at its 60th line (871 `read` (63) routes of 4,096
  bytes, then `lseek` (62) offsets). Only these four failing routes are in the ring, so the third refused call is not
  identified.
  The phase is still unexplained: on the host, phase 142 needs no temporary file.
- **Next: a matched pair in one boot** (`nested/out/nst1-s100acc`, the same flags). Arm uacc is the unprot source
  with one change: SQLite's `access` system-call entry is replaced by a function that returns `stat`'s result
  (`nested/st1/build-acc.sh`; image `1bc95737…`). The same recipe rebuilds unprot to its staged `817b1652…`. uacc
  runs first and unprot second, each at `--size 100` with a fresh database. If the refused `faccessat` is the cause,
  uacc passes phase 142 and unprot stops there. If uacc also stops at 142, the cause is elsewhere.

### Result of the matched pair: the refused `faccessat` is the cause (`nested/out/nst1-s100acc`, 14:50-17:38)

- **uacc** (`1bc95737…`) returns rc 0 after all 32 phases, in 657.8 s of guest time. Each phase's rows and hash equal
  the host reference (`2026-11-22-size100-host/phases.json`): 32 of 32. Its st.err is empty.
- **unprot** (`817b1652…`), in the same boot, prints phases 100-140, all five equal to the reference, then stops at
  phase 142 with `step error: disk I/O error`, rc 1, as in every earlier boot.
  `native_refused_calls` reads 4 at the end of the boot.
- The two arms differ only in SQLite's `access` entry. So the error comes from `unixTempFileDir` finding no usable
  temporary directory, because the native boundary answers `faccessat` (48) with ENOSYS.
- The comparator was run against a reference with one hash changed: it reports 31 of 32 and names phase 170.
- Still not explained: why phase 142 needs a temporary file on the twin, when the host run needs none there. A sort
  that spills to a temporary file is the likely reader, and that was not checked.
- What follows:
  - **Measured run.** Either the boundary routes `faccessat`, which is the runtime lane's module (`native_supported`
    in the natio tree; not this lane's), or every arm carries the same `access` shim. The shim changes the program, so
    it needs its own pre-registration and a protected arm built the same way.
  - **Until then,** an `--size 100` result on this boundary is uacc's, a diagnostic arm, and it is not a result arm.

### Pre-registration: unprot on a kernel that routes `faccessat` (written 2026-11-22 17:49, before the runs)

The runtime lane routes `faccessat` (48) on the native path: llvm-capstone `native-faccessat` 80bfe800dfc9, off
table-tree 722416c35e77. Its twin kernel is `perf/facc/images-facc`, Image `bf95cde78f45c0a6…`, with
images-tree's rootfs. It is images-tree's kernel (`perf/tree/images-tree`, Image `8bc7dc8d722fe906…`) with three
files replaced, as that lane states. This lane has not read that change.
- **Runs.** Two twin boots on slot 28, in this order, both with stage `nested/st1/stages/s100diag`: unprot
  (`817b1652…`, no shim), command u1 alone, the route trace on, `--heap 33554432 64 --size 100`. Both use the module
  flags of `nst1-s100acc` (`dom_local_tmp=1`) and 262,144 records.
  - `nst1-s100facc` on images-facc;
  - `nst1-s100tree` on images-tree.
  The two boots differ only in the kernel image.
- **Predictions:**
  - facc: rc 0, all 32 phases equal to the host reference (`2026-11-22-size100-host/phases.json`, by
    `cmp-s100`'s rule: rows and hash per phase), and no `faccessat` line with result -38 among the failing routes.
  - tree: phase 142 with `step error: disk I/O error` and rc 1, as on `perf/natio/images`, with `faccessat` -38
    among the failing routes. That arm is the control: it shows that this check of the failing routes can name
    `faccessat`.
- **Readings:** rc, the phase lines against the reference, `native_refused_calls`, and the failing-route tail
  (`@@DIAGT`).
- **Branches:**
  - facc passes and tree stops at 142: the routed `faccessat` is enough for `--size 100` on the twin. The runtime
    lane then builds the board kernel and kit with it, and the measured run is pre-registered separately.
  - facc stops at 142: the routed call does not satisfy `unixTempFileDir`, or something else follows. The failing
    routes are read before anything else.
  - tree passes: the earlier failures belong to `perf/natio/images`, not to images-tree's kernel. Then the pair
    says nothing about `faccessat`, and this is written down as such.
  - Either boot ends without a phase line or rc: that boot is void and is rerun once.

### Result of the pair: the routed `faccessat` is enough for `--size 100` on the twin (17:49-00:09)

Every prediction held, so the first branch applies.
- **facc** (`nst1-s100facc`, images-facc): rc 0 after 790.3 s of guest time. All 32 phases equal the host
  reference by `cmp-s100` (none differ, none missing, none extra), and the route trace is on (`@@DIAG
  route_trace=Y`). Its 14 `faccessat` (48) routes return 0 (7) and -2 (7), and none returns -38.
  `native_refused_calls` reads 1. The only route returning -38 is an exchange query, `nr=3235774468`
  (0xc0de0004, count 16), at 0.33 s; it is the first failing route of the boot and is not attributed here.
- **tree** (`nst1-s100tree`, images-tree): phases 100-140 are equal to the reference (5 of 5). The boot then stops
  at phase 142 with `step error: disk I/O error`, rc 1. Both of its `faccessat` routes return -38, at 8.96 s.
  `native_refused_calls` reads 3: those two plus the same exchange query. So the control fires. The check of the
  failing routes names `faccessat` when the kernel refuses it, and the -38 count in facc is a reading, not a blind
  zero.
- The two boots differ only in the kernel image. Read, not taken from the runtime lane: the two image directories
  hold `Image` (`bf95cde78f45c0a6…` against `8bc7dc8d722fe906…`) and the same `rootfs.ext2` symlink
  (`9903242c37a72425…`). The runtime lane's `faccessat` route is therefore what
  `--size 100` needs on the twin. As the branch says, the runtime lane builds the board kernel and kit with it, and
  the measured run is pre-registered separately.
- Not settled by this pair: why phase 142 is the first to need a temporary directory on the twin. The earlier note
  (a sort that spills) still stands unchecked. This lane has not read the route's change (80bfe800dfc9).

### Amendment 2: the measured twin run on the `faccessat` kernel (written 2026-11-23 00:16, before the run)

- **Run.** `nst1-s100c`: the pre-registration of 06:05 (`nst1-s100`) with two changes. The kernel is images-facc
  (`bf95cde78f45c0a6…`, the same rootfs), and the module flags carry `capstone_vm.dom_local_tmp=1` (Amendment 1).
  Everything else is unchanged: stage `nested/st1/stages/s100` (u1, p1, p2, u2, each `--heap 33554432 64 --size 100
  st.db`), images `817b1652…` and `945fb69f…`, slot 28, 262,144 records, PROF=1 with `nested/prof/capmixn.so` at
  shift 24, as `nst1-s100b` ran.
- **Predictions and failure branches:** those of 06:05, unchanged. In particular:
  - 32/32 in every command;
  - 1 M to 6 M CREVOKEs in each protected command, against fewer than 100 in each unprotected one;
  - program instructions +0.3 % to +3 %, with repeats within 0.01 %;
  - QEMU cause 30 in a protected command means the 262,144 records are too few.
- **Why before the board.** The runtime lane's board kit with this route (`t3-smoke/bkit107-k2n-native-p1-tree-facc`,
  a lane build, not on dev) runs 262,144 records too. So the board pre-registration waits for this run's answers,
  lifetime counts and the table branch.

### Result of amendment 2: every prediction held (`nested/out/nst1-s100c`, 00:17-04:47)

- **Answers:** all four commands rc 0, and 32/32 phases equal the host reference in each by `cmp-s100` (none differ,
  none missing, none extra). Guest seconds per command: 658.3, 660.1, 660.1, 658.3.
- **Lifetimes** (`capmixn-read.py`):
  - each protected command executes 2,112,363 CDERIVE and 2,112,353 CREVOKE, inside the band of 1 M to 6 M;
  - each unprotected command executes 9 and 2, under 100.
- **No cause 30:** no command faulted. A refused derive would have stopped the program, and every command ran all
  32 phases to rc 0. So 262,144 records are enough for `--size 100`. The runtime lane asked to hear before the board
  pre-registration only if cause 30 appeared, so the 2^20 cut is not needed.
- **Program instructions:**
  - unprotected: 32,294,252,503;
  - protected: 32,441,745,777;
  - **+0.457 %** (+147,493,274), inside the band of +0.3 % to +3 %. That is 69.8 program instructions per CREVOKE.
  - Repeats are equal to the instruction (u1 = u2, p1 = p2).
- **Kernel instructions:** +0.265 % (+1,658,030,557, about 785 per CREVOKE). Repeats agree within 0.0002 %. Not
  priced here; the band was written for the program only.
- **Branch:** the board pre-registration, u1 + p1 on bkit107, with the known-good control first.

### Pre-registration: speedtest1 `--size 100` on P1, u1 + p1 on bkit107 (written 2026-11-23, before the boot)

- **Kit:** `t3-smoke/bkit107-k2n-native-p1-tree-facc`. It is bkit91 with the native kernel tree changed alone, to
  `perf/k2n/linux-board-k2n-native-hpm2tree-facc`: hpm2tree plus the three files of the runtime lane's
  `native-faccessat` 80bfe800dfc9.
  - **The kernel is a lane build.** That route is on the runtime lane's branch, not on dev.
  - It has not booted on a board before, so this boot is also its Tier 1.
  - Bitstream P1 `…-tagoverlap6c-pagedtbl_d83d3a633` (sha256 `dc411e36…`).
  - The module flags include `node_flat_records=262144`, `dom_local_tmp=1`, `dom_grant_root=/tmp/nt/sr` and
    `dom_null=1`. That is the record table, and the flags, the twin ran with in `nst1-s100c`.
- **Stage:** `nested/p1s100/s100` (`s100.SHA256SUMS` `358cf568…`). It is bkit107's `make-train-stage.sh` around
  `make-stage.py`'s stage, with three commands in this order:
  1. **c0, the known-good control:** unprotected, `--heap 8388608 64 --size 1 st.db`, the command whose 32/32
     answers P1 gave on bkit91 (board 2, 2026-11-17);
  2. **u1:** unprotected, `--heap 33554432 64 --size 100 st.db`;
  3. **p1:** protected, the same arguments.
  - Images: the `--size 1` ones, unchanged: `817b1652…` and `945fb69f…`.
  - The kit's own controls come first, as in every boot: `b0-stats`, `b0-hello` and the native smoke.
- **Before the boot:**
  - the rule-8 recheck: a real bake, BOARD_STUB=2 on a transcript cut that does not return, and the firmware under
    the 26 MB cap;
  - a dry run of the chain that prints the effective idle limit and the runner's timeout.
  - Each command is silent for hours: the SPEC lane put `--size 100` at about 6.5 h per command on silicon. So the
    idle limit must exceed 8 h, and the stage timeout must exceed 16 h.
- **Predictions:**
  - **VOID** if `b0-stats` or `b0-hello` fails, the smoke fails, or c0 is not 32/32. A c0 failure is a finding about
    the kernel, not about protection.
  - u1 and p1: 32/32 answers equal to the host reference (`2026-11-22-size100-host/phases.json`). Any command that
    does not return, or that differs, gives no cost.
  - Protected vs not, in-context: +0.3 % to +5 %. The twin read +0.457 % in program instructions. On P1 at
    `--size 1`, the in-context cost was +2.21 %.
  - Record-TLB misses (hpm3): more in p1 than in u1.
  - QEMU's cause 30 did not appear on the twin. A refused derive on the board, in p1 alone, is read as a table
    result: the record limit, or the paged table's grow (event 33, hpm6).
- **ABBA, conditional, from the runtime lane's design:** a second boot of p1 then u1 follows only if p1 − u1 lies
  within three times the noise band. The noise band is the repeat rule on F (1 % `run_cycles`). Otherwise the pair
  is the reading.
- **Board time:** about 13 h for u1 + p1, plus about 10 min for c0 and the controls. The board slot is asked of the
  runtime lane. Nothing runs before this text is pushed.

### Result: c0 passes; u1 was cut by this lane's idle limit before it could have finished, so no `--size 100` reading (board 3, 2026-11-23 13:38 - 11-24 06:25)

Transcript `spec/board/out-nat-p1s100-b3r2/s100/board.txt` (`bfc1dc01c6b6be48…`).

- **Controls pass.**
  - `b0-stats` and `b0-hello` return 0, and the native smoke passes all 9 steps.
  - c0 (`--size 1`) returns rc 0 in 229.06 s (uptime 125.04 to 354.10). All 32 phases equal the host's native
    `--size 1` reference (`ports/sqlite/app/results/2026-10-11-same-answers/phases.json`).
  - The comparison fires: against the `--size 100` reference the same output equals only 18 of 32 phases.
  - So the new kernel (bkit107's facc tree) boots and runs speedtest1 on P1. This boot is its Tier 1.
- **u1 did not return.** It began at 354 s. The driver ended the rung after its idle limit of 60,000 s without
  console output (`idle limit 60000 s`, SEQ), at 06:25:20. p1 never started, and the board was unbooked.
- **That is not a hang. u1 could not have finished inside the limit.**
  - On the twin, `--size 1` takes 2.26 guest seconds (`nst1-q5`, command u). The `--size 100` commands of
    `nst1-s100c` take 658.27, a ratio of 291.
  - At that ratio, c0's 229.06 s on P1 gives about 66,700 s, or 18.5 h, for u1, above the 60,000 s limit.
  - The apertures read after the cut changed between two reads (`UNSTABLE`, 5 switches), so the core was still
    running.
  - The pre-registration sized its limits from the SPEC lane's "about 6.5 h per command". This lane had a measured
    P1 `--size 1` time (bkit91, 2026-11-17) and the twin's size ratio, and did not compute from them.
  - The estimate assumes board time scales with the twin's guest time. The twin's commands are dominated by kernel
    instructions (+0.265 % being 1.66 G means about 625 G), so this is an estimate, not a reading.
- **No cost or answer for `--size 100` on silicon.** About 16.8 h of board 3 were spent on this.
- **What a measured run would take:**
  - about 18.5 h for u1 and about the same for p1, so about 37 h, or two boots of about 19 h, one command each;
  - an idle limit above the command, or a heartbeat line with the bytes written;
  - before that, the board-time cost is put to the runtime lane, which owns the queue.

### Pre-registration: speedtest1 `--size 100` on P1, one command a boot, stock first (written 2026-11-24 06:34, before the boots)

The runtime lane agreed to the pair on 2026-11-24, after the first boot's result.

**Why the first boot's limit was too short.**
- `bkit107`'s `board-run.sh` sets a SPEC stage's idle limit to its batch budget: the list's timeout × 4. The first
  list gave 15,000, so 60,000 s.
- That figure came from the SPEC lane's "about 6.5 h per command", not from a measurement.
- Two measurements were available: P1's `--size 1` time (c0, 229 s in the first boot) and the twin's ratio between
  the two sizes for the same arguments, 658.27 / 2.26 = 291. Together they give about 66,700 s.
- The first boot also gives a lower bound: u1 ran for at least about 59,900 s without returning.

**Boots.** Two stages, each derived from the first boot's stage `nested/p1s100/s100` (`s100.SHA256SUMS` 358cf568…).
- `s100u` (`s100u.SHA256SUMS`, sha256 of the list `5ba7714d…`) runs c0, then u1.
- `s100p` (`fb65fa93…`) runs c0, then p1.
- Changes to `train/run.sh`, and nothing else:
  - the other `--size 100` command is removed, with `commands=2` and p1's index renumbered;
  - a heartbeat runs during the `--size 100` command, every 900 s:
    `@@BEAT <uptime> st.db=<bytes> <native counters>`. It starts before `mark1` and is killed after the command
    returns, so the same shape runs in both arms.
- Images: unchanged, `817b1652…` (unprot) and `945fb69f…` (prot). Kit: unchanged, `bkit107-k2n-native-p1-tree-facc`
  on P1.
- c0 stays in each boot as that boot's known-good control.
- **Budgets each boot gets** (from the kit's code and the dry run):
  - the list timeout is 25,000, so the batch budget and the idle limit are both 100,000 s (`BAKED_TIMEOUT=100000` in
    the dry run; `board-run.sh:155` gives a SPEC stage IDLE = TO and no entry-stall abort);
  - the driver gets 101,800 s;
  - the booking is 1,740 min;
  - the guest stage has no timeout of its own.
- Chain: `nested/p1s100/nat-run3p.sh <board> <wait pid> p1s100pair.conf`, list `p1s100pair.list` (s100u, then
  s100p), with `NR_PREREG` set to this commit.
  - It is `nat-run3n.sh` with the wait made optional. `NR_WAITLOG` is checked only when set. `NR_WAITMARK=<file>`
    waits for a marker newer than the chain's start. A wait pid of 1 means no pid to wait for.
  - The runtime lane arms it behind its board-3 queue, or on whichever board frees first. The chain flashes P1 if
    another bitstream is resident.

**Dry runs** (`nested/p1s100/pair-dry/`, `s100{u,p}-recheck/`):
- Rule-8 recheck of each stage:
  - real bake;
  - `BOARD_STUB=2` on a returning and a non-returning transcript, both rc 0;
  - real firmware, 19,566,088 B, under the 26 MB cap.
- The chain with `NR_DRY=1` reaches DONE through both stages, each with `BAKED_TIMEOUT=100000`.
- The wait block, run alone:
  - pid only: proceeds;
  - pid plus a marker written later: waits for it;
  - a marker older than the start: ignored until rewritten;
  - `NR_WAITLOG` without an END line: STOP, rc 2.
- The heartbeat under busybox sh: it beats during the command and stops after it.
- The reader `read-s100pair.py`:
  - on the first boot it reads c0 32/32 and "u1 NO END", rc 1;
  - on a transcript with the twin's u1 output and three beats, c0 and u1 are 32/32, rc 0;
  - with c0's output put in u1's place, it reads 18/32 with 14 phases named, rc 1;
  - an empty file gives NO DATA, rc 2.
  - The kit's `spec-classify.py` reads every native transcript as NO RESULT (the known classifier gap), so this
    reader gives the answer.

**Predictions.**
- **VOID** if `b0-stats`, `b0-hello` or the smoke fails, or if c0 is not 32/32.
- **Answers:** u1 and p1 are each 32/32 against `2026-11-22-size100-host/phases.json`.
- **Time:** u1 takes 60,000 to 93,000 s. The low end is the first boot's lower bound. The high end is 1.4 times
  c0 × 291, because the twin's guest time is dominated by kernel instructions and may scale differently on silicon.
  p1 is in the same band.
- **Heartbeats:** about one every 900 s, so 66 to 103 per command, with `st.db` not shrinking.
- **Cost, p1 against u1, in-context** (`sys=` of `@@SPEC-END`): +0.3 % to +5 %, the first pre-registration's band.
  The twin read +0.457 % in program instructions; P1 at `--size 1` read +2.21 %. This is one boot per arm with no
  repeat, so the noise is not measured. The F repeat rule (1 % `run_cycles`) is the only noise figure, and a
  difference inside 3 % is reported as not resolved.
- **Cause 30 or a refused derive in p1** is read as a table result (the 262,144-record limit or the paged table's
  grow), as before.

**Branches.**
- A command that does not return within 100,000 s gives no cost. Its beats tell three cases apart:
  - beats continue and `st.db` grows: slow;
  - beats continue and `st.db` is frozen: stopped in the program or kernel;
  - beats stop: the core or kernel stopped.
- u1 failing does not stop the chain, because p1's boot is separate. The pair is read only if both return.

### Amendment during the run, before any reading (written 2026-11-24 23:35): the heartbeat's `st.db` field reads nothing

s100 (u1) has been running on board 2 since 15:31 (booking 1478). It was at uptime 28,322 s at the last beat, inside the
registered band of 60,000-93,000 s. The runtime lane asked whether the run had overrun. It has not.
- **Effective limits, read from the live runner:** `timeout 101800` around the driver, with `BAKED_TIMEOUT=100000`,
  `BAKED_IDLE_S=100000` and `BOARD_TIMEOUT_MULT=4`. The beats print every 900 s, so the idle limit cannot fire, and
  the 100,000 s batch budget is the cap, as registered.

**The `st.db` field is void.**
- Every beat so far reads `st.db=0`, from the first one at uptime 1,267 s. (Corrected 2026-11-25: the first version said
  3,071 s, which is the first beat my parser read whole from the split UART chunks.) The cause is in the heartbeat's command:
  `stat -c %s st.db 2>/dev/null || echo 0`.
- The board's busybox has no `stat` applet, so the command fails and prints 0. The rootfs is this boot's
  `image/rootfs.cpio`. Its applet table runs `sort`, `start-stop-daemon`, `strings` with no `stat`, while `sleep`,
  `ls` and `wc` are present. The cpio has no other `stat` binary.
- The dry run's heartbeat test checked that the beat runs during the command and stops after it. The reader was tested
  on synthetic beats. Neither checked the size field against the board's busybox, so the field was never shown to
  fire.
- So a zero says nothing about whether speedtest1 writes its database. The registered branches "`st.db` grows"
  against "`st.db` frozen" cannot be read.

**What the beats still read.**
- The native counters in each beat do grow, so the process is still being served by the module:
  - `register` 241,347 at 3,071 s, 1,478,309 at 12,992 s, 1,842,154 at 28,322 s;
  - since 15,697 s, a steady 15,735-15,741 per 900 s beat;
  - `exchange` constant at 382 since 15,697 s.
- **If u1 does not return within 100,000 s**, the three branches become two:
  - **beats continue** with `register` growing: the program is still running and being served. This cannot tell
    slow work from a loop;
  - **beats stop:** the core or kernel stopped.
- **If u1 returns,** the field never enters the reading. The answers, the time and the cost are read as registered.
- The running stage is not changed. s100p carries the same heartbeat, so the same applies to it.

### Stopped by the project lead's ruling, before u1 returned (2026-11-25 00:22): no reading

- **The ruling.** The runtime lane relayed it on 2026-11-25. SPEC train runs only in the final campaign, and
  speedtest1 `--size 100` moves there with it (spec-silicon `bbcc54172beb`, PREREG Addendum 236, for the train half).
  That is a deviation from 84b7a23c5935 and its amendment a8cfc596bd86, by the lead's instruction.
- **The stop, at 00:22.**
  - Before it, every process of the chain was listed: chain `nat-run3p.sh` pid 1617419, its two `board-run.sh`
    shells, the `timeout 101800` runner group, and the watchdog. One other process waited on the chain pid: this
    lane's own log watcher.
  - The chain was stopped first, so it could not start s100p. Then the runner's process group was stopped (driver
    rc 130). `board-run.sh` released the lease, and the watchdog ended with it.
  - Booking 1478 was unbooked (HTTP 200), and board 2 reads FREE. There was no flash.
- **State at the stop.** u1 began at about uptime 365 s (its first beat was at 1,267 s). It had run about 31,100 s of
  its registered 60,000-93,000 s. The last beat was at uptime 31,028 s, with `register` 1,889,377 (still rising at 15,741 a beat) and `exchange` 382. `st.db` read 0, which is void
  (amendment above).
- **No reading.** u1 did not return, and p1 never ran, so there is no answer, time or cost for `--size 100` on silicon.
  The stage, the reader and these limits stand for the final campaign.
  - Before that run, the heartbeat's size field needs a command that the board's busybox has, for example `wc -c`.
    It must also print `NA` instead of 0 when the command fails.

## PHP against a pristine second binary (2026-11-22)

Item 4 of the programme: Fig. 10's PHP pairing compares the protected image with a pristine second binary. The P1 row
(2026-11-18) compared two run-time modes of one image instead, mode 4 against mode 6.

### Pre-registration: QEMU twin, PHP pristine vs protected (written 2026-11-22 06:10, before the run)

- **Images.** Both come from the same `perf/appd/php/build.sh` build (FRCDKMS SDK, the F/P1 flags at -O2):
  - not protected: `php-stock.strip.dom` `13d187b0…`, pristine `zend_alloc.c` over mallocng (`PHP_REALPHP_ALLOC=stock`).
    It carries no `PHP_CAP_MODE` or `ZEND_CAP_MODE` string;
  - protected: `php-nested.strip.dom` `d63abbc5…` in mode 6, the P1 row's protected arm.
- **Stage.** `nested/appd/php10s`, made with `make-abba-stage.py`: u1 p1 p2 u2, the P1 row's input and workload
  (`bench10-check.php` `655fd6d4…`, `-n`), the same normalisation and output capture. Its `run.sh` differs from
  `nested/appd/php10`'s only in the stage name and in the stock arm's `PHP_CAP_MODE=4`, which the pristine binary does not
  read.
- **Run.** One twin boot on slot 28, the flags of `nphp10-1`, no plugin. Read with `parse-abba.py`, as `nphp10-1` was.

Predictions:
- **Answers.** Every command's normalised output equals the native reference `8f692e64…`. The protected commands print
  `ZEND_CAP_MODE 6 exact=0 revoke=1`, and the stock commands print no ZEND_CAP_MODE line.
- **Repeats.** Within 0.1 % per arm on in-context cycles.
- **Cost, protected vs pristine, in-context.** UNCERTAIN in size. On the twin the lifetime alone (mode 6 vs mode 4) was
  +0.16 %. Mode 4 also gives every block back to mallocng, where pristine zend_alloc keeps its own caches, so this
  pairing adds the geometry difference. I predict +0.5 % to +15 %. A result below mode 6 vs mode 4's +0.16 % would mean
  that pristine zend_alloc costs more than mode 4's geometry; that is to be explained, not absorbed.
- **The geometry alone.** This boot gives protected vs pristine. Pristine vs mode 4 then comes from comparing this
  boot's stock arm with `nphp10-1`'s mode-4 arm, which are two boots. It is labelled as such, not read as a paired
  result.

VOID if any command's answer differs, or if the protected commands do not print mode 6.

### Result: protected PHP costs +15.43 % against pristine zend_alloc, just above the predicted band (`nested/out/nphp10s-1`, 06:40-10:12)

| command | rc | answer | in-context cycles | kernel + other | runs |
|---|---|---|---:|---:|---:|
| u1 pristine | 0 | `8f692e64…` | 7,605,688,343 | 37,199,838 | 4,949 |
| p1 mode 6 | 0 | `8f692e64…` | 8,779,246,529 | 49,800,409 | 6,478 |
| p2 mode 6 | 0 | `8f692e64…` | 8,779,247,394 | 49,935,512 | 6,478 |
| u2 pristine | 0 | `8f692e64…` | 7,605,688,196 | 36,830,655 | 4,949 |

Read with `parse-abba.py` from the boot's serial log (sha256 `6f661677…`); the stderr of each command decoded from it.
- **Answers: met.** Every normalised output equals the native reference. p1 and p2 print `ZEND_CAP_MODE 6 exact=0
  revoke=1`, and their allocator line reads 3,442,469 issues, retires and revokes with 0 live. u1 and u2 print
  nothing on stderr. Not VOID.
- **Repeats: met**, 0.00 % in both arms (147 and 865 cycles apart).
- **Cost, protected vs pristine, in-context: +15.43 %. Predicted +0.5 % to +15 %: missed, just above the band.**
- The protected arm reads the same as `nphp10-1`'s protected arm (8,779.2 M), so the two boots agree where they
  share an image.
- **The geometry alone** (two boots, not a paired result): pristine reads 7,605.7 M against `nphp10-1`'s mode 4,
  8,765.5 M. On the twin, mode 4's geometry (every block back to mallocng) therefore costs +15.25 % over pristine
  zend_alloc's own caches, and the lifetime adds +0.16 % on top. The protected arm also makes 1,529 more trips out of
  context per command (6,478 against 4,949 runs) and spends 12.8 M more cycles outside it. That fits mallocng mapping
  and unmapping what zend_alloc would keep, but this run does not show the calls themselves.
- So on the twin almost all of PHP's cost against pristine is the allocator substitution, not the lifetimes. A protected
  zend_alloc that keeps its own chunk caches (lifetimes carved from its chunks, as the perlbench and gcc ports do) is
  the variant that would separate them. It has not been built. The P1 row's unprotected arm (`nested/appd/php10`) runs
  mode 4, not pristine; which comparison the paper uses is the lead's decision.

### Pre-registration: PHP pristine against protected mode 6 on P1, u1 p1 p2 u2 in one boot (written 2026-11-25 00:28, before the boot)

**Question.** The twin's result above puts protected PHP at +15.43 % against pristine zend_alloc. Almost all of that is
the allocator substitution, not the lifetimes. This boot reads the same pairing on P1. That is Fig. 10's pairing, a
pristine second binary, which the P1 row (mode 6 against mode 4 of one image, +2.34 %) does not provide.

**Configuration.**
- **Stage** `nested/p1ps/php10s` (`php10s.SHA256SUMS`, 20 files), made by bkit91's `make-train-stage.sh` from
  `nested/appd/php10s`, the twin's stage.
  - Against the P1 row's board stage `nested/p1a/php10` (`diff -rq`), three files differ:
    - `php_unprot` is the pristine `13d187b0…` in place of `d63abbc5…`;
    - `run.sh` drops `PHP_CAP_MODE=4` from the two u commands and renames the stage;
    - `plan.json` names the stage and the u image.
  - `php_prot` (`d63abbc5…`, mode 6), the workload (`bench10-check.php` `655fd6d4…`), the normalisation and the smoke
    are the P1 row's, byte for byte.
- **Kit** `bkit91-k2n-native-p1-tree-cscfix` (SHA256SUMS verified), the P1 row's kit: 262,144 records.
- **Bitstream** P1 `…-tagoverlap6c-pagedtbl_d83d3a633`, board 2, where it is resident. The chain reads it back by name
  and flashes it, sha checked, only if it is not.
- **Chain** `nested/p1s100/nat-run3p.sh 2 1 nested/p1ps/p1ps-b2.conf`, list line `php10s 2400 120`. That gives
  `BAKED_TIMEOUT=9600`, as the row's, and a booking of 120 min.
- **Checked off the board** (rule 8, `nested/p1/recheck-p1.sh` with `RD=nested/p1ps`):
  - real bake rc 0;
  - `BOARD_STUB=2` rc 0 on the P1 row's real transcript (`out-nat-p1a-php-b2/php10`) and on its cut after
    `@@SPEC-BEGIN`, both PASS all 9 steps and print `BAKED_TIMEOUT=9600`;
  - real firmware 19,566,088 B (`9dc4cadc…`);
  - an `NR_DRY=1` pass of the chain to DONE (`nested/p1ps/dry.log`).
- **Reader:** `nested/appd/parse-abba.py` (`1d0459d5…`) against the native reference `8f692e64…`, then
  `nested/p1/abba.py` (`fe8e1531…`).
  - On the P1 row's transcript it reproduces that row's figures (+2.32 % span, +151.69 % record-TLB misses).
  - On the cut transcript it exits 2.

**Rules.**
- **Answers:** every command rc 0 with the native normalised output. p1 and p2 print `ZEND_CAP_MODE 6 exact=0
  revoke=1` and retire all 3,442,469 issues. u1 and u2 print no `ZEND_CAP_MODE` line. Otherwise VOID.
- **Repeats:** `run_cycles` within 1 % in each arm, or the cost is not read.

**Predictions** (in-context cycles, ABBA means).
1. **The protected arm repeats the P1 row's protected arm** (22,903.2 M) within 1 %: the same image, kit and
   bitstream in a different boot.
2. **Cost, protected against pristine: +8 % to +31 %, point +18 %.**
   - On P1 the row's mode-4 arm ran 2.553 times its twin cycles.
   - The pristine arm at 2.3 to 2.8 times its twin 7,605.7 M gives 17,493-21,296 M. Against 22,903 M that is +7.5 %
     to +30.9 %, and +18.1 % at 2.553.
3. **The geometry share:** P1's mode 4 against pristine (two boots, not a pair) is the substitution's cost on silicon.
   The row's +2.34 % is the lifetime's. The prediction is that the substitution is most of the total, as on the twin.
4. **Duration:** 12 to 16 min per pristine command (the row's mode-4 commands took 15.2 min), and about 75 min for the
   boot.

**Limits.**
- The pristine binary differs from mode 4 in block geometry and caches, not only in the lifetimes. So prediction 3
  rests on two boots, with no pair between them.
- One boot.
- hpm6 (grow) can read nonzero in the first commands, as on the row. It is reported, not read.

### Result: protected PHP costs +29.78 % against pristine zend_alloc on P1, inside the band (board 2, 2026-11-25 00:28-01:32, read against 246d1c4f383b)

The chain (pid 2204581) read P1 back by name as resident (no flash), booked 1491, and unbooked it. The smoke passed all
9 steps. The chain's `SPEC rc=2` is the generic classifier, which does not read this stage's markers; it is not a
verdict. Transcript `…/out-nat-p1ps-php-b2/php10s/board.txt` (sha256 `8506e1ca…`), read by `parse-abba.py`
(`1d0459d5…`) against `8f692e64…` and `abba.py` (`fe8e1531…`).

| command | rc | answer | in-context cycles | kernel + other | context entries |
|---|---|---|---:|---:|---:|
| u1 pristine | 0 | `8f692e64…` | 17,637,636,211 | 296,702,525 | 5,624 |
| p1 mode 6 | 0 | `8f692e64…` | 22,887,566,222 | 427,165,472 | 7,569 |
| p2 mode 6 | 0 | `8f692e64…` | 22,868,235,362 | 426,930,663 | 7,558 |
| u2 pristine | 0 | `8f692e64…` | 17,619,420,646 | 295,729,335 | 5,613 |

- **Answers: met.** Every normalised output equals the native reference. p1 and p2 print `ZEND_CAP_MODE 6 exact=0
  revoke=1`, and their allocator line reads 3,442,469 issues, retires and revokes with 0 live. u1 and u2 print nothing
  on stderr. Not VOID.
- **Repeats: met**, 0.10 % (u) and 0.08 % (p).

**Against the predictions.**
1. **The protected arm repeats the row: met.** 22,877.9 M against the row's 22,903.2 M, −0.11 %.
2. **Cost: +29.78 % (+5,249.4 M), inside the band** (+8 % to +31 %), well above the point (+18 %).
   - The pristine arm ran 2.318 times its twin cycles (17,628.5 M over 7,605.7 M), at the low end of the 2.3-2.8 range.
     The point assumed the row's mode-4 ratio, 2.553. The protected arm ran 2.606 times its twin's 8,779.2 M.
   - So on silicon pristine zend_alloc gains less over the twin than mode 4 and mode 6 do, and the cost in percent is
     nearly twice the twin's +15.43 %.
3. **The substitution is most of the total: met.** It is a cross-boot reading, as registered.
   - Mode 4 (the row's u, 22,379.1 M) against pristine (this boot, 17,628.5 M) is +4,750.6 M, +26.95 %.
   - The row's lifetime cost (mode 6 against mode 4) is +524.1 M, +2.34 %.
   - Of their sum, 5,274.7 M, the substitution is 90.1 %. The sum differs from this boot's own p−u (5,249.4 M) by
     25.3 M, 0.11 % of p, which is the cross-boot difference in the protected arm.
   - On the twin the same split was +15.25 % and +0.16 %.
4. **Duration: missed, narrowly.** The pristine commands took 11.98 and 11.97 min (predicted 12 to 16 min), and the
   protected ones 15.57 and 15.56 min. The boot took 63.8 min, against about 75.

**Reported, not read.**
- The protected arm enters context 34.6 % more often (7,569 against 5,624) and spends +44.2 % more cycles in kernel and
  other (+130.8 M). On the twin it was 1,529 more entries per command.
- Record-TLB misses +162 %, walk cycles +176 %, L1D misses +341 % (+77.6 M), event-23 replays +14 %.
- hpm6 (grow) reads 10 and 8 in u1 and p1 and 0 after, as on the row.

**Limits.** One boot. Prediction 3 rests on two boots with no pair between them. The pristine binary differs from mode
4 in block geometry and caches, not only in the lifetimes.

## Item 5: the SQLite and PHP bug corpora for the board (2026-11-23)

The programme's item 5: prepare the images and the pre-registration, and the runtime lane queues them on a P1 board.
Neither corpus has run on a board with its nested arm. SQLite's control arm (vp2-malloc) ran on vp2. PHP has no board
reading at all.

**Board kit (updated 2026-11-23 13:47): route (c) of the runtime lane's plan, a twin read, the board kit not yet
cut.** `docs/plans/2026-11-23-p1-virtual-corpus-kit.md` on spec-silicon (099555b5d3c5) settled it. The table-tree module
has no ioctl interface, so it cannot run these images, and dev-built corpus images exec'd as `.dom` under it die
before any output. The kit therefore runs the images' own adapter: dev 1a051da6e82b's module and launcher, the module
built FLAT for ids55 (`-DCAPSTONE_VM_FLAT_TABLE=1 -DCAPSTONE_VM_IDS=55 -DCAPSTONE_VM_INDEX_BITS=25`). That is P1's
flat table, which P1 has run on silicon (bkit88/89, 2026-11-16), not its paged tree. Its twin is kit-p1v (readings
below); the board kit is the runtime lane's next build. The text from here to the kit-p1v readings is as written
before that decision.

Earlier text: no P1 board kit runs the virtual profile today. The runtime lane checked every t3-smoke
kit: the non-native ones are on older bitstreams or riscv-base. P1's lifetime hardware is the paged table, which
the native kits drive with the table-tree module (llvm-capstone table-tree 722416c35e77, not on dev). So a P1 virtual
kit needs dev's virtual module plus the table-tree code, with a VM ABI matching kit-devq's runtime. That kit, its ABI
check and its Tier 1 boot are the runtime lane's next build. Every board claim below waits for it. The board
prediction will be the QEMU verdict on the same image bytes (Tier 3's rule).

**Images**, from the dev worktree at deb4ebf0b06b with the llvm-agp compiler (clang d9fa11a409d2), and corpus SDKs
(`-fno-builtin`, mallocng, virtual profile, application ABI 2):
- **SQLite engine-repros**, 33 cases per arm: `ports/sqlite/repro322/build-virtual.py` over the adapted 3.22.0
  amalgamation (`7fc92e19…`, the bundles' input), `--ext-src` the 3.22.0 full tree, and the two memsys5 controls
  (`control_uaf_mem5.c`, `control_bounds_mem5.c`) built in.
  - `virtual-malloc` (`--memsys5`): 33 built, 0 failed.
  - `virtual-nested-pools` (`--sublet`): 33 built, 0 failed.
  - SDK `sdk.json` `6ff09cfbe6ad7b5d…`.
  - By encoding, case 00's image carries CDERIVE 7 and CREVOKE 3 on the malloc arm (mallocng's own lifetimes),
    and 9 and 4 on the pools arm. Every CREVOKE has two operands, which is the form P1 decodes.
- **PHP engine-repros**: one interpreter image for the four `PHP_CAP_MODE` arms, built by
  `ports/php/app/build-virtual.sh realphp` on PHP 5.0.0 from `prepare-sources.sh` (tarball `5783e0c0…`, patch set
  `ac187130…`):
  - `php-real-nested.dom` `6321dcc8993ac0af…`;
  - the bound control's probe, `probe-nested.dom` `f6b1e7085a423dd7…`;
  - `probe-putenv.dom` `7367104ffe1552bb…`.
  - The build counts nested capability instructions as derive=8, revoke=4.
- **QEMU readings on kit-devq**, queued behind the twin's runs:
  - SQLite: one persistent VM per arm, `shared/run-virtual.py`;
  - PHP: `run-realphp.py`, one boot per population.
  - What those runs will read is not predicted here. Their verdicts become the board's predictions, and each is
    compared with the 2026-10-29 (SQLite) and 2026-10-28 run18 (PHP) bundles. A cell that differs from those
    bundles is reported as a difference of the newer toolchain and runtime before any board pre-registration.

### QEMU readings (2026-11-23 07:47-08:46)

Three kit-devq defects came out of these runs. The runtime lane owns kit-devq and is fixing all three. Until then,
every reading below says which QEMU ran it and which `-cpu` it applied.
- **The QEMU drops two properties.** kit-devq's `qemu-system-riscv64` execs kit-profq's wrapper around a Q-34
  binary (capstone-qemu 46b69c03a4).
  - It refuses `x-capstone-addr`, which `ports/php/app/run-realphp.py` sets, so on it no PHP arm booted.
  - It strips `x-capstone-exact-bounds=true` on purpose: that twin is RTL-pinned, and the silicon has no exact bounds.
    The bundle's platform record copies the requested `-cpu`, so every kit-devq bundle says `"exact_bounds": true`.
    The applied line is in kit-profq's `applied.log`; read the field as false.
- **A timeout does not end the guest job.** `appvm.run_app`'s 300 s timeout ends the host side only. SQLite case 11
  (fts3_zterm) spins on in the 1-CPU guest on the malloc arm. Every later case on that VM then read "runner timeout"
  with an empty output (cases 11-25 of the first attempt, discarded as no data). In the 2026-10-29 bundle, case 11 is
  the only timeout, and all 21 later cases began and completed or faulted. Its pools arm caught 11.
- **So the readings below are on capstone-qemu 0542c5556c, run directly.** That is the pin of dev 1a051da6e82b, which
  kit-devq's adapter comes from: binary `52f3daa98e3a487e…`, from a clean worktree. They use kit-devq's images and
  adapter (module `af7484e5e39019a6…`), and SQLite's case 11 runs alone and last on each arm.

**SQLite engine-repros, against 2026-10-29-virtual.**
- **Exact bounds on, 66 of 66 cells the same.** The `-cpu` applied was
  `rv64,sstc=false,h=false,sv48=false,sv57=false,x-capstone-u-mode=true,x-capstone-exact-bounds=true`. Same means
  verdict and reason; the controls read as expected on both arms.
  - Added 2026-11-23 13:47: "same" was verdict and reason, not the observation. One cell's observation differs. On
    the malloc arm, case 24 (`writable_schema_stale_table`) completed in 10-29 and faults today with cause 24 in
    `sqlite3OpenTable`. It reads that way in all three of today's runs (both QEMU profiles and kit-p1v below), and both
    verdicts are MISSED. The images differ (case 24's malloc image `60b732d36a08…` today, `51ba0b0170c8…` in 10-29),
    and vp2's board read the same fault in 2026-10-18-vp2-rdx0 (`vp2-malloc-sq0e`: cause 24 in `sqlite3OpenTable`).
    So it is the later image's reading, the program's own consequence fault on the arm with no mechanism.
  - virtual-malloc: MISSED 29, NO-READING inconclusive 1 (06), unattributed 2 (02, 20), infra 1 (11, the timeout).
  - virtual-nested-pools: CAUGHT 29, MISSED 4.
- **Exact bounds off (kit-devq): 64 of 66.** The two other cells are both pools-arm overreads:
  - 29 (`sqlite3Get4byte`) reads cause 24 where 10-29 read cause 28 CAUGHT;
  - 31 (`statDecodePage`) does the same.
- **The pair that differs in the property alone.** Same binary 0542c5556c, same images, pools arm, cases 25, 29 and
  31:
  - `x-capstone-exact-bounds=true`: 25, 29 and 31 all cause 28 (`RISCV_EXCP_CAP_OOB`, 0x1c), CAUGHT;
  - the property absent: 25 cause 28 CAUGHT; 29 and 31 cause 24 (`RISCV_EXCP_UNEXP_OP_TYPE`, 0x18, `cpu_bits.h:699`)
    in the same function, which the arm's judge does not count as its mechanism, so MISSED.
- **Where the cause-24 traps are.** Both halves of the pair trap at the same instruction and the same address,
  on the same image (sha256 `59b4941d…` for 29, `9f6d8fb7…` for 31). Offsets below are from the code window's
  base, which is ELF 0x10000:
  - 29: pc +0x643fc, address +0x35a3a2, ra +0x2041f0 in both halves. That is ELF 0x743fc, `lbu a0, 0(a1)` in
    `sqlite3Get4byte`, the first byte load of the 4-byte read.
  - 31: pc +0x204290, address +0x3585e0 in both. That is ELF 0x214290, `lbu a1, 0(a0)` in `statDecodePage`, right
    after `cincoffset a0, a0, a1` with an offset read from the page.
  - In the exact-bounds half the same pc and address are cause 28, so that instruction is the out-of-bounds load
    and the address is outside the capability.
- **Reading.** The property explains the two cells, and under compact bounds, the profile the silicon has, both
  overflowing loads are still stopped, at the load itself. Only the cause differs. Cause 24 there fits the reference's
  moved-capability rule (`op_helper.c:845`): a cursor that is not representable leaves an untagged integer, and its
  dereference traps with 24. That the tag was dropped is inferred from the cause; the operand's tag was not read.
  So the bugs are caught, and the judge's mechanism list for the pools arm lacks 24 under the compact profile. This
  is a finding about the judge against the 10-29 bundle's exact-bounds verdicts, for the lead. Which cause the board
  will read is the board's question, and no paper text changes here.

**PHP engine-repros, against 2026-10-28-run18.** Both populations, the four arms each.
- Every arm booted.
- "all cross-arm gates pass" in both logs (image capability instructions derive=8 revoke=4).
- The `-cpu` was run-realphp.py's own, with `x-capstone-exact-bounds=true` and `x-capstone-addr=sv39`, applied
  unchanged by the bare binary.

| arm | asan-blind (39): CAUGHT / MISSED | temporal-asan-visible (31, 25 out of denominator): CAUGHT / MISSED |
|---|---|---|
| stock | 5 / 29 | 1 / 4 |
| system allocator (block) | 12 / 21 | 5 / 0 |
| nested (object) | 26 / 10 | 5 / 0 |
| nested exact | 31 / 4 | 5 / 0 |

- 274 of 280 cells equal run18's in verdict and reason. The six that differ are two cases:
  - **30 (crash-092): the per-run watchdog moved.** In run18 it killed the two nested arms (infra), and stock and
    block read not-reached. Today it kills stock and block (infra), and both nested arms read CAUGHT (cause 25 in
    `_zval_ptr_dtor`). A timing case, on a different QEMU.
  - **41 (crash-139, an `efree` of the `.rodata` literal "message").** In run18 the nested layer refused the free
    (NO-READING unattributed, code=3). Today both nested arms read MISSED. But in today's image no arm performs that
    free: `blocks=342 frees=342` in all four, which read the same. So the MISSED is a non-reach the judge cannot see,
    not a free the layer accepted.
    - Not attributed. A candidate: the trigger's member zval is never refcount-initialised (its header), so whether
      `zval_ptr_dtor` frees depends on stack contents. Those can move with the image: dev's nested layer since
      bbedfbd40c6c, `php-real-nested.dom` `6321dcc8…` against run18's `101c6524…`.
    - So today's nested arm has one temporal MISSED where run18's has none, and that cell is a difference of reach.
- These verdicts are the QEMU side of the board's predictions. The board pre-registration still waits for the P1
  virtual kit above.
- Superseded as the board's prediction by the kit-p1v readings below: kit-devq is not the board's module.

### Readings on kit-p1v, the P1 twin (2026-11-23 12:12-13:41)

kit-p1v is the runtime lane's `perf/p1v/kit-p1v` (its gate: 11:12, 25/28/28, `ids=55 index_bits=25
node_flat_records=1048576`):
- QEMU: a wrapper (`e855047c…`) over capstone-qemu 0542c5556c (binary `52f3daa98e3a487e…`). It adds
  `x-capstone-ids=55` and `x-capstone-addr=sv39`, and strips exact bounds and test CSRs.
- Images: kit-devq's (Image `e5861359…`, fw_jump `6f2b082c…`, rootfs `9903242c…`).
- Adapter: launcher `2488e5d2…` and the ids55 flat module `0d5a7d627787af56…`.
- The corpus images are the ones above, unchanged; every verdict's `image_sha256` is in the manifests below.

Committed per corpus under `results/2026-11-23-p1v-twin/`: the bundles (`inputs.json`, `verdicts.jsonl`, with account
paths shown as `<hdd>/`) and `images.sha256`. SQLite's manifest has 70 lines, the 33 cases and two controls per arm
(`267960f79021a726…`). PHP's has the interpreter and two probes.

**SQLite** (`run-sqlite-p1v.sh`; one VM per arm, `capstone_vm up`, `-m 8G`, `cma=512M`, `supervisor_slots=1`; case 11 alone
and last):
- applied `-cpu` (kit-p1v's `applied.log`, 12:12:48 and 12:19:31): `rv64,sstc=false,h=false,sv48=false,sv57=false,
  x-capstone-u-mode=true,svadu=false,x-capstone-ids=55,x-capstone-addr=sv39`. No exact bounds, and the bundles say
  `"exact_bounds": "False"`, which here is what was requested and what was applied.
- dmesg: `flat table 1048576 records, 16777216 bytes, from CMA`.
- Controls on every row: malloc arm `uaf-mem5` and `bounds-mem5` complete (33/33), pools arm both fault (33/33).
- virtual-malloc: MISSED 29, NO-READING 4 (02 and 20 unattributed, 06 inconclusive, 11 infra, the 300 s timeout).
  Faults among the MISSED: 05, 19, 22 and 24, cause 24 each.
- virtual-nested-pools: CAUGHT 27 (cause 25: 20; cause 28: 7), MISSED 6 (01, 04, 08, 09; 29 and 31 with cause 24).
- **Against kit-devq (compact bounds): 66 of 66 the same** in verdict and cause. **Against exact bounds (0542 run
  with the property) and against 10-29: 64 of 66**, the two cells being pools 29 and 31. That is what the compact
  profile predicted, from the pair above.

**PHP** (`run-php-p1v2.sh`; `run-realphp.py`, one boot per arm, `-m 2G`, module args `node_initial_pages=1024
supervisor_slots=32`):
- First attempt void (`res-p1v`): no arm ran. The flat module takes its 16 MiB table from CMA, and run-realphp's VM
  boots with `-append "root=/dev/vda ro"`, which has no CMA. dmesg read `flat table ... from CMA: ALLOCATION FAILED`,
  and every case read `capstone-vexec: open: Cannot allocate memory`. `capstone_vm up` appends `cma=`, so SQLite was
  not affected.
- Second attempt: QEMU is `nested/corpus5/qemu-p1v-cma.sh` (`e3badf60…`). It adds `cma=256M`, the board's value
  (`capstone_vm.c:485`), to that `-append` and execs kit-p1v's wrapper. It booted all eight arms (13:30:00-13:39:57
  in kit-p1v's `applied.log`, with `x-capstone-supervisor-slots=32` and no exact bounds). dmesg read `262144K
  cma-reserved` and `flat table 1048576 records ... from CMA`.
- The platform record copies run-realphp's requested `-cpu`, which asks for exact bounds and test CSRs; neither was
  applied (the lines above).
- "all cross-arm gates pass" in both logs (derive=8, revoke=4).

| arm | asan-blind (39): CAUGHT / MISSED | temporal-asan-visible (31, 25 out of denominator): CAUGHT / MISSED |
|---|---|---|
| stock | 5 / 29 | 1 / 5 |
| system allocator (block) | 12 / 21 | 6 / 0 |
| nested (object) | 26 / 10 | 5 / 0, infra 1 |
| nested exact | 31 / 4 | 5 / 0, infra 1 |

- **Against today's kit-devq run (res2): 278 of 280 cells the same** in verdict, reason, cause and function. All 156
  asan-blind cells match, and 122 of 124 temporal ones. The two others are one case, 21 (crash-075):
  - res2: infra on all four arms, a cause-30 node-table exhaustion (the judge's "not a reading about the program");
  - kit-p1v: stock MISSED (completed, 550,844 nodes allocated), block CAUGHT (cause 25 in `php_array_walk`, the
    declared function), both nested arms still infra with cause 30.
  - The table is the difference: capacity 262,144 records on kit-devq's module, 1,048,576 on kit-p1v's (each bundle's
    notes). No other case allocates more than 4,344 nodes, on either kit.
- So against run18, the cells that differ are 30 and 41 as before, plus 21 on stock and block.

Not measured here: memory and cycles. These runs read verdicts only.

### Pre-registration: item 5 on a P1 board (written 2026-11-23 13:47, before any board kit or run)

What runs: both corpora's images above, the same bytes (`results/2026-11-23-p1v-twin/images.sha256` per corpus), on a
P1 board (bitstream `caplifive_ids55-…-pagedtbl_d83d3a633`) under the runtime lane's board kit for route (c). The
kit is dev 1a051da6e82b's launcher `2488e5d2…` and flat ids55 module `0d5a7d627787af56…`, on a board kernel with no
built-in capstone_vm. **This is P1's FLAT table, not its paged tree:** no reading below is about the paged table.

**The prediction is the kit-p1v twin's cell, on the same image bytes and module.** That covers verdict, reason, cause
and faulting function, for 66 SQLite cells (`2026-11-23-p1v-twin/{malloc,malloc-11,pools,pools-11}`) and 280 PHP
cells (`2026-11-23-p1v-twin/{asan-blind,temporal-asan-visible}`), with these declared exceptions:
- **Timing cells: verdict class only.** These are SQLite malloc 11 (the 300 s timeout, infra on the twin) and PHP 30
  (crash-092, the watchdog moved between kits). Predicted: not CAUGHT on malloc 11; PHP 30 is not predicted.
- **Consequence-fault causes on arms with no catch.** On silicon, an untagged dereference is reported as cause 4 or
  13 as often as 24 (`arms.json`, silicon-sqlite-memsys5's `catch_causes_why`). vp2 read 4 and 13 on malloc 19, 20
  and 22 (2026-10-18-vp2-rdx0). So on the malloc arm only the verdict is predicted for the faulting cells 05, 19, 22
  and 24 (MISSED), and for 20 (NO-READING). On PHP's stock and block arms, a MISSED with a fault is predicted as MISSED
  with any cause outside the mechanism set.
- **SQLite pools 29 and 31: MISSED, with a cause outside the mechanism set,** at the twin's instruction if the pc is
  read (ELF 0x743fc `lbu` in `sqlite3Get4byte`; ELF 0x214290 `lbu` after `cincoffset` in `statDecodePage`).
  - Branch: CAUGHT with the bounds cause on either is a twin-vs-silicon difference in CINCOFFSET on an
    unrepresentable cursor (the twin follows the reference's moved-capability rule, `op_helper.c:845`). It goes to the
    RTL lane, and it is not a protection reading.
- **PHP temporal 21 (crash-075): infra (cause-30 table exhaustion) on both nested arms.** That reading depends on the
  table's capacity, which the geometry gate fixes. On stock and block it is the twin's cell.

**Gates on the board's first boot, before any cell is compared:**
1. **Cause numbering.** The judge's `catch_causes` is `[25, 28]` in the twin's numbering (cpu_bits.h: 25 INVALID_CAP,
   28 CAP_OOB). The launcher prints the module's step cause unchanged (`exec.c:1122-1127`, `step->cause` from the
   supervisor frame, `capstone_vm.c:2706`). On silicon that is the hardware's number. `docs/ref/README.md:22-23`
   gives 26 for INVALID_CAPABILITY and 29 for OUT_OF_BOUNDS. vp2's board read 24 where the twin reads 24, for the
   untagged dereference, and it had no mechanism arm to show the other two.
   - So the pools arm's two controls (`uaf-mem5`, `bounds-mem5`) and a SQLite case-25 positive read the numbers.
   - If the stale access reads 25 and the bound 28, the judge applies as is.
   - If they read otherwise, no cell is judged until an amendment naming the mapping is pushed.
   - Predicted: the stale access 26 and the bound 29. That is README's table, with the launcher passing the frame
     through.
2. **Geometry.** On the board, `/sys/module/capstone_vm/parameters/` must read `ids=55`, `index_bits=25` and
   `node_flat_records=1048576`, as on the twin's gate. A different capacity voids the comparison of PHP 21 and of
   any infra cell with cause 30.
3. **CMA.** The board kernel's command line must reserve CMA of at least 16 MiB; `capstone_vm.c:485` gives
   `cma=256M`. The boot's dmesg must show `flat table 1048576 records, 16777216 bytes, from CMA` without `ALLOCATION
   FAILED`. Without it, every case reads ENOMEM, as PHP's first twin attempt did.
4. **Slots and memory.** PHP's twin ran `supervisor_slots=32` against a QEMU with 32 slots, and SQLite's ran 1. The
   board's module argument must equal the bitstream's slot count. A module with more slots than the hardware has halts
   the guest at child entry. If the board needs a different count, or less memory than the twin's (2 GiB PHP, 8 GiB
   SQLite), the kit's README says so. Any cell that then differs is read as a candidate kit difference first.
5. Each boot runs the corpus's virtual positive control first, and puts at most one wedge risk last (Tier 3's rule).

**Branches:** a cell outside the exceptions that differs from the twin is a silicon-vs-twin difference. It is reported
with both cells and the image hash before any count is said. A whole arm that differs, such as every pools cell, is
read as a kit fault first: gates 1-4.

Memory and cycles are not measured by this run.

### Amendment 1 to the item 5 board pre-registration (written 2026-11-23 15:11, before any board kit has booted)

This replaces gates 1, 2 and 4 above and adds gate 6. The rest of the pre-registration stands, including the
exceptions and branches.

**Withdrawn: gate 1's "Predicted: the stale access 26 and the bound 29".** This lane wrote it from
`docs/ref/README.md`'s cause table, which predates capstone-ariane c77c65324 (2026-09-15), the renumbering. dev
8b3d792e6b5d corrects the README. P1's own source agrees with the twin's numbering:
- `riscv_pkg.sv:358-363` at the bitstream's revision d83d3a633 gives 24 UNEXPECTED_OPERAND_TYPE, 25 INVALID_CAP, 26
  UNEXP_CAP_TYPE, 27 INSUF_PERM, 28 OOB and 29 ILLEGAL_OPERAND_VALUE.
- Those are cpu_bits.h's numbers.

**Gate 1, as amended.**
- Predicted: the stale access reads 25 and the bound reads 28, and the judge's `catch_causes [25, 28]` applies
  unchanged.
- A 26 or 27 on either control means the arm's capability is not the one it assumes: wrong type, or wrong
  permissions. A 29 means a wrong operand. In each case no cell is judged until it is read.
- **A 25/28 swap is not a verdict difference.** Some cells read 25 on the twin and 28 on the board, or the reverse.
  Such a cell is reported as a check-order difference: same verdict, both causes listed. These are the cells whose
  access is both stale and out of bounds.
- Where the order is known:
  - At d83d3a633, the M-mode capmode load/store check (`load_store_unit.sv:1390-1465`) raises 28 for bounds before
    25 for lifetime.
  - For LDC/STC, ex_stage applies the lifetime deny (25) only when the DYN unit raised nothing (`ex_stage.sv:1528-1533`).
  - The order for the corpora's own accesses in a supervised user context is not established here.

**Gate 4, as amended: P1 has one saved context.**
- The rtl lane reports that P1 holds one saved virtual context: `csr_regfile.sv:392-393`, "the one saved virtual
  context (QEMU keeps 32 slots; the plan takes one)".
- The board's module therefore runs `supervisor_slots=1`.
- PHP's twin cells were read at 32 slots, so both corpora were re-read at P1's geometry (below). The memory part of
  gate 4 is unchanged.

**Gate 2, as amended.** The module also runs `resolve_dirty=1 resolve_pte=1`. On silicon, the flat ids55 module
matches its twin only when loaded with `resolve_dirty=1`, S-24's fix, which defaults off. The source is dev
`tests/rtl-smoke/ids55-flat-module/RESULTS-twin.md`, "Board": bitstream 325966ee4, boot b4-t1c, kit bkit14-b4r-b2, both
parameters set. Loaded bare, the module read 28/45. The twin never takes that branch. So on the board, `/sys/module/capstone_vm/parameters/` must also
read `supervisor_slots=1`, `resolve_dirty=Y` and `resolve_pte=Y`.

**Gate 6, new: module identity.**
- The board module cannot be the twin's bytes. It is built for the board kernel 6.4.14; the twin's was built for 6.1.
- The runtime lane's board build of 13:58 (`perf/p1v/buildb.sh`, output `b55dev`) is `74b4cceb69d159c1…`.
  - Source: the same `capstone_vm.c`, `f256348b49d59426…`.
  - Flags: the same `KCFLAGS`, `-march=rv64g -mabi=lp64d -DCAPSTONE_VM_FLAT_TABLE=1 -DCAPSTONE_VM_BENCH_COUNTERS`.
  - vermagic `6.4.14`; no 2-byte instructions among its 8,071.
- The module the board loads must be the one the runtime lane's kit README names, with this source hash. If it is
  not this build, the kit README says what differs.

**The P1-geometry re-read (14:49-15:07), now the prediction of record.**
- What ran:
  - The same image bytes and kit-p1v.
  - Module arguments `supervisor_slots=1 resolve_dirty=1 resolve_pte=1`; QEMU with `x-capstone-supervisor-slots=1`.
  - The CMA reserve was 256 MiB; the flat table came from CMA on every boot.
  - The kit's applied `-cpu` (`ids=55`, `sv39`, no exact bounds) was read from each PHP boot's `KIT-P1V-APPLIED` line
    and from the running SQLite QEMU's argv. The parameters were read back from sysfs on both SQLite VMs.
- Bundles: `results/2026-11-23-p1v-twin-p1geom/`, per corpus.
- **Every cell equals the 12:12-13:41 readings.** SQLite: 66 of 66 in verdict, reason, cause, function and offset
  (only absolute addresses differ, with the load base). PHP: 280 of 280 in verdict, reason, cause and function.
- So the predicted cells and the exceptions above are unchanged. Neither the slot count, nor `resolve_*`, nor the
  smaller CMA moves any cell on the twin. SQLite malloc 11 is again the runner timeout (rc 75), as the exception
  already allows.

### Amendment 2 to the item 5 board pre-registration (written 2026-11-23 15:25, before any board kit has booted)

The runtime lane cut the board kit as `t3-smoke/bkit109-p1v-corpus-flat`. Its README answers gate 6. Two corrections
and one addition follow.

- **Gate 6's flag list was incomplete.**
  - The module's identity defines are four: `CAPSTONE_VM_FLAT_TABLE=1`, `CAPSTONE_VM_IDS=55`,
    `CAPSTONE_VM_INDEX_BITS=25` and `CAPSTONE_VM_BENCH_COUNTERS`.
  - The source alone would give ids 30 and index bits 16 (`capstone_vm.c:35-36`, `:245-246`), so ids and index bits
    come from the build's extra `KCFLAGS`.
  - The runtime lane compared the twin and board builds' defines and found them identical. Gate 2's sysfs read of
    `ids` and `index_bits` checks the same thing on the board.
- **`supervisor_slots` is not a board parameter.**
  - It is read-only after load, with default 1 (`capstone_vm.c:616-617`, mode 0444).
  - So the kit's `params` sets only `resolve_dirty=1 resolve_pte=1`. Gate 2 still reads `supervisor_slots=1` from
    sysfs.
- **Stages.**
  - The firmware payload is capped at 26,000,000 bytes, and an image is about 3.3 MB (PHP's nested build 4.9 MB). So
    each corpus runs as several boots of a few images.
  - Every stage boot runs its corpus's virtual positive control first (gate 5). A stage whose control fails is void,
    and none of its cells is compared.
  - The twin ran each arm's cases in one VM. So when a board cell differs from the twin, its position in the stage is
    reported beside it. Before anything else, the same case is read as the first case of a stage (boot history).

### Amendment 3 to the item 5 board pre-registration (written 2026-11-23 17:27, before any board kit has booted)

This amendment replaces Amendment 2's stage split with the route agreed with the runtime lane. It also states the stop
gate, the budgets each boot gets and the sources of those budgets, and two facts about reading the board bundles.

- **Route.**
  - SQLite runs as one boot per arm, pools first, then malloc. A boot runs the runner's two controls, then every case
    except 11, then case 11 alone, last. This is the twin's order, where case 11 also ran last in its VM.
  - Both arms go through `appvm-session.py`: phase collect, then one board boot, then phase read. The kit is
    `bkit109-p1v-corpus-flat`, run with `BOARD_STAGE_XZ=1`, which packs the stage's images into one xz archive. With
    the archive, an arm's 33 images fit under the 26,000,000-byte firmware cap. Without it, they do not.
  - PHP runs through the runtime lane's board runner `app-p1b`. Its `run-realphp.py`, `count-cap-insns.py` and
    `run-virtual-twin.py` are byte-identical to the twin's. That makes four boots per population, one per arm, for
    eight PHP boots in all.
  - Every boot carries its own controls: SQLite's `uaf-mem5` and `bounds-mem5`, and PHP's sanity and probe arms. So
    Amendment 2's rule holds for each boot: a boot whose control fails is void.
  - Amendment 2's position rule still applies. A board cell that differs from the twin is reported with its position in
    its boot, then re-read as the first case of a boot.
- **The session mechanism's own QEMU control.** Pools as one `appvm-session` boot on `vkit-p1v` (kit-p1v, with the
  board's two parameters) equals the P1-geometry twin in 33 of 33 cells. Malloc's control is queued behind the QEMU
  lock and is reported when it lands. The board boots do not wait for it.
- **Stop gate after pools.**
  - The chain reads the pools boot's controls from its `verdicts.jsonl`. The gate passes only when `uaf-mem5` reads
    fault 25 and `bounds-mem5` reads fault 28. Anything else stops the chain before malloc and before any PHP boot.
  - Both branches were dry-run: the empty transcript reached the STOP branch, and a transcript built from the QEMU
    session control reached the PASS branch. The PASS-branch dry run goes as far as the chain's marker.
  - Dry-run readings (17:28-17:31, every board call stubbed: lab-book, including one rc 3, and the board boot):
    - pools through the board read path equals the P1-geometry twin in 33 of 33 cells;
    - the same pools transcript fed to the malloc arm reads `NO-READING:control-failed` in all 32 cases, because
      stock's controls expect completion. That is the void rule firing;
    - PHP equals the twin in 280 of 280 cells on verdict, reason, cause and symbol;
    - the marker was written after the last unbook.
  - The marker's per-step codes are the scripts' exit codes, not readings. A void boot shows in its bundle, not in
    the marker.
- **Budgets each boot actually gets.** Each figure is read from the staged `gate.sh` or the kit, not from a runner
  default.
  - SQLite.
    - The boot timeout is `appvm-session`'s default, 300 + Σ min(step, 600). For 35 steps of 300 s that is 10,800 s.
    - The in-guest budget is that timeout minus boardvm's 300 s margin, so 10,500 s. Each step's bound is min(the
      remaining budget − 60, 6,000) s.
    - The driver gets 10,800 × `BOARD_TIMEOUT_MULT` 4 = 43,200 s, plus 1,800 s.
    - The in-guest budget is not multiplied. So the binding limit is 10,500 s, not the 43,200 s the multiplier
      suggests.
  - Source of the SQLite budget figures.
    - The QEMU session control on `vkit-p1v` ran all 35 steps of pools in 62 s of guest time. The kernel clock reads
      62.2 s at the last step.
    - The kit's README puts a board step at about 18 times its QEMU time (vp2, pym0).
    - So the expected pools boot is about 19 min of steps, about nine times inside the budget.
    - Case 11 faulted in 2 s on the twin and exits on the QEMU control with status 139. It is not the spin that
      the earlier kit-devq attempts had to work around.
  - PHP.
    - Each run's in-guest watchdog is the twin's run timeout times 20, so 800 s. The idle limit is 1,100 s. Both
      figures are read from the dry-run bundle's `board` field.
    - The twin's arms ran 48 to 111 s of guest time each. At 18 times that, a boot runs up to about 35 min.
- **Booking.**
  - Board 2 is booked through `lab-book.py` once per corpus step: pools 180 min, malloc 240 min, and each PHP
    population 300 min.
  - Each booking is released when its step's boot exits, before the next is booked.
  - A held window (rc 3) is retried every 300 s, up to 12 h.
  - On every exit path, the chain unbooks and then writes `corpus5/board/ITEM5-BOARD-DONE` with its status. Those
    paths are done, stopped, and SIGINT/SIGTERM. The runtime lane's h16s and the board lane's CSMINT kit wait on that
    marker.
  - The chain starts only after the runtime lane's mcf train on board 2 has exited.
- **Gate 4 gets one more candidate kit difference: memory.**
  - The board has 962 MiB, with 256 MiB of CMA.
  - The twin had 8 GiB for SQLite and 2 GiB for PHP.
  - A board cell that differs from the twin through an allocation failure is read as this difference before anything
    else.
- **Reading the board bundles.**
  - The board arm is recorded as `vp2-<arm>` (`boardvm.record_arm`), even on P1. The bitstream comes from the
    bundle's firmware record, not from the label.
  - In PHP, `platform.json` hashes the `--qemu` and `--images` inputs, which the board does not run. The board's
    identity is the `board` field of each `<arm>-run.json`.

### Amendment 4 to the item 5 board pre-registration (written 2026-11-23 20:26, before any board kit has booted)

This amendment corrects one statement in Amendment 3 and records malloc's QEMU session control.

- **Correction: what Amendment 3 says about case 11 holds for pools only.**
  - On malloc (stock), case 11 does not fault.
    - The P1-geometry twin reads it `NO-READING:infra` (runner timeout).
    - In the QEMU session control, the step began and never ended. The QEMU gate has no per-step bound, so the boot
      ran to its overall timeout (`boot DID NOT COMPLETE`).
  - On the board, the step is bounded by the gate's watcher, at min(remaining − 60, 6,000) s. If a spinning domain
    starves the watcher and the heartbeat, the driver's 200 s idle limit ends the rung instead.
  - Case 11 runs last, so the spin costs no other cell. The board's malloc boot is still expected to fit its 240-minute
    booking, with up to about 100 minutes of that spent in case 11.
  - The prediction for malloc's case 11 is the twin's `NO-READING:infra`. A board boot that does not complete because
    of this last step is not void for its other cells. Their readings are read from the rung segment, as the QEMU
    control's were.
- **Malloc's QEMU session control** equals the P1-geometry twin in 33 of 33 cells on verdict, reason, cause, symbol and
  offset. Its case 11 matches the twin's `NO-READING:infra`.

### Amendment 5 to the item 5 board pre-registration (written 2026-11-23 21:58, after the first board boot, before the re-slot)

**The first boot read nothing.**
- The chain booked board 2 at 21:39:55. The pools boot, `p1b1`, ran on bkit109 and resolved no corpus cell.
  - Both b0 controls returned 0.
  - The batch rung printed `@@XZSTAGE unpack 35 images` and then stayed silent. bkit109's wrapper unpacks 103,522,721
    bytes of images (3,934,579 compressed) and hashes them without printing.
  - After the driver's 200 s idle limit, the driver called the rung wedged.
- Several apertures changed between two reads, so the core was probably still running. This is not established.
- The stop gate read both controls as absent and stopped the chain before malloc. The board was unbooked at 21:51:05,
  and the marker was written as `STOP-GATE1`.
- No cell, control or verdict is taken from that boot.
- The stubbed dry runs could not show this, because the unpack runs only on the board.

**The kit becomes bkit111** (`t3-smoke/bkit111-p1v-corpus-xzv`, `board-run.sh` 359e90d6fc3dc265), cut by the runtime lane.
- It is bkit109 with one change, to the wrapper `gate.sh`. The wrapper now prints a line for each file it unpacks, an
  `unpacked` stamp, and one line per image hash. It also checks that the hash count equals the file count. So the
  longest silence is one image of about 3 MB.
- The rest of the kit is unchanged. The adapter, module, params, bitstream and monitor files are byte-identical to
  bkit109's (`diff -rq`). `board.env` and `base.manifest` differ only in naming the kit's own path. The kit's
  `SHA256SUMS` checks clean.
- The runtime lane ran the wrapper under busybox 1.36.1 against four inputs:
  - the positive case;
  - a corrupted hash, which fails as `hash`;
  - a truncated archive, which fails as `unpack`;
  - an empty list, which fails as `count 0`.
- It also ran the wrapper over a real bake of the pools stage: 35 file lines, then 35 hash lines, then `ok`.
- The PHP boots use the same wrapper.

**The SQLite boots' idle limit is 600 s** (`BOARD_IDLE_S`). That leaves a margin over one image's hash, while a real
wedge still costs at most 10 minutes. PHP's runner sets its own limit (1,100 s, Amendment 3).

**Dry run on bkit111** (21:58), with every board call stubbed:
- the chain reached its marker;
- pools matched the twin in 33 of 33 cells;
- PHP matched the twin in 280 of 280 cells;
- the bundles record the new kit.

Everything else in Amendments 1-4 stands.

### Amendment 6 to the item 5 board pre-registration (written 2026-11-23 22:03, before the re-slot boots)

**The re-slot runs on board 2, and the chain now flashes P1 itself.**
- After the board lane's CSMINT work, board 2 holds `…-pagedtbl-r76-r75d-b5b1_d0afdb40f`. A read-only readback at
  22:03 reported a mismatch against P1.
- bkit111's `board-run.sh` does not flash. So after each booking the chain runs these steps:
  1. It checks the local P1 `.bit` against sha256 `dc411e36…`.
  2. It reads back the resident name. If the name is not P1, it flashes P1 with `flash-bitstream.py` and reads the
     name back again.
  3. It parks the switches at 0 (`pre_boot_park.py`).
- A failed check, flash, readback or park unbooks the board, writes the marker as `STOP-P1` and stops the chain.
- The known-good fallback stays the console's default, bitstream 1.
- Dry runs with the readback, flash and park stubbed:
  - a failed flash stops the chain, unbooks the board and writes `STOP-P1`;
  - a mismatch followed by a successful flash reaches the marker. Pools reads 33 of 33.

### Amendment 7 to the item 5 board pre-registration (written 2026-11-24 05:17, after the re-slot, before the asan-blind re-run)

**The re-slot's PHP asan-blind population read nothing.** Board 2, P1, bookings 1454 (01:00-02:56).
- The stock and block boots ran cases 01 to 29, 25 of the population's 39. Then both were silent in case 30
  (crash-092). Without protection that case spins until the runner's per-case guard kills it at 800 s. The
  P1-geometry twin reads it `NO-READING:infra` for that reason, and the pre-registration names it a timing cell.
- bkit111's `board-run.sh` starts `board-watchdog.sh` with `ABORT_ON_ENTRY_STALL=1 ENTRY_STALL_S=420`. A vexec batch
  never prints `SHA6`, so after the boot's `SHA5`, 420 s of silence reads as an entry stall. The watchdog aborted
  each runner (at 1,605 s and 1,636 s) before the guard could fire, and no transcript was written.
- The object and exact boots completed, with every exit read (42 of 42 each). But the runner's cross-arm gate needs
  all four arms. It refused every verdict of the population ("the arms are not the arms they claim to be").
- So no asan-blind cell is taken from the re-slot. Amendment 3's budget list read the guard (800 s) and the driver's
  idle limit (1,100 s), not the kit's watchdog. That watchdog was the binding limit for case 30.

**The asan-blind population runs again, on bkit112** (`t3-smoke/bkit112-p1v-corpus-stall`, `board-run.sh`
699dca0696779aa6).
- bkit112 is bkit111 with one line changed: `ENTRY_STALL_S=${BOARD_ENTRY_STALL_S:-420}`. The stub mode also prints
  the effective limit. `board.env` and `base.manifest` name the kit's own path, and the rest is byte-identical
  (`diff -r`).
- `php-board.sh` sets `BOARD_ENTRY_STALL_S=1000`. That is above the guard's 800 s and below the driver's idle limit
  of 1,100 s. A real entry stall now costs up to 17 minutes instead of 7.
- Controls of the watchdog line, run from bkit112's own text on the re-slot's recorded stock `driver.log`:
  - with a limit of 30 s it aborted the runner at 45 s (positive);
  - with 1,000 s it did not fire while the runner lived (negative).
- Dry runs (05:16-05:17), with the board edge stubbed:
  - the re-run chain (`item5-rerun.sh`, tag `p1b3`) reaches its marker through a held window, a flash and a park;
  - a failed flash unbooks and writes `STOP-P1`;
  - with `BOARD_STUB=1`, the real runner reaches bkit112's `board-run.sh` for all four arms. Each prints
    `ENTRY_STALL_S=1000`, a driver idle limit of 1,100 s and a batch budget of 16,800 s.
- The booking, P1 check, flash, readback and park are Amendment 6's. The marker is `corpus5/board/ITEM5-RERUN-DONE`,
  and no other lane waits on it.

**Prediction.**
- Every asan-blind cell is the P1-geometry twin's (156 cells), with the pre-registration's exceptions.
- Case 30 on stock and block, which the pre-registration did not predict, is now predicted as the guard's timeout:
  `NO-READING:infra`, as on the twin.
- The re-slot's complete object and exact boots are compared exit by exit with the re-run's, as a repeat.

### Result: item 5 on a P1 board, SQLite and PHP temporal-asan-visible (re-slot `p1b2`, 2026-11-23 22:03 - 11-24 05:10)

Board 2, P1 (`…-pagedtbl_d83d3a633`), flashed by the chain at 22:05:56 (sha checked before, name read back after) and
read back resident before every later booking. Bundles: `bug-corpora/{sqlite,php}/engine-repros/results/2026-11-24-p1-board/`.

**Every cell read matches its prediction: 190 cells, against the P1-geometry twin.**
- SQLite pools (one boot, 22:05-22:42): 33 of 33 in verdict, reason, cause, function and offset.
  - Gate 1 read on this boot: `uaf-mem5` faults with 25 and `bounds-mem5` with 28, and case 25 (fts5 getvarint)
    is CAUGHT with 28. So the judge's `[25, 28]` applies unchanged (Amendment 1).
  - Pools 29 and 31 read MISSED with cause 24, in `sqlite3Get4byte` and `statDecodePage`. These are the twin's
    cells, so the CINCOFFSET branch did not fire.
  - 27 CAUGHT, 6 MISSED.
- SQLite malloc (one boot, 22:42-01:00): 33 of 33 in verdict, reason, function and offset.
  - 30 cells also match in cause. Cases 19, 20 and 22 read cause 4, 4 and 13 where the twin reads 24. These are the
    cells the pre-registration excepted ("consequence-fault causes on arms with no catch"). vp2 read the same 4 and
    13 on them.
  - Case 11 spun until the gate's watcher ended it at its 6,000 s bound (`@@STEP-BOUND … 6000`, exit 137). It reads
    `NO-READING:infra`, as Amendment 4 predicted.
- PHP temporal-asan-visible (four boots, 02:56-05:10): 124 of 124 in verdict, reason, cause and function. Stock has
  1 CAUGHT and 5 MISSED. Block has 6 CAUGHT. Object and exact have 5 CAUGHT each, plus case 21's `NO-READING:infra`
  (the excepted table-capacity cell). 25 cases per arm are out of the denominator, as on the twin.
- The comparisons have controls:
  - The PHP comparison run against the other population reports 160 differences, rc 1.
  - The SQLite comparison reports the three malloc cause cells above.

**Gates.**
- Gate 2 was read in full on each PHP boot, from sysfs: `ids=55`, `index_bits=25`, `node_flat_records=1048576`,
  `supervisor_slots=1`, `resolve_dirty=Y`, `resolve_pte=Y`.
  - The SQLite boots print only the two `resolve_*` parameters (`@@PARAM`). They also print `supervisor_slots=1` in
    the module's load line, and a flat table of 1,048,576 records.
  - The SQLite and PHP boots load the same module, `74b4cceb…` (kit README).
- Gate 3: every boot's dmesg shows `flat table 1048576 records, 16777216 bytes, from CMA`.
- Gate 4: no cell differs outside an exception, so the memory candidate (962 MiB against the twin's 2-8 GiB) is not
  needed to explain anything.
- Gate 5: every boot ran its controls first.
  - SQLite: two controls, both returned.
  - PHP: the sanity arm completed on all four boots, with 6,375 revokes on object and exact. The bound probe faulted
    as declared on object and exact.

**What this does not say.**
- It is P1's FLAT table, not the paged tree.
- It says nothing about memory or cycles.
- It covers 190 of the 346 pre-registered cells. PHP asan-blind (156 cells) read nothing on the re-slot
  (Amendment 7). Its re-run is the next section.
- The xz unpack timings: `@@XZSTAGE ok` came at board times 954 s and 953 s, which puts each stage's unpack and
  hashing at about 15 minutes.

### Result: item 5's PHP asan-blind population on a P1 board, every cell as predicted (re-run `p1b3`, 2026-11-24 05:18-07:47)

Board 2, P1 (`…-pagedtbl_d83d3a633`), booking 1458. The chain flashed P1 at 05:18:41-05:20:53 (sha checked before, name
read back after, readback rc 3), parked the switches, ran the four boots through `php-board.sh` on bkit112, and
unbooked at 07:47:39. Bundles: `bug-corpora/php/engine-repros/results/2026-11-24-p1-board/asan-blind/`.

**All 156 cells match the P1-geometry twin**, in verdict, reason, cause and function. Each arm's count is the twin's:

| arm | asan-blind (39): CAUGHT / MISSED | NO-READING |
|---|---|---|
| stock | 5 / 29 | out of denominator 3, infra 1 (case 30), not reached 1 |
| system allocator (block) | 12 / 21 | out of denominator 3, infra 1 (case 30), not reached 1, unattributed 1 |
| nested (object) | 26 / 10 | out of denominator 3 |
| nested exact | 31 / 4 | out of denominator 3, unattributed 1 |

- Case 30 (crash-092) on stock and block reads `NO-READING:infra`, as Amendment 7 predicted. Both boots print
  `VM_EXIT_TIMEOUT` and `VM_EXIT:…crash-092:137`. In each boot, the kernel line that closes case 29 and the one that
  closes case 30 are 801 s apart: 958.8 s to 1,759.9 s on stock, 978.5 s to 1,779.6 s on block. That is the
  runner's 800 s guard. Both boots then ran cases 31 to 41 to the end.
- The repeat: the re-run's object and exact boots match the re-slot's (`p1b2`) in every exit, 42 of 42 each.
- The landing check compares verdict and reason against the twin again, and finds no difference.

**The watchdog fix is what changed.** bkit112's watchdog ran in all four boots (119 to 170 `ALIVE`/`QUIET` lines each).
In stock and block it logged 780 s without UART during case 30, above bkit111's 420 s limit, and did not abort. No
watchdog log has a stall line.

**Gates**, read the same way as on the re-slot:
- Gate 2: each boot's sysfs gives `ids=55`, `index_bits=25`, `node_flat_records=1048576`, `supervisor_slots=1`,
  `resolve_dirty=Y` and `resolve_pte=Y`.
- Gate 3: each boot's dmesg shows `flat table 1048576 records, 16777216 bytes, from CMA`.
- Gate 5: every boot completed the sanity arm first (`sanity 573 200 17 abc …`). It shows 6,375 revokes on object and
  exact, and none on stock and block. The bound probe exited 139 on object and exact, as declared.

**So item 5 on the board is complete: 346 of 346 pre-registered cells read, each as predicted.** That is 190 from
the re-slot and 156 from this run. Like the re-slot, this is P1's flat table, not the paged tree, and it reads verdicts
only, not memory or cycles.

## Fig. 10's one-operand revisions in the one-operand environment (2026-11-23)

The programme's item 4 asks for Fig. 10's one-operand revisions, in the environment of item 2 (revision 71ce199d8fea,
QEMU 9e69171f1c, rebuilt as `nested/oneop`), so that they are ready for silicon when bitstream 4 exists.
- **Which revisions.** At native-kernel-new-qemu 5060011844e, Fig. 10's tree, nine lifetime patches use one-operand
  CREVOKE (`capstone_cap_release`) or CRENEW (`capstone_cap_renew`):
  - SQLite `sublet-3220000-memsys5.patch`;
  - CPython 0014;
  - FFmpeg 0003;
  - memcached 0006;
  - mruby 0008;
  - Perl 0001;
  - PHP's zend-alloc layer;
  - PostgreSQL 0003;
  - Wireshark wmem 0001.
- **The API.** 5060011844e pins QEMU 35de150cef, not the environment's 9e69171f1c. The two revisions'
  `runtime/include/capstone/capability.h` define the same functions with the same encodings: `capstone_cap_release`
  is `.insn r 0x5b, 0x1, 0x52, x0, obj, x0`, and `capstone_cap_renew` is funct7 0x54. So the patches build
  against this environment's SDK.
- **Order.** SQLite first: it has the smallest workload and this lane's longest record.

### Pre-registration: SQLite speedtest1 `--size 1`, Fig. 10's revision, in oneop (written 2026-11-23 00:34, before the runs)

- **Images** (`nested/oneop/fig10/build-sqlite.sh`, the environment's SDK and compiler, clang cba01ed3090d, `-O2`):
  the adapted amalgamation `7fc92e19…`, and `speedtest1.c` `16750ede…`.
  - unprot: stock memsys5, `speedtest1.dom` `f0ea7e65dbca76a2…`.
  - prot: `sublet-3220000-memsys5.patch` of 5060011844e (`8fa2d094e4edc6a3…`) applied with `-F0`, `8e166381d039dc2b…`.
  - By encoding, `sqlite3.o` holds no lifetime instruction on the unprot arm. On the prot arm it holds CDERIVE 10,
    one-operand CREVOKE 5 and CRENEW 14.
- **Runs.** `fig10/chain-sqlite.sh`: six boots of the environment's kit (`oneop/kit`, lock domain A), one run each
  of `--heap 8388608 64 --size 1 st.db` under `-icount` shift 0.
  - Boots 1-2: `opcount.so`, executed 0x5B instructions by region and funct7, on unprot and then prot.
  - Boots 3-6: `zones3.so`, exact instructions by code address, as ABBA u1, p1, p2, u2.
  - The stubbed dry run reached all six rungs. Its comparator refused empty result files, and read 31 of 32 against
    a reference with one changed hash. The gate's fault branch fired on a stub job that exited 3.
- **Predictions:**
  - **Answers:** in every boot, all 32 phases equal `2026-10-11-same-answers/phases.json`'s native phases (rows and
    hash), with status 0.
  - **Lifetime instructions** (opcount, image region):
    - prot executes CDERIVE (funct7 0x51) and CREVOKE (0x52) each between 10,000 and 100,000 times, and CRENEW
      (0x54) at least once;
    - unprot executes fewer than 1,000 of each and no CRENEW.
    - This is the control: if prot's CREVOKE count is not 10 times unprot's, the patch did not take effect, and
      the run says nothing about it.
  - **Image instructions, prot against unprot:** +0.3 % to +3 % (the two-operand revision on the twin was +0.63 %).
    Repeats of an arm agree within 0.01 %.
- **Branches:**
  - A prot fault is a finding about this revision in this environment, localised before anything else. One
    candidate is known in advance: on this table CRENEW cannot renew in place (the gcc section above). The patch
    takes a zero result as a failed renew, and what follows that path is not predicted.
  - An answer that differs in either arm stops the comparison.
  - A boot that returns no SQ_TEST line is void and is rerun once.

### Result: Fig. 10's SQLite revision runs in oneop at +0.56 % image instructions, every prediction met (00:35-00:36)

Six boots, `nested/oneop/fig10/runs/fig10-sq-*`, each lock grant immediate. Every boot returned status 0 with all 32
phases equal to the native reference (`cmp-sq.py`: none differ, none missing, none extra).

| boot | arm | plugin | image instructions | CDERIVE | CREVOKE | CRENEW |
|---|---|---|---:|---:|---:|---:|
| opc-u | unprot | opcount | 223,245,566 | 6 | 1 | 0 |
| opc-p | prot | opcount | 224,493,724 | 28,228 | 28,215 | 539 |
| u1 | unprot | zones3 | 223,245,669 | | | |
| p1 | prot | zones3 | 224,493,621 | | | |
| p2 | prot | zones3 | 224,493,805 | | | |
| u2 | unprot | zones3 | 223,245,566 | | | |

- **Control: met.** prot executes 28,215 CREVOKEs against unprot's 1, and 28,228 CDERIVEs and 539 CRENEWs, inside
  the predicted ranges. Unprot's 6 and 1 are mallocng's own lifetimes. The two plugins agree on each arm's image
  count: opcount's `ALL image` and zones3's `ZONE image` differ by at most 103 instructions.
- **Image instructions: +1,248,095.5 (+0.559 %)**, ABBA means 224,493,713 against 223,245,617.5. That is inside
  the +0.3 % to +3 % band. The repeats agree within 0.0001 %. That is 21.9 image instructions per executed lifetime
  instruction, not attributed per function.
- Against the two-operand revision: on the twin, the dev-era patch cost +0.63 % in program instructions over
  56,442 lifetime instructions (`--size 1` section above). That was a different SDK, compiler, kit and counting rule, so
  the two numbers are not a difference of the revisions.
- CRENEW ran 539 times, and nothing faulted. Whether each one renewed in place or fell back on this table was not
  read.
- **For silicon:** these images run one-operand CREVOKE and CRENEW, which P1 does not decode (funct7 0x54 is CSNODE
  there). They wait for bitstream 4. What this run settles is the answers and the instruction cost on the ISA
  bitstream 4 is to implement.

### Pre-registration: mruby `bm_ao_render.rb 16`, Fig. 10's revision, in oneop (written 2026-11-23 00:48, before the runs)

- **Build worktree.** `nested/oneop/fig10/wt` is revision 71ce199d8fea, so the recipes, runtime and SDK are the
  environment's. Fig. 10's patch file replaces that revision's file there, and is never committed from it. For mruby
  that is `0008-gc-slots-as-sublet-lifetimes.patch` of 5060011844e (`4a1314e3ab82f318…`). It differs from
  71ce199d8fea's 0008 only in raising out-of-memory when `gc_issue` fails.
- **A builder trap, caught before any run.** `ports/common/application/build-virtual.sh` sets `MRBD_PIN=4.0.0-rc2`
  only for the lifetime arm. The stock arm then builds mruby head (ad98f216e), with 174 objects and 2.42 MB of text
  against the protected arm's 124 objects and 1.69 MB. Both arms are now built with `MRBD_PIN=4.0.0-rc2`, the pin
  Fig. 10 and P1 used. The head build is kept as `fig10/mruby/stock-head-unmatched` and is not run.
- **Images** (`fig10/build-mruby.sh`, the environment's compiler clang cba01ed3090d). Both apply 0001-0003, 0008,
  0009 and 0010, and 0008 is live only under `-DMRB_CAPSTONE_SUBLET`:
  - stock `6fd8fe7b2b2b6246…`, text 1,691,804 bytes;
  - prot `02584cf74a3fb56e…`, text 1,692,224 bytes.
  - By encoding, `gc.o` holds no lifetime instruction on stock, and CDERIVE 2 and one-operand CREVOKE 2 on prot.
    Each whole image adds mallocng's 6 CDERIVE and 3 CREVOKE.
- **Workload, Fig. 10's:** `bm_ao_render.rb 16` (`e44efffd…`, the same file in both trees). The answer is the PPM
  on stdout. The reference `cfd73bb1…` (781 bytes) is this tree's native build, and equals the P1 record's.
- **Runs.** `fig10/chain-mruby.sh`, with the SQLite chain's design: opcount on unprot, then on prot, then zones3
  ABBA. The runner is `fig10/run-app-zones.sh`; its gate's MATCH and MISMATCH branches were exercised on stubs.
  The stubbed chain reached all six rungs, and its first dry run caught a truncated EXPECT, fixed before this
  commit.
- **Predictions:**
  - **Answers:** every boot prints the reference PPM, status 0.
  - **Lifetime instructions** (opcount, image region):
    - prot executes at least 100,000 CDERIVE and 100,000 CREVOKE, and no CRENEW;
    - stock executes fewer than 1,000 of each.
    - The control: prot's CREVOKEs are at least 100 times stock's.
  - **Image instructions, prot against stock:** +0.5 % to +5 % (the twin's in-context +1.64 % on P1's images).
    Repeats agree within 0.01 %.
- **Branches:**
  - A prot fault with cause 29 at `add_heap`'s CDERIVE is Q-15 in this environment: the page's 80 KiB life is not
    representable. It is recorded, and the Q-15-repaired variant (P1's) is pre-registered separately.
  - Any other prot fault is localised first.
  - An answer that differs stops the comparison.

### Result: the mruby chain is void on two instrument errors of this lane; nothing is read from it (00:48-01:00)

- **Answers were not checked.** `bm_ao_render.rb`'s PPM does not end in a newline, so `icount`'s own
  `ICOUNT instructions=… status=…` line was appended to its last line. The gate then hashed 821 bytes, the 781-byte
  PPM plus that line, and every boot reads MISMATCH with a different hash. The count itself was lost the same way.
  The gate was tested on stub output that ended in a newline, so the stub could not show this. The answer files
  stayed in the guest, so the PPMs cannot be checked afterwards.
- **The control was mis-specified.** Stock executes 1,173,238 CDERIVE and 1,172,318 CREVOKE, because mruby's
  `mrb_malloc` sits on mallocng, whose lifetimes are on in both arms (`heap_lifetimes` in the SDK). The
  pre-registered "stock below 1,000, prot at least 100 times stock" cannot hold. That is my prediction's error, not
  the patch's: prot executes 916,189 CDERIVE and 916,101 CREVOKE more than stock.
- By the pre-registered rule, a run whose answers differ or whose control fails is not read, so no cost is read.
  Seen anyway: image zones stock 4,721,740,155 in each of its three boots, prot 4,798,649,926 in each of its three
  boots.
- **A process slip as well.** I edited `run-app-zones.sh` while the chain's first boot was running it. That boot's
  data (serial log, opcount file) was complete. The tail of the script then ran from a shifted offset
  (`timeout: failed to run command '--icount'`) and reported rc 1. The rule is already in this lane's notes:
  never edit a running script. The fixed runner is a new file, `run-app-zones2.sh`.

### Amendment 1 to the mruby pre-registration, and the Perl pre-registration (written 2026-11-23 01:01, before the runs)

- **The runner, for both:** `fig10/run-app-zones2.sh`. `icount` runs `sh -c "exec ./capstone-vexec ./app … > answer.txt"`
  with its own line sent to `count.txt`, so the answer file holds only the program's stdout. On a stub whose output
  has no final newline it reads MATCH. Both chains' stubbed dry runs reached all six rungs on it.
- **The control, restated for any port whose allocator sits on malloc:** the lifetime instructions prot executes
  beyond stock. It is not a ratio, because mallocng's own lifetimes are in both arms. Plus the static census: the
  ported object holds lifetime instructions in prot and none in stock.
- **mruby rerun** (`fig10-mrb-r2`, otherwise as pre-registered):
  - the reference PPM in every boot;
  - prot beyond stock: at least 100,000 CDERIVE and 100,000 CREVOKE, and no CRENEW in either arm;
  - image instructions +0.5 % to +5 %, the band written before the void chain;
  - repeats within 0.01 %.
  - The void chain's zones were seen before this amendment. So the cost result of the rerun is a confirmation on the
    same images, not a blind prediction.
- **Perl**, Fig. 10's revision, `0001-sv-heads-as-sublet-lifetimes.patch` of 5060011844e (`83501a59ae08d54a…`),
  copied into `fig10/wt` (`fig10/build-perl.sh`):
  - Images: Perl 5.36.3 with the port's tree patches 0001-0012 in both arms; prot adds the sublet 0001.
    - stock `9ebe0d0456894107…`, text 7,023,436 bytes;
    - prot `3c1962c60f1ce918…`, text 7,009,156 bytes.
    - By encoding, `sv.o` holds no lifetime instruction on stock, and CDERIVE 28 and one-operand CREVOKE 3 on
      prot.
  - Workload, Fig. 10's: perlbench's train script as SPEC runs it, `perl -I. -I./lib suns.pl` with
    `PERL_HASH_SEED=0`. The staged files are SPEC's `suns.pl`, `WORDS` and `lib`, licensed, and staged outside
    the repository. The answer is stdout against SPEC's `suns.out` (`01688ea0c1058063…`). The validate file is
    not compared.
  - Runs: `fig10/chain-perl.sh`, six boots as for mruby.
  - **Predictions:**
    - SPEC's `suns.out` in every boot, status 0;
    - prot beyond stock: at least 100,000 CDERIVE and 100,000 CREVOKE, and no CRENEW;
    - image instructions +1 % to +10 %. The twin's train zone with SV heads protected, all five commands, was
      +5.619 % against stock. suns.pl alone was not measured.
    - repeats within 0.01 %.
  - Branches as for mruby: a fault is localised first, and an answer that differs stops the comparison.

### Pre-registration: CPython, Fig. 10's pyperformance subset, Fig. 10's revision, in oneop (written 2026-11-23 01:11, before the runs)

- **Images** (`fig10/build-cpython.sh`). 5060011844e's `cpython-3.13.7-0009-pymalloc-arena-pointer.patch`
  (`8456fb1d48304444…`) and `0014-pymalloc-sublet-lifetimes.patch` (`fc63552d6c4d9eaf…`) are copied into
  `fig10/wt`. Both arms apply the port's series and prot adds 0014; the logs' patch lists differ by 0014 alone.
  - stock `9966571644f86eb9…`, text 6,520,620 bytes;
  - prot `d14a767ceb839438…`, text 6,521,532 bytes.
  - By encoding, the cross-built `Objects/obmalloc.o` holds no lifetime instruction on stock, and CDERIVE 6 and
    one-operand CREVOKE 2 on prot.
- **Workload, Fig. 10's:** `pyperformance-subset.py` of 5060011844e (`544de26d…`), over pyperformance 1.14.0 and
  pyperf 2.10.0 from the pinned PyPI wheels (sha256 `94068ac2…` and `79196bc4…`, both checked).
  - The run is `-S`, with `PYTHONHASHSEED=0`.
  - One stage, built from stock's build (`fig10/tools/stage-cpython.py`), serves both arms, so the arms differ
    only in the image. It holds the Lib and `_sysconfigdata`, compiled by the native 3.13.7 with
    `--invalidation-mode unchecked-hash`, because the guest's copy of the stage does not keep mtimes. Two stagings
    give the same trees (stdlib `c5549c6d…`, bench `2d53e3a0…`).
  - The answer is stdout. It is the native build's own stdout on the same driver (`58211793…`): eight benchmark
    lines, then `PYPERFORMANCE-OK 8`.
- **Runs:** `fig10/chain-cpython.sh`, six boots as for mruby, on `run-app-zones2.sh`. The stubbed chain reached
  all six rungs.
- **Predictions:**
  - **Answers:** the native stdout in every boot, status 0.
  - **Lifetime instructions:** prot beyond stock, at least 100,000 CDERIVE and 100,000 CREVOKE, and no CRENEW. The
    twin's dev-era runs of seven benchmarks issued 2.9 M CDERIVEs, so this floor is deliberately low.
  - **Image instructions, prot against stock:** +0.5 % to +5 % (the twin's in-context +1.63 % summed, dev-era patch).
    Repeats agree within 0.01 %.
- **Branches** as for mruby.

### Result: mruby rerun (fig10-mrb-r3): reference PPM in all six boots, +1.63 % image instructions (01:03-01:15)

- A second rerun was needed first. `fig10-mrb-r2` (and the Perl chain launched with it) ran `icount sh -c …`, and
  `icount` calls `execv`, which does not search PATH: every job returned status 32512 (127 << 8) with image zone 0.
  Those 12 boots ran no image and are void. My host stub had searched PATH, so it could not show this. The runner
  now names `/bin/sh`, and the stub now refuses a bare name, so it reproduces 32512 on the old form.
- **Answers: met.** All six boots print the reference PPM (781 bytes, MATCH), status 0.
- **Control: met.** prot executes 916,189 CDERIVE and 916,101 CREVOKE beyond stock (2,089,427 and 2,088,419
  against 1,173,238 and 1,172,318). Neither arm executes a CRENEW.
- **Image instructions: +76,909,771 (+1.629 %)**, stock 4,721,740,155 and prot 4,798,649,926. Every boot of an arm
  reads the same count to the instruction, and opcount's `ALL image` agrees. Inside +0.5 % to +5 %. That is 42
  image instructions per added lifetime instruction. As amended, this confirms on images whose zones had been seen.

### Result: Perl, Fig. 10's revision on suns.pl: SPEC's answer, and +34.9 % image instructions, far above the band (01:15-01:26)

- **Answers: met.** All six boots print SPEC's `suns.out` (53 bytes, MATCH), status 0.
- **Control: met.** prot executes 1,077,695 CDERIVE and 864,583 CREVOKE beyond stock (2,064,268 and 1,684,005
  against 986,573 and 819,422), and no CRENEW.
- **Image instructions: +726,556,662.5 (+34.88 %)**, stock 2,082,932,016 and 2,082,932,020, prot 2,809,488,676
  and 2,809,488,685 (repeats within 0.000001 %). **The prediction, +1 % to +10 %, is missed.** That is 374 image
  instructions per added lifetime instruction, against mruby's 42 and SQLite's 22.
- **Candidate, from the source, not yet from the counts.** Fig. 10's 0001 finds a head's arena with
  `S_sv_arena_of`, a walk of `PL_sv_arenaroot` (the patch's lines 108-113), at every issue and release. That is the
  walk pending patch 0002 replaces with a sorted array and bisection ("Perl with the arena lookup repaired"
  above). A walk over hundreds of arenas per lifetime instruction would give a price of this size. Two diagnostic boots with per-function counts (zones3 with
  each image's symbols, `fig10-perl-fn-*`) are queued to read where the 726.6 M fall. No mechanism is claimed
  before they are read.
- **Per function** (`fig10-perl-fn-stock` and `-prot`, zones3 with each image's symbols; both give SPEC's answer, and
  the zones equal the chain's within 3 instructions). The named functions sum to +726,556,497:
  - `Perl_sv_arena_head` +434,181,209 at 80,084 entries (absent from stock): 5,421 instructions per call. prot
    allocates 2,536 arenas (`Perl_more_sv` entries), so a call that walks the arena list most of the way is this
    size;
  - `Perl_sv_arena_issue` +87,051,697 at 238,418 entries;
  - `Perl_newSV_type` +104,610,611 (entries 53,215 to 890,464, so inlining moved);
  - `Perl_newSVsv_flags` +76,284,648 at the same 53,139 entries;
  - `Perl_sv_free2` +33,644,294 and `Perl_sv_clear` +12,122,082 at the same entries;
  - `Perl_newSVpvn_flags` -25,979,116 at the same entries.
- **Reading.** The price is mainly the arena lookup. `Perl_sv_arena_head` alone is 60 % of it, at a per-call
  size that fits a walk, and the patch's source has the walk (`S_sv_arena_of` over `PL_sv_arenaroot`). That the
  other increases are the same walk inlined elsewhere is inferred from the entries, and was not read from the
  disassembly. Fig. 10's revision predates the repair in pending patch 0002, so on silicon this revision would carry
  the walk too. The variant worth pricing is 0001 plus 0002. 0002 has since landed (2026-11-23): on the perl
  corpus, 0001 and 0001+0002 agree in all 11 cases and all 47 qualify files
  (`bug-corpora/perl/release-differential/results/2026-11-22-virtual-0002/README.md`), and it is now
  `ports/perl/sublet/patches/5.36.3/0002`.

### Pre-registration: Perl, Fig. 10's revision plus 0002, the variant worth pricing (written 2026-11-23 06:37, before the runs)

- **The variant.**
  - Patch: `fig10/patches/fig10-perl-0002-on-5060-0001.patch` (`d93308435dd5d4f6…`).
  - Pending 0002 (`sublet/pending/5.36.3/0002-sv-arena-lookup-by-index.patch` of the perl0002 worktree) was written
    on dev's 0001, not on Fig. 10's. Two of its three hunks apply to Fig. 10's 0001 unchanged.
  - The third registers each new arena in `Perl_more_sv`. Its context differs, so the one line
    `S_sv_arena_register((SV *)chunk);` is placed after `sv_add_arena` under `#ifdef PERL_CAPSTONE_SUBLET`.
    `Perl_more_sv` is the only place either revision adds an arena.
  - Built in a second scratch worktree (`fig10/wt2`, 71ce with the same swapped-in files as `fig10/wt`, plus this
    patch as `sublet/patches/5.36.3/0002`), by `fig10/build-perl0002.sh`.
  - Image `528d7f1e35a1cd74…`. Against prot's tree, only `sv.c` differs, apart from generated files that differ by
    their miniperl path.
  - By encoding, `sv.o` holds CDERIVE 2 and one-operand CREVOKE 3; prot's holds 28 and 3. The derive sites likely
    fell because `S_sv_arena_of` grew and stopped being inlined. That was not read from the disassembly; the dynamic
    counts below are the check.
  - 16 MiB, so it maps at 0x3ff7000000 like prot (symbols `fig10/syms/perl-p0002.syms`).
- **Runs:** `fig10/chain-perl0002.sh`, four boots:
  - opcount p0002;
  - zones3 p0002 twice, with FN lines;
  - a zones3 stock control, against `fig10-perl-r2`'s stock readings.
  - Same workload and answer as the Perl chain. The stubbed chain reached all four (`fig10/perl/chain02-dry.txt`).
  - The chain queues behind the kit-devq corpus runs, as the tshark chain does.
- **Predictions:**
  - SPEC's `suns.out` in every boot, status 0;
  - the stock control within 0.001 % of 2,082,932,016-2,082,932,020;
  - **lifetimes unchanged:** p0002's CDERIVE and CREVOKE counts equal prot's (2,064,268 and 1,684,005) within 1 %,
    and no CRENEW. The index changes how an arena is found, not which heads get lifetimes. This stands in for a
    fixture here; the protection fixture for 0002 is the perl0002 corpus;
  - **image instructions: +2 % to +12 % against stock**, and at least 400 M fewer than prot's 2,809,488,676;
  - `Perl_sv_arena_head`: under 200 instructions per call (prot: 5,421);
  - repeats within 0.01 %.
- **Branches:**
  - lifetime counts that differ from prot's by more than 1 %: the variant changed what is protected. It is not priced,
    and the difference is localised first;
  - above +12 %: attributed per function before anything else is said;
  - below +2 %: it is reported as it stands, with the per-function deltas.

### Result: Perl, Fig. 10's revision plus 0002 (08:45-09:07)

All six predictions hold.

- **The runs.** The chain first started at 08:33 and was stopped four minutes into its first boot, to let three short
  SQLite corpus runs take the QEMU lock first. That partial boot is kept apart (`fig10/runs-aborted/`). The chain
  then ran 08:45-09:07, four boots, rc 0.
- **Answer.** `suns.out` in every boot: stdout sha256 `01688ea0…`, MATCH, status 0.
- **Stock control.** `fig10-perl02-u3` reads 2,082,931,984 against 2,082,932,016-020, which is -32 to -36 (0.000002 %).
- **Lifetimes unchanged.** opcount p0002: CDERIVE (funct7 0x51) 2,064,278 and CREVOKE (0x52) 1,684,011, against
  prot's 2,064,268 and 1,684,005, so +0.0005 % and +0.0004 %. No CRENEW (0x54) line.
- **Image instructions: +174,659,199 (+8.385 %) against stock, inside the band.** p0002 reads 2,257,591,183 and
  2,257,591,178 (zones3 p1, p2) and 2,257,591,179 (opcount). That is 551,897,493 below prot's 2,809,488,676, against
  the predicted 400 M. So 0002 removes 76 % of Fig. 10's revision's +726.6 M.
- **`Perl_sv_arena_head`: 147.1 instructions per call** (11,782,336 at the same 80,084 entries), against prot's
  5,421.
- **Repeats:** p1 and p2 within 5 instructions.
- **Per function, against stock** (`fig10-perl02-p1` against `-u3`, each with its own symbols). The named functions
  sum to +169,111,580 of the zone's +174,659,199, so 5,547,619 falls outside every FN range.
  - `Perl_sv_arena_issue` +77,477,237 at 1,077,570 entries (absent from stock). That is 71.9 per call; prot had
    87,051,697 at 238,418 entries;
  - `Perl_sv_free2` +44,139,039 at the same 864,579 entries;
  - `memmove` +24,696,528 (entries 877,558 against 875,022);
  - `Perl_sv_arena_head` +11,782,336;
  - `Perl_newSVpvn_flags` +5,842,025 at the same entries;
  - `memset` +2,395,488, `calloc` +1,625,273, `Perl_more_sv` +1,058,115 (2,536 arenas against stock's 1,590), and
    `aligned_alloc` +765,647;
  - against these, `Perl_vivify_ref` -2,750,681 and `Perl_sv_clear` -864,273 at the same entries.
  - `Perl_sv_arena_issue` is entered 4.5 times as often as in prot. Read as inlining moved by the lookup's growth,
    which fits `sv.o`'s CDERIVE words falling from 28 to 2. That is inferred, not read from the disassembly.
- **Memory** was not measured in this chain: oneop's runner reads instructions only. Stock's peak for this workload
  is not on record either, so the memory condition is open for this variant.

### Pre-registration: Perl's peak memory on suns.pl in oneop, stock, prot and p0002 (written 2026-11-24 05:28, before the runs)

- **Method.** `fig10/run-app-mem.sh` is `run-app-zones3.sh` without the plugin. It runs the image under busybox
  `time -v` and reads `Maximum resident set size`, the vexec process's `ru_maxrss`.
  - busybox 1.36.1 prints `%M` as `ptok(pagesize, ru_maxrss)` (`miscutils/time.c:283-284`), which multiplies by
    pagesize/1024. Linux already reports `ru_maxrss` in kB, so the printed value is 4 times the peak. Every figure
    below is the printed value divided by 4.
  - The script also samples MemTotal − MemAvailable once a second. That is the census's "peak used", but it is too
    coarse for a run of about 3 s, so it is recorded and not judged.
- **Controls of the method** (stock image, one boot each, 05:25-05:27):
  - positive: a script that builds a 50,000,000-byte string reads 107,036 kB;
  - negative: a one-line script reads 8,896 kB.
  - The difference, 98,140 kB, matches the two copies of the string that `x` produces (97,656 kB). So the peak
    includes the domain's heap.
  - The 1 s sampler saw only +40,572 kB of the same run, which is why it is not judged.
- **Runs.** `fig10/chain-perl-mem.sh`: stock, prot and p0002, interleaved, two boots each. The workload and answer
  are `chain-perl.sh`'s (suns.pl, `PERL_HASH_SEED=0`). The images are the chains' own: stock `9ebe0d04…`, prot
  `3c1962c6…` and p0002 `528d7f1e…`.
- **Predictions.**
  - SPEC's `suns.out` in every boot, status 0.
  - Repeats of an arm within 64 kB.
  - **prot against stock: +4,000 to +12,000 kB.** The counted arenas give 2,536 against 1,590 (`Perl_more_sv`
    entries, above). prot's arena is 4,080 + 84 × 16 = 5,424 bytes (`SV_ARENA_HEADS` = 4080/48 − 1, plus one
    capability per head, `sv.c:367`), and stock's is 4,080. That is +7,268,064 bytes, before allocator overhead.
  - **p0002 against prot: −64 to +256 kB.** The index holds one 16-byte entry per arena, 2,536 entries, and grows
    by at most double.
- **Branches.** A peak outside a band is reported with both arms' readings before any reading of why. A mismatch
  in the answer stops the comparison for that arm.

### Result: Perl's peak memory on suns.pl: p0002 is +4,800 kB (+10.3 %) over stock, every prediction met (05:28-05:37)

- **Answers: met.** All six boots print SPEC's `suns.out`, status 0. The readings are in `fig10/runs-mem/perl-mem-*`.
- **Peak RSS of the vexec process** (printed `ru_maxrss` / 4):

  | arm | boot 1 | boot 2 | against stock |
  |---|---|---|---|
  | stock `9ebe0d04…` | 46,744 kB | 46,744 kB | |
  | prot `3c1962c6…` | 51,408 kB | 51,412 kB | +4,666 kB (+9.98 %) |
  | p0002 `528d7f1e…` | 51,544 kB | 51,544 kB | +4,800 kB (+10.27 %) |

- **Repeats: met**, within 4 kB.
- **prot against stock: met** (band +4,000 to +12,000 kB). The arena arithmetic gave +7,098 kB, so the reading sits
  2,432 kB below that estimate. The estimate assumed every arena stays resident at the peak, and no reading here says
  why it falls short.
- **p0002 against prot: met**, +134 kB (band −64 to +256). That is about the index's size, 2,536 entries of 16 bytes
  grown by doubling.
- The census's system-wide "peak used", sampled once a second, reads 175,516 and 175,768 kB for stock, 184,352 kB for
  both prot boots, and 184,092 and 184,596 kB for p0002. That is about +8.7 MB against stock. It counts the kernel and
  the module as well as the process, and it is sampled too coarsely to judge. It is recorded and not read further.
- **So the memory condition for p0002 is answered:** its peak is 10.3 % above stock's on this workload. Fig. 10's
  revision accounts for 10.0 % of that, and the index for 0.3 %.

### Result: the CPython chain is void: the environment refuses `ioctl(FIOCLEX)`, so no script can be run by path (01:26-01:28)

- Every boot exited with status 2 in 0.8 s and printed only `./app: can't open file
  '/tmp/ap/bench/pyperformance-subset.py': [Errno 38] Function not implemented`. Stock's image ran 43.3 M
  instructions: start-up and its imports worked.
- Diagnostic boots (`fig10-cpy-probe-*`, stock image), with the probe imported as a module:
  - `os.open`, `O_CLOEXEC`, `stat`, `fstat`, `open(…, 'rb')`, `realpath`, `listdir` and `lstat` all
    succeed on that file;
  - `os.set_inheritable` (either value) fails with errno 38, and so does `fcntl.ioctl(fd, FIOCLEX)`;
  - `fcntl(F_SETFD, FD_CLOEXEC)` succeeds.
  - So this environment's native route answers `ioctl(FIOCLEX)` with ENOSYS. CPython's run-file path calls
    `_Py_set_inheritable`, which does not fall back on ENOSYS. Imports open with `O_CLOEXEC` and never reach it.
    Fig. 10's own runs used a different platform (the trap route of 35de150cef).

### Amendment 2: CPython run as a module (written 2026-11-23 01:34, before the run)

- The only change: the driver runs as `-S -m pyperformance_subset`, with `PYTHONPATH=/tmp/ap/bench`. The driver
  is the same file under a module name, staged and compiled with the rest (stage `stage2`: stdlib tree
  `c5549c6d…` as before, bench tree `bf5cc1a0…`). It finds its benchmarks through `__file__`, which `-m` sets.
- The native build's stdout run the same way equals its script-mode stdout byte for byte (`58211793…`). That is
  still the reference.
- Predictions and branches as pre-registered. The stubbed chain reached all six rungs.

### Result of amendment 2: CPython stops again, at `os.uname()`; Fig. 10's CPython workload does not run in this environment (01:34-01:37)

- Every boot now starts the driver and fails in pyperf, at `platform.machine()` → `os.uname()`: `OSError: [Errno 38]
  Function not implemented`, status 1. The images ran 439 M (stock) and 446 M (prot) instructions, so nothing is
  read from them.
- So this environment's native route refuses at least two calls Fig. 10's CPython workload needs:
  `ioctl(FIOCLEX)` and `uname`. Both were found only by running, one at a time, so the list is not known to be
  complete. Removing either from the workload would make it a different workload from Fig. 10's, so that is not
  done.
- **Stopped here for CPython.** Its Fig. 10 revision needs an environment that routes these calls, or the trap
  route Fig. 10 used (35de150cef). The image pair and the stage stay built (`fig10/cpython`, `stage2`).

### Pre-registration: Fig. 10's Wireshark revision in oneop (written 2026-11-23 06:15, before the protected arm has run)

- **Revision:** `wireshark-4.6.8-0001-wmem-sublet-lifetimes.patch` of 5060011844e (`ca4ba97e22407f4a…`, the copy in
  `fig10/wt` equal to it). It changes `wmem_allocator.h`, `wmem_allocator_block.c` and `wmem_allocator_block_fast.c`.
  - Built by `fig10/build-tshark.sh`: tshark 4.6.8 with the port's patches in both arms; the dependencies come from
    a local copy of the cached source tarballs, because their URLs return 404.
  - Image `xbuild/run/tshark`: stock `352e193c849f6a16…`, prot `d026193831257cf6…`.
- **Static census, by encoding** (`fig10/tools/census-lifetime.py`, llvm-objdump words, opcode 0x5B funct3 1):
  - stock: none of the 16 wmem objects holds a lifetime instruction;
  - prot: `wmem_allocator_block.c.o` holds CDERIVE 10, one-operand CREVOKE 2 and CRENEW 1; `wmem_allocator_block_fast.c.o`
    holds 10, 1 and 1. The other 14 objects hold the same counts in both arms.
  - So Fig. 10's wmem revision uses CRENEW, and the "no CRENEW" prediction of the other apps is not made here.
- **Workload:** the one 5060011844e's plan `2026-11-15-wireshark-established-workload.md` defines.
  - `tshark -r C -V -n` with `TZ=UTC` and an empty `HOME` and `WIRESHARK_CONFIG_DIR`, one process per capture.
  - The 169 captures of `test-captures-4.6.8.txt` (`c8d1114f…`), in that order, all in one boot. They are copied
    from the build's `xsrc/test/captures`, including the 13 in `mysql/` and `zabbix/`.
- **Runner:** `fig10/run-app-zones4.sh`.
  - It is `run-app-zones3.sh` with a loop: every process runs inside one `icount`, and each capture's stdout is
    appended under a `=== C rc=N` header. A nonzero exit is therefore part of the answer.
  - `APP_SET` prints the inputs and the nonzero count; an `APP_PROG` line goes to the console per capture.
  - The zones3 window is `IMG_LO`/`IMG_HI`.
- **Where the image is mapped.** The image is 32 MiB (`roundup_pow_of_two` of text plus data, `binfmt-dom.h:156`) and
  is aligned to its size, so it is mapped at 0x3ff6000000, not at 0x3ff7000000. Its `.text` runs from 0x3ff6000000 to
  0x3ff71a941c.
  - The chain's zones3 window is 0x3ff0000000-0x3ff8000000, opcount's image range.
  - Per-function lines use symbols at that base (`fig10/syms/tshark-{stock,prot}.syms`, `tools/mksyms-base.py`).
  - Checked on the stock probe below: the FN lines sum to 99.985 % of the image zone, and `__set_thread_area` is
    entered 169 times, once per process.
- **Stock probes (they establish the answer; seen before this text):**
  - `fig10-ts-probe1`/`probe2`: two captures. The output is tshark's `-V` dissection in UTC.
  - `fig10-ts-probe3` (opcount, timeout 3,600 s): the boot did not complete within the timeout. It is void.
  - `fig10-ts-probe4` (zones3, timeout 21,600 s), 05:06-06:14:
    - 169 inputs, 0 nonzero;
    - stdout 34,055,148 bytes, sha256 `b190069b62f6265ebfb63e5991443886028440a22f5df364c244330734690347`;
    - 13,357,016,045 instructions in the icount, image zone 6,312,779,721.
- **Runs:** `fig10/chain-tshark.sh`, six boots (opcount stock, prot; zones3 u1 p1 p2 u2), timeout 21,600 s each.
  - The stubbed chain reached all six rungs with the right image, plugin, window and symbols
    (`fig10/tshark/chain-dry.txt`).
  - About 70 min per boot.
  - It starts only after the kit-devq corpus runs (perl0002, SQLite, PHP) have had the QEMU lock. Those wait for an
    exclusive lock, and this lane's own oneop and twin runs kept them out for 5-10 h.
- **Predictions:**
  - every boot: 169 inputs, 0 nonzero, stdout sha256 `b190069b…`;
  - **control:** prot beyond stock executes at least 100,000 CDERIVE and at least 1,000 CREVOKE. Stock runs musl
    mallocng's lifetimes too (`fig10-ts-probe3`'s partial count: 1.66 M CDERIVE), so the control is the difference.
    CRENEW: stock 0; prot is recorded, not predicted;
  - **image instructions, prot vs stock: +0.3 % to +6 %.** The twin's tshark on 9,000 frames, at dev's revision, was
    +2.69 % in-context. Here every process also pays tshark's start-up, which wmem touches less.
  - repeats within 0.01 %.
- **Branches:**
  - a prot boot that differs from `b190069b…`, or has a nonzero capture: no cost is read, and the capture is named
    from the answer's headers;
  - control not met: the patch did not take effect, and the run says nothing about protection;
  - cost outside the band: attributed per function from the FN lines before anything else is said.

### Result: Fig. 10's Wireshark revision in oneop (read 2026-11-23 22:44)

The six boots of `fig10/chain-tshark.sh` ran from 12:22 to 22:41, with a 21,600 s timeout each (the chain prints
`timeout=` per boot). Every runner exited 0.

**Deviation from the pre-registration.** The pre-registration gives every boot 21,600 s, but the chain's first launch
did not export `TIMEOUT`. `run-app-zones4.sh` therefore used its own 3,600 s default, and three boots (opc-u from 09:07,
opc-p from 10:12, then u1) ran to that cap. Those three are void and nothing is read from them. The chain now exports
the value. The relaunch reran all six boots under the same names, so the void boots' directories were overwritten.

Readings against the predictions:
1. **Every boot: holds.** All six report 169 inputs, 0 nonzero, and stdout sha256 `b190069b…` (MATCH).
2. **Control: holds.** In the image zone, prot minus stock is:
   - CDERIVE 4,091,559 − 2,016,854 = 2,074,705 (at least 100,000 was predicted);
   - CREVOKE 1,992,502 − 1,963,762 = 28,740 (at least 1,000 was predicted);
   - CRENEW: stock 0 (no funct7 0x54 line), prot 9,471.
3. **Image instructions, prot vs stock: +2.27 %, inside the +0.3 % to +6 % band.**
   - Stock u1 6,317,790,819 and u2 6,314,379,942; prot p1 6,460,856,200 and p2 6,458,616,090.
   - The mean difference is 143,650,765 instructions.
4. **Repeats within 0.01 %: does not hold.**
   - Stock repeats agree within 0.054 % (3,410,877) and prot within 0.035 % (2,240,110). That is about a fortieth of
     the measured effect.
   - The spread is in musl mallocng, not in wmem: u1 − u2 is `alloc_slot` +1,635,543, `nontrivial_free` +919,091, and
     `get_meta`, `__malloc_alloc_meta` and `enframe` at about +195,000 each. The wmem functions differ by 0 to 5
     instructions between repeats.
   - Why mallocng varies between identical boots is not established here.

**Where the +2.27 % goes.** The FN lines cover 99.985 % of the image zone in every boot. Mean prot − stock:

| function | instructions | share of the difference | entries stock / prot |
|---|--:|--:|--:|
| `wmem_block_fast_sublet_alloc` | 114,319,526 | 79.6 % | 0 / 1,905,321 |
| `wmem_block_sublet_alloc` | 11,597,546 | 8.1 % | 0 / 153,761 |
| `wmem_block_fast_alloc` | 9,626,582 | 6.7 % | 1,905,321 / 1,907,699 |
| `memset` | 4,271,186 | 3.0 % | 976,197 / 1,134,526 |
| `wmem_block_sublet_realloc` + `_fast_sublet_realloc` − the plain reallocs | 750,985 | 0.5 % | 0 / 14,505 |
| `wmem_sublet_issue_at` | 689,462 | 0.5 % | 0 / 14,505 |
| all other functions | 2,395,478 | 1.7 % | |

So about 94 % of the difference is in the patched block allocators' alloc paths. Of the rest, `memset` is entered
158,329 more times on prot. Its callers are not attributed here.

## perlbench item 3: the port's code without its lifetime instructions (2026-11-24)

### Pre-registration: Perl suns.pl on P1, U P N N P U in one boot, N = the port with its lifetime instructions neutralised (written 2026-11-24 08:18, before the boot)

**Question.** The programme (item 3) asks where Perl 5.36's +788.8 M in-context cycles on suns.pl go. The attribution
in "Perl with the arena lookup repaired" puts 491.7 M (62 %) in the port's +173.3 M twin instructions, priced at the
stock arm's mean of 2.838 cycles per instruction. It leaves 32-33 % unattributed, and it calls that price the softest
number: no board run has priced the port's instructions apart from the lifetimes they create. This boot adds an arm N
that differs from P only in its lifetime instructions:
- N−U is the port's other instructions at their real price;
- P−N is the lifetime instructions together with their records.

**The images.**
- U: the stock image of every P1 Perl boot, `3012f392…` (built 2026-11-17).
- P: `perl-protre.strip.dom` `09f8b716…`. This is patch 0001 + 0002, rebuilt today by `nested/perl/build-arms2.sh`
  (`f8232cb6…`; FRCDKMS SDK, clang `ecd6d8c4…`, Perl tarball `f2a1ad88…` as in `PROVENANCE.txt`).
  - It is not byte-identical to the 2026-11-18 image `693568f6…` (protfix): 10,393 bytes differ, in `.rodata`,
    `.text` and `.data`. One known source is `__DATE__`/`__TIME__` ("Compiled at Nov 24 2026 08:07:20" against
    "Nov 18 2026 11:03:21"). A changed string can move `.rodata` and the `.text` offsets that reach it. That every
    differing byte comes from it is not shown.
  - Section sizes are equal. A full opcode census of `.text` (`nested/perl/tools/capop-census.py`, `568d72b5…`) is
    identical, and on the twin P reads the same instructions as protfix (below).
- N: `perl-protnop.strip.dom` `d2d79d42…`. It is P plus `nested/perl/lifenop/0003-lifetime-instructions-neutralised.patch`
  (`caf0cf33…`), an experiment patch kept outside `ports/`. In `sv.c` it redirects `capstone_cap_derive` and
  `capstone_cap_revoke_child` to copies of their asm:
  - the live CDERIVE (funct7 0x51) becomes MOVC (0x0a). The bytes were checked with `llvm-mc`: `movc a0, a1` and
    `.insn r 0x5b, 0x1, 0x0a, a0, a1, x0` are both `5b 95 05 14`;
  - CREVOKE (0x52) becomes `add x0, rs1, rs2`, a 4-byte word with the same two source registers;
  - the refusing zero-size CDERIVE and every operand and clobber are unchanged.
  - So a head is handed out as a copy of its arena's capability, with the cursor at the head and no record of its own.
    The arena's own derive in `Perl_more_sv` is neutralised the same way, so each arena carries the capability that
    `Newxz` returned. Nothing in the port reads a child's bounds.
- Static census, P to N: CDERIVE 42 → 26 (−16, the live derive at each inlined site), MOVC +16, CREVOKE 7 → 6, ADD-to-x0
  0 → 1. Total 4-byte words (957,966) and RVC words (273,062) are equal. The census of U to P differs in many classes,
  so the census can fire.

**Twin control** (`nested/out/npn-1`, 2^20 records, P in the stage's "unprot" slot and N in its "prot" slot, order P N
N P): every command rc 0 with `suns.out`.
- In-context instructions:
  - P 1,905,129,684 and 1,905,131,815;
  - N 1,905,130,755 and 1,905,130,627.
  - N−P is −58 instructions on the means, inside the repeats' own spread of 2,131.
- P equals the 2026-11-18 protfix twin run (1,905,130,536 and 1,905,132,087) within that spread.
- So N executes P's instructions, apart from the substitutions. N−U on the twin is P−U, +173.3 M.

**Configuration.**
- Bitstream P1 (`…-pagedtbl_d83d3a633`), kit `nested/kits/bkit91-rec1m` (`node_flat_records=1048576`, SHA256SUMS
  verified), board 1.
- The runtime lane arms the chain `nested/p1n/p1n-b1.*` behind pid 893413 (board 1's current run and the board lane's
  booking). The chain reads P1 back by name, and flashes it sha-checked if it is not resident.
- Stage `nested/p1n/upn` (`upn.SHA256SUMS`) is the kit's `make-train-stage.sh` around `nested/appd/perl-upn`, made by
  `appd/make-abba3-stage.py` (`a82bdfc5…`, make-abba-stage.py with a third arm).
  - The smoke files are byte-identical to the perlfix stage's.
  - Commands u1 p1 n1 n2 p2 u2, each in a fresh tmpfs copy of the inputs, with `PERL_HASH_SEED=0`.
  - List line `upn 1200 60`, so `BAKED_TIMEOUT=4800` (the dry run prints it). Expected wall time: about 40 min of
    commands.
- Checked off the board (rule 8, `nested/p1n/recheck-p1k.sh` = recheck-p1.sh with the kit as a parameter):
  - real bake rc 0;
  - `BOARD_STUB=2` rc 0 on the perlfix boot's real transcript and on its cut, PASS all 9 steps;
  - real firmware build 23,760,392 B, under the cap;
  - an `NR_DRY=1` pass of the chain to DONE (`nested/p1n/dry.log`).
- Read by `appd/parse-abba.py` per command, with the readings computed per arm pair.

**Rules.**
- Answers: every command returns `suns.out`. Otherwise that arm is not read.
- Repeats: 1 % in-context and 2 % kernel + other, per arm. Otherwise that arm's difference is not read.
- Stock control: U within ±1 % of the perlfix boot's 4,914.4 M. Outside it, no comparison with that boot is made, but
  this boot's own differences are still read.

**Predictions.**
1. **Records:**
   - N's record counters sit at U's: hpm3, hpm4, hpm5 and hpm7 each within 10 % of the U-to-P gap from U. On perlfix
     the gaps were +0.455 M, +12.3 M, +16.6 M and +1.45 M.
   - N's L1D misses (hpm8) lie between U's and P's. N keeps the larger arenas and the `issued` array.
2. **N−U, the port's other instructions on silicon: 250 M to 600 M** in-context cycles.
   - At the mean price (2.838 a twin instruction) it would be 491.7 M.
   - At the lookup pair's measured price (1.67) it would be 289 M.
3. **P−N**, the lifetime instructions with their records, is the rest of P−U. Today's P−U is expected within ±5 % of
   +788.8 M.

**What each outcome says.**
- **N−U near 491.7 M (418 to 565 M):** the mean price holds for the port's code. P−N is the 32-33 % remainder plus
  the walks and replays.
- **N−U below 418 M:** the port's code is cheaper than the mean, and the lifetimes cost more than the table
  attributed.
- **N−U above 565 M:** the code is dearer, for example through the `issued` array's cache traffic. Then the remainder
  belongs to the code, not to the records.
- **P−N:**
  - From it come the measured walks (P−N hpm4) and the replays (P−N hpm7 at 17-20 cycles each).
  - What is left is divided by the lifetime instructions P executes beyond N: on perlfix, 1.94 M CDERIVEs and
    CREVOKEs. That gives a price per lifetime instruction, including its record traffic beyond walks and replays.
  - This pair does not split the instruction from its record.

**Limits.**
- One run per arm, U first, not interleaved, which is the perlfix boot's design.
- N's heads are not protected. N is an instrument, not a variant anyone would ship.
- Item 3's CRENEW and liveness-test variant is no longer deferred behind item 5, which is done. It follows this pair,
  in the one-operand environment ("The CRENEW and liveness-test variant" above).

### Result: U P N N P U on P1 (board 1, 2026-11-24 09:29-10:08, read against d1a4526e5f8e)

**The boot.**
- Chain `nested/p1n/p1n-b1.*` (pid 2048772) on board 1. It read P1 back by name as resident (no flash) and booked
  1465, then unbooked.
- Smoke PASS on all 9 steps. The watchdog sampled throughout (151 ALIVE/QUIET lines).
- The command line carries `capstone_vm.node_flat_records=1048576`.
- The staged images decompress to the registered hashes: U `3012f392`, P `09f8b716`, N `d2d79d42`.
- The chain's generic SPEC classifier logged `SPEC rc=2 MALFORMED: @@SPEC-DONE 6 but 0 commands ended`. It reads
  SPEC's own per-command markers, which this stage does not print, so this line is not a verdict.
- The pre-registered reader is `appd/parse-abba.py` (`1d0459d5…`) on `board.txt`, then `nested/p1n/read-upn.py`
  (`4bec0c0e…`). Its output is `nested/p1n/upn.json` (`647f93bb…`).

**The checks.**
- Answers: all six commands rc 0, each with `suns.out`.
- Repeats:
  - In-context cycles, all three arms: U 0.06 %, P 0.06 %, N 0.02 %, all within 1 %.
  - Kernel + other: U 2.11 % and P 2.78 %, both over 2 %, so they are not read. N is 0.04 %.
- Stock control: U 4,915.1 M against the perlfix boot's 4,914.4 M, +0.02 %. Met.

**In-context cycles, means of each arm's two commands:**

| pair | difference | share of P−U |
|---|---:|---:|
| P−U | +789.1 M (+16.05 %) | 100 % |
| N−U, the port's code without its lifetime instructions | +564.9 M (+11.49 %) | 71.6 % |
| P−N, the lifetime instructions with their records | +224.2 M (+4.09 % of N) | 28.4 % |

**Predictions.**
1. **Records.**
   - hpm3, hpm5 and hpm7 met: N−U is 6.1 %, 0.3 % and 4.1 % of the U-to-P gap.
   - **hpm4 missed:** N's walk cycles are 11.1 % of the gap from U, against the pre-registered 10 % (U 18.51 M,
     N 19.88 M, P 30.87 M).
   - hpm8 met: N's L1D misses (19.93 M) lie between U's (14.52 M) and P's (21.98 M).
   - So N has U's record traffic, as it should with no records of its own. It keeps 72 % of P's added L1D misses
     (+5.41 M of +7.46 M).
2. **N−U, +564.9 M, met** (band 250-600 M).
   - It falls in the outcome range "the mean price holds" (418-565 M), 0.1 M below its upper edge.
   - The implied price is 3.260 cycles per twin instruction (564.9 M over +173,265,263). Stock's mean is 2.838, so
     this is 15 % above it.
   - The reading sits on the boundary between "the mean holds" and "the code is dearer". The repeats (0.06 %, about
     3 M) cannot place it on either side.
3. **P−U, +789.1 M, met** (+0.0 % from +788.8 M, band ±5 %).

**The revised attribution of P−U, +789.1 M in-context cycles:**

| part | cycles | share | price and its source |
|---|---:|---:|---|
| the port's code, measured | 564.9 M | 71.6 % | N−U on the board. The twin counts it at +173.3 M instructions, so 3.260 cycles each (measured, the replacement for the 2.838 mean) |
| record-table walks | 11.0 M | 1.4 % | P−N hpm4, read as cycles (measured) |
| event-23 replays | 23.7-27.9 M | 3.0-3.5 % | P−N hpm7, +1.39 M at 17-20 cycles each (Addendum 1's RTL estimate, from source) |
| the lifetime instructions and their other record traffic | 185.4-189.5 M | 23.5-24.0 % | P−N less the two rows above |
| **not attributed** | **none at this resolution** | | The rows add to P−U by construction: N−U + P−N = P−U |

- **A price per lifetime instruction.**
  - The port's executed CDERIVEs and CREVOKEs come from the one-operand environment's opcode counts of the same
    patches on suns.pl. Protected minus stock is +1,077,705 CDERIVE and +864,589 CREVOKE, 1,942,294 together.
    - Protected: 0001 + 0002, `nested/oneop/fig10/runs/fig10-perl02-opc-p`.
    - Stock: the same `perl/stock` image, `fig10-perl-r2-opc-u`.
    - (Corrected before landing. The first version cited `fig10-perl-r2-opc-p`, which is 0001 alone with the linear
      arena search. Its counts differ by 10 and 6, so no figure moves.)
    - That count is from the one-operand build of the patches, not from the board image's own trace.
  - That makes 95.5-97.6 cycles per lifetime instruction. N runs a MOVC or an ADD in each one's place, so this price
    is over a 1-cycle substitute.
  - It includes each instruction's record traffic beyond the walks and replays. This pair does not split the
    instruction from its record.
  - It sits at the top of the 63-100 cycles assumed in the perlfix pre-registration.
- **Where the previously unattributed 255.7-260.1 M (32.4-33.0 %) goes.**
  - At the 2.838 mean the code took 491.7 M. Measured, it takes 564.9 M, so 73.2 M of the remainder is the code's.
  - The lifetime instructions take 185.4-189.5 M, most of it.
  - The walks (12.3 → 11.0 M) and the replays (24.7-29.0 → 23.7-27.9 M) each move by about 1 M against the earlier
    table, because P−N leaves out the small part of each that N−U carries.
- **The L1D misses are not priced.** N−U carries +5.41 M of them, and P−N +2.05 M. They are inside the code's and the
  lifetimes' rows above, not separate from them.
- **Context entries are the code's, not the lifetimes'.** N−U is +1,819 entries (+39.7 %) and P−N is +32. What the
  port's code does that enters the context more often is not examined here.
- Uptime: N−U +27 s and P−N +10 s.

**Limits.**
- One run per arm, U first, not interleaved. The two U commands bracket the boot and agree within 0.06 %, so drift
  over it is no larger than that.
- Kernel + other is not read: U's and P's repeats fail the 2 % rule.
- N is an instrument. Its heads are unprotected copies of their arena's capability.

## M-16 on P1: how often the protected tshark returns (2026-11-24)

### Pre-registration: the protected tshark stage in five consecutive boots on P1 (written 2026-11-24 08:07, before the boots)

**Question.** The programme (item 4) asks for the Wireshark non-return to be root-caused. On P1 the protected stage
returned in 2 of 3 boots (the heartbeat boot and tsu-p2; tsu-p1 did not), and on F in 0 of 2. Those counts can't
separate an intermittent fault from one that depends on the boot. This run reads the return rate of the same bytes on
P1. For each boot that does not return, it also records where the core stopped, compared with tsu-p1's reading.

**Configuration.**
- P1 `…-tagoverlap6c-pagedtbl_d83d3a633`, kit bkit91 (`t3-smoke/bkit91-k2n-native-p1-tree-cscfix`, its
  SHA256SUMS verified), board 1.
- The runtime lane arms the chain `nested/p1tr/p1tr-b1.*` behind this lane's Perl chain (the section above), which
  waits for board 1's current runs. P1 is restored and read back before then; the chain reads it back by name again
  and flashes it, sha checked, if it is not resident.
- Five boots, `tsu-r1` to `tsu-r5`, each a `cp -a` of `nested/p1tu/tsu-p1`. `diff -r` against it is empty, and
  `nested/p1tu/tsu-p1.SHA256SUMS` verifies:
  - the image is `ts_prot.xz` (`c1a1293f…`, which unpacks to `bcc251f3…`), wmem 0001 with the Q-15 repair;
  - the capture is 9,000 frames (`5ba7788d…`) and the command is `-r w.pcap -V -n`, with no heartbeat.
- List lines are `<stage> 900 75`. The kit multiplies by 4, giving `BAKED_TIMEOUT=3600` with no entry-stall abort
  on a SPEC stage. That is the limit tsu-p1 ran under. The booking is 75 min, because tsu-p1's non-return held the
  board for 68 min.
- Board time: about 39 min a boot that returns and 68 min one that doesn't, so 3.3 to 5.7 h in all.
- Each boot is baked again on the shared native kernel tree. Its kernel Image hash is recorded from the bake, because
  tsu-p2's kernel differed from tsu-p1's by four objects that another relink had recompiled.

**Checked off the board** (rule 8): `nested/p1/recheck-p1.sh` on `tsu-r1` (`nested/p1tr/tsu-r1-recheck/`):
- real bake rc 0;
- `BOARD_STUB=2` rc 0 on the P1 heartbeat boot's real transcript (`out-nat-p1ts-nested-b1/tsph`, which returns) and on
  its cut after `@@SPEC-BEGIN` (which does not), both PASS all 9 smoke steps, both `BAKED_TIMEOUT=3600`;
- real firmware build 23,760,392 B;
- an `NR_DRY=1` pass of the chain through five stub boots to DONE (`nested/p1tr/dry.log`).

**Readings, per boot.**
- **Returns:** `@@SPEC-END` with rc 0, and stdout equal to stock (`056c1f8c…`, 1,140,005 lines, 9,000 frames).
- **Does not return:** no `@@SPEC-END` before the runner's 3,600 s limit. For each such boot, these are read from
  the runner's apertures (`BAKED_WEDGE_APERTURES=1`):
  - the commit pc;
  - sw=225 and sw=220;
  - the serving index (`rev_node_debug_ex`).
  Each is compared with tsu-p1's readings: commit pc `0x3ff6780f04` (slot-0 residue at the CDERIVE of
  `wmem_block_fast_sublet_alloc`), sw=225 `0x96`, sw=220 `0x8c`.
- **Anything else** (rc != 0, or other output) is a third class and is reported as itself.
- A boot whose smoke does not pass stops the chain. Its count stops there, and the boots that ran are read.

**What is read.**
- k returns of n boots read, with the Clopper-Pearson 95 % interval for the return rate. Pooled with tsu-p1/tsu-p2
  and the heartbeat boot (same image, kit and capture; the heartbeat boot also had the beat), that gives k + 2 of
  n + 3.
- If every boot returns, the non-return rate of this stage on P1 is below 45 % (the upper bound for 0 of 5), or 53 %
  pooled (1 of 8). Then M-16 on P1 is rare, and the folder for the hardware side needs a stress form.
- If none returns, the non-return on P1 is the rule, and the two returns are the exception.
- **Where it stops:** if every non-return reads tsu-p1's commit pc, the stops are at one site. That says where the core
  stopped, not why: a pc says which instruction faulted, not where its operand came from.
- Returning boots' in-context cycles are reported against tsu-p2's 24,073.7 M as a repeat. No cost is read: there is
  no stock arm in this run, and the kernels may differ.

**Prediction.** Intermittent: between 2 and 4 returns of 5, and every non-return at tsu-p1's commit pc.

### Result: 3 returns of 5, and both non-returns at tsu-p1's commit pc (board 1, 10:08-14:19, read against d1a4526e5f8e)

The chain ran all five boots in order on resident P1 (read back by name, no flash). Every bake returned rc 0, and every
boot passed all 9 smoke steps. `nested/p1tr/read-m16.py` read `…/out-nat-p1tr-m16-b1`. The chain's `SPEC rc=2` lines
are spec-classify's verdict on a one-arm tshark stage (`NO RESULT` or `MALFORMED: @@SPEC-DONE 1 but 0 commands ended`),
not a reading. The readings below come from the transcripts.

| boot | transcript sha256 | outcome | board time |
|---|---|---|---:|
| tsu-r1 | `ded32bc6…` | no `@@SPEC-END` within 3,600 s | 67.5 min |
| tsu-r2 | `f621eb0a…` | no `@@SPEC-END` within 3,600 s | 67.5 min |
| tsu-r3 | `d76782aa…` | rc 0, stock's output | 38.6 min |
| tsu-r4 | `5c1eceb8…` | rc 0, stock's output | 38.8 min |
| tsu-r5 | `14af70f1…` | rc 0, stock's output | 38.7 min |

**Returns** (`nested/tshark/parse-ts.py` against `wl/ref1000.out`): each has `056c1f8c…`, 1,140,005 lines and 9,000
frames, with 25,862 context entries, the same as tsu-p2.

| boot | in-context cycles | against tsu-p2 | `between_cycles` | hpm3-8 |
|---|---:|---:|---:|---|
| tsu-r3 | 24,104,428,222 | +0.13 % | 3,830,570,613 | 7,230,615 / 87,340,438 / 98,809,032 / 8 / 11,166,889 / 126,486,217 |
| tsu-r4 | 24,325,028,579 | +1.04 % | 3,843,566,462 | 8,340,791 / 98,256,595 / 98,602,109 / 8 / 11,061,161 / 126,828,181 |
| tsu-r5 | 24,350,085,335 | +1.15 % | 3,837,625,730 | 7,202,222 / 87,661,214 / 98,824,174 / 8 / 11,192,682 / 127,482,944 |

These are repeats only, as registered. There is no stock arm in this run, so no cost is read.

**Non-returns.** The runner's apertures after the timeout, verbatim:
- **tsu-r1:**
  - trap log `0x89`; sw=224 `0x05`; **sw=225 `0x92`**; sw=226 `0x00`; sw=227 `0x07`; sw=228 `0x9f`; sw=229 `0x00`;
  - sw=192-195 all `0x00`; sw=219 `0xd7`; sw=220 `0x8c`; sw=221 `0x02`; sw=222 `0x00`;
  - commit pc `0x0000003ff6780f04`; trap-log mepc `0xffffffff80007376`;
  - rev_node_debug_ex `0x0000067f40001131`, head `0x0071`, serving index `0x000014b9`.
- **tsu-r2:**
  - trap log `0x89`; sw=224 `0x05`; sw=225 `0x96`; sw=226 `0x00`; sw=227 `0x07`; sw=228 `0x9f`; sw=229 `0x00`;
  - sw=192-195 all `0x00`; sw=219 `0xd7`; sw=220 `0x8c`; sw=221 `0x02`; sw=222 `0x00`;
  - commit pc `0x0000003ff6780f04`; trap-log mepc `0xffffffff80007376`;
  - rev_node_debug_ex `0x0000059240001134`, head `0x0071`, serving index `0x000019fb`.
- **Against tsu-p1:**
  - Both read tsu-p1's commit pc, sw=220, trap log, mepc, head, and sw=224 and sw=226-229.
  - tsu-r2 also reads tsu-p1's sw=225.
  - tsu-r1's sw=225 is `0x92` against `0x96`. The one bit that differs is `stall`.
  - The serving indices differ in both, as F's two non-returns did.

**Kernels.** All five boots carry the same `CONFIG_CMDLINE` as tsu-p1, and the same dmesg command line.
- The Image hashes differ in every boot (`2ac5a9f4`, `0d44744f`, `df390e1a`, `ff070840`, `ea4b600f`). Every bake
  re-embeds its own initramfs (`usr/initramfs_data.o`) and recompiles `init/version-timestamp.o`.
- **tsu-r2's bake also recompiled the four `CONFIG_CMDLINE` objects**, the same four as tsu-p2's: `setup.o`, `fdt.o`,
  `cmdline_early.o` and `efi-stub.o`.
  - The cause is this lane's own perlbench N recheck at 10:24 (`nested/p1pn/pairn-recheck`). It relinked the shared
    tree with bkit91-rec1m's `node_flat_records=1048576` between tsu-r1's bake (10:09) and tsu-r2's (11:16).
  - tsu-r2's bake put 262144 back. This is the same mechanism as tsu-p2's.
- **The recompile does not separate the outcomes:** tsu-r1 did not return without it, and tsu-p2 returned with it.

**Against the pre-registration.**
- **Return rate:** 3 of 5, Clopper-Pearson 95 % interval **0.147-0.947**. Pooled with tsu-p1, tsu-p2 and the heartbeat
  boot it is 5 of 8, **0.245-0.915**. Neither registered extreme holds: the stage returns in neither every boot nor no
  boot.
- **Where it stops:** every non-return reads tsu-p1's commit pc, `0x3ff6780f04`. That is slot-0 residue at the CDERIVE of
  `wmem_block_fast_sublet_alloc` (`commit_instr[0].valid` = 0 in all three). So the stops are at one site, now 3 of 3
  on P1. As registered, the pc names where the core stopped, not why.
- **Prediction** (2-4 returns of 5, every non-return at tsu-p1's commit pc): **met**.

**Reading.**
- On P1 the protected tshark's non-return is intermittent across boots of the same bytes and kit. Its rate is
  between about 9 % and 76 % on the pooled interval (3 non-returns of 8). Whenever it does not return, the core is
  held at one site.
- So a reproducer for the hardware side has to be a stress form: many issues of that allocation path in one boot,
  rather than one tshark run per boot. One tshark run per boot reads this rate at about 40-68 min per sample.
- **Run order, reported and not read:** the two non-returns are the first two boots and the three returns the last
  three, as tsu-p1 (no return) came before tsu-p2 (return). Eight boots in three runs cannot test an effect of order or
  time, and none is claimed.
- No mechanism is claimed. In particular:
  - the serving-index differences say only that the record table's state differs between stops;
  - tsu-r1's clear `stall` bit is one sample of a live signal.

## perlbench item 3: the CRENEW variant, Perl 5.36's SV heads in the one-operand environment (2026-11-24)

### Pre-registration: Fig. 10's Perl revision plus 0002, with heads renewed instead of released, in oneop (written 2026-11-24 08:37, before the runs)

**Question.** The programme (item 3) asks for a variant with CRENEW and the liveness test.
- **The liveness half needs no run.** As "The CRENEW and liveness-test variant" records, no QEMU decodes a liveness
  test. Its instruction case is one probe replaced by one instruction, 0.032 % of Perl's instructions. So the variant
  has no instruction saving to measure, and its case is semantic.
- **The CRENEW half** runs here, on Perl 5.36's SV heads with suns.pl, where Fig. 10's revision and 0002 already have
  oneop readings. perlbench's bodies port does not build in oneop yet (its SPEC build covers gcc only), so the
  bodies half stays the open line of this item.
- What a renew saves: when a head is freed, p0002 releases its lifetime (CREVOKE), and the next issue of that head
  derives a new one (CDERIVE).
- This variant renews the lifetime at the free (CRENEW). Every old copy goes stale as before, and the arena keeps the
  new lifetime, so the head's next issue hands it out with no CDERIVE.

**The arm.**
- `fig10/perl/renew`, built by `fig10/build-perlrenew.sh` on `fig10/wt3`: a worktree at 71ce199d8fea with wt2's two
  Perl patches (0001 `83501a59…`, 0002 `d9330843…`, the p0002 build's) plus
  `fig10/renew/0003-sv-heads-renewed.patch` (`4b502316…`). The tarball is `f2a1ad88…` and the compiler is the
  environment's.
- The patch changes two functions in `sv.c`:
  - `S_sv_arena_retire` sets `*issued = capstone_cap_renew(*issued)` where p0002 released and cleared;
  - `Perl_sv_arena_issue` returns a non-NULL `issued` entry as it is, and derives only for a head never issued.
  - `SVf_BREAK` heads are untouched, as before.
  - Arena teardown stays `Safefree` of the arena. If that frees the arena with a release, the release ends every
    lifetime derived inside it, renewed ones included. Whether mallocng's free releases it is not checked here, and a
    lifetime left live at exit costs nothing these runs read.
- Static census (`fig10/tools/census-lifetime.py`), p0002 to renew:
  - one-operand CREVOKE 6 → 3, CRENEW 0 → 3, CDERIVE 8 → 8;
  - 1,753,484 → 1,753,486 instructions.
- **This image never goes to P1.** P1 decodes funct7 0x54 as CSNODE.

**Runs.** `fig10/chain-perlrenew.sh`, chain-perl0002.sh's four oneop boots with `suns.pl` and `PERL_HASH_SEED=0`:
- opcount renew;
- zones3 renew twice (symbols `fig10/syms/perl-renew.syms`);
- zones3 p0002 as the same-day control.
A dry run with an echo runner reaches `CHAIN-PERLRN-DONE` through all four.

**What is read: image-zone instructions only.**
- oneop's table has no generations. CRENEW cannot renew in place there, and the table's cost of a renew falls in the
  kit's sweeps, outside the zone.
- So this reads what renewing saves in the program's own instructions, not its cost on a record table.
- Cycles need a bitstream with CRENEW, and none is written.

**Predictions.**
1. **Answers:** `suns.out` in every boot.
2. **Control:** q3 reads p0002's 2,257,591,178-183 within ±40.
3. **Lifetime instructions** (opcount renew against p0002's 2,064,278 CDERIVE and 1,684,011 CREVOKE):
   - CRENEW (0x54) equals CREVOKE's fall exactly, because each retire that released now renews.
   - CDERIVE falls by 600,000 to 1,077,570. The upper bound is p0002's issue count. Only first issues still derive,
     and those are bounded by the heads in its 2,536 arenas.
4. **Image instructions: renew minus p0002 between −20 M and −2 M.** The derive path skipped on a reissue is about
   12 instructions, against a load and a branch added.
5. **Repeats:** r1 and r2 within 10 instructions.

**What follows.**
- If held, the saving per reissue (zone delta ÷ CDERIVE fall) is what a renew is worth in the port's code.
- The bodies half is the same change on perlbench's body free lists, once perlbench builds in oneop.

### Result: the CRENEW variant saves 11.68 M image instructions (−0.52 %), 13.5 per reissue; every prediction but one holds (08:38-08:44)

Chain `fig10/chain-perlrenew.sh`, four boots, rc 0, read by `fig10/read-renew.py`.

- **Answers:** `suns.out` in every boot (sha256 `01688ea0…`, MATCH, status 0). **Held.**
- **Control:** q3 (p0002, same day) reads 2,257,591,178, inside p0002's 2,257,591,178-183. **Held.**
- **Repeats:** r1 2,245,912,933 and r2 2,245,912,932, 1 instruction apart. **Held.**
- **Image instructions: renew − p0002 = −11,678,248 (−0.52 %), inside the −20 M to −2 M band. Held.**
  - Against stock's 2,082,931,984, the variant costs +7.82 %, against p0002's +8.385 %.
  - The per-function profile against q3 puts −11,677,721 of it in `Perl_sv_arena_issue`, at the same 1,077,570
    entries (77,477,237 → 65,799,516). No other function moves by more than 700.
- **Lifetime instructions** (opcount renew against p0002's opcount):
  - CDERIVE 2,064,278 → 1,199,660, which is −864,618, inside the band. **Held.**
  - CREVOKE 1,684,011 → 819,422 (−864,589). CRENEW 0 → 864,662.
  - **"CRENEW equals CREVOKE's fall exactly": not held.** CRENEW is 73 higher.
- **Saving per reissue: 13.51 image instructions per CDERIVE removed** (11,677,721 ÷ 864,618). The band assumed
  about 12 for the derive path, less a load and a branch.

**The 73, as far as this run reads them.**
- `Perl_sv_free2` executes 73 more instructions in renew at the same 864,579 entries, and no other function moves
  by a count near 73.
- That fits 73 retires that find a non-NULL `issued` entry in renew where p0002 finds NULL, and so renew a lifetime
  p0002 would not release. p0002 clears the entry when it releases. The renew variant never clears it, so a head
  that comes back to a retire without passing through `Perl_sv_arena_issue` is renewed again.
- Which Perl path hands a head out without the issue call is not identified here, and the reading above is
  inferred, not traced.
- **The counts do not close, so the 73 are not fully accounted for.**
  - With 73 second renews, at most 864,589 distinct entries could be reissued without a derive. Yet CDERIVE fell by
    864,618.
  - The opcode totals include mallocng's own CDERIVEs and CREVOKEs. Those moved by +10 and +6 between the prot and
    p0002 builds, whose head code differs only in the lookup. So a few dozen in these totals is not attributable to
    the heads from opcounts alone.
  - Per-function opcode counts would settle it, and this runner does not give them.
- If the hypothesis holds, it costs nothing in correctness here, since every old copy is already stale. But a
  renew-at-free design would then have to clear or mark the entry, or count on every handout going through the issue
  call.

**Reading.**
- Renewing a head at its free instead of releasing it removes 864,618 of Perl's 2,064,278 CDERIVEs on suns.pl. In
  the image zone that is worth 13.5 instructions a reissue, 0.52 % of the zone. It is a seventh of p0002's
  +174.7 M over stock.
- That is the program's side only. On oneop's table, without generations, a renew is a release plus an allocation
  in the kit, and its record cost is not in these numbers. Cycles need a bitstream with CRENEW.
- The bodies half (perlbench's body free lists) remains open. perlbench does not build in oneop yet.

## perlbench item 3: the CRENEW variant of the SV bodies port, SPEC's test.pl in the one-operand environment (2026-11-24)

### Pre-registration: bodies renewed instead of revoked, perlbench test.pl in oneop (written 2026-11-24 10:33, before the runs)

**Question.** This is item 3's open CRENEW line for bodies, after the heads half above.
- When a body is freed, the bodies port (`own-allocators/500.perlbench_r/port-sv-bodies.py`, `3933328d…`) revokes it
  (CREVOKE), and the free list keeps the segment's pointer.
- When the body is popped again, the port derives a new child (CDERIVE).
- The variant renews the body at the free (CRENEW). Every old copy goes stale, as after the revoke, and the free list
  keeps the renewed capability. A reissue hands that capability out with no CDERIVE.

**perlbench in oneop, new today.**
- `oneop/build-spec-perl.sh` is `build-spec.sh` for 500.perlbench_r, in a fresh copy of the pristine tree
  (`spec/tree-perl`), with label `oneopperl` and the environment's SDK. Its executable is `ba295e57…`.
- `fig10/build-perlbench-arms.sh` makes four copies of the build directory and rebuilds each with `make.out`'s own 69
  compile commands and link command:
  - **stock**, unchanged: byte-identical to the build's executable (`ba295e57…`), the replay control;
  - **heads** (`408defc6…`): `port-sv-heads.py` (`f55e5096…`), `fix-lookup.py` (`f5ccf26e…`) and
    `fix-lookup-pages.py` (`ddcfced5…`), with `-DPERL_CAPSTONE_SUBLET`. Its sources equal the twin's prot6 tree
    (`perlspec/fx2/prot6`, the board's prot6 image) except `Makefile.spec`, which names the compiler;
  - **bodies** (`7d5f41fa…`): heads plus `port-sv-bodies.py`, with `-DPERL_CAPSTONE_SUBLET_BODIES`. Its sources equal
    the twin's bodies tree (`perlspec/fx2/bodies`), except `Makefile.spec`;
  - **bodren** (`36c66ae5…`): bodies plus `fig10/renew/bodies-renew.py` (`9945e787…`, an experiment script, never a
    port script), with `-DPERL_CAPSTONE_SUBLET_BODIES_RENEW`.
- The variant changes two functions of the bodies port:
  - `S_sv_body_retire` returns `capstone_cap_renew(body)` where the port revoked and returned the segment's pointer.
    `del_body` stores that capability as the free-list link, in the word upstream already uses for it.
  - `S_sv_body_issue` hands a popped capability out as it is when its bounds are narrower than a segment (a renewed
    body). Otherwise, for a body fresh from an arena, it derives as before.
- Static census, bodies to bodren (`capop-census.py`): CREVOKE 9 → 6, CRENEW 0 → 3, funct7 0x04 (the bounds reads)
  +10, funct7 0x0c −3, funct3 3 −6, and 4-byte words +10.
- **This image never goes to P1.** P1 decodes funct7 0x54 as CSNODE.

**Runs.** `fig10/chain-perlbench-renew.sh`, five oneop boots, one image each, SPEC's test.pl (`-I. -I./lib test.pl`):
- zones3 bodies;
- zones3 bodren;
- opcount bodren;
- opcount bodies;
- zones3 heads, the environment's check against the twin.

The answer is the twin's own stock `test.out` (`7990ff94…`). On the twin, SPEC's reference differs in three known
scripts. A dry run with an echo runner reaches `CHAIN-PBRN-DONE` through all five.

**What is read: image-zone instructions only,** as in the heads half. oneop's table has no generations, and cycles
need a bitstream with CRENEW.

**Predictions.**
1. **Answers:** every boot's `test.out` is the same file.
   - If it equals the twin's stock output, it is read as is.
   - If all five agree but differ from the twin's, the instruction readings are still read, and the difference is
     recorded.
   - If an arm's output differs from the others, that arm is not read.
2. **Environment check:** bodies minus heads (image zone) within ±30 % of the twin's +636,442,127. The compiler
   differs: oneop's builds have no RVC words.
3. **Lifetime instructions** (opcount bodren against opcount bodies):
   - CRENEW appears, and CREVOKE falls by exactly the CRENEW count. Each body retire with a segment renews instead of
     revoking.
   - CDERIVE falls by at most the CRENEW count: only reissues of renewed bodies skip it.
   - The heads half read CRENEW 73 above CREVOKE's fall, from a cause it did not trace. So "exactly" is the claim
     here, and a miss is recorded as one.
4. **Image instructions: bodren minus bodies between −2 and −14 instructions per removed CDERIVE.**
   - The derive skipped on a reissue is the asm's 10 instructions plus its operand set-up, against 4-5 added for the
     bounds test.
   - The retire swaps a CREVOKE and its address arithmetic for one CRENEW.
   - The heads half read 13.51 per removed CDERIVE, with no bounds test.

**What follows.**
- If held, the saving per reissue is what a renew is worth in the bodies port's code.
- Its share of the bodies port's +0.733 % is the reading for item 3.
- The hash-entry and op-slab ports are the same change in principle. They are not built here.

### Amendment before any reading (written 2026-11-24 10:41): suns.pl, and the port's one-operand revision

The first boot (`fig10-pbrn-z-bod`) ended at once, and the chain was stopped. No arm was read. Two causes, each found
in source:

1. **test.pl cannot run in oneop.** SPEC's test.pl runs every test as a new process of `$^X`. The twin's stage
   provides one, and oneop's single-image runner cannot exec one.
   - A check of the stock arm with the twin's driver line (`$^X="../sw/perl"; do "./test.pl"`, `fig10-pbrn-chk-stock3`)
     prints `can't run '../../sw/perl base/cond.t': Permission denied` for all 317 tests, at 0.25 G image
     instructions against the twin's 87 G.
   - **The workload is now suns.pl** (`-I. -I./lib suns.pl`, SPEC's train input). It runs in one process and is what the
     5.36 heads half used.
   - The answer is SPEC's `suns.out` (`01688ea0…`). The stock arm matches it in oneop (`fig10-pbrn-chk-suns`:
     2,488,000,453 image instructions, a feasibility check made before this amendment).
   - The environment check against the twin's test.pl figure is dropped, because no twin reading of perlbench on
     suns.pl exists. Bodies minus heads is still read, with nothing to check it against.
2. **The 5.22 port emits the two-operand CREVOKE, which oneop's model refuses as illegal.** Its `capability.h` says so:
   use `capstone_cap_release`.
   - The heads arm faulted with cause 2 after 0.1 M image instructions, and bodies after 0.37 M.
   - The fix is the same one-operand revision Fig. 10 made to the 5.36 patch. `fig10/perlbench/oneop-shim.h`
     (`335b499c…`) is force-included into the port arms and maps `capstone_cap_revoke_child(parent, child)` to
     `capstone_cap_release(child)`. On this model a release ends the child's lifetime and every lifetime derived
     inside it.
   - The rebuilt arms are heads `453cbfed…`, bodies `0f6b36fe…` and bodren `618d7bf7…`
     (`build-perlbench-arms.sh` `0a175b46…`). The stock replay is again byte-identical to the build's executable.
   - A scan of each image's CREVOKE words (rs2 = x0 means one operand):
     - stock: 3 one-operand, 0 two-operand;
     - the first heads build: 3 one-operand, 3 two-operand (the fault; the scan fires);
     - rebuilt heads 6/0, bodies 9/0, bodren 6/0.

**Runs.**
- The chain is unchanged apart from the workload and fresh run names `fig10-pbrn2-*` (`chain-perlbench-renew.sh`
  `6db35678…`, an echo-runner dry run to `CHAIN-PBRN-DONE`).
- The boots are as before: zones3 bodies, zones3 bodren, opcount bodren, opcount bodies, zones3 heads.

**Predictions, as amended.**
1. **Answers:** SPEC's `suns.out` in every boot. A boot that differs is not read.
2. **Dropped:** the twin test.pl check.
3. **Kept unchanged:**
   - Lifetime instructions: CRENEW = CREVOKE's fall exactly, and CDERIVE falls by at most the CRENEW count.
   - Image instructions: bodren minus bodies at −2 to −14 per removed CDERIVE.

### Result: perlbench's SV bodies renewed instead of revoked, suns.pl in oneop (2026-11-24 10:41-10:50, read against 71e4e38d4b66 as amended by 057682723325)

All five boots (`fig10/runs/fig10-pbrn2-*`) returned status 0 with SPEC's `suns.out` (`output=MATCH`).

| arm | image-zone instructions | against stock |
|---|---:|---:|
| stock (`fig10-pbrn-chk-suns`, the feasibility run before the amendment) | 2,488,000,453 | |
| heads (`453cbfed`) | 2,648,354,139 | +6.445 % |
| bodies (`0f6b36fe`) | 2,699,578,683 | +8.504 % |
| bodren (`618d7bf7`) | 2,691,163,440 | +8.166 % |

The opcount boots agree with the zone boots to within 8 instructions: bodies 2,699,578,691, bodren 2,691,163,441.

**Executed lifetime instructions (opcount):**

| | CDERIVE (0x51) | CREVOKE (0x52) | CRENEW (0x54) |
|---|---:|---:|---:|
| bodies | 3,114,591 | 2,574,839 | 0 |
| bodren | 2,276,822 | 1,737,087 | 837,815 |
| change | −837,769 | −837,752 | +837,815 |

**Against the predictions.**
1. Answers: **met**, `suns.out` in all five.
2. **CRENEW equals CREVOKE's fall exactly: missed by 63.** CRENEW is 837,815 and the fall 837,752.
   - The heads half missed the same claim by 73, in the same direction.
   - Where the 63 renews come from is not traced here. A renew where the bodies arm made no revoke at all would give
     this, as would a revoke elsewhere that the bodies arm makes and bodren does not. Neither is checked.
3. **CDERIVE falls by at most the CRENEW count: met.** The fall is 837,769, so 46 renewed bodies were not handed out
   again before the end.
4. **Image instructions, −2 to −14 per removed CDERIVE: met.** Bodren minus bodies is −8,415,243 (−0.312 % of the
   bodies arm), **10.04 instructions per removed CDERIVE**. The heads half read 13.51 with no bounds test; this
   variant adds one.

**Reading.**
- Renewing a freed body instead of revoking it removes 16.4 % of the bodies port's own instructions on suns.pl:
  8.42 M of the +51.22 M that bodies adds to heads.
- The protected image goes from +8.504 % to +8.166 % against stock.
- These are instructions of the program only. oneop's table has no generations, and cycles need a bitstream that
  decodes CRENEW, as for the heads half.
- The arms here are the 5.22 port with the one-operand revision (`oneop-shim.h`), on suns.pl. The board's perlbench
  images are the two-operand port on test.pl. So these figures say what a renew saves in the port's code, not what
  any board image would read.
- Hash entries and op slabs are not built here. Their free paths have the same shape, a revoke at the free and a
  derive at the next issue, so the same change applies. Its saving there is not measured.

## perlbench item 3: test.pl's port code without its lifetime instructions (2026-11-24)

### Pre-registration: perlbench test.pl on P1, u = stock and n = prot6 with its lifetime instructions neutralised, one boot (written 2026-11-24 11:07, before the boot)

**Question.**
- The brief's attribution item for perlbench is test.pl's cost. prot6's +13,583.8 M in-context cycles (pair6) leave
  34.4-35.4 % unattributed at the mean instruction price.
- On suns.pl the N arm settled the same question for Perl 5.36 (result above): the code took 71.6 % at 3.260 cycles per
  twin instruction, and the lifetimes took 95.5-97.6 cycles per instruction.
- This boot runs the same instrument on test.pl. N is prot6 with its two lifetime instructions neutralised, so:
  - N−U is the port's code at its board price;
  - P−N is the lifetime instructions with their records. P comes from pair6's boot, through the stock check.

**The images.**
- u: stock `c1744e87…` (the paper's Full image), as in pair6.
- n: `perlbench_r.protnop.strip.dom` `9acc61bc…` (full `48620b2d…`), built in `nested/perlspec/fx2/protnop`:
  - a copy of prot6's tree (`fx2/prot6`, the heads port + `fix-lookup.py` + `fix-lookup-pages.py`);
  - `sv.c` gets the lifenop block of `nested/perl/lifenop/0003` (`caf0cf33…`), inserted after
    `#include <capstone/capability.h>`. The diff is `fx2/lifenop-522.patch` (`3ef306a8…`);
  - only `sv.o` is recompiled, with prot6's own compile line, and the image is linked with prot6's link line;
  - it is stripped as prot6 was: llvm-c4 `llvm-objcopy --strip-all`, keeping `.capstone_virtual`,
    `.capstone_application` and `.capstone_domreq`. The same command on prot6's full image reproduces `37381b03`.
- Checks on the build:
  - **Control.** prot6's `sv.c`, recompiled unchanged with that line, gives a `sv.o` byte-identical to prot6's.
  - **The header's asm** (`capstone/runtime/include/capstone/capability.h`, unchanged since 2026-10-17) is the asm
    the 0003 block copies, apart from the two substituted instructions.
  - **Census, prot6 to N** (`capop-census.py`): CDERIVE 72 → 41 (−31, the live derive at each inlined site), MOVC
    +31, CREVOKE 9 → 6, ADD-to-x0 0 → 3. Total 4-byte words (407,149) and RVC words (146,562) are equal, and the
    symbol table is identical (`mksyms.py` output equal to `perl-prot6.syms`). The census of prot6 against stock
    differs in many classes, so it can fire.
- **Twin control** (`nested/census/out/zperl-protnop-9acc61bc`, `fx2/stage-ptn`, prot6's twin stage with only the image
  replaced, run as prot6's: zones4, 2^20 records). makerand.out and test.out equal the stock twin run's (the twin's three
  known scripts). Image zone: N 86,774,083,025 against prot6's 86,774,090,874, **N−P = −7,849** (0.00001 %). So N executes
  prot6's instructions apart from the substitutions, and N−U on the twin is +2,440,555,323.
- **Lifetime instructions:** the port's executed CDERIVEs and CREVOKEs on test.pl, as prot6 minus N in the twin's opcode
  counts (`nested/p1pn/opc.sh`, `396dd4d1…`: opcount boots of `stage-pt6` and `stage-ptn`, running at this writing). The
  count is fixed by those two runs before the boot, enters the reader as its fourth argument, and is quoted with its runs
  in the result.

**Configuration.**
- Bitstream P1 (`…-pagedtbl_d83d3a633`), kit `nested/kits/bkit91-rec1m` (2^20 records, SHA256SUMS verified). The
  board is the runtime lane's to choose.
- Chain: `nested/p1pn/p1pn.conf` and `p1pn.list` on `nested/p1s100/nat-run3p.sh`. The chain reads P1 back by name,
  and flashes it sha-checked if it is not resident.
- Stage `nested/p1pn/pairn` (`pairn.SHA256SUMS`, 574 files). It is the kit's `make-train-stage.sh` around
  `fx2/stage-pairn`, which is `fx2/stage-pair6` with only its prot image replaced. Against pair6's board stage, one
  file differs, the prot `.xz`. So run.sh, the commands and the u image are pair6's byte for byte.
  - The stage's second arm is labelled "p" and runs N.
  - Commands: makerand.pl then test.pl, for u, then for n.
- List line `pairn 86400 660`, so `BAKED_TIMEOUT=345600` (the dry run prints it), as in pair6. Expected wall time:
  about 7.5 h of commands.
- Checked off the board (rule 8, `nested/p1n/recheck-p1k.sh` with `RD=nested/p1pn`):
  - real bake rc 0;
  - `BOARD_STUB=2` rc 0 on pair6's real transcript and on its cut, PASS all 9 steps;
  - real firmware build 23,760,392 B (`da2a5384…`);
  - an `NR_DRY=1` pass of the chain to DONE (`nested/p1pn/dry.log`).
- Read by `nested/p1pn/read-pairn.py` (`732d220d…`): this boot's transcript, pair6's, SPEC's references and the
  lifetime count.
  - Tested on pair6's own transcript as both inputs. It reproduces pair6's +13,583.8 M and counters, and its band
    and record checks fire (N−U "OUT", 100 % of every gap).
  - On a transcript cut after the first arm it prints NOT READ and exits 1.

**Rules.**
- Answers: every command rc 0, and both arms' makerand.out and test.out byte-equal to SPEC's references, with empty
  `.err` files. Otherwise nothing is read.
- Stock check: u's test.pl in-context cycles within ±1 % of pair6's u (267,219.2 M).
  - Outside it, N−U is still read within this boot.
  - P−N is not read: it needs pair6's P.

**Predictions** (test.pl, in-context cycles).
1. **N−U: 4,000 M to 11,000 M.** The twin counts it at +2,440.6 M instructions (prot6 minus stock).
   - At suns.pl's measured ratio to the mean (3.260 / 2.838 = 1.149, so 3.640 here) it would be 8,884 M.
   - At the stock mean (3.169) it would be 7,733 M.
   - At the lookup pair's 1.67 it would be 4,076 M.
2. **Records:** N's hpm3, hpm4, hpm5 and hpm7 each within 15 % of pair6's U-to-P gap from U. 15 % because suns.pl's N
   read hpm4 at 11.1 %. N's L1D misses (hpm8) lie between U's and pair6's P.
3. **P−N** is the rest of pair6's +13,583.8 M.
   - From it come the walks (P−N hpm4) and the replays (P−N hpm7 at 17-20 cycles each).
   - What is left is divided by the lifetime instructions above.

**What each outcome says.**
- **N−U 8,000-9,770 M** (suns.pl's ratio ±10 %): the code's price ratio carries over from Perl 5.36 to perlbench.
- **Below 8,000 M:** the code is cheaper here, and the lifetimes cost more per instruction than on suns.pl.
- **Above 9,770 M:** the code is dearer here.
- **The per-lifetime-instruction price set against suns.pl's 95.5-97.6:** close means the lifetime cost is a property
  of the instructions and their records. Far means it depends on the program's record working set (test.pl's record
  TLB misses are 89 % above stock, suns.pl's 66 %).

**Limits.**
- One run per arm, u first, not interleaved, as in pair6.
- P−N is cross-boot (pair6's P against this boot's N), each taken against its own boot's u. pair6's u reproduced st2's
  u to +0.009 % across boots.
- N is an instrument: its heads are unprotected copies of their arena's capability.
- Kernel + other is printed but not part of any prediction.

**The lifetime count, fixed before the boot** (2026-11-24 12:48; the p1pn boot has not started). `nested/p1pn/opc.out`,
opcount boots of `stage-pt6` (`opc-prot6-37381b03`, 11:07-11:54) and `stage-ptn` (`opc-protnop-9acc61bc`,
11:54-12:41), both rc 0, image zone:
- CDERIVE (0x51) 34,414,810 → 20,620,015, prot6 minus N **13,794,795**;
- CREVOKE (0x52) 30,966,183 → 18,717,505, prot6 minus N **12,248,678**;
- **lifetime instructions 26,043,473**, the reader's fourth argument.
- Cross-checks: MOVC (funct7 0x0a) rises by 13,794,897, 102 above CDERIVE's fall. All image instructions differ by
  −4,062 (the zone runs read −7,849). Neither enters the reading.

### Result: N−U +9,092.9 M, P−N +4,490.9 M, 129.1-133.9 cycles per lifetime instruction (board 1, 2026-11-24 14:20-22:00, read against 021cd655daec)

**The boot.**
- Chain `nested/p1pn/p1pn.*` (pid 2329273) ran on board 1 after M-16. It read P1 back by name as resident (no flash),
  booked 1476, and unbooked it (HTTP 200).
- The smoke passed all 9 steps. The watchdog sampled throughout (1,439 ALIVE/QUIET lines).
- The command line carries `capstone_vm.node_flat_records=1048576`.
- The stage still verifies against `pairn.SHA256SUMS` (rc 0, no mismatches). Its images decompress to the registered
  hashes: u `c1744e87`, n `9acc61bc`.
- The chain's `SPEC rc=2` is the generic classifier, which does not read this stage's markers. It is not a verdict.
- Transcript `…/out-nat-p1pn-perlbench-b1/pairn/board.txt` (sha256 `3c1c76f7…`). It was read by `read-pairn.py`
  (`732d220d…`) with pair6's transcript, SPEC's references and the lifetime count 26,043,473 fixed above.

**The checks.**
- **Answers: met.** All four commands rc 0. Both arms' `makerand.out` (`539d02dc…`) and `test.out` (`b41ee90a…`) equal
  SPEC's references, and every `.err` is empty.
- **Stock check: met.** u's test.pl reads 267,083.1 M against pair6's u 267,219.2 M, −0.051 %.

**In-context cycles, test.pl:**

| | cycles | difference | share of P−U |
|---|---:|---:|---:|
| u (this boot) | 267,083,082,253 | | |
| n (this boot) | 276,176,023,556 | N−U **+9,092.9 M (+3.40 %)** | 66.9 % |
| P (pair6, against pair6's u) | 280,802,971,216 | P−U +13,583.8 M | 100 % |
| | | P−N = P−U − N−U **+4,490.9 M** | 33.1 % |

**Predictions.**
1. **N−U, +9,092.9 M: met** (band 4,000-11,000 M).
   - That is 3.726 cycles per twin instruction (over +2,440,555,323), 1.176 times the stock mean of 3.169. suns.pl
     read 1.149.
   - It falls in the registered outcome "the code's price ratio carries over from Perl 5.36" (8,000-9,770 M).
2. **Records: met.**
   - N's distance from u, as a share of pair6's u-to-P gap: hpm3 (record-TLB misses) 1.7 %, hpm4 (walk cycles) 4.4 %,
     hpm5 0.7 %, hpm7 (event-23 replays) 3.9 %. All are under 15 %.
   - N's L1D misses (hpm8, 1,343.1 M) lie between u's (1,305.6 M) and pair6's P (1,416.2 M). N−U carries +37.5 M of
     them, 33.9 % of pair6's P−U +110.5 M.
3. **P−N** is the rest of P−U, +4,490.9 M. It divides as follows:

| part | cycles | share of P−U | price and its source |
|---|---:|---:|---|
| the port's code, measured | 9,092.9 M | 66.9 % | N−U on the board, 3.726 cycles per twin instruction |
| record-table walks | 295.6 M | 2.2 % | P−N hpm4, read as cycles (measured; each side against its own boot's u) |
| event-23 replays | 707.3-832.2 M | 5.2-6.1 % | P−N hpm7, +41.61 M at 17-20 cycles each (Addendum 1's RTL estimate, from source) |
| the lifetime instructions and their other record traffic | 3,363.1-3,487.9 M | 24.8-25.7 % | P−N less the two rows above |
| **not attributed** | **none at this resolution** | | N−U + P−N = P−U by construction |

- **A price per lifetime instruction: 129.1-133.9 cycles.** It is the last row over 26,043,473 executed CDERIVEs and
  CREVOKEs: prot6 minus N in the twin's opcode counts, fixed before the boot.
  - It covers the instruction and its record traffic beyond the walks and the replays, as on suns.pl.
  - N runs a MOVC or an ADD in each one's place, so the price is over a 1-cycle substitute.
- **Where the unattributed 34.4-35.4 % goes.**
  - At the stock mean the code would have taken 7,733 M (the pre-registration's figure). Measured, it takes
    9,092.9 M, so 1,359.9 M of the remainder is the code's.
  - The lifetime instructions take 3,363.1-3,487.9 M, most of the rest.
  - The walks and the replays move slightly against the earlier table, because P−N leaves out the small part of each
    that N−U carries.
- **Against suns.pl's 95.5-97.6.** The test.pl price is 1.32-1.40 times suns.pl's. No threshold for "close" was
  registered, and 32-40 % above is not close.
  - On the registered reading, the lifetime cost is not a fixed property of the instructions. It depends on the
    program's record working set.
  - test.pl's record-TLB misses are 89 % above stock and suns.pl's 66 %. That is consistent with this reading.
  - It is two programs, a consistency rather than a test. The counts also come from different instruments: suns.pl's
    from the one-operand build, test.pl's from the twin.
- **Reported, not part of any prediction:** kernel + other N−U +1,339.5 M; uptime span N−U +421.1 s.

**Limits.**
- As registered: one run per arm, u first, not interleaved.
- P−N is cross-boot: pair6's P against this boot's N, each taken against its own boot's u. The two u's agree to 0.051 %.
- N is an instrument. Its heads are unprotected copies of their arena's capability.

## perlbench item 3: CRENEW for hash entries and op slabs, and the protection fixtures of all three CRENEW variants, in oneop (2026-11-24)

### Pre-registration (written 2026-11-24 12:48, before the runs)

**Question.**
- Item 3's two remaining CRENEW lines, after the bodies half above, use the same change and the same reading.
- The fixtures answer the runtime lane's standing condition for all three variants. It was not checked for the bodies
  variant before its result: is protection unchanged when a renew replaces the revoke?

**The arms** (`fig10/build-perlbench-arms2.sh`, from the same oneopperl build directory, `make.out`'s compile and
link commands, and the one-operand shim `oneop-shim.h` as before):
- **he** (`42cb82ec…`): heads + `port-sv-bodies.py` + `port-he.py`, with `-DPERL_CAPSTONE_SUBLET -DPERL_CAPSTONE_SUBLET_HE`,
  as the twin's he arm.
- **heren** (`087516c3…`): he + `fig10/renew/he-renew.py`, an experiment script. In `capstone_sv_segs.h` (hv.c's
  copy of the segment helpers) it makes the bodies variant's two changes:
  - the retire returns `capstone_cap_renew(entry)`;
  - the issue hands a popped capability out as it is when its bounds are narrower than a segment.
- **ops** (`a1c60222…`): heads + `port-op-slabs.py` (revision 4), with `-DPERL_CAPSTONE_SUBLET -DPERL_CAPSTONE_SUBLET_OPS`.
- **opsren** (`e8338bba…`): ops + `fig10/renew/ops-renew.py`, also an experiment script:
  - `S_op_retire` returns `capstone_cap_renew(op)` where it revoked and returned the slab's own pointer. The free chain
    (`opslab_freed`) then holds the renewed child.
  - `S_op_issue` hands a reused slot out as it is when its bounds are narrower than its home slab's lifetime. A
    renewed child spans its slot, header included, which is what a derive would give.
  - Both paths still find the holding slab first, as in the port.
- **Static census** (4-byte words of `.text`):
  - he 9 CREVOKE, heren 6 CREVOKE and 3 CRENEW (`del_HE`'s three sites), +13 words;
  - ops 7 CREVOKE, opsren 6 CREVOKE and 1 CRENEW, +10 words.
- Neither image goes to P1, which decodes funct7 0x54 as CSNODE.

**Fixtures** (`fig10/build-renew-fixtures.sh`):
- The port's own fixture scripts, each compiled with its arm's `perlmain.o` line and linked in `perlmain.o`'s place:
  - `make-bodies-fixture.py` on bodies (`9151c874…`), bodren (`3150a3aa…`) and the control bodies-nr (`f04275c7…`);
  - `make-he-fixture.py` on he (`fd8f6afa…`), heren (`f0841be5…`) and he-nr (`dc50e9af…`);
  - `make-ops-fixture.py` on heads (`a7ba2720…`, the reference for mode 5), ops (`558ff148…`), opsren (`e3300a7e…`)
    and ops-nr (`27f93d9d…`).
- **The controls:**
  - each is a copy of its port arm rebuilt with the port's NOREVOKE define on every compile:
    `PERL_CAPSTONE_SUBLET_BODIES_NOREVOKE` for bodies and he, `PERL_CAPSTONE_SUBLET_OPS_NOREVOKE` for ops;
  - their CREVOKE counts fall to the shim's 6, 6 and 6.
- One oneop boot (`fig10/run-renew-fx.sh`) runs every fixture in modes 0-7. It prints each run's lines, its exit status
  and the kernel fault line that run added. The bodies and HE fixtures return 2 above mode 4.

**Runs.** `fig10/chain-perlbench-renew3.sh` (TAG `fig10-pbrn3`), with a dry run through both stubs to
`CHAIN-PBRN3-DONE` (`perlbench/dry3.log`). The boots, in order:
- the fixture boot;
- suns.pl, one image per boot: zones3 he, zones3 heren, opcount heren, opcount he, zones3 ops, zones3 opsren,
  opcount opsren, opcount ops.

The answer is SPEC's `suns.out` (`01688ea0…`), as in the bodies half.

**Predictions.**
1. **Fixtures.**
   - **Each renew arm faults exactly where its port arm faults.** A fault is read from the run's fault line, and the
     two arms are compared run by run.
   - **bodies and bodren:** mode 0 reads, and modes 1-4 fault.
   - **he and heren:** mode 0 reads, and modes 1-4 fault.
   - **ops and opsren:** mode 0 reads, and modes 1-4 fault. In mode 2, the op made in the freed op's slot reads, and
     so does the second op. Modes 5 and 6 return.
   - **Controls:**
     - bodies-nr reads in modes 0-3 and faults in mode 4 (the twin's control did);
     - he-nr reads in modes 0-4;
     - ops-nr reads in modes 0-3, faults in mode 4, and returns in modes 5 and 6.
   - **heads with the ops fixture:** reads in modes 0-3 and returns in modes 5 and 6. Its mode 4 is not predicted.
   - Mode 7 is recorded, not predicted. Its positive control was the twin sandbox's refusal of `/dev/null`, which
     oneop's need not share.
   - If a control does not read where predicted, its fixture's protected modes are not read.
2. **Answers:** `suns.out` in every suns.pl boot. A boot that differs is not read.
3. **Lifetime instructions** (opcount, renew arm against port arm, for he and for ops):
   - CRENEW appears;
   - CREVOKE falls by the CRENEW count to within 0.1 % of it. The two halves so far missed "exactly" by 73 and 63,
     untraced, so the band is stated here instead;
   - CDERIVE falls by at most the CRENEW count.
4. **Image instructions:** renew arm minus port arm between −2 and −14 per removed CDERIVE, for each. Bodies read 10.04
   and heads 13.51.

**What is read.**
- Each variant's saving, and its share of its port's own instructions (port arm minus heads, from a heads zones3
  reading; `fig10-pbrn2-z-heads`, 2,648,354,139).
- Not read: cycles, since no bitstream decodes CRENEW, and memory (not measured in oneop).
- The free lists hold the same words in the variant and the port, a renewed capability in place of the own pointer.

### Result: CRENEW for hash entries and op slabs, and the three variants' fixtures (2026-11-24 12:50-13:06, read against 835248d344fa)

Read with `fig10/read-renew3.py`, which was checked on a synthetic transcript with one mode altered (its checks fire).
All nine boots ran. The fixture boot is `oneop/runs/fig10-pbrn3-fx`, and the suns.pl boots are `fig10/runs/fig10-pbrn3-*`.

**Fixtures** (RET0 = returned 0 with no fault line; FAULT = the run added a fault line, every one cause 25):

| image | m0 | m1 | m2 | m3 | m4 | m5 | m6 | m7 |
|---|---|---|---|---|---|---|---|---|
| bodies, bodren | RET0 | FAULT | FAULT | FAULT | FAULT | 2 | 2 | 2 |
| bodies-nr | RET0 | RET0 | RET0 | RET0 | FAULT | 2 | 2 | 2 |
| he, heren | RET0 | FAULT | FAULT | FAULT | FAULT | 2 | 2 | 2 |
| he-nr | RET0 | RET0 | RET0 | RET0 | RET0 | 2 | 2 | 2 |
| ops, opsren | RET0 | FAULT | FAULT | FAULT | FAULT | RET0 | RET0 | 3 |
| ops-nr | RET0 | RET0 | RET0 | RET0 | FAULT | RET0 | RET0 | 3 |
| heads (ops fixture) | RET0 | RET0 | RET0 | RET0 | FAULT | RET0 | RET0 | 3 |

- **Prediction 1: met, all 15 checks.**
  - Every renew arm faults in the same modes as its port arm, at the fixture's own read in `main`.
  - The bodies pair faults at the same four pcs. The he and ops pairs' pcs differ by a constant 0x34 and 0x28
    across modes, a constant shift, consistent with the renew images' longer code.
  - Mode 0 reads its value in every image: `read cur=5`, `val_ok=1`, `ok=1 slabbed=1`.
  - Mode 2 of the ops fixture prints `reused_slot same=1 o3_ok=1 o2_ok=1` in ops and opsren. The op made in the freed
    op's slot reads, then the freed op faults. In opsren that slot is handed out as its renewed capability.
  - Modes 5 and 6 print `errors=3` and `errors=0` in all four ops-fixture images.
- **Not predicted:**
  - heads faults in mode 4, a kept op after its CV is freed. The heads arm has no op lifetimes, so this fault comes
    from elsewhere, not traced.
  - Mode 7 fired: `boot failed parse=13` and exit 3 in all four ops-fixture images. So oneop, like the twin, refuses
    `-e 0`'s `/dev/null`.
  - It follows that the bodies and HE fixtures, which boot from `-e 0` and do not read the parse's status, ran on
    an interpreter whose parse failed, as on the twin. Their controls read where predicted, so the C API reads they
    make work. Their fixture code is the port's, unchanged here.
- **Reading.** Replacing the revoke by a renew leaves the fixtures' protection unchanged in all three ports. This
  includes the bodies variant, whose result above was read without a fixture.

**suns.pl** (every answer `suns.out`; the opcount boots agree with the zone boots to within 13 instructions):

| | image zone | CDERIVE | CREVOKE | CRENEW |
|---|---:|---:|---:|---:|
| heads (`fig10-pbrn2-z-heads`; opcount `fig10-pbrn3-posthoc-opc-heads`) | 2,648,354,139 | 2,119,561 | 1,737,086 | 0 |
| he | 2,649,186,665 | 2,171,952 | 1,737,094 | 0 |
| heren | 2,649,451,709 | 2,171,945 | 1,737,086 | 8 |
| ops | 2,648,821,424 | 2,120,404 | 1,737,802 | 0 |
| opsren | 2,648,823,355 | 2,120,270 | 1,737,086 | 716 |

- **Prediction 2, answers: met.**
- **Prediction 3, lifetime instructions: met for both.**
  - CRENEW equals CREVOKE's fall exactly: 8 = 8, and 716 = 716.
  - CDERIVE falls by 7 and 134, each at most the CRENEW count.
- **Prediction 4, −2 to −14 instructions per removed CDERIVE: missed for both, with the sign reversed.** Both
  variants cost instructions instead of saving them:
  - heren +265,044 (+0.010 %) for 7 removed derives;
  - opsren +1,931 for 134, +14.41 per removed derive.
- **Why, for hash entries** (from the heads opcount, a boot added after the reading, not pre-registered):
  - suns.pl hands out 52,391 hash entries (he's CDERIVE minus heads') and frees 8 of them (CREVOKE, likewise).
  - The variant's bounds test runs at every issue. If all of the +265,044 is that test, it costs 265,044 / 52,391 =
    **5.06 instructions per issue**, the size the bodies half's pre-registration put on it (4-5). The split is
    inferred, not read from a per-function count.
  - A renew can save at most a derive per freed entry, and suns.pl frees almost none.
- **For op slabs:** 843 ops issued and 716 freed. The +1,931 is not traced. The test runs only on the reuse path, so
  its cost per reused slot is not known from these counts.
- **Reading.**
  - On suns.pl, the hash-entry and op-slab ports have little free traffic: +832,526 and +467,285 image instructions
    over heads, against the bodies port's +51.22 M.
  - Their CRENEW variants therefore read only the fixed cost of telling a renewed capability from an arena pointer.
  - What a renew saves where these allocators free often (test.pl, where hash entries cost +97.2 M on the twin) is not
    measured: oneop cannot run test.pl.
  - This result says nothing for or against the bodies variant's 10.04 saving.
- Not read: cycles, and memory.

## perlbench item 3: the full port, bodies, hash entries and op slabs together on the heads port (2026-11-25)

### Pre-registration: the combined image's fixture, test.pl and memory on the twin (written 2026-11-25 00:32, before the runs)

**Question.** Each of the three allocator ports was priced alone on the twin, with its own fixture. The board pair
of stock against the whole port needs one image that carries all three. This section checks that image on the twin
before any board pre-registration: protection, answers, instructions and memory.

**The images** (`nested/perlspec/fx2/build-full.sh`, `build-full-fixtures.sh`):
- **full** (`perlbench_r.full.strip.dom` `ce3a48e5…`, full `44ce64f2…`):
  - a copy of prot6's tree, then the committed `port-sv-bodies.py`, `port-he.py` and `port-op-slabs.py` (revision 4);
  - every object recompiled with prot6's compile lines plus `-DPERL_CAPSTONE_SUBLET_BODIES -DPERL_CAPSTONE_SUBLET_HE
    -DPERL_CAPSTONE_SUBLET_OPS`, linked with prot6's link line;
  - stripped as prot6 was. The same strip of prot6's full image reproduces prot6's `.dom` (control, IDENTICAL).
- **Census against prot6:** CDERIVE 72 → 100 (+28) and CREVOKE 9 → 16 (+7). The CREVOKE rise equals the three
  arms' own (+3, +3, +1).
- **full-nr**, the control (`7d6bb54e…`): full rebuilt with `PERL_CAPSTONE_SUBLET_BODIES_NOREVOKE` and
  `PERL_CAPSTONE_SUBLET_OPS_NOREVOKE` on all 69 compiles. Its CREVOKE count is prot6's 9, so all seven of the ports'
  CREVOKEs are gone (the census fires).
- **Fixtures:** the committed `make-{bodies,he,ops}-fixture.py`, each generated from the tree's own `perlmain.c`,
  compiled with its `perlmain.o` line and linked in its place, then stripped. Six images, `fixture-full/bin`.
- **Twin stage** `stage-full`: `stage-pt6` with only the prot image replaced (`diff -rq`: the `.xz` and plan.json's
  note).

**Runs** (`nested/census/chain-full.sh`, slot 28, 2^20 records, `dom_grant_root`):
1. Fixture `perlfxfull-ce3a48e5`, 35 commands: bodies and HE modes 0-4 on full and full-nr; ops modes 0-7 on full
   and 0-6 on full-nr.
2. test.pl zones `zperl-full-ce3a48e5` (zones4, `perl-full.syms`).
3. Memory `rss-full-ce3a48e5`, only if test.out and makerand.out equal the stock twin run's.
- The chain's gate took both branches in an echo dry run: on ops4's results it ran the memory step, and on revision 1's
  and on a missing directory it did not.

**Predictions.**
1. **Fixture**, each as its own arm read alone:
   - bodies on full: mode 0 reads, modes 1-4 fault with cause 25;
   - bodies on full-nr: modes 0-3 read, and mode 4 faults (the segment's end);
   - HE on full: mode 0 reads, modes 1-4 fault;
   - HE on full-nr: modes 0-4 read;
   - ops on full: mode 0 reads, modes 1-4 fault, mode 5 returns `errors=3`, mode 6 `errors=0`, and mode 7 prints
     `FIXTURE boot failed` and exits 3;
   - ops on full-nr: modes 0-3 read, mode 4 faults, and modes 5-6 return as on full.
   - Any other reading is reported as itself, and the board pair is not pre-registered until it is explained.
2. **Answers:** test.out and makerand.out byte-equal to the stock twin run's (specdiff 1/2, the twin's three known
   scripts).
3. **Image zone:** the arms' sum is +636.4 M (bodies) + 97.2 M (HE) + 628.0 M (ops) = +1,361.7 M over prot6's
   86,774,090,874 (+1.569 %). Predicted within ±15 % of that sum: +1,157 M to +1,566 M (+1.33 % to +1.80 %).
   - Below the band: the ports share work (the HE arm already carved from the bodies' segments).
   - Above it: they interact, for example through the lookup paths.
4. **Memory:** peak used within ±1 % of prot6's 102,620 kB (101,594-103,646 kB). Each arm read alone moved it by less
   than 250 kB.

### Result: the combined image passes its fixture, keeps stock's answers and costs +1,384.1 M in the image zone, inside the band (twin, 2026-11-25 00:32-02:17, read against e9cdb6841c9e)

**1. Fixture `perlfxfull-ce3a48e5`: met, all 35 commands as predicted.** Serial log sha256 `1f639564…`.

| arm | full | full-nr (no CREVOKE) |
|---|---|---|
| bodies | mode 0 reads (`cur=5`); modes 1-4 fault, cause 25 | modes 0-3 read (1 and 3 read the freed body, 2 the old body); mode 4 faults, cause 25 |
| HE | mode 0 reads; modes 1-4 fault, cause 25 | modes 0-4 read (1-2 `read_after_delete`, 3-4 `read_after_free`) |
| ops | mode 0 reads; modes 1-4 fault, cause 25; mode 5 `errors=3`, mode 6 `errors=0`; mode 7 `FIXTURE boot failed parse=13 run=0`, rc 3 | modes 0-3 read (1-3 `read_after_free`); mode 4 faults, cause 25; modes 5 and 6 as on full |

- Every fault is `kind=2 cause=25` with rc 139. Modes 3 and 4 fault only after their live reads (4,999, 2,999 and 3,000
  good reads, 0 bad), so the live objects stay readable while the freed ones fault.
- So each of the three ports' protection holds in the combined image, and removing the seven CREVOKEs is what lets
  the stale reads through. The control's two faults (bodies mode 4, at the segment's end, and ops mode 4) were
  predicted, as registered, and do not depend on a revoke.
- stderr carries the harness's `Can't open /dev/null: Permission denied`, as in the earlier perl fixtures.

**2. Answers: met.** `zperl-full-ce3a48e5`'s test.out and makerand.out are byte-equal to the stock twin run's
(`zperl-stock`), specdiff 1/2 as there, and the chain's gate ran the memory step.

**3. Image zone: +1,384.1 M, met** (band +1,157 M to +1,566 M).

| zone | prot6 | full | full − prot6 |
|---|---:|---:|---:|
| image | 86,774,090,874 | 88,158,235,556 | **+1,384,144,682 (+1.595 %)** |
| user | 5,638,059,112 | 5,666,285,925 | +28.2 M |
| kernel | 26,155,713,688 | 26,279,319,186 | +123.6 M |
| firmware | 23,300,788 | 23,286,332 | −0.01 M |

- `capmix.txt` sha256 `61f4deeb…`.
- It is 22.4 M (1.6 %) above the arms' sum of +1,361.7 M: within the band, so neither a shared-work saving nor an
  interaction is read.

**4. Memory: met.** `rss-full-ce3a48e5`: peak used 101,844 kB, −776 kB (−0.76 %) against prot6's 102,620 kB (band
101,594-103,646 kB), and +3.96 % against stock's 97,968 kB. The three arms read alone were −156 kB (bodies), −116 kB
(HE) and −236 kB (ops 4), −508 kB together. The sampler's VmHWM field reads 2,868 kB against prot6's 2,724; it is
reported, not read.

**5. Lifetime instructions, counted for the board pair** (not a prediction): `opc-full-ce3a48e5` (opcount, slot 29,
`nested/p1pf/opc29.sh`, run beside the memory run; the queued `opc.sh` was stopped before it started) against
`opc-prot6-37381b03`:

| executed on test.pl (image) | prot6 | full | full − prot6 |
|---|---:|---:|---:|
| all instructions | 86,774,070,441 | 88,158,232,696 | +1,384,162,255 |
| CDERIVE (funct7 0x51) | 34,414,810 | 53,440,173 | +19,025,363 |
| CREVOKE (funct7 0x52) | 30,966,183 | 45,503,771 | +14,537,588 |

- So the three allocators execute **33,562,951** more lifetime instructions than the heads port, 1.29 times the heads
  port's own 26,043,473 (pairn).
- The opcount's all-instruction difference agrees with the zones' (+1,384,144,682) to 17,573, 0.001 %.

**So the board pair is pre-registered** (next section): every check on the combined image came out as registered.


### Pre-registration: perlbench test.pl on P1, u = stock and f = the full port, one boot on board 1 (written 2026-11-25 02:28, before the boot)

**Question.** pair6 read the heads port (prot6) against stock on test.pl: +13,583.8 M in-context cycles, +5.08 %. (The
paper's figure is still prot2's +5.84 %; a change is the lead's.) The full port adds the three allocators (bodies, hash
entries, op slabs). This boot reads the full port
against stock in one boot (F−U), and, through pair6's boot, the three allocators' cost over the heads port (F−P6).

**The images.**
- u: stock `c1744e87…` (the paper's Full image), as in pair6 and pairn.
- f: `perlbench_r.full.strip.dom` `ce3a48e5…`, the twin result above: protection fixture met (35 of 35), answers equal
  to stock's, image zone +1,384.1 M twin instructions over prot6, memory 101,844 kB (−0.76 % against prot6).
- **Lifetime instructions** (twin opcode counts on test.pl, `opc-full-ce3a48e5` against `opc-prot6-37381b03`):
  +1,384,162,255 instructions in all, of which +19,025,363 CDERIVE and +14,537,588 CREVOKE: **ΔL = 33,562,951**.

**Configuration.**
- Bitstream P1 (`…-pagedtbl_d83d3a633`). Board 1 holds another seal (B5b-2) at this writing, so the chain flashes P1
  from its sealed bit, sha256 checked against `SEAL-…-pagedtbl_d83d3a633.sha256`, and reads the resident name back
  before the bake. The runtime lane reserved board 1 for this boot.
- Kit `nested/kits/bkit91-rec1m` (2^20 records, SHA256SUMS verified), pairn's.
- Chain: `nested/p1pf/p1pf.conf` and `p1pf.list` on `nested/p1s100/nat-run3p.sh`, list line `pairf 86400 660`, so
  `BAKED_TIMEOUT=345600`, as in pair6 and pairn. Expected: about 7.5 h of commands, plus the flash.
- Stage `nested/p1pf/pairf` (`pairf.SHA256SUMS`, 574 files), the kit's `make-train-stage.sh` around `fx2/stage-pairf`.
  Against pairn's board stage (`diff -rq`) one file differs, the prot `.xz`. So run.sh, the commands and the u image are
  pair6's byte for byte. Commands: makerand.pl then test.pl, for u, then for f.
- Checked off the board (rule 8, `nested/p1n/recheck-p1k.sh` with `RD=nested/p1pf`):
  - real bake rc 0;
  - `BOARD_STUB=2` rc 0 on pair6's real transcript and on its cut, PASS all 9 steps, `BAKED_TIMEOUT=345600`;
  - real firmware build 23,760,392 B (`95b1c6ce…`);
  - an `NR_DRY=1` pass of the chain to DONE (`nested/p1pf/dry.log`).
  - **Disclosed:** the recheck relinked the shared kernel tree `perf/k2n/linux-board-k2n-native-hpm2tree` with
    bkit91-rec1m's `node_flat_records=1048576`. The next 262,144-record bake from that tree recompiles the four
    `CONFIG_CMDLINE` objects (the mechanism seen in M-16's tsu-r2). This boot's own bake relinks it the same way.
- Read by `nested/p1pf/read-pairf.py` (`2906ef92…`): this boot's transcript, pair6's, and SPEC's references.
  - On pair6's own transcript as both inputs it reproduces pair6's +13,583.8 M and exits 0.
  - On a transcript cut after the first arm it exits 1.

**Rules.**
- Answers: every command rc 0, and both arms' makerand.out and test.out byte-equal to SPEC's references, with empty
  `.err` files. Otherwise nothing is read.
- Stock check: u's test.pl in-context cycles within ±1 % of pair6's u (267,219.2 M).
  - Outside it, F−U is still read within this boot.
  - F−P6 is not read: it needs pair6's P.

**Predictions** (test.pl, in-context cycles).
Prices from pairn (test.pl, P1, same kit): the heads port's code at 3.726 cycles per twin instruction (stock mean
3.169), and its lifetime instructions at 172.4 cycles each all in (P−N 4,490.9 M over 26,043,473, walks and replays
included), of which 129.1-133.9 is the remainder after the walks and replays.
1. **F−P6: +8,700 M to +11,800 M, point +10,945 M.**
   - Point: 3.726 × 1,384.16 M = 5,157 M of code, plus 172.4 × 33.56 M = 5,788 M of lifetimes.
   - Low edge: the code at the stock mean (4,386 M) and the lifetimes at the remainder-only price 129.1 (4,333 M).
   - High edge: the code at 3.726 and the lifetimes at 1.15 times the all-in price (6,656 M), for a larger record
     working set than the heads port's alone.
2. **F−U: +22,300 M to +25,400 M (+8.3 % to +9.5 %), point +24,529 M (+9.18 %)**: pair6's P6−U6 (+13,583.8 M) plus 1.
3. **Records:** F's hpm3 (record-TLB misses) and hpm4 (walk cycles) above pair6's P, each against its own boot's u.
   F's L1D misses (hpm8) above pair6's P.

**What each outcome says** (the reading is the lifetime price left over after the code at 3.726: (F−P6 − 5,157 M)
over 33,562,951):
- **Within ±15 % of 172.4** (146.6-198.3 cycles): the allocators' lifetimes cost what the heads' do per instruction.
- **Below:** the allocators' records are cheaper, for example because they share pages with the heads' records.
- **Above:** the record working set grows with the extra lifetimes, as on test.pl against suns.pl.
- If the stock check fails, only F−U is read.

**Limits.**
- One run per arm, u first, not interleaved, as in pair6 and pairn.
- F−P6 is cross-boot: this boot's F against pair6's P6, each taken against its own boot's u. pairn's u reproduced pair6's
  to −0.051 %.
- The prices come from pairn (one boot) and the twin's counts. Kernel + other is printed but not part of any prediction.

### Result: F−U +25,600.5 M (+9.58 %), F−P6 +12,016.7 M; both just above their bands; 204.4 cycles per lifetime instruction (board 1, 2026-11-25 02:29-10:24, read against 3c0441a27b8e)

**The boot.**
- The chain `nested/p1pf/p1pf.*` (pid 2403759) found board 1 on another seal. It flashed P1 from its sealed bit: local
  sha256 equal to the seal's, `FLASH END rc=0`, and the resident name read back as `…-pagedtbl_d83d3a633`. It booked
  1492 and unbooked it at the end (HTTP 200). The board is left on P1.
- The smoke passed all 9 steps. The watchdog sampled throughout (1,475 ALIVE/QUIET lines). The command line carries
  `capstone_vm.node_flat_records=1048576`.
- The stage still verifies against `pairf.SHA256SUMS` (rc 0). Its images decompress to the registered hashes: u
  `c1744e87`, f `ce3a48e5`.
- The chain's `SPEC rc=2` is the generic classifier, which does not read this stage's markers. It is not a verdict.
- Transcript `…/out-nat-p1pf-perlbench-b1/pairf/board.txt` (sha256 `5a1597c46680ee7c…`). It was read by
  `read-pairf.py` (`2906ef92…`) with pair6's transcript and SPEC's references, rc 0.
- Board time: boot 02:31:34 to 10:24:02, 7 h 52 min, against about 7.5 h expected.

**The checks.**
- **Answers: met.** All four commands rc 0. Both arms' `makerand.out` (`539d02dc…`) and `test.out` (`b41ee90a…`) equal
  SPEC's references, and every `.err` is empty.
- **Stock check: met.** u's test.pl reads 267,243.5 M against pair6's u 267,219.2 M, +0.009 %. So F−P6 is read.

**In-context cycles, test.pl:**

| | cycles | difference |
|---|---:|---:|
| u (this boot) | 267,243,452,830 | |
| f (this boot) | 292,843,972,843 | F−U **+25,600.5 M (+9.58 %)** |
| P6 (pair6, against pair6's u) | 280,802,971,216 | P6−U6 +13,583.8 M (+5.08 %) |
| | | F−P6 = F−U − P6−U6 **+12,016.7 M** (+4.50 % of u) |

**Predictions.**
1. **F−P6, +12,016.7 M: missed**, 216.7 M (1.8 %) above the band's high edge of 11,800 M. Point +10,945 M.
2. **F−U, +25,600.5 M (+9.58 %): missed**, 200.5 M above the band's high edge of 25,400 M (+9.5 %). It is prediction 1
   plus pair6's P6−U6, so it is the same miss counted once more.
3. **Records: met.** Each is f's distance from its own boot's u, against pair6's P from its u:

| counter | F−U | P6−U6 | ratio |
|---|---:|---:|---:|
| hpm3, record-TLB misses | +47,086,857 | +21,445,211 | 2.20 |
| hpm4, walk cycles | +692,425,980 | +309,100,291 | 2.24 |
| hpm5 | +587,350,062 | +247,290,028 | 2.38 |
| hpm7, event-23 replays | +126,937,338 | +43,288,874 | 2.93 |
| hpm8, L1D misses | +257,345,625 | +110,524,171 | 2.33 |

**The reading** registered for this outcome is the lifetime price after the code at 3.726:
(12,016.7 M − 5,157.4 M) / 33,562,951 = **204.4 cycles per lifetime instruction**. That is 1.185 times the heads'
172.4, above the registered 146.6-198.3, so it is the registered outcome **"above"**.

What the price depends on, stated before it is read:
- **It is under-determined by the code price.** The full port's code is priced at the heads port's 3.726 cycles per
  twin instruction, which was not measured for these allocators. At 3.873, 3.9 % more, the price reaches the band's edge,
  198.3, and at 3.88 it is inside it (198.0). At the stock mean 3.169 it is 227.3. This boot cannot separate "the lifetimes cost more" from "the
  allocators' code costs a few per cent more per instruction". Only an N-style instrument build of the full port
  (its lifetime instructions replaced, as pairn did for the heads) would.
- **The record counters do not show a larger record working set per lifetime instruction.** The registered gloss of
  "above" was "the record working set grows with the extra lifetimes". The lifetime instructions grow 2.29 times over
  the heads port's (26,043,473 + 33,562,951 against 26,043,473). The record-TLB misses grow 2.20 times and the walk
  cycles 2.24 times, slightly less than the lifetime count. So the gloss is not supported. What grows faster than the
  lifetime count is the event-23 replays (2.93 times): 2.49 per added lifetime instruction against 1.66 for the heads'.
  No mechanism is claimed.

**F−P6, attributed:**

| part | cycles | share of F−P6 | price and its source |
|---|---:|---:|---|
| the allocators' code | 5,157.4 M | 42.9 % | 3.726 cycles per twin instruction × 1,384,162,255 (pairn's N−U price for the heads' code; **assumed**, not measured here) |
| record-table walks | 383.3 M | 3.2 % | F−P6 hpm4, read as cycles (measured; each side against its own boot's u) |
| event-23 replays | 1,422.0-1,673.0 M | 11.8-13.9 % | F−P6 hpm7, +83,648,464 at 17-20 cycles each (Addendum 1's RTL estimate, from source) |
| the lifetime instructions and their other record traffic | 4,803.0-5,054.0 M | 40.0-42.1 % | F−P6 less the three rows above |
| **not attributed** | **none at this resolution** | | the last row is the remainder by construction |

- The remainder is **143.1-150.6 cycles per lifetime instruction**, against pairn's 129.1-133.9 for the heads'.
- It carries the code row's assumption: every 0.1 cycle per twin instruction moves 138.4 M between the first and the
  last rows, 4.1 cycles per lifetime instruction.

**Memory**, from the twin (not measured on the board): 101,844 kB, +3.96 % against stock's 97,968 kB.

**Reported, not part of any prediction:** kernel + other F−U +1,955.3 M (pairn's N−U +1,339.5 M); uptime span F−U
+1,133.0 s.

**Limits.**
- As registered: one run per arm, u first, not interleaved.
- F−P6 is cross-boot: this boot's F against pair6's P6, each taken against its own boot's u. The two u's agree to
  0.009 %.
- Both misses are within 2 % of the band's edge, on one boot per arm. No run-to-run spread for f has been measured.
- Reported, not read (no noise band was registered, so "missed" stands): the three boots' stock u's (pair6, pairn and
  this one) span 267,083.1-267,243.5 M, 160.4 M. The misses are 200.5 and 216.7 M, and F−P6 is cross-boot.

### Note: the s100 heartbeat (2026-11-25, board-free)

The `@@BEAT` field of the s100 stages read `$(stat -c %s st.db 2>/dev/null || echo 0)`. The board's busybox has no `stat`
applet, so every beat of the board-2 run read 0 and the fallback hid it ("The `st.db` field is void" above). The heartbeat line of `s100u/train/run.sh` and
`s100p/train/run.sh` now reads the size with `wc -c` (present in the baked rootfs as `usr/bin/wc`), and prints `NA` when
the file is absent, never a number. The manifests were regenerated (`sha256sum -c` rc 0). `read-s100pair.py` accepts
`NA` and the padded `wc` output. It was tested on `pair-dry/ret-twin-u1.txt` and on a copy with `NA` beats. The s100
rechecks predate the change, so rule 8 is re-run on them before the final campaign.

## M-16: a stress form of the allocation path (2026-11-25)

### Pre-registration: the protected allocation path alone, 100 then 400,000 packets in one boot on P1, board 1 (written 2026-11-25 11:40, before the boot)

**Question.** M-16's five boots (above) read the protected tshark's non-return as intermittent on P1. Pooled, 5 of 8
boots return. Every non-return holds the core at the CDERIVE of `wmem_block_fast_sublet_alloc`. One tshark run per
boot reads that rate at 40-68 min a sample. The result named the next form: many issues of that allocation path in
one boot. This boot runs that path alone, about 41 times as often as one tshark run, and asks whether it stops.

**The program** (`nested/tshark/stress/tsapp-stress.c`, `42b7ad10…`). It is not a port file, and it is not tshark.
- **Scopes.** It uses tshark's two scope kinds:
  - a packet scope (BLOCK_FAST) with K = 212 allocations per packet, then `wmem_free_all`;
  - a file scope (BLOCK). One allocation in 12 goes there, half of those are freed again with `wmem_free`, and the
    scope is reset every 1,000 packets.
- **Sizes.** A fixed LCG draws them: 70 % 8-64 bytes, 25 % 65-512, 5 % 513-4096. Each allocation's first and last
  bytes are written and read back into a checksum, so the checksum is a function of argv only.
- **Output.** A `STRESS pkt=…` heartbeat every N packets (N = 10 for 100 packets, 10,000 for 400,000) places a stop
  to within N packets.
- **Build** (`build-stress.sh`, `ad719e29…`):
  - compiled with work-sublet's own `tshark.c.o` compile line, and linked as `build-domain-native.sh` links
    `tshark_m5`: the same inputs, `tsapp-heap.o`, the SDK's libc and `-Wl,--no-relax`;
  - so the wmem 0001 objects (with the Q-15 repair), glib and the runtime are tshark's own objects.
  - **Control:** the same link of work-sublet's `tshark_m5.o` reproduces `tshark_m5.dom` byte for byte. It does in
    work-none too.
  - Image `c93cc972…`, stripped as tshark's is: `84d347d8…`.
- **The site.** `wmem_block_fast_sublet_alloc` holds the same word at the same offset as in tshark. tshark's stop pc
  `0x3ff6780f04` is offset 0x45c, word `0xa283955b`, the CDERIVE. In this image it is vaddr `0x77c9ee`, which loads
  at **`0x3ff676c9ee`**, assuming the same load base as tshark's.
- Census against the unprotected build: +36 CDERIVE and +5 CREVOKE.
- **Twin** (slot 28, 2^18 records, `nst-stress2`): the protected arm's 9,000-packet output equals the unprotected
  arm's, line for line.
  - It issues 1,748,489 fast allocations. tshark issues 1,905,321 per run (twin zones).

**Configuration.**
- P1 `…-pagedtbl_d83d3a633` is resident on board 1; the chain reads it back by name.
- Kit `t3-smoke/bkit91-k2n-native-p1-tree-cscfix` (2^18 records), M-16's.
- Chain `nested/p1st/p1st.conf` and `p1st.list` on `nested/p1s100/nat-run3p.sh`. The list line is
  `stress1 2700 210`, so `BAKED_TIMEOUT=10800`: a stop holds the board 3 h.
- Stage `nested/p1st/stress1` (`stress1.SHA256SUMS`, 19 files), from the kit's `make-train-stage.sh` around
  `tshark/stress/stage-board` (`make-stage-stress.py`, `e1c0ecaf…`). Commands, in order:
  - **s0:** 100 packets, the protected image;
  - **s1:** 400,000 packets, the protected image.
- **No unprotected arm on the board.** With two images the firmware is 30,051,848 B, over the kit's 26,000,000 B
  cap. The first stage was set aside for that reason.
  - The rule-8 recheck had built that firmware without refusing it: the cap is checked only by the real board run.
  - With one image the firmware is 23,760,392 B.
  - The unprotected control is on the twin, and in M-16's board record, where the unprotected tshark returned in
    every boot.
- Checked off the board (rule 8, `nested/p1n/recheck-p1k.sh` with this kit):
  - real bake rc 0;
  - `BOARD_STUB=2` rc 0 on tsu-r3's real transcript and on its cut, PASS all 9 steps, `BAKED_TIMEOUT=10800`;
  - real firmware build 23,760,392 B (`b1c1c36d…`);
  - an `NR_DRY=1` pass of the chain to DONE (`p1st/dry.log`).
- **Reference.** The twin ran this exact stage (`nested/out/nst-stress-board`, serial `153ffc99…`).
  - Both commands return.
  - s1 ends `packets=400000 fast=77731920 block=7068080 free=3535783 sum=8b070641d4aab910`.
- **Reader** `nested/p1st/read-stress.py` (`0384cfba…`). For each command it reports RETURNED (rc 0 and every
  STRESS line equal to the twin's), NO RETURN (the last heartbeat and the wedge apertures) or OTHER.
  - On the twin's serial as both inputs: both RETURNED.
  - On a copy cut after packet 150,000: NO RETURN at packet 150,000 (29,148,417 fast allocations).
  - On a copy with one checksum digit changed: OTHER, mismatches at the last two lines.
  - Its aperture patterns, run on tsu-r1's real transcript, read M-16's published values (`0x3ff6780f04`, `0x92`,
    `0x8c`, `0x14b9`).

**The hypothesis it tests (H1).** The non-return is a per-issue hazard of this allocation path, at M-16's rate and
independent of the rest of tshark.
- M-16's pooled non-return rate per tshark run is 0.085-0.755 (95 %). Over 1.905 M fast allocations a run, that is a
  hazard of 4.7e-8 to 7.4e-7 per allocation.
- s1 issues 77.7 M:
  - the expected number of stops is 3.6 to 57;
  - the chance s1 returns under H1 is at most 0.027;
  - the median stop falls between packets 4,800 and 76,000.
- s0's 19,360 allocations return under H1 with probability at least 0.986.

**Predictions.**
1. s0 returns, with every STRESS line equal to the twin's.
2. **s1 does not return.** Its last heartbeat lies before packet 400,000. The commit pc reads the stress image's
   CDERIVE, `0x3ff676c9ee`, and sw=225 and sw=220 read as in M-16.

**What each outcome says.**
- **s1 stops at `0x3ff676c9ee`:** the stop needs nothing of tshark beyond this path. The heartbeat places it to within
  10,000 packets, and this program is then the hardware side's reproducer, at minutes a sample instead of an hour.
- **s1 stops elsewhere:** reported as itself, with its pc.
- **s1 returns, equal to the twin:** H1 is rejected. At 77.7 M issues without a stop, the hazard of this path alone
  is below 3.9e-8 per allocation (95 %), under the lower bound M-16's rate needs (4.7e-8).
  - The non-return then needs some of the rest of tshark: its dissection, its heap and memory size, its I/O, or its
    run time.
  - The next stress form would add those, one at a time.
- **s0 does not return:** the boot is read at s0.

**Limits.**
- One boot. A return is a bound, not a proof.
- The stress program differs from tshark in everything except the allocators and their callers' shape. That is
  the point of it, but a return says nothing about which difference matters.
- The board's speed for this program is not known before the boot. The twin's s1 took 14.4 s of guest time; on the
  board it is estimated at 10-40 min. If the 3 h budget ends while the heartbeat still advances, that is reported as
  "no stop within T", not as a stop.

### Result: the allocation path stops on its own; s1 holds the core at the stress image's CDERIVE between packets 100,000 and 110,000 (board 1, 2026-11-25 11:41-14:49, read against 6659e74b24ca)

**Run.** P1 `…-pagedtbl_d83d3a633` read back by name before the boot and again at DONE. The chain ran its bake,
smoke (PASS, all 9 steps) and the stage, and unbooked board 1 (`UNBOOK 1500: 200`). The transcript is
`spec/board/out-nat-p1st-m16stress-b1/stress1/board.txt` (`6141714c…`). The reader (`read-stress.py`, `0384cfba…`) was
run on it against the twin's serial (`153ffc99…`).

**Readings.**
- **s0: RETURNED.** rc 0, and all 12 STRESS lines equal the twin's, ending `done packets=100 fast=19360 block=1840
  free=954 sum=1fe6df7c8e2b0e1f`. Span 4.7 s.
- **s1: NO RETURN.** 11 STRESS lines, every one equal to the twin's. The last is `pkt=100000 fast=19432859
  block=1767141 free=884810 sum=8b64d9b857f7aee5`. The twin's next line is `pkt=110000 fast=21375862`.
  - So the stop falls between 19,432,859 and 21,375,862 fast allocations. That is 10.2 to 11.2 tshark runs' worth
    (1,905,321 a run).
- **It is a stop, not slow progress.** The heartbeats reached the console about 60-75 s apart (the watchdog's UART-age
  resets, 15 s sampling). After the 100,000 line the console stayed silent for the rest of the 10,800 s budget. The
  watchdog's last line reads `no UART for 10209s`.
- **Apertures after the timeout, verbatim:**
  - trap log `0x89`; sw=224 `0x05`; sw=225 `0x92`; sw=226 `0x00`; sw=227 `0x03`; sw=228 `0xb8`; sw=229 `0x00`;
  - sw=192-195 all `0x00`; sw=219 `0xd7`; sw=220 `0x8c`; sw=221 `0x02`; sw=222 `0x00`;
  - commit pc `0x0000003ff676c9ee`; trap-log mepc `0xffffffff80007376`, tval `0x73`;
  - rev_node_debug_ex `0x00001346400007fa`, head `0x0071`, serving index `0x00001381`.

**Against the predictions.**
1. s0 returns, every line equal to the twin's: **met.**
2. s1 does not return, and its last heartbeat lies before packet 400,000: **met.**
   - The commit pc reads the stress image's CDERIVE, `0x3ff676c9ee`: **met.**
   - sw=220 `0x8c`, as in every M-16 non-return: **met.**
   - sw=225 `0x92` is tsu-r1's reading, not tsu-p1's `0x96`. The bit that differs is `stall`, as between M-16's own
     boots. M-16 read both values, so the prediction ("as in M-16") is met.
     - The reader compares sw=225 with tsu-p1's value only, so its line prints `[False, True]`. That is the reader's
       reference, not a different reading.
   - Trap log, mepc, sw=224, sw=226, sw=229, sw=192-195, sw=219, sw=221, sw=222 and head equal M-16's.
   - Two apertures differ from M-16's:
     - sw=227 `0x03` against `0x07`. The bit that differs is `data_req.write_en`.
     - sw=228 `0xb8` against `0x9f`. It holds the domain-switch index; this boot's domain set is not tshark's.
   - The serving index differs, as it did between every pair of M-16 non-returns.
3. The registered median stop was packets 4,800-76,000. This stop came later, at 100,000-110,000. That is one draw.

**What it says.**
- **The stop needs nothing of tshark beyond this allocation path.** The program has no dissection, no capture input,
  a far smaller live heap and no I/O beyond the heartbeat. It holds the core at the same instruction of the same
  function, with M-16's aperture signature.
- **The hazard.** One stop after about 2.0e7 allocations puts the per-allocation hazard at about 4.9e-8. The 95 %
  interval from a single draw is 1.2e-9 to 1.9e-7.
  - This lies at the low end of M-16's band (4.7e-8 to 7.4e-7).
  - At the band's lower bound, a first stop this late has probability 0.37-0.40. At its upper bound it is below 1e-6.
  - So this boot agrees with H1 at the lower part of M-16's rate, and argues against the upper part. It cannot
    separate H1 from a hazard below M-16's band, because it is one draw.
- **A reproducer.** As the pre-registration named, the stress program is now the hardware side's reproducer of M-16.
  A stop came about 11 min into s1 here, against 40-68 min per tshark sample.
  - It is not yet a directed test (CLAUDE.md rule 8). The next reduction is fewer allocator paths: BLOCK_FAST alone,
    with no file scope, then a fixed size.

**Limits.**
- One boot and one stop. The stop's position is one draw from an exponential, so the hazard interval is wide.
- No unprotected arm ran on the board (the cap, as registered). The unprotected control is the twin, where s1
  returns, and M-16's board record, where the unprotected tshark returned in every boot.
- The stress program is not tshark. It shares the allocators and their callers' shape and nothing else. Its stop
  therefore says that the path suffices. It does not say that tshark's stops come from this path alone, although the
  pc, sw=220 and the trap log agree.

### Note: what M-16's apertures can and cannot say (2026-11-25, board-free)

This corrects the labels in the stress result above and in M-16's earlier non-return readings. The drivers are traced
in `capstone-ariane` at `715bdd1fe` (P1's line); the details are in `tests/fpga-repros/M16-wmem-fast-alloc-stops-at-cderive/`
(d402073e4b14).
- **sw=220's bit 6 is `lsu_ready_o`, not "bypass empty"** (`load_store_unit.sv:341`). So `0x8c` reads: a valid
  CAPSTONE_DYN op at the LSU head, with the LSU not ready. Wherever this doc says "valid at the bypass head", read
  "at the LSU head, LSU not ready".
- **sw=195 and sw=229 are undriven.** Both read `load_unit.sv`'s `state_o`, which is declared and never assigned. Their
  0x00 carries no information.
- **sw=227's differing bit** (stress `0x03` against tshark's `0x07`) is `dom_switch_data_req.write_en`, the domain
  switcher's request. It is not the LSU's `data_req.write_en`, as the stress result above names it.
- **Not traced to an assignment:** sw=224-228, the trap log and the multi-byte rows. They are connected to submodule
  ports only, so their labels are unverified.
- **On B5b-2 (`904cd9c6b`) some rows are constants.** The rev-node debug word, head and serving index, and the switcher's
  bits of 225-228, are tied off (`capstone_sublet_life.sv:226-229`, `cva6.sv:2398-2411`). There they are no reading.

None of these changes a prediction or a verdict above. The stress stop was graded on the commit pc and sw=220, and
sw=220's value is unchanged; only its meaning is corrected.

### Pre-registration: the stress form on B5b-2, protected only, 100 then 2 × 300,000 packets in one boot on board 2 (written 2026-11-25 15:18, before the boot)

**Question.** Does M-16 exist on B5b-2, the design going forward? On P1 the stress form stopped after about 2·10^7
allocations (result above). B5b-2 is a different bitstream, `…-r75d-b5b2n_904cd9c6b`. It is also a different
allocator revision. B5b-2's trap route makes the two-operand CREVOKE illegal, so the image carries Fig. 10's wmem
0001, the one-operand revision from trap/eval, not the dev-era revision that P1 ran. **So a difference between the two
boots cannot be given to the hardware alone.**

**The image.** `nested/b5b2/build-stress-trap.sh` (`f928ae1c…`) builds `tsapp-stress.c` (unchanged, `42b7ad10…`) in
each B5b-2 tshark tree (`b5b2/tshark/{stock,prot}`, as `build-tshark.sh` built them for boots B):
- compiled with the tree's own `tshark.c.o` compile line;
- linked with the tree's recorded image link (`image/commands.json` [2] and [3]), with `tsapp_stress.o` in
  `tshark.c.o`'s place.
- **Controls, both trees:**
  - the recorded compile line recompiles `tshark.c` to the tree's `tshark.c.o` byte for byte;
  - the recorded link reproduces `image/tshark.dom` byte for byte.
- Images:
  - prot `118ac0e7…`, strip `12c63983…` (20,982,240 B);
  - stock `074f2756…`, strip `8647e28f…`.
- Scan (`scan_illegal.py`): ILLEGAL=0 in both. CDERIVE is 24 in prot and 4 in stock. Both carry the trap runtime
  (`trap_altstack`).
- **The site.** `wmem_block_fast_sublet_alloc` holds word `0xa283955b`, the CDERIVE that P1's stop names, at offset
  0x4f0, vaddr `0x7cc234`. In P1's image it was offset 0x45c. The same search on P1's image finds P1's `0x77c9ee`.
  - The twin's loader places vaddr 0x10000 at `0x3ff6000000` (`sublet: stress[…] enters 0x3ff6000000`). So this
    image's CDERIVE runs at **`0x3ff67bc234`**.

**Payload.** `nested/b5b2/boardS/out/fw_payload.bin`, sha256
`2b19fc2abae4916851cf028b4a2a51fd15c820e52b15196ad197995dd65b3ef5`, 21,657,608 B, under the 26,000,000 B cap. Built by
`trapk/fpga/make-payload.sh` (KEEP=y), with compiler's boot-5 `/init` and `smoke.dom` unchanged, and an empty probe list.
- **The unprotected image did not ride along.** Both images put the payload at 27,949,064 B, over the cap (that build
  is set aside in `attic/boardS-both`).
- The overlay (`make-boardS.py`, `dc44c9f2…`) has two stages; each runs under `/init`'s 3,600 s watchdog:
  - **N2:** p0 = `100 212 10 1`, then p1 = `300000 212 10000 1`;
  - **N3:** p2 = `300000 212 10000 1`, with the image read from N2's stage.
- Each command runs in a fresh tmpfs, with stdout live on the console.
- `/init`'s 120 s tick continues while the kernel runs. So a stop in the program (ticks continue) and a stop of the
  core or kernel (ticks stop) read differently. On P1 the console went silent.

**Twin.** The trap-route twin (`twinS`, `trapk/run-trap.sh`'s recipe with `boot-3600.py`; log `c9522bec…`) ran this
exact overlay:
- the control reads `smoke.dom rc=7`;
- all three commands return rc 0;
- every STRESS line equals the P1 native twin's (`pkt=100000 … sum=8b64d9b857f7aee5`, `pkt=300000 … sum=f6ed64e0d51b9d04`);
- spans 1.9 / 369.5 / 371.1 s of guest time.

The unprotected arm's twin (`twinS-both`, the over-cap overlay) is still running at this writing, and is reported with
the result.

**Reader** `nested/b5b2/read-boardS.py` (`d22d1262…`), run as `read-boardS.py <console log> twinS/boot.log`:
- **RETURNED:** rc 0, with every STRESS line equal to the twin's;
- **NO RETURN:** the last heartbeat, the twin's next one, and whether ticks continued after it;
- **OTHER:** a line differs, or rc is nonzero;
- **NOT RUN:** the command is absent.

It was tested five ways:
- the twin's log as both inputs: 3 RETURNED, exit 0;
- a copy cut after p1's packet-100,000 line, with two ticks appended: NO RETURN, "the kernel ran on";
- the same cut with no ticks: NO RETURN, "the core or kernel stopped";
- the cut with a watchdog line: reported as the stage bound;
- a copy with one checksum digit changed: OTHER, exit 1.

**Run.** Board 2, after boot B-prot. The board lane runs it with its B5b-2 capture runner. Runtime granted the board
for this run in its own message. The resident bitstream is read back before and after, and the bound is 2.5 h.
- After a stop, before any reset, the board lane reads the aperture snapshot M-16 was read with:
  - commit pc, trap log, mepc and tval;
  - sw=219, 220, 221, 222, 224-229 and 192-195 (193 is `store_buf_commit_cnt`);
  - rev-node head and serving index;
  - then it parks the switches at 0.
- The labels come from cva6.sv at 715bdd1fe; their meaning on B5b-2 is the RTL lane's to confirm.

**The hypothesis (H1 on B5b-2).** The stop is a per-allocation hazard of this path, at P1's rate.
- p0 + p1 + p2 issue 116,616,408 fast allocations.
- At P1's estimate, 4.9·10^-8:
  - 5.7 stops are expected, and a return of all three has probability 0.0033;
  - the first stop falls in p1 with probability 0.94, with its median near packet 72,800.
- At the lower end of P1's single-draw interval, 1.2·10^-9, a return has probability 0.87. So a return cannot show
  that M-16 is absent on B5b-2.

**Predictions.**
1. The control reads `smoke.dom rc=7`, and p0 returns equal to the twin's lines.
2. **Under H1, p1 does not return.** Its last heartbeat lies before packet 300,000, and the commit pc reads
   `0x3ff67bc234` with sw=220 `0x8c`.

**What each outcome says.**
- **A stop at `0x3ff67bc234` with M-16's signature:** M-16 exists on B5b-2 too, with this allocator revision.
  Whether the ticks continue says whether the kernel survives it.
- **A stop elsewhere, or with another signature (sw=221, for example):** reported as itself. It is not merged with
  M-16, nor with the B5T mcf wedge (sw=221 `0xf8`).
- **All three return, equal to the twin:** no stop in 1.17·10^8 allocations. The hazard on B5b-2 with this revision
  is below 2.6·10^-8 (95 %).
  - That is about half P1's point estimate, and inside P1's interval. It is a bound, not an absence.
  - It does not say whether the hardware or the allocator revision made the difference.
- **A watchdog line while the heartbeat still advances:** "no stop within T" for that stage, not a stop.

**Limits.**
- One boot.
- No unprotected arm on the board (the cap).
- The allocator revision differs from P1's.
- The board's speed on this route is not known before the boot. At P1's speed, 10,000 packets take about 65 s, so
  a stage takes about 1,950 s, under the 3,600 s watchdog.

## perlbench item 3: the full port's code without its lifetime instructions, test.pl on P1 (2026-11-25)

### Pre-registration: perlbench test.pl on P1, u = stock and n = the full port with its lifetime instructions neutralised, one boot on board 1 (written 2026-11-25 12:31, before the boot)

**Question.** The full port's board pair (result above, d50760796c1a) read F−P6 +12,016.7 M. It priced the lifetimes
at 204.4 cycles each, but only by assuming the allocators' code costs the heads' 3.726 cycles per twin instruction.
At 3.88 the price falls inside the band. This boot measures the allocators' code on the board, as pairn did for the
heads. N is the full port with its lifetime instructions neutralised, so:
- the allocators' code price is (NF−N)/(Z_NF−Z_N);
- the allocators' own lifetime price is ((F−NF) − (P6−N))/(L_FULL−L_HEADS).

**The instrument.**
- `nested/perlspec/fx2/build-fullnop2.sh` builds n, `perlbench_r.fullnop2.strip.dom` `3fec4963…` (full `943dfb3e…`):
  - a copy of the full port's tree (`fx2/full`, `ce3a48e5`);
  - `lifenop-522.patch`'s block (pairn's) after the capability.h include in the three translation units that call the
    lifetime helpers: `sv.c` (heads and bodies), `op.c` (op slabs) and `capstone_sv_segs.h` (hash entries, in `hv.c`);
  - only `sv.o`, `op.o` and `hv.o` recompiled, with the full port's own compile lines, linked with its link line and
    stripped as it was.
- **Controls:**
  - the full port's full image strips to its `.dom`;
  - the three objects, recompiled unchanged, are byte-identical to the full port's;
  - the compile warnings equal the control's (105, 24 and 17);
  - `mksyms.py` gives `perl-full.syms` exactly.
- **Census, full port to n** (`capop-census.py`): CDERIVE 100 → 56 (−44), MOVC +44, CREVOKE 16 → 6 (−10),
  ADD-to-x0 0 → 10. Word counts are equal (407,693 and 146,812).
- **One derive stays real: the body segment's, `s->life` of a chunk at its aligned address.**
  - The bodies and hash-entry issue paths take each child at `a − s->lo` from that life's base. That is the aligned
    address only for a real derive: a MOVC copy keeps the chunk's base.
  - The first instrument, `fullnop` (`82fd9265…`, `build-fullnop.sh`), neutralised that derive too. Every body was
    handed out the alignment pad off, and test.pl failed 58 output lines on the twin (40.0 G instructions against
    88.2 G).
  - It is set aside and not used. pairn's heads port had no such derive: its lives start at offset 0.
- **Twin** (`fx2/stage-fnop2`, the full port's twin stage with only the image replaced, zones4 and opcount, 2^20
  records):
  - makerand.out and test.out equal the stock twin run's.
  - Image zone: Z_NF 88,154,230,752 against the full port's 88,158,235,556, NF−F = −4,004,804 (−0.005 %). The
    neutralised derives skip a few instructions of their checks.
  - **Lifetime instructions, F minus NF** (opcount; `opc-full-ce3a48e5` against `opc-fullnop2-3fec4963`): CDERIVE
    32,847,623 and CREVOKE 26,758,241, so **L_FULL = 59,605,864**.
    - Against prot6's L_HEADS 26,043,473, the difference is 33,562,391. The twin's earlier opcode difference of the
      two images was 33,562,951.
    - MOVC rises 32,848,149.
  - Z_NF − Z_N = 1,380,147,727 and Z_NF − Z_U = 3,820,703,050.

**Configuration.**
- P1 is resident on board 1. The chain reads it back by name.
- Kit `nested/kits/bkit91-rec1m` (2^20 records), pairf's.
- Chain `nested/p1pnf/p1pnf.conf` and `p1pnf.list` on `nested/p1s100/nat-run3p.sh`, behind the M-16 stress chain
  (pid 3170600). The list line is `pairnf 86400 660`, so `BAKED_TIMEOUT=345600`.
- Stage `nested/p1pnf/pairnf` (`pairnf.SHA256SUMS`, 574 files). Against pairf's stage one file differs, the prot
  `.xz`, which unpacks to `3fec4963…`. Commands: makerand.pl then test.pl, for u, then for n.
- Checked off the board (rule 8, `recheck-p1k.sh` with `RD=nested/p1pnf`):
  - real bake rc 0;
  - `BOARD_STUB=2` rc 0 on pairf's real transcript and on its cut, PASS all 9 steps, `BAKED_TIMEOUT=345600`;
  - real firmware build 23,760,392 B;
  - an `NR_DRY=1` pass of the chain to DONE.
- **Reader** `nested/p1pnf/read-pairnf.py` (`3ed8e1d1…`). Its inputs are this boot, pairf's, pairn's and pair6's
  transcripts, SPEC's references, and `nested/p1pnf/counts.txt` (`5786cc10…`, the twin numbers above, fixed now).
  - With pairn's transcript as this boot it reproduces pairn's N−U +9,092.9 M and P6−N 172.44.
  - On pair6's cut it exits 1. With a count missing it exits 1.
- Disclosed as for pairf: this boot's bake relinks the shared kernel tree with `node_flat_records=1048576`.

**Rules.**
- Answers: as pairf's.
- Stock check: u within ±1 % of pair6's u. Outside it, only NF−U and the full port's code price are read.

**Readings and their outcomes.** Every arm is taken against its own boot's u.
- **NF−U** prices the full port's code: NF−U over Z_NF−Z_U, against the heads' 3.726 and the stock mean 3.169.
- **The allocators' code price** c_a = (NF−N)/(Z_NF−Z_N), with N from pairn.
- **The allocators' lifetime price** ((F−NF) − (P6−N))/33,562,391, with F from pairf and P6 from pair6.
  - It equals (12,016.7 M − c_a × 1,380.15 M)/33.56 M, so the two readings are one measurement seen twice.
  - Against the heads' 172.4 ± 15 % (146.6-198.3), as pairf registered:
    - **c_a ≤ 3.885, so the price is ≥ 198.3 (NF−U ≤ 14,454 M):** pairf's "above" stands, measured. The allocators'
      lifetimes cost more per instruction than the heads'.
    - **c_a between 3.885 and 5.142 (NF−U 14,454-16,189 M):** the allocators' lifetimes cost what the heads' do,
      and pairf's miss was its code price.
    - **c_a above 5.142:** the allocators' lifetimes are cheaper than the heads'.

**Prediction.** NF−U +13,500 to +16,800 M. The low edge puts the allocators' code at the stock mean, 3.169; the high
edge puts the lifetimes at the heads' remainder-only price, 129.1. Point +14,249 M: c_a = 3.736, the heads' price,
which is pairf's assumption. The pairf result's counters favour it: the record-TLB misses and walk cycles there scale
with the lifetime count, the replays faster.
- Records: n's distance from u in hpm3, hpm4 and hpm7, each under 15 % of pairf's F−U in the same counter, as N's
  were under pair6's.

**Limits.**
- One run per arm, u first, not interleaved.
- The allocators' code and lifetime prices take four boots: this one, pairf, pairn and pair6. Their u's span
  267,083.1-267,243.5 M so far.
- n is an instrument. Its children are unprotected copies of their parents' capabilities, and one derive per body
  segment stays real.

### Result: pairnf, board 1, P1 (2026-11-25 14:49-22:33)

The chain (pid 461875) read P1 back by name and booted pairnf 14:49:52-22:33:46: BOOT rc 0, smoke PASS on all 9
steps, unbooked (HTTP 200). The chain's `SPEC rc=2` is the generic classifier's known format gap (it wants
`@@SPEC-DONE <n>`); the transcript has all four `@@SPEC-END` lines with rc 0 and `@@SPEC-DONE 500.perlbench_r test`.
Read by `read-pairnf.py` (`3ed8e1d1…`, exit 0) with pairf's, pairn's and pair6's transcripts and `counts.txt`.

- **Answers:** makerand.out and test.out equal SPEC's references for u and n (test.out `b41ee90a…`), with empty
  stderr.
- **Stock check met:** u 267,214.1 M against pair6's 267,219.2 M (−0.002 %). Under the same check, pairf is +0.009 %
  and pairn −0.051 %.
- **NF−U +13,556.0 M (+5.07 %)**, inside the predicted +13,500 to +16,800 M, at its low edge. The point was +14,249.
  The span is +625.5 s, and kernel and other +1,826.9 M.
- **The full port's code price:** 13,556.0 M / 3,820,703,050 = **3.548 cycles per twin instruction**, between the
  stock mean 3.169 and the heads' 3.726.
- **The allocators' code price:** c_a = (NF−N)/(Z_NF−Z_N) = (13,556.0 − 9,092.9) M / 1,380,147,727 = **3.234**.
- **The allocators' own lifetime price:** ((F−NF) − (P6−N))/33,562,391 = (12,044.5 − 4,490.9) M / 33,562,391 =
  **225.1 cycles per lifetime instruction**.
  - The full port's lifetimes all in, (F−NF)/L_FULL, are 202.1. The heads' (P6−N)/L_HEADS reproduce 172.44.
  - The identity with pairf's registered form holds: (12,016.7 − 3.234 × 1,380.15)/33.56 = 225.1.
- **Outcome: the first branch.** c_a ≤ 3.885, so the price is ≥ 198.3. pairf's "above" stands, measured: the
  allocators' lifetimes cost more per instruction than the heads' (225.1 against 172.4 ± 15 %, band 146.6-198.3).
- **Records met:** n's distance from u in hpm3, hpm4 and hpm7 is −12.2 %, −10.8 % and −8.5 % of pairf's F−U in the
  same counter, each under 15 % in size. All three are negative: n reads slightly fewer of those events than u. In
  hpm8, which is not registered, n is +14.5 %.
- Source of every cycle price above: P1's `run_cycles` for each command on board 1, as the pre-registration names.
  The twin supplies only the instruction counts (`counts.txt`).

## Fig. 10's one-operand revisions for B5b-2 (2026-11-25)

### The images: rebuilt on the trap route, every one scanned clean

The compiler lane found that the FRCDKMSa3 images carry two-operand CREVOKE, which B5b-2 makes illegal, and that they
link the vexec runtime, which is the wrong one. The runtime lane relayed compiler's recipe:
- the trap-route SDK `c4/b5b2spec/sdk-trap` (origin/trap/eval `ed8cc375c2b1`'s `build-sdk.sh`, llvm-c9 clang
  `29027126f39b`) and the Full arm's flags;
- every image scanned with dev's `capstone/tests/b5b2-illegal-scan/scan_illegal.py` (`937208e0991f`), where exit 0
  is clean.

The one-operand ports needed no source change. Scripts are in `nested/b5b2/` (`env.sh` `1c2879b0…`, the Full flags
being the cclever base set plus `FRCDKMSa3/extra-flags`). Each port keeps its oneop recipe; only the SDK, the runtime
tree and the flags change.

| program | script | arms (sha256) |
|---|---|---|
| SQLite speedtest1, memsys5 patch | `build-sqlite.sh` | unprot `00d66489`, prot `03ca5713` |
| mruby 4.0.0-rc2, 0008 | `build-mruby.sh` | stock `576bc0ee`, prot `636e4cab` |
| Perl 5.36.3, 0001 / 0001 + 0002 | `build-perl.sh` | stock `87ce6988`, prot `c27c9c9c`, p0002 `73f02969` |
| gcc 502, the collaborator's three allocators and gcc's own free lists | `build-gcc.sh` | stock `48bd30a0`, fix `6c78dc88`, all `06e61814`, all-abc-bm-ls-tfl-phi `0b88e520` |
| tshark 4.6.8, wmem 0001 | `build-tshark.sh` | stock `c209a353`, prot `4e01172f` |

- **Every image scans ILLEGAL=0** (`nested/b5b2/MANIFEST.txt`, `b4228bba…`).
  - The scanner fires on prot6 (9 two-operand CREVOKEs, exit 1).
  - The oneop SQLite image scans clean too, so the scan alone does not separate the SDKs.
- **Every image links the trap runtime.** `trap_altstack` is defined in each, and absent from a FRCDKMSa3 image.
- **The Full flags reach every image.**
  - mruby has SHRINKTO 7,601, where the oneop image had none; Perl and tshark have their own.
  - Perl's `make.log` carries them on its compile lines, and tshark's SDK `sdk.json` lists them.
- Each port's own route to the flags:
  - **mruby.** `MRBD_OPT` also reaches the native host build, so the flags come from a wrapper `capstone-cc` in a copy
    of the SDK (`b5b2/sdk`; the wrapper's sha256 is `de4d3de2…`). `build-mruby-domain.sh` writes into its SDK
    directory, so it is never given the compiler lane's own.
  - **Perl and tshark.** Each builds its own SDK from the trap tree (`b5b2/wt-trap`, a worktree at `ed8cc375c2b1`).
    - Perl's `build-perl-domain.sh` is called directly from Fig. 10's port trees, because `build-virtual.sh` takes the
      port from `CAPSTONE_REPO_ROOT`.
    - tshark is the trap tree's own build: its wmem 0001 equals Fig. 10's. Its flags go through
      `CAPSTONE_SDK_EXTRA_CODEGEN` in the joined form `-mllvm=X`, because the split form reached clang without its
      `-mllvm`.
  - **gcc's arms** are oneop's build trees, with their `make.out` retargeted to the trap SDK and the flags. Every one
    of the 387 compiles and the link is asserted, and the trees are rebuilt by the port's `replay-build.sh`.
- Disclosed: llvm-c9's freshness gate prints STALE (176 ninja steps pending, outside the revision chain), as in
  compiler's own SDK build.

### Pre-registration: B5b-2 boot A, SQLite, mruby and gcc, stock against protected, board 2 (written 2026-11-25 13:14, before the boot)

**Question.** Do Fig. 10's one-operand revisions run on B5b-2 silicon, on the trap route, with the stock answers, and
what do they cost there in wall-clock time? No board has run them before.

**The image.** `nested/b5b2/boardA/out/fw_payload.bin`, sha256
`298b89daa59d6186daaf7039558f1c98ab9b8835317ce0a3c76fc3cece991416`, 25,851,912 B, under the 26,000,000 B cap.
- Built by the runtime lane's `trapk/fpga/make-payload.sh` with KEEP=y: the board's rootfs unchanged, the trap-route
  kernel relinked, and the keep-tags firmware.
- The overlay (`make-boardA.py`, `0d8b028b…`) carries compiler's `/init` and `smoke.dom` from boot 5, unchanged, and
  an empty probe list. Three stages, each command in a fresh tmpfs directory with its image xz-unpacked there:
  - **N1 SQLite:** `speedtest1 --heap 8388608 64 --size 1 st.db`, u p p u;
  - **N2 mruby:** `bm_ao_render.rb 16`, u p p u;
  - **N3 gcc:** SPEC's test `t1.c -O3 -finline-limit=50000 -o out.s`, u p. gcc's p is the port with gcc's own free
    lists (`all-abc-bm-ls-tfl-phi`).
- Stripped images (`nested/b5b2/strip`, `llvm-objcopy --strip-all` keeping the three capstone sections).

**Run.** The board lane runs it with its B5b-2 capture runner on board 2, resident `…-r75d-b5b2n_904cd9c6b.bit`,
read back before and after. The bound is 3 h; `/init`'s watchdog bounds each stage at 3,600 s. The gate is boot 5's
`control smoke.dom rc=7` on board 3, which the board lane reports.

**Reader** `nested/b5b2/read-boardA.py` (`48d72b29…`):
- the control line;
- SQLite's STUDY-ORACLE phases per arm against `phases.json`'s native reference (32 phases);
- mruby's stdout against `cfd73bb1…` (781 B, the native reference);
- gcc's `out.s` against SPEC's `t1.opts-O3_-finline-limit_50000.s` (`58111208…`);
- per program, the protected arms' mean span against the stock arms'.

It was tested three ways: a twin log cut in mruby (exit 1: NO RESULT, NOT RUN), the complete twin log (exit 0), and
that log with one phase hash and one mruby hash altered (exit 1, both marked WRONG).

**Twin evidence** (`nested/b5b2/twinA`, the trap-route QEMU twin as `trapk/run-trap.sh` boots, `boot.py` with its
per-line timeout raised from 300 s to 3,600 s; log `f63c48cc…`):
- control `smoke.dom rc=7`;
- every arm rc 0 with its reference answer: SQLite 32/32 phases ×4, mruby `cfd73bb1` ×4, gcc `out.s` equal ×2;
- twin spans, not a prediction of silicon: SQLite p/u +9.0 %, mruby +13.2 %, gcc +18.7 %.
- gcc prints the SPEC input's own warning (`conflicting types for built-in function 'printf'`, 68 B), as on any
  platform.
- An earlier twin run of the same overlay was cut by `boot.py`'s 300 s line limit during mruby. The twin's clock runs
  slower than the guest's, so `/init`'s 120 s tick does not keep it. Its four SQLite and two mruby arms matched.

**Predictions.**
1. The control reads `smoke.dom rc=7`.
2. Every arm returns rc 0 with its reference answer.
3. **Cost** (wall-clock span, the only time this image reads): p against u in each program, between −2 % and +30 %,
   point at the twin's value. No counter in this image is in-context, so a span includes the kernel and the trap
   route's own work.

**What each outcome says.**
- **All answers met:** the one-operand revisions run on B5b-2 silicon with stock's answers. Item 4 then has its
  first silicon row for these three, priced in wall-clock time.
- **A protected arm faults or answers differently while its stock arm returns:** a B5b-2 defect or a port defect,
  reported with its dmesg and stderr. The scan has already ruled out an illegal encoding.
- **The control fails:** the boot is void (loader or kit).

**Limits.**
- One boot. The spans are from `/proc/uptime`, without in-context counters.
- The arms are ABBA for SQLite and mruby, and one each for gcc.

### Result: boot A, every arm returns with its reference answer; protected against stock −2.3 % SQLite, +5.2 % mruby, +9.0 % gcc in wall-clock time (board 2, 2026-11-25 13:18-14:06, read against de6399c1dfff)

**Run.** The board lane ran the image on board 2 with its B5b-2 capture runner. Resident
`…-r75d-b5b2n_904cd9c6b.bit`, read back by name before and after (rc 3 both times: the name matches and the console
exposes no sha256, which the runner accepts). The run ended on `TRAPTEST: done` (rc 0), and board 2 was unbooked
(`UNBOOK 1505: 200`).

- **Deviation.** The registered gate, boot 5's control on board 3, did not run first. The runtime lane waived it for
  this boot. The in-boot control, the same `smoke.dom` from boot 5, ran before every stage.
- The console capture is `b5b2linux/b2bootA-20261125T0518/uart.txt` (`922be290…`, the board lane's tree). I re-ran
  the reader (`48d72b29…`) on it: exit 0, with output identical to the board lane's grade file.

**Readings.**
- **Control:** `TRAPTEST: control smoke.dom rc=7`.
- **N1 SQLite**, stage rc 0. Each arm rc 0, 32/32 STUDY-ORACLE phases equal to the native reference, stderr 0 B.
  - Spans u1 35.57 s, p1 33.06 s, p2 33.42 s, u2 32.49 s. Mean p 33.24 s against mean u 34.03 s: **−2.32 %**.
- **N2 mruby**, stage rc 0. Each arm rc 0, stdout `cfd73bb1…` (781 B) equal to the native reference, stderr 0 B.
  - Spans u1 506.94 s, p1 532.93 s, p2 533.34 s, u2 507.03 s. Mean p against mean u: **+5.16 %**.
- **N3 gcc**, stage rc 0. Each arm rc 0, `out.s` `58111208…` equal to SPEC's reference.
  - stderr 68 B in both arms. It is the SPEC input's own warning (`t1.c:2:5: warning: conflicting types for built-in
    function 'printf'`), as on the twin.
  - Spans u1 7.20 s, p1 7.85 s: **+9.03 %**.
- dmesg shows each image entering through the `.dom` loader (`sublet: <name>[pid] enters …`), one line per arm.

**Against the predictions.**
1. The control reads `smoke.dom rc=7`: **met.**
2. Every arm returns rc 0 with its reference answer: **met**, all ten arms.
3. Cost between −2 % and +30 % in each program:
   - mruby (+5.2 %) and gcc (+9.0 %): **met**;
   - SQLite (−2.3 %): **not met**, 0.3 points below the band's floor.
   - All three are well below the twin's values (+9.0 %, +13.2 %, +18.7 %), which were the registered points.

**What it says.**
- **The one-operand revisions of SQLite, mruby and gcc run on B5b-2 silicon, on the trap route, with stock's answers.**
  These are the first silicon rows for these three.
- **mruby's cost is resolved.** The two stock arms agree to 0.09 s and the two protected arms to 0.41 s, against a
  26 s gap. +5.2 % is the protected revision's wall-clock price in this boot.
- **SQLite's is not.** The stock arms differ by 3.08 s between u1 and u2, and the protected arms lie between them. The
  first arm of the boot is the slowest. So the −2.3 % sits inside this boot's own order effect, and it reads as "no
  cost resolved at about ±5 %". It does not say the protected revision is faster.
- **gcc's +9.0 %** is one arm each, so it has no spread of its own.
- The twin's spans overstated every cost. Its clock is not the board's, as the pre-registration said; its spans were
  never a prediction of silicon.

**Limits.**
- One boot. Wall-clock spans from `/proc/uptime`, including the kernel and the trap route's work. No in-context
  counters.
- SQLite is ABBA with a first-arm effect larger than the difference read. gcc is one arm each.

### Pre-registration: B5b-2 boots B-stock and B-prot, Fig. 10's Wireshark workload one arm per boot, with Perl makerand.pl, board 2 (written 2026-11-25 13:38, before the boots)

**Question.** Does Fig. 10's Wireshark revision run on B5b-2 silicon with stock's answer? What does it cost in
wall-clock time across the two boots? And do Perl 5.36's stock and 0001 + 0002 images run there at all?

**Why two boots.**
- The base payload without an overlay is about 17.2 MB. The two stripped tshark images are 5.2 MB each in xz, so
  they do not fit under the 26,000,000 B cap together.
- So each boot carries one tshark arm and one Perl arm.

**The images** (`make-payload.sh`, KEEP=y; overlays by `make-boardB.py`, `07323665…`):
- **B-stock:** `nested/b5b2/boardB-stock/out/fw_payload.bin`, sha256
  `c89a676d3d67ec537ec770f530c34dc53755f505274c65e7d5c4e52f4f3b6272`. Contents: Perl stock, then tshark stock.
- **B-prot:** `nested/b5b2/boardB-prot/out/fw_payload.bin`, sha256
  `2effea0096ee964f4239300ce66eeea707920acaf57d1e70d33bcc867fa4d9fa`. Contents: Perl p0002, then tshark prot.
- Both are 25,851,912 B, as boot A's image is. The payload pads to its size, so they are told apart by hash, and their
  initramfs files differ from the overlay on.
- **Stage N0, Perl 5.36 on SPEC's test `makerand.pl`.** This is does-it-run only, with no cost reading, at about 0.1 s
  of work.
  - SPEC's `test.pl` cannot serve Perl 5.36. It runs each test as `..//$^X`, and upstream 5.36 makes `$^X` absolute,
    so every child fails (317 of 317 on the native build).
  - The reference is native 5.36's own `makerand.out`, `acbf40c2…`: deterministic, and not SPEC's 43,841 B because
    5.36's rand differs.
  - Fig. 10's suns.pl row is a train input and waits for the final campaign (runtime lane, the lead's rule).
- **Stage N1, Fig. 10's Wireshark workload, as oneop's `run-app-zones4.sh` runs it.**
  - Each of the 169 captures of `set169.txt` runs as `tshark -r caps/<c> -V -n`, with TZ=UTC and an empty
    configuration directory.
  - Each stdout is appended after a `=== <c> rc=<n>` header. The answer is that file's sha256, against oneop's
    `b190069b…`, which matched in all six oneop boots.
  - A heartbeat is printed every 20 captures.

**Run.** The board lane runs both boots with its B5b-2 capture runner on board 2, stock first, each bounded at 2 h,
with the resident read back. `/init` bounds each stage at 3,600 s.

**Reader** `nested/b5b2/read-boardB.py` (`122f1974…`): the control line, the Perl arm, then the tshark arm's 169
inputs, 0 nonzero, the answer's hash, and its span.
- On both twin logs it exits 0.
- On a twin log of the overlay without the Perl stage it exits 1 (`perl: NO STAGE`).

**Twin evidence** (the trap-route QEMU twin, as boot A's; logs `a9ec37c9…` and `f469853f…`): both boots give control
rc=7, Perl's `acbf40c2…`, tshark's 169 inputs with 0 nonzero, and `b190069b…`.
- The twin's spans are not repeatable. tshark stock read 166.3 s and 193.1 s in two twin runs of the same image, and
  prot 194.3 s and 181.7 s. None is read.

**Predictions.**
1. Both controls read `rc=7`.
2. Both Perl arms answer `acbf40c2…` with rc 0.
3. Both tshark arms answer `b190069b…` with 169 inputs and 0 nonzero.
4. **Cost** (cross-boot, wall-clock span, kernel and trap route included): prot against stock between −5 % and +15 %,
   point +2.3 %, oneop's image-instruction cost.

**What each outcome says.**
- **All met:** Fig. 10's Wireshark revision runs on B5b-2 silicon with stock's answer, and so does Perl 5.36.
- **tshark prot does not return, or answers differently:** reported with the heartbeat and dmesg. M-16's stop on P1
  was in this allocator's CDERIVE, but B5b-2 is another bitstream with another route, so nothing is predicted from it.
- **A control fails:** that boot is void.

**Limits.**
- The cost is cross-boot, one boot per arm, spans only.
- Perl is does-it-run only.

### Result: boots B-stock and B-prot, Wireshark and Perl run with stock's answers; tshark prot against stock +2.5 % (board 2, 2026-11-25 14:06-15:56, read against 17d859cc89e4)

**Run.** The board lane ran each image on board 2, `…-r75d-b5b2n_904cd9c6b.bit`, read back by name before and after each
boot (rc 3: the name matches, no sha256 exposed). Each run ended on `TRAPTEST: done` and board 2 was unbooked
(`UNBOOK 1506: 200`, `1508: 200`). The captures are `b5b2linux/b2bootB-20261125T0540-{stock,prot}/uart.txt`
(`f36522b0…`, `4ac7b197…`, the board lane's tree). I re-ran the reader (`122f1974…`) on each: exit 0, and the output is
identical to the board lane's grade files.

**Readings.**

| boot | control | Perl makerand.pl | tshark arm | 169 inputs, nonzero | answer | span |
|---|---|---|---|---|---|---|
| B-stock | `rc=7` | perl-stock: rc 0, `acbf40c2…` (43,824 B), 276.4 s | u1 | 169, 0 | `b190069b…` (34,055,148 B) | 1,766.43 s |
| B-prot | `rc=7` | perl-p0002: rc 0, `acbf40c2…`, 272.2 s | p1 | 169, 0 | `b190069b…` | 1,811.13 s |

- tshark's stderr is 1,758,495 B in both arms. It is the same dissector-registration `OOPS` lines in both, starting
  `dissector table "arcnet.protocol_id" doesn't exist`. The twin's arms wrote 1,758,498 B each. The 3-byte
  difference was not read further: the answer, which is stdout, is equal.
- Heartbeats: 8 per arm, each every 20 captures.

**Against the predictions.**
1. Both controls read `rc=7`: **met.**
2. Both Perl arms answer `acbf40c2…` with rc 0: **met** (does-it-run only).
3. Both tshark arms answer `b190069b…` with 169 inputs and 0 nonzero: **met.**
4. Cost between −5 % and +15 %, point +2.3 %: **met.** Prot against stock is **+2.53 %** (1,811.13 s against
   1,766.43 s), cross-boot.

**What it says.**
- **Fig. 10's Wireshark revision runs on B5b-2 silicon, on the trap route, with stock's answer, and so does Perl 5.36.**
  With boot A, all five one-operand applications have a silicon row with stock's answers.
- **The protected tshark returned on B5b-2 in its one boot.** Its protected wmem path issues about 1.9·10^6
  allocations a run. At P1's stress-form estimate (4.9·10^-8 per allocation), a run returns with probability about
  0.91. So this return says little about M-16 on B5b-2; the stress boot S is the test for that.
- The +2.53 % lies at oneop's image-instruction point (+2.3 %). It comes from one boot per arm, across boots.

**Limits.**
- One boot per arm, so the cost carries the boot-to-boot spread, which is not measured here.
- The wall-clock spans include the kernel and the trap route.
- The Perl arms are does-it-run only.

### Result: boot S, M-16's stress form returns three times on B5b-2; no stop in 1.17·10^8 protected allocations, so the hazard there is below 2.6·10^-8 (95 %) (board 2, 2026-11-25 15:57-16:57, read against 9b7c370ba5c7)

**Run.** The board lane ran the payload (`2b19fc2a…`, 21,657,608 B) on board 2, `…-r75d-b5b2n_904cd9c6b.bit`, read back
by name before and after (rc 3: the name matches, no sha256 exposed), with no flash. The run ended on `TRAPTEST: done`
and board 2 was unbooked (`UNBOOK 1510: 200`). The capture is `b5b2linux/b2bootS-20261125T0723/uart.txt` (`796a8769…`,
the board lane's tree). I re-ran the reader (`d22d1262…`) on it against `twinS/boot.log`: exit 0, and the output is
identical to the board lane's grade file. There was no stop, so no aperture snapshot was taken.

**Readings.**

| command | args | verdict | STRESS lines equal to the twin's | last line | span (uptime) |
|---|---|---|---|---|---|
| p0 | `100 212 10 1` | RETURNED, rc 0 | 12 of 12 | `done packets=100 fast=19360 … sum=1fe6df7c8e2b0e1f` | 3.7 s |
| p1 | `300000 212 10000 1` | RETURNED, rc 0 | 32 of 32 | `done packets=300000 fast=58298524 … sum=f6ed64e0d51b9d04` | 1,432.4 s |
| p2 | `300000 212 10000 1` | RETURNED, rc 0 | 32 of 32 | the same line as p1 | 1,432.7 s |

- The control reads `TRAPTEST: control smoke.dom rc=7`. Both stages end `@@QSTEP-END … rc=0`, with no watchdog line.
  The console carries 27 of `/init`'s 120 s ticks.
- The protected commands completed 116,616,408 fast allocations, the number the pre-registration computed.
- Speed: 47.7 s per 10,000 packets on this route, against about 65 s on P1's native route. The twin took 369.5 s of
  guest time for p1.

**Against the predictions.**
1. The control reads `rc=7`, and p0 returns equal to the twin's lines: **met.**
2. **Under H1, p1 does not return: not met.** p1 and p2 both returned, with every heartbeat equal to the twin's. At P1's
   point estimate (4.9·10^-8), the probability of this was 0.0033.

**What it says.** This is the pre-registration's third outcome:
- **No stop in 1.17·10^8 allocations. The hazard on B5b-2 with this allocator revision is below 2.6·10^-8 at 95 %.**
  That is about half P1's point estimate, and inside P1's single-draw interval (1.2·10^-9 to 1.9·10^-7). It is a
  bound, not an absence.
- **The two boots do not yet show different rates.** P1 had one stop in about 2.0·10^7 allocations, and B5b-2 none
  in 1.17·10^8. If both boots shared one rate, the single stop would fall in P1's boot with probability
  2.0/(2.0 + 11.7) = 0.15 (one-sided, conditional on one stop in all). This test is computed after the boot; the
  pre-registration did not name it.
- It does not say whether the hardware or the allocator revision made the difference, as pre-registered. B5b-2 runs
  the one-operand wmem 0001 and the trap route; P1 ran the dev-era revision natively.

**Limits.**
- One boot.
- No unprotected arm on the board (the payload cap). On the twin (`twinS-both`), the unprotected u0 and u1 returned,
  and u1's last line carries the same counts and `sum=8b64d9b857f7aee5` as the protected commands' `pkt=100000`
  heartbeat. Its p1 took
  358.7 s against u1's 21.4 s for a third of the packets, so the protected path is 5.6× slower per packet there.
- No aperture reading, since nothing stopped.

## M-16's first two reductions on P1 (2026-11-25)

### Pre-registration: M-16's stress form with its first two reductions, one boot on board 1 (P1), protected only (written 2026-11-25 17:53, before the boot)

**Question.** Which parts of the stress form does P1's stop need? On P1 the full form stopped once, after about 2·10^7
allocations (result above, `832faa980e3b`). On B5b-2 it did not stop in 1.17·10^8 (boot S, `5d902b54a271`). So the
reductions run on P1, the bitstream where the stop has been seen. Runtime granted board 1 for this run in its own message.

**The program.** `tshark/stress/red/tsapp-stress-red.c` (`1f0825a6…`) is `tsapp-stress.c` with a mode argument. Its
argv is `mode packets K report seed`. One image carries all three modes, because the 26,000,000 B firmware cap fits one
such image per boot.
The program and its reader are committed beside M-16's other files, in
`tests/fpga-repros/M16-wmem-fast-alloc-stops-at-cderive/src/`. The other paths here are under `<hdd>/capstone/nested/`.
- **a:** `tsapp-stress.c`'s loop, unchanged. On the twin its 41 STRESS lines for `400000 212 10000 1` equal
  `tsapp-stress.c`'s, and its 100-packet lines equal too. This is the positive control.
- **f (reduction 1):** BLOCK_FAST alone. Every allocation goes to the packet scope, with the same LCG sizes, and there is
  no file scope.
- **s (reduction 2):** BLOCK_FAST alone, at one fixed size, 32 B.

**Build.** `tshark/stress/red/build-red.sh` (`bea9fe6a…`) is `build-stress.sh` with only the source and the output
directory changed.
- **Controls, both trees:** the relink of each tree's `tshark_m5.o` reproduces `tshark_m5.dom` byte for byte.
- Images:
  - protected `959f8509…`, strip `a15d30b2…` (20,208,504 B);
  - unprotected `eb418c7d…`, strip `813a099f…` (twin only).
- **The allocator is P1's.** `wmem_block_fast_sublet_alloc` is 0x4b2 bytes in both images. Its instruction bytes hash to
  `bee25b8c…` in this image and in P1's stress image `c93cc972…`. The function sits 0x7a higher here, at `0x77c60c`.
- **The site.** The function has two CDERIVE words: `0xa283955b` at +0x45c and `0xa205955b` at +0x462. Both are
  `capstone_cap_derive`'s inlined tail in `wmem_sublet_issue_at`:
  - +0x45c is the derive of the child: `scc t2` and then `cderive child, t2, size`;
  - +0x462 is the zero-size refusal branch.
  So every successful allocation executes +0x45c once, whatever its size. Mode s therefore runs P1's stop instruction
  once per allocation, as a and f do.
  - At P1's load base, +0x45c runs at **`0x3ff676ca68`** (P1's image: `0x77c9ee` → `0x3ff676c9ee`).

**Stage.** `tshark/stress/red/stage-board` (`make-stage-red.py`, `a65700ac…`), around bkit91's own smoke:
- `p1red/red1` (`mk-red-stage.sh` `f4ad092e…`; `red1.SHA256SUMS` `443aa317…`);
- its files differ from stress1's (`p1st/stress1`) in exactly three: the image, `plan.json` and `run.sh`.

Commands, in order, each in a fresh run directory, with K = 212 and seed 1:

| | mode | packets | fast allocations | heartbeat |
|---|---|---|---|---|
| r0 | s | 100 | 21,200 | every 10 |
| r1 | s | 600,000 | 127,200,000 | every 10,000 |
| r2 | f | 300,000 | 63,600,000 | every 10,000 |
| r3 | a | 300,000 | 58,298,524 | every 10,000 |

**Ordering.** One stop ends the boot. So the most-reduced form runs first, and the full form runs last as the control.
The first command that does not return is the reading.

**Board-free checks, done.**
- The recheck on bkit91 (`p1red/recheck.sh`, as `p1s100/recheck.sh` with the kit changed):
  - the real bake and both stubs ran, rc 0;
  - the real firmware is 23,760,392 B, under the cap, `e4be27b5…`.
- The chain's own dry run (`nat-run3p.sh`, `NR_DRY=1`) reached `DONE`. It prints `BAKED_TIMEOUT=10800`, as stress1's
  did.
- On this kit a stage with `spec.sh` sets the idle limit to the timeout. So a stop is recognised only when the
  10,800 s budget ends, and the runner then reads the wedge apertures (`BAKED_WEDGE_APERTURES=1`), as for stress1.

**Time.** On P1, mode a ran 10,000 packets in 60-75 s. The twin's per-packet time for s and f is 0.71 and 0.84 of a's.
At 7.5 ms a packet for a, the four commands take about 7,300 s, plus about 140 s to unpack the image each time. That
is under the 10,800 s budget, with about 2,000 s to spare after the smoke and the boot.

**Twin.** The twin (`nst-red-board2`, slot 28, 2^18 records) ran this stage (`serial.log` `02055366…`). All four commands return rc 0:
- r0 `done … sum=144131f731537378`;
- r1 62 lines, `done packets=600000 fast=127200000 … sum=53c790819c8ca080`;
- r2 32 lines, `sum=b1ad28789dad93bd`;
- r3 32 lines, `sum=f6ed64e0d51b9d04`, the line boot S's p1 printed on B5b-2.

On the earlier twin run (`nst-red-twin`), the unprotected image's s and f lines equal the protected ones.

**Reader.** `p1red/read-red.py` (`64782291…`), run as `read-red.py <board.txt> <twin serial.log>`:
- **RETURNED:** rc 0, with every STRESS line equal to the twin's;
- **NO RETURN:** the last heartbeat and the twin's next one, every aperture row verbatim, and a comparison of the commit
  pc with `0x3ff676ca68` and of sw=220 with `0x8c`;
- **OTHER:** a line differs, or rc is nonzero;
- **NOT STARTED:** the command is absent.

It was tested six ways:
- the twin's log as both inputs: 4 RETURNED, exit 0;
- a copy cut after r1's packet-60,000 line, with P1's aperture block and its commit pc set to `0x3ff676ca68`: NO
  RETURN, a stop between 12,720,000 and 14,840,000 allocations, pc True, sw=220 True;
- the same with P1's own pc: pc False;
- the cut with no aperture block: "NO APERTURE BLOCK";
- one checksum digit changed: OTHER, exit 1;
- a twin log with no complete command: exit 2.

On P1's real stress1 transcript against its twin, it reads s0 RETURNED and s1 NO RETURN between 19,432,859 and
21,375,862. Its pc is False, as it should be, since that is the other image's pc, and sw=220 is True. That is the
result recorded above, so the reader parses the board's real format.

**The hypothesis (H1).** The stop is a per-allocation hazard of the derive at +0x45c, at P1's estimate of 4.9·10^-8, in
all three modes.
- Under H1, r1 stops with probability 0.998. Its first stop falls at about 1.4·10^7 allocations (median), near packet
  67,000.
- At the lower end of P1's interval (1.2·10^-9), r1 returns with probability 0.86. So no return here shows absence.

**Predictions.**
1. The smoke passes, and r0 returns equal to the twin's lines.
2. **Under H1, r1 does not return.** Its last heartbeat lies before packet 600,000. The commit pc reads `0x3ff676ca68`,
   with sw=220 `0x8c`.

**What each outcome says.**
- **r1 stops at `0x3ff676ca68` with sw=220 `0x8c`.** BLOCK_FAST alone, at one fixed size, is enough. That removes the
  file scope, the BLOCK allocator, the free path, the size mix and the jumbo path.
  - Next is reduction 3, `capstone_cap_derive` on a fixed parent outside wmem. Then a directed `.S` for simulation.
  - No more board time goes to r2 or r3.
- **r1 returns and r2 stops there.** Removing the size mix removed the stop in this boot. r1's return bounds the
  fixed-size hazard below 2.4·10^-8 (95 %).
  - That is a bound, and it is about half P1's point estimate. So "the size mix is needed" is a hypothesis for the
    next boot, not a finding.
- **r1 and r2 return and r3 stops.** The file scope's interleaving is needed, with the same caveat. r2's return bounds f
  below 4.7·10^-8.
- **All four return.** Then the positive control did not stop in 5.8·10^7 allocations either. Under H1 that has
  probability 0.057. Together with boot S this would put the full form at no stop in 1.7·10^8 allocations since P1's one
  stop, so P1's rate is the question first, not the reductions.
- **A stop elsewhere, or with another signature:** reported as itself, and not merged with M-16.
- **A command cut by the budget** (the heartbeat still advancing at the budget's end): "no stop within T" for that
  command. The other commands' verdicts stand.

**Limits.**
- One boot.
- No unprotected arm on the board (the cap). On the twin, the protected and unprotected s and f lines are equal.
- The apertures are read once, after the budget.
- A return is a bound, not an absence.

### Result: red1, BLOCK_FAST alone returns twice and the full form stops in P1's first stop's window again; H1 refuted (board 1, 2026-11-25 22:34 to 2026-11-26 03:39, the stop near 01:35 on 11-26, read against 71eb7ac97286)

**Corrected 2026-11-26** (an audit of the first version, e47c05659fd5):
- the stop time was given as 23:37 on 11-25. That figure was the bake's start plus the watchdog's seconds, and the watchdog's clock starts at the upload (00:32).
- "20 of 21 aperture rows equal" overstated the evidence. 15 of those rows read the same on all seven M-16 stops, so the state evidence is the window plus the serving index.
- The boot is UNGATED (PREFLIGHT=0), as s1's was.

Detail, lines and apertures: `tests/fpga-repros/M16-wmem-fast-alloc-stops-at-cderive/` ("On P1 again",
`results/p1-board1-red1.txt`, `results/twin-red1.txt`).
- The smoke passed. P1 was read back by name, and the transcript sha256 is `9b032344068a31cf…`.
- **r0, r1 and r2 returned with rc 0**, every STRESS line equal to the twin's:
  - r1, s, 127,200,000 allocations in 1,583 s;
  - r2, f, 63,600,000 allocations in 918 s.
- **r3, a, did not return.** Its 11 lines equal the twin's, and the last is `pkt=100000 fast=19432859 … sum=8b64d9b857f7aee5`.
- The commit pc is this image's CDERIVE `0x3ff676ca68` (True), and sw=220 is `0x8c` (True).
- Predictions:
  - prediction 1 held;
  - **prediction 2 failed:** r1 returned.
- The outcome matched the registered branch "r1 and r2 return and r3 stops".
- **Beyond the registered branches:** r3 stopped after exactly the last line of P1's first stop (2026-11-25, s1 of
  `tsapp-stress.c`), in the same 10,000-packet window. The serving index `0x1381` is the same in both, and it differed
  on every other M-16 stop. Of the other aperture rows, 15 read the same on all seven M-16 stops, and two follow the
  stage's domain set. The rev-node rows are traced at `d83d3a633` in the folder. The commit pc differs, and in each
  boot it is that image's own CDERIVE.
- Both stress stops came about 360 s after their process started, so for mode a, position and elapsed time are
  confounded.
- **H1 is refuted.** Under H1, r1 returns with probability 0.0020 and r2 with 0.044. A stop of r3 in P1's window has
  probability 0.035, the maximum over every rate. Post hoc, "any window" gives 0.047 to 0.097. All of these were
  computed after the boot. The hypothesis this leaves (H2, at N = 2) is that on P1 the stop is tied
  to a point in the full form's sequence, between packets 100,000 and 110,000.
- r1 and r2 never exercise the file-scope BLOCK allocator or the free path (block = free = 0). So their returns say
  that BLOCK_FAST alone does not reach that state. They give no rate, and the registered "the file scope's
  interleaving is needed" stands in that sense.
- **Withdrawn (2026-11-26):** the constant-rate reading of the stress form. To be exact: a hazard **the same in every
  mode** is refuted, by r1 and r2 (0.0020, 0.044). A constant rate **for the full form alone** is disfavoured, not
  refuted (0.035; post hoc 0.05-0.1). The figure is withdrawn as a reading either way.
  - That covers P1's 4.9·10^-8 per allocation, and boot S's "hazard below 2.6·10^-8" read as a rate. Boot S reads
    instead: on B5b-2, with the one-operand revision, the same full-form sequence passed this point twice without
    stopping.
  - The Line B Wireshark pre-registration below (19:29) took its first two prediction rows from that model. Those
    rows are void. Its third row (P1's tshark 5 of 8) and its readings are unaffected.
- Next (the folder's "Next reductions"):
  - the twin locates the window at 100-packet resolution, which costs no board time;
  - on P1, ordered with each command expected to return before the one expected to stop:
    - a control cut before packet 100,000, then seed 1 with a fine heartbeat, last;
    - another seed in a boot of its own, since it may also stop.

### Pre-registration: M-16's window on P1, boots loc1 and loc2 on board 1, protected only (written 2026-11-26 06:46 host time (+08), 22:46Z on Nov 25, before either boot)

**Question.** Is P1's stop tied to a point in the full form's seed-1 sequence (H2, from red1 at N = 2), and where in
the window does it fall? Runtime granted board 1 for M-16's window localisation, to follow +Spatial Tier 1 there.

**The program and the image** are red1's, unchanged: `tsapp-stress-red.c`, protected, strip `a15d30b2…`
(20,208,504 B), packed as `stress_sublet.xz` `46396e7d…` in both kits. The stop instruction is this image's CDERIVE
in `wmem_block_fast_sublet_alloc`, which runs at `0x3ff676ca68`.

**Stages and kits.** Each stage is made by `make-stage-red.py` (`a65700ac…`) and wrapped in bkit91's own smoke, as red1
was. Paths are under `<hdd>/capstone/nested/`.
- loc1: `tshark/stress/red/stage-board-loc` → `p1red/loc1` (`mk-loc-stage.sh` `7aaa0010…`, `loc1.SHA256SUMS`
  `c79a1a76…`). Its files differ from red1's in exactly two: `train/plan.json` and `train/run.sh`.
- loc2: `tshark/stress/red/stage-board-loc2` → `p1red/loc2` (`mk-loc2-stage.sh` `f486bdb4…`, `loc2.SHA256SUMS`
  `02e75cc3…`). Its files differ from loc1's in the same two.

Commands, in order, each a new process with K = 212:

| boot | command | mode | packets | heartbeat | seed |
|---|---|---|---|---|---|
| loc1 | c0 | a | 99,000 | every 1,000 | 1 |
| loc1 | l1 | a | 300,000 | every 100 | 1 |
| loc2 | c0 | a | 99,000 | every 1,000 | 1 |
| loc2 | s2 | a | 300,000 | every 1,000 | 2 |

c0 is the control: the full form, cut before the window, so it must return. The command expected to stop runs last in
each boot. s2 may stop as well, so it has a boot of its own.

**Chain.** `p1s100/nat-run3p.sh` (`9c3525d4…`) on board 1 with:
- `p1red/p1loc.conf` (`36f1ca29…`): P1's bitstream `…-tagoverlap6c-pagedtbl_d83d3a633`, TAG `p1loc-m16loc`, kit bkit91;
- `p1red/p1loc.list` (`68c90c1d…`): loc1, then loc2, each 2,700 s with a 210-minute booking.

The chain flashes P1 if it is not resident, with the seal's sha256 checked, and reads the bitstream back by name. It
leaves the board on P1.
- The kit multiplies 2,700 s into `BAKED_TIMEOUT=10800`, which the dry run printed for both boots.
- On this kit the idle limit equals the timeout. So a stop is recognised only when the 10,800 s budget ends, and the
  wedge apertures are read once, after it.
- **Both boots will be UNGATED.** The kit's `board-run.sh:159` sets `PREFLIGHT=0`, as it did for red1 and s1. The
  checks that apply are the chain's: the kit's SHA256SUMS, the bitstream named in `board.env`, and the seal.

**Board-free checks, done.**
- `p1red/recheck.sh` (`65b83a46…`) on each kit:
  - the real bake, rc 0;
  - `BOARD_STUB=2` on a real P1 transcript that returns (`d76782aa…`, the one red1's recheck used) and on its cut
    that does not, rc 0 each;
  - the real firmware, 23,760,392 B, under the cap: loc1 `0a96ab58…`, loc2 `43019be3…`.
- The chain's own dry run (`NR_DRY=1`) reached `DONE` through both boots.

**Time.** On P1, mode a runs about 3.6 ms a packet, from red1's r2 (918 s for 300,000 packets) and the twin's f/a ratio
of 0.84. red1's pre-registration said 60-75 s per 10,000 packets, and that figure was wrong.
- So c0 takes about 360 s, and l1 or s2 about 1,090 s if it returns. Each command also unpacks the image, about 140 s.
- A boot in which everything returns takes about 1,800 s. A stop holds the boot to its 10,800 s budget.

**Twin references.** A command's lines are a function of its argv, and the heartbeat interval changes only which lines
print. So the lines that matter are already in hand:
- c0's last heartbeat is the locate twin's `pkt=99000 fast=19238362 block=1749638 free=876156 sum=c7bcb25f7f6a9209`
  (`nst-red-locate`, serial `cc185143…`; M-16's `results/twin-locate.txt`).
- l1's lines up to 110,000 are the locate twin's. Its end is red1's twin r3 line (`nst-red-board2`, serial
  `02055366…`): `done packets=300000 fast=58298524 … sum=f6ed64e0d51b9d04`.
- The reader's references are the twin runs of these exact stages, `nst-loc1-twin` and `nst-loc2-twin`, which are
  running.
- **Before the boots, an addendum is pushed** with:
  - their serial hashes;
  - a check that loc1's twin lines equal the lines above;
  - seed 2's counts at its packets 100,000 and 110,000 against seed 1's;
  - the reader's re-test.
- Board gets the arming data only once that addendum is on origin.

**Reader.** `p1red/read-red.py` (`64782291…`), red1's, run as `read-red.py <board.txt> <twin serial.log>`. It reads by
command number, so the names c0, l1 and s2 need no change. In l1 it brackets a stop between two 100-packet lines.

**Hypotheses.**
- **H2:** the stop is tied to a point in the seed-1 full form's sequence, between packets 100,000 and 110,000.
- **T:** the stop is tied to the time elapsed in the process, from about 364 s (red1's last line) to about 400 s (the
  twin's next line, at 3.6 ms a packet). r1 and r2 ran longer without stopping, so T is about mode a only.
- At seed 1 both put the stop in the same window. **loc1 cannot separate them.** Its finer heartbeat is not a planned change of
  time per packet, and its print cost on P1 is not measured.

**Predictions.**
1. The smoke passes in both boots, and c0 returns in both, equal to the twin's lines.
2. **loc1: l1 does not return.** Its last heartbeat lies between packets 100,000 and 109,900. The commit pc reads
   `0x3ff676ca68`, and sw=220 reads `0x8c`. The stop then lies within the 100 packets after that line.
3. **loc2: H2 predicts that s2 does not stop between its packets 100,000 and 110,000.** T predicts a stop at about the
   same elapsed time.

**What each outcome says.**
- **c0 does not return:** a stop before the window, near 360 s or earlier, and that boot's last command is lost.
  - That is against H2's window.
  - c0 ends at about 360 s, at the edge of T's window, so a stop near its end is consistent with T.
  - Its last heartbeat is reported, with the elapsed time by the watchdog's clock.
- **l1 stops in the window, with the signature:** H2 and T both hold, at N = 3. The window narrows to 100 packets, which
  the twin's lines bracket. Next is the boot that separates position from time (the folder's "Next reductions", step
  3), before any directed `.S`.
- **l1 stops before packet 100,000:** against H2. It is consistent with T only if its elapsed time, by the watchdog's
  clock, falls near 364-400 s.
- **l1 returns:** neither H2 nor T predicts the stop at that point. Three runs of one sequence, two stops in the same
  window and one return: the stop is not deterministic there, and the rate question reopens.
- **s2 returns:** against T, and consistent with a point specific to seed 1's sequence (H2).
- **s2 stops between its packets 100,000 and 110,000:** against a point specific to seed 1's sequence. It is UNDECIDED
  between T and a count trigger (packets or allocations), if the addendum shows seed 2's counts there close to seed
  1's.
- **s2 stops elsewhere:** against a count trigger set by seed 1's counts. It is against T unless its elapsed time
  falls near 364-400 s. That is consistent with a seed-dependent point in the sequence.
- **A stop elsewhere in the code, or with another signature:** reported as itself, and not merged with M-16.

**Limits.**
- Two boots, so one draw for each question.
- Both boots are ungated.
- The elapsed time comes from the watchdog's clock, which times console arrival, not the core.
- T is one form of a time trigger: a time counted from the process's start.

**Amendment 1 (2026-11-26 06:49 host time (+08), before either boot).** The twin references no longer gate the boots;
they gate the grading.
- Why: the loc twins have not started. They wait on the shared QEMU lock, and each needs about 10 h of run time.
  Board 1 is free in about 15 min (the board lane's word). The twins are deterministic and cannot see the board's
  result, so reading them after the boot biases nothing. The one place the earlier text left a judgement to the
  addendum is fixed here instead.
- **"Close" is fixed now:** seed 2's counts are close to seed 1's if, at each of packets 100,000 and 110,000, its
  `fast` and its `block` each lie within 2 % of seed 1's: fast 19,432,859 and 21,375,862, block 1,767,141 and
  1,944,138. Otherwise a stop of s2 in that window is read against a count trigger, not UNDECIDED.
- The addendum (serial hashes, the check of loc1's twin lines against the lines above, seed 2's counts, the reader's
  re-test) is pushed before either boot is graded. A boot graded without it is not read.
- The chain waits for the board lane's session marker on board 1 and starts only if that session's log ends
  `SESSION END`.

**Addendum: the twin references (2026-11-26 09:08 host time (+08), 01:08Z; loc1 still running on board 1, neither boot graded).**
- The serials, on twin runs of the exact stages:
  - `nst-loc1-twin` 21612e7b8c2aeb66… (264,118 B);
  - `nst-loc2-twin` 13bbfaa41ef25231… (48,618 B).
  - Both rc 0, and every command returns.
- **loc1's twin equals the lines above:**
  - c0's last heartbeat and its end are `pkt=99000 fast=19238362 block=1749638 free=876156 sum=c7bcb25f7f6a9209`.
  - l1's heartbeats on packets common to the locate twin (`nst-red-locate`): all 1,100 equal, packets 100 to 110,000.
  - l1's end is `done packets=300000 fast=58298524 block=5301476 free=2652582 sum=f6ed64e0d51b9d04`, red1's r3 line.
  - loc2's c0 is the same as loc1's.
- **Seed 2 against seed 1** (s2 against l1):

  | packet | fast, seed 2 / seed 1 | block, seed 2 / seed 1 |
  |---|---|---|
  | 100,000 | 19,434,904 / 19,432,859 (+0.011 %) | 1,765,096 / 1,767,141 (−0.116 %) |
  | 110,000 | 21,378,605 / 21,375,862 (+0.013 %) | 1,941,395 / 1,944,138 (−0.141 %) |

  - All four are within Amendment 1's 2 %, so seed 2's counts are **close**. A stop of s2 between its packets 100,000 and 110,000 is read as UNDECIDED between T and a count trigger, as registered.
  - A stop elsewhere is still read as registered.
  - s2's end is `done packets=300000 fast=58303004 block=5296996 free=2649579 sum=c2d00342ae2d27e5`.
- **The reader's re-test** (`read-red.py` `64782291…`, `<hdd>/nested/p1red/rtest-loc/`):
  - loc1's twin against itself: exit 0, c0 and l1 RETURNED.
  - loc2's twin against itself: exit 0, c0 and s2 RETURNED.
  - loc1's twin cut after l1's `pkt=100000` line, followed by the aperture block of red1's matching test cut (`cut-l1-100k.txt` `bb32d87a…`): exit 0.
    - c0 RETURNED. l1 NO RETURN, with the stop bracketed between 19,432,859 and 19,452,273 fast allocations (packets 100,000 and 100,100).
    - The commit pc equals the CDERIVE, and sw=220 equals P1's 0x8c.
  - loc2's twin read against loc1's: exit 1, s2 OTHER at its first line (seed 2). So the comparison fires.

### Result: loc1 and loc2. c0 returns in both, then l1 and s2 each stop early, about 34,000-36,300 packets into their run (board 1, P1, 2026-11-26 07:10-13:26 host time, read against 553f00e2239f as amended, with addendum 4b606bdca102)

**Run.**
- Board 1, on P1 (`…-tagoverlap6c-pagedtbl_d83d3a633`). The chain's own bitstream check passed, and the board was
  released (UNBOOK 200).
- loc1 ran 07:10-10:18 and loc2 10:18-13:26. Each held its 10,800 s budget.
- **Both boots are UNGATED** (the kit's `PREFLIGHT=0`), as registered.
- Captures: loc1 `board.txt` `83afe3ce…`, loc2 `board.txt` `a4a94787…`. The result lines and aperture blocks are in M-16's folder
  (`results/p1-board1-loc1.txt`, `results/p1-board1-loc2.txt`).

**The registered reader** (`read-red.py` `64782291…`, against `nst-loc1-twin` and `nst-loc2-twin`): **exit 1 on both
boots**.
- c0 is RETURNED in both.
- l1 and s2 are each OTHER, "first STRESS line differing from the twin".
- **What the OTHER is:**
  - Every complete STRESS line of l1 (363) and of s2 (34) equals the twin's.
  - The one differing line is the command's last. It is cut mid-line, and it is a strict prefix of the twin's:
    `pkt=36300 fast=7053623 bl` and `pkt=34000 fast=6608300 block=599700 free=`.
  - Neither command has a `@@SPEC-END`.
  - So both have the NO RETURN shape. The reader has no partial-line case (`read-red.py:25`, `:56-59`), and its re-test
    cut (`cut-l1-100k.txt`) ended on a whole line, so the re-test never exercised one.
  - The NO RETURN reading below is by hand, with the facts above. The registered exit stays 1.
- **A core stop, not a lost console** (checked by an auditor, then here):
  - the console socket kept delivering lease events for about 10,000 s after the last UART chunk;
  - the switches stayed at 0 throughout;
  - a returning P1 boot's final apertures read differently: tsu-r3 reads sw=220 `0x40` and a kernel pc, where these read
    `0x8c` and a wmem allocator pc.

**Readings.**

| | loc1 | loc2 |
|---|---|---|
| smoke | PASS, 9 steps | PASS, 9 steps |
| c0 (seed 1, cut at 99,000) | RETURNED, 101 lines equal to the twin's, 369.35 s | RETURNED, 101 lines, 369.29 s |
| stopping command | l1: seed 1, heartbeat every 100 packets | s2: seed 2, heartbeat every 1,000 |
| last complete line | `pkt=36200 fast=7034256 block=640144 free=320568` | `pkt=33000 fast=6414056 block=581944 free=291055` |
| cut line | `pkt=36300 fast=7053623 bl` | `pkt=34000 fast=6608300 block=599700 free=` |
| elapsed at the stop, by the watchdog's clock | 135-165 s | 120-150 s |
| commit pc | `0x3ff676acfe` | `0x3ff676ca68` (the registered CDERIVE) |
| sw=220 | `0x8c` | `0x8c` |
| sw=225 | `0x92` | `0x96` |
| trap-log mepc / tval | `0xffffffff802100a2` / `0x09093783` | `0xffffffff80007376` / `0x73` |
| rev-node serving index | `0x10e6` | `0x1147` |

- **Elapsed time.**
  - The watchdog counts console chunks in 15 s bins, and `run.sh` unpacks the image before each `@@SPEC-BEGIN`.
  - c0's chunks span the bins that bracket its 369.3 s, which is the positive control.
  - l1 ran from the 1,006-1,021 s bin to the 1,156-1,171 s bin, and s2 from 990-1,005 s to 1,125-1,140 s.
  - At c0's 3.73 ms a packet, the packet counts give about 135 s and 127 s.
- **l1's commit pc.**
  - It maps, with the bias that places this image's registered CDERIVE at `0x3ff676ca68` (vaddr + `0x3ff5ff0000`), to
    vaddr `0x77acfe` = `wmem_block_sublet_alloc`+0x452.
  - That is the BLOCK allocator, not BLOCK_FAST.
  - Its bytes there (`add t0,t0,a2; scc t2,a1,t0; 5b 95 83 a2; j`) are identical to the registered site's at
    `wmem_block_fast_sublet_alloc`+0x45c. So it is the same inlined derive in the sibling function.
  - The commit pc is slot-0 residue, so it names a site, not the faulting instruction.
- **l1's trap-log row also differs.** Every earlier M-16 stop with a block (stress1, tsu-r1, tsu-r2, red1), and loc2,
  reads mepc `0x…80007376` / tval `0x73`. loc1 reads `0x…802100a2` / `0x09093783`.
  - The decode is UNRESOLVED. An auditor resolved it against loc2's kernel build to `ld a5,0x90(s2)` in
    `cv_step_locked` and assumed the text did not move between the two builds.
  - That row is the last M-mode trap, which a returning boot reads too, not a stop marker.
- **Both cuts fall on the UART's 16-byte boundaries.**
  - loc1's last chunk is the 32 bytes `STRESS pkt=36300 fast=7053623 bl`.
  - loc2's last three are 16 bytes each: `STRESS pkt=34000`, ` fast=6608300 bl` and `ock=599700 free=`.
  - So the UART sent what the kernel had loaded, and the next refill never came. The program's stdout is line-buffered
    (`_IOLBF`).
  - Every earlier stress stop (heartbeat every 10,000 packets) ended on a whole line.
  - This is an observation, not a mechanism.

**Against the predictions.**
1. **Met.** Smoke passes in both boots, and c0 returns in both, equal to the twin.
2. **loc1, missed except for "does not return".** l1 does not return, but:
   - its last line is at 36,200 packets, outside 100,000-109,900;
   - its commit pc is not `0x3ff676ca68`;
   - sw=220 reads `0x8c`, as predicted.
3. **loc2.** s2 does not stop between its packets 100,000 and 110,000, which is what H2 predicted. It stops at about
   34,000 packets, at 120-150 s, far from T's about 364-400 s.

**What each outcome says, read as registered.**
- **c0 returns:** no stop before 99,000 packets in either seed-1 control.
- **l1: two registered rules apply, and the text gives no order between them.**
  - "l1 stops before packet 100,000: against H2. It is consistent with T only if its elapsed time falls near 364-400 s",
    and it does not.
  - "A stop elsewhere in the code, or with another signature: reported as itself, and not merged with M-16." Its commit
    pc and its trap-log row both differ from M-16's.
  - Read plainly, "not merged" makes l1 no M-16 draw. Then **loc1 leaves H2 and T untested** rather than refuted: l1
    stopped before it reached the window, at another site.
  - Reported as itself: a stop of the full form at seed 1, about 36,300 packets in, with the BLOCK allocator's derive
    as the slot-0 site and sw=220 `0x8c`.
- **s2: "stops elsewhere".** That is against a count trigger set by seed 1's counts: s2 stopped at 6.6·10^6 fast
  allocations, where seed 1's window starts at 19.4·10^6.
  - It is against T: 120-150 s, not 364-400 s.
  - As registered, it is "consistent with a seed-dependent point in the sequence". **That attribution is not earned
    here.** s2 changed two things from red1's r3: the seed, and the heartbeat (every 1,000 packets, against r3's
    10,000).
  - l1 kept seed 1 with a finer heartbeat still (every 100) and also stopped early.
  - A stop that depends on the print pattern survives as an alternative, and so does one tied to console or interrupt
    activity, given the mid-line cuts.

**So, after loc1 and loc2.**
- **On the record now:**
  - every full-form run on P1 with a heartbeat every 10,000 packets stopped between packets 100,000 and 110,000
    (s1, red1);
  - both runs with a finer heartbeat stopped at 34,000-36,300 packets and 120-165 s, mid-line.
- T as registered, a fixed time from process start near 364-400 s, is against s2. Neither boot tests H2 at seed 1.
- **The next boot** (not pre-registered here) would hold seed and sequence fixed and change only the heartbeat: seed 1,
  every 10,000 packets, against seed 1, every 100. That tests whether printing moves the stop. Step 3's spinner
  (`Next reductions`) is the other half.

**Limits.**
- Two boots, one draw each, both ungated.
- The elapsed times come from the console's arrival, binned at 15 s.
- The commit pc names a site only.
- The trap-log decode is unresolved.

### Pre-registration: M-16's heartbeat pair on P1, boots hbA and hbB on board 2: does printing move the stop? (written 2026-11-26 15:46 host time (+08), before either boot)

**Question.**
- On P1, every full-form run with a heartbeat every 10,000 packets stopped between packets 100,000 and 110,000: s1 and
  red1's r3.
- Both runs with a finer heartbeat stopped at 34,000-36,300 packets, mid-line on a UART load: loc1's l1 (every 100) and
  loc2's s2 (every 1,000, seed 2). See "Result: loc1 and loc2".
- Does the heartbeat alone move the stop, with the seed and the sequence held fixed?
- runtime granted board 2 for this, after the four Full + ports boots, with a P1 flash.

**The two boots.** Each command is a new process, K = 212, seed 1, protected image only (`a15d30b2…`, red1's).

| boot | kit (bkit91 stage) | command 0 | command 1, last |
|---|---|---|---|
| hbA | `p1red/hbA` (`hbA.SHA256SUMS` `fb21b798…`) | c0: mode a, 99,000 packets, heartbeat every 1,000 | h10k: mode a, 300,000, every 10,000 |
| hbB | `p1red/hbB` (`hbB.SHA256SUMS` `c79a1a76…`) | c0, the same | h100: mode a, 300,000, every 100 |

- **hbB is loc1's kit byte for byte** (`hbB.SHA256SUMS` equals `loc1.SHA256SUMS`), so hbB is a second draw of loc1.
- hbA's files differ from loc1's in exactly `train/plan.json` and `train/run.sh`. Its stage is `make-stage-red.py`
  (`a65700ac…`) with `--cmd c0:sublet:a:99000:212:1000:1 --cmd h10k:sublet:a:300000:212:10000:1`, built by
  `mk-hbA-stage.sh`.
- h10k's arguments are red1's r3's (mode a, 300,000, K 212, report 10,000, seed 1). In red1, r3 ran after r0-r2 in
  the same boot. Here it runs after c0 alone.
- Two boots, because both command-1 arms may stop. By the ordering rule, a boot holds at most one arm that may wedge,
  and it runs last.

**Chain.**
- `p1s100/nat-run3p.sh` (`9c3525d4…`) on board 2 with `p1red/p1hb.conf` (loc's conf with TAG `p1hb-m16hb` and the list
  changed) and `p1red/p1hb.list` (hbA, then hbB, each 2,700 s with a 210-minute booking).
- It flashes P1 (`…-tagoverlap6c-pagedtbl_d83d3a633`), with the seal's sha256 checked, and reads the bitstream back by
  name. It leaves the board on P1. The board lane then puts b42c8c644 back on board 2.
- The kit multiplies 2,700 s into `BAKED_TIMEOUT=10800`, and a stop holds the full budget.
- **Both boots will be UNGATED** (the kit's `PREFLIGHT=0`), as loc1 and loc2 were.
- Board-free checks (`p1red/recheck.sh`, run for real on each kit):
  - hbA: bake rc 0, stubbed returning and non-returning runs rc 0, firmware rc 0, 23,760,392 B, `664b9d1ef18d54b3`;
  - hbB: the same, firmware `d11fee562f7b1e29`;
  - both: the dom_stack gate PASS (25,872 B of 32,768 B), monitor `22a907f`.
- Stubbed dry run of the chain (`p1red/hb-dry/`, board calls stubbed): it reached `DONE board 2`, and the runner
  read `BAKED_TIMEOUT=10800` for both boots.

**Twin references.**
- hbA: `nst-hbA-twin`, the twin run of this exact stage (`nst-run.sh` slot 28, the natio images, record size 262,144).
  - It is queued on the QEMU lock behind another lane's run at this writing. Its serial's sha256 goes in an addendum
    before either boot is graded.
  - It must reproduce both references, by `p1red/check-hbA-twin.py` (`dae47152…`): c0's STRESS lines and rc equal
    `nst-loc1-twin`'s c0, and h10k's equal `nst-red-board2`'s r3 (`02055366…`). Exit 0 means both are equal, 1 means
    a difference, and 2 means a command is missing.
  - Controls: a log spliced from those two references exits 0. The same with one STRESS line changed exits 1 and names
    the line. loc1's twin standing in for hbA exits 2, with h10k missing.
  - If it does not exit 0, the twin's difference is reported first, and hbA is not graded against that twin.
- hbB: `nst-loc1-twin` (`21612e7b…`), the twin run of this exact stage.

**Addendum 1 (2026-11-26 16:30 host time (+08), during hbA's boot, before either boot is graded): hbA's twin reference.**
- `nst-hbA-twin` ended at 16:30:06, rc 0, `specdiff 0/0`. Its serial log's sha256 is `d12cb8b8c606c94f…`.
- `check-hbA-twin.py` exits 0. c0 is EQUAL to `nst-loc1-twin`'s c0 (101 STRESS lines, rc 0), and h10k is EQUAL to
  `nst-red-board2`'s r3 (32 lines, rc 0, `done packets=300000`).
- `read-red2.py`, the twin against itself: c0 and h10k RETURNED, exit 0.
- It is the binding hbA reference, as registered.
- **Disclosed:** the run was launched without `OUTROOT`, so `nst-run.sh` wrote it to its default root, `perf/faultesc`
  (BEGIN/END lines in that root's `RESULTS`). Its directory was then moved, unchanged, beside the other twins in
  `<hdd>/nested/out/`. The hash above is the moved file's.

**Reader.**
- `p1red/read-red2.py` (`3d2b9c35…`), run as `read-red2.py <board.txt> <twin serial.log>`.
- It is read-red.py (`64782291…`) with one change. A command with no `@@SPEC-END` whose last STRESS line is a strict
  prefix of the twin's line at that index is read as NO RETURN on its complete lines, and the cut text is printed. It
  also names the BLOCK allocator's derive (`0x3ff676acfe`) when the commit pc reads it.
- Re-test (`p1red/rtest-loc/r2-*.out`):
  - loc1's and loc2's board captures: c0 RETURNED, and l1 and s2 NO RETURN with their cuts printed, exit 0. read-red.py
    read OTHER here.
  - Both twins against themselves: RETURNED, exit 0.
  - The 100k cut control: NO RETURN at packet 100,000, exit 0.
  - loc2's twin against loc1's: OTHER, exit 1.
  - loc1's capture with its cut line changed so that it is no longer a prefix (`fast=7053624 bl`): OTHER, exit 1.

**Hypotheses.**
- **HW (window):** the full form at seed 1 stops at a point between packets 100,000 and 110,000, whatever the heartbeat.
- **HP (print):** the stop depends on the heartbeat, through the printing itself or the console and interrupt activity
  it brings. A finer heartbeat stops it early.
- At a heartbeat of 10,000 both predict a stop in the window. At 100, HW predicts the window and HP an early stop.

**Predictions.**
1. The smoke passes in both boots, and c0 returns in both, equal to the twin.
2. **hbA, h10k does not return.** Its last complete line is at packet 100,000, with the stop before the twin's 110,000.
   The commit pc reads `0x3ff676ca68` and sw=220 `0x8c` (HW and HP alike).
3. **hbB, h100:**
   - HP predicts a stop before packet 100,000, as loc1's at 36,300;
   - HW predicts a stop between 100,000 and 109,900.

**What each outcome says.**
- **c0 does not return:** a stop before 99,000 at a heartbeat of 1,000. That boot's command 1 is lost, and the stop is
  reported with its last line and elapsed time.
- **hbA:**
  - **h10k stops in the window with M-16's signature:** the window holds at a heartbeat of 10,000 for the third time,
    and the first with only c0 before it in the boot. It cannot separate HW from HP alone.
  - **h10k stops before packet 100,000:** against HW. The window at 10,000 was not a fixed point.
  - **h10k returns:** the window is not deterministic at 10,000, against HW, and the stop's rate reopens.
- **hbB:**
  - **h100 stops before 100,000:** against HW at this heartbeat. A second early stop at a heartbeat of 100 (loc1 was the
    first).
    - Its signature is read as for loc1: M-16's commit pc, or the BLOCK allocator's derive (then reported as itself), or
      another.
    - Its packet count against l1's 36,300 says whether the early stop sits at a point too.
  - **h100 stops in 100,000-110,000:** loc1's early stop did not recur. That is consistent with HW, and against HP.
  - **h100 returns:** neither predicts it.
- **Together:**
  - hbA in the window and hbB early: **the heartbeat moves the stop** (HP), at N = 1 for each arm. The window is then a
    property of the 10,000 heartbeat, not of the sequence alone.
  - Both in the window: HW, and loc1's early stop does not recur.
  - Any other pair is reported as itself.
- **Elapsed time** at a stop is read from the watchdog's 15 s bins, as for loc1 and loc2. It is not a hypothesis test
  here.
- **A stop elsewhere in the code, or with another signature,** is reported as itself and not merged with M-16, as before.

**Limits.**
- One draw per arm, and both boots are ungated.
- The heartbeat changes both the print volume and the console's interrupt load. This pair cannot tell which of those
  matters.
- The commit pc names a site only.

## M-16's return rate on Line B: the protected tshark in repeated B5b-2 boots (2026-11-25)

### Pre-registration: Fig. 10's protected Wireshark workload in five boots on a Line B board, queued (written 2026-11-25 19:29, before any of the boots)

**Status.** Queued, not granted. The runtime lane asked for this run, to go after the combined seal's kits. No board or
time is granted for it yet, and this section claims none.

**Question.** How often does the protected tshark return on Line B? On P1 it returned in 5 of 8 boots (tsu-p1, tsu-p2,
the heartbeat boot and tsu-r1-r5; 95 % interval 0.245-0.915), and every non-return held the core at the CDERIVE of
`wmem_block_fast_sublet_alloc`. On B5b-2 it has run once (boot B-prot) and returned. The stress form returned three
times there (boot S), which bounds the per-allocation hazard below 2.6·10^-8.

**Configuration.**
- The bitstream is Line B's resident one, read back by name before and after each boot.
  - If it is `…-r75d-b5b2n_904cd9c6b`, the payload is boot B-prot's, unchanged: `boardB-prot/out/fw_payload.bin`,
    sha256 `2effea0096ee964f4239300ce66eeea707920acaf57d1e70d33bcc867fa4d9fa`. It holds Perl p0002 on makerand.pl, then
    tshark prot (`4e01172f…`) on Fig. 10's 169 captures, with the default DTB.
  - **If the resident is a later Line B bitstream, this section does not cover it as written.** The images must
    first be re-scanned against that bitstream's illegal set, and an amendment must name the bitstream and the scan,
    before the first boot.
- Five boots, B-prot-r1 to B-prot-r5, one after another, read with `read-boardB.py` (`122f1974…`). Each boot is
  bounded at 2 h, as B-prot was, and `/init` bounds each stage at 3,600 s.
- On a non-return, before any reset or power-off, the board lane reads the aperture snapshot of boot S's request.
  Its labels are read with the board lane's driven/constant table for 904cd9c6b. On B5b-2, sw=195, sw=229 and the
  rev-node rows are no reading.

**Exposure per boot.** In oneop, Fig. 10's workload issued 2,074,705 more CDERIVEs on prot than on stock (image zone,
the oneop result above). Those are wmem's derives, BLOCK_FAST and BLOCK together. So about 2.07·10^6 is an upper bound
on the boot's BLOCK_FAST allocations. P1's stage was a different workload, one tshark over 9,000 frames. Its BLOCK_FAST count is about
1.9·10^6 a run, by the twin's zones (the figure `tsapp-stress.c`'s header records). The two exposures agree within about 10 %, so the per-boot rates
are compared directly, with that caveat.

**Predictions** (per-boot return probability p; Fig. 10's workload issues at most 2.07·10^6 allocations):

| hypothesis | p per boot | P(5 of 5) |
|---|---:|---:|
| boot S's bound, below 2.6·10^-8 per allocation | at least 0.947 | at least 0.76 |
| P1's stress-form estimate, 4.9·10^-8 | 0.903 | 0.60 |
| P1's tshark rate, 5 of 8 | 0.625 | 0.095 |

**Amendment (2026-11-26, 06:47, before any of these boots: none is granted, and the board lane's output tree holds no B-prot-r boot).** The first two rows came
from a constant per-allocation rate of the stress form, which red1 withdrew (the result above). They are void, and no
row replaces them: on P1 the stress form's stop looks tied to one point in its own sequence (H2, N = 2), so it gives no
per-boot rate for tshark.
- The prediction is the third row alone: at P1's tshark rate, p = 0.625 per boot and P(5 of 5) = 0.095.
- On B5b-2 the stress form passed P1's stop point twice (boot S). That is a fact about that sequence, not a rate, and it
  predicts nothing here.
- "What each outcome says" stands, read against the third row only.

**What each outcome says.**
- **5 of 5 return:** together with B-prot, that is 6 of 6, with a Clopper-Pearson 95 % interval of 0.54-1.00 on p.
  P1's 0.625 lies inside it, so the rates are not shown to differ. Line B's p would be read as above about 0.5, not
  as 1.
- **A non-return:** reported with its heartbeat, its dmesg and its snapshot.
  - It is M-16 on Line B only if the commit pc reads the image's CDERIVE in `wmem_block_fast_sublet_alloc` and
    sw=220 reads `0x8c`. Anything else is reported as itself.
  - That CDERIVE is computed from `4e01172f…`, as boot S's was. It is word `0xa283955b` at offset +0x4f0, vaddr
    `0x7e2cd0`; the refusal branch is at +0x4f8. The loader places vaddr 0x10000 at `0x3ff6000000`, so it runs at
    **`0x3ff67d2cd0`**.
- **A control or Perl arm failing:** that boot is void and is not counted.

**Limits.**
- Five boots.
- The workload differs from P1's.
- The exposure is an upper bound from oneop, not this route's count.
- The apertures on B5b-2 are partly constant.

## Note: B5T-g4's perl cause-28 is perl's own write past argv[0] (2026-11-25, CPU only)

**The event.** B5T-g4 ran on board 3 (`904cd9c6b`). Inside SPEC's test.pl harness, one `.perl-native` child faulted
(`b5b2linux/b3chain2-20261125T0716-B5T-g4/uart.txt`, the board lane's tree). The child is a copy of
`perlbench_r_base` `38ce666f…`, and `perf/cclever/arms/B5T/exes/perlbench_r.full` (`67a24421…`) strips to it. The
fault line:
`sublet: fault cause=28 pc=0x3ff7c01480 address=0x3ff7bfff78 pid=246 ra=0x3ff7c342a6 code=0x3ff7c00000-0x3ff7e50000`.

**Where it stops.**
- **pc** vaddr 0x11480 is `memset`+0x64, the third `sd` of the 32-byte loop. The stores at a3 = `…ff68` and +8
  passed, and +0x10 = `…ff78` faulted. So the capability ends in [`…ff78`, `…ff7f`].
- **ra** vaddr 0x442a6 is in `Perl_magic_set`, at its `jal memset` with a1 = 0x20 and a2 = PL_origalen − len − 1.
  That is SPEC's `mg.c:3170`, `case '0'`:
  `memset(PL_origargv[0] + len + 1, ' ', PL_origalen - len - 1)`.
- **Why the padding goes out of bounds.**
  - `perl.c:1516-1602` sets PL_origalen to the whole contiguous argv[] + environ[] string area.
  - On this route, argv[0]'s capability covers only its own string. That string is `/tmp/nt/sr/.perl-native`,
    24 bytes. The store offsets put its start in (`…ff5b`, `…ff63`], so its end lands at the bound.
  - The address sits 0x88 below the code base because the initial stack's string area lies just under the image.
- **Neither deep recursion nor a small stack is involved.**

**Which test: `t/op/leaky-magic.t`**, by two independent lines of evidence.
- **Ordinal.** pid 246 is the 130th `.perl-native` of test.pl's command in the capture.
  - The stage's test.pl command was replayed on the host. It ran under user-mode qemu-riscv64 8.2.2, built from the
    release tarball (`847346c1…`), with the stage's own `natperl` (`5737866a…`) as the harness and as every child.
  - `$^X` is a logging wrapper that re-enters through `qemu -0`, so grandchildren are logged too.
  - Invocation 130 is `-I.. -MTestInit op/leaky-magic.t`.
- **Mechanism.** In the stage's `in/` tree, a python search finds one write to `$0` that the harness runs:
  leaky-magic.t lines 71-75 do `local $$_ = 'thew'` for each name in `qw< : ? ! - | ^ ~ = % . \ / ; 0 >`.
  - The search's other hits do not write `$0` when the harness runs:
    - `TestInit.pm:30`, `$0 =~ s/\.dp$//`, matches nothing here, and a failed `s///` sets no magic;
    - `TestInit.pm:65` runs only from the top level, which the harness is not;
    - `t/japh/abigail.t` runs only under torture.
  - For the name `0`, that is a symbolic `local $0 = "thew"`.
  - The search's patterns were checked against this line and against six forms of a direct write.
- **Unexplained:** the replay made 317 invocations in all, while the board had made 372 when `/init` killed the stage
  at 3,600 s. So after some point the board spawned children the native run did not. The ordinal match rests on the
  two sequences agreeing up to 130. The mechanism evidence does not.

**Native and twin.**
- **Native:** in the replay, every test passes ("All tests successful", 208 s), op/leaky-magic.t included, and
  `local $0 = "thew"` runs.
- **Twin (B5T, the trap kernel):** the stage `F1-perl-dollar0` ran B5T's image against natperl, with the board's default
  boot arguments (no wide grants). It ran on `-e 'local $0="thew"'` and on leaky-magic.t, from the board's path
  `/tmp/nt/sr/.perl-native`, on Image `749fed26…`. The run finished at 2026-11-25 22:39:26 (its lock.log).
  - B5T faulted in both:
    `sublet: fault cause=28 pc=0x3ff7c01480 address=0x3ff7bfff88 … ra=0x3ff7c342a6 code=0x3ff7c00000-0x3ff7e50000`,
    rc 139.
  - natperl returned rc 0 on both.
  - So the twin reads the board's fault at the same memset store (vaddr 0x11480), from the same `jal memset`
    (vaddr 0x442a6), in the same code window.
  - The address is 0x10 higher than the board's `…ff78`. argv[0]'s string sits at a position set by the argv and
    environ strings above it, and those differ between SPEC's harness child and this stage. That fits the mechanism,
    but the twin's argv and environ were not read, so the 0x10 is not traced.

**Reading (the runtime lane's ruling).**
- This is a spatial check catching perl's deliberate write past argv[0]'s string. C does not give argv[0] the
  argv + environ block, so it is the protection working as designed, not a defect of the route.
- Nothing is fixed. A perl port patch would change the workload. Widening argv[0]'s capability over the block is a
  loader semantics change, and it would weaken exactly the check that fired.

**Reach.** Does any perlbench train input write `$0`? None does. The python search covered all 941 files of
`data/train/input` and `data/all/input`: diffmail, perfect, scrabbl, splitmail and suns, with `lib/`.
- It found no direct write, list write, symbolic-name write, `local` of `$0` or `PROGRAM_NAME`, and each of those
  patterns was checked against a positive control.
- Its hits are comments, a `Config.pm` value and mail text in `lib/spam.pl`.
- The symbolic dereference hits are reference variables (`$$data`, `$$abr`, …), not the name `0`.
- So the fault cannot reach a measured train row. test.pl is not measured.

**Files** (`<hdd>/capstone/nested/b5b2/perlfault/`): `run-native.sh`, `children.log` (the 317 invocations),
`sr/test.out`, and the twin overlay `ovl/`.

## B5b-2's bug corpora, memcached and CPython on the trap-route twin, and the cause-24 rule (2026-11-25)

These are the twin readings that become the board's predictions for runtime's B5b-2 items 1-3, plus a judging rule
for the corpus boots. That rule is pre-registered here, written 21:42, before any board run of the corpus. No board
has run any of this.

**The run.** `twin-all`, one boot on the trap-route QEMU twin:
- QEMU: qemu-crv `ea5c05f3…`, `-m 512M`, one hart, no exact bounds, the silicon's setting.
- Firmware: fw_jump `67115ce6…`.
- Kernel: trapk Image `4fc06dde…` with `sublet.dom_wide_grants=1`, which is what `caplifive-b5b2-wide.dtb`
  (`3d223e80…`) gives the board.
- Root filesystem: base-rootfs `dd2dd8ff…`, with compiler's /init.
- Lock: granted 20:29, done 21:19. Console log sha256 `08c735e24a4907bc…`.
- Stages: C30/C31 CPython, M20/M21 memcached, N1/N2 SQLite engine-repros, P10-P17 PHP engine-repros. Every stage
  reached `@@QSTEP-END rc=0`.

**CPython** (stock `e47ab197…`, prot `a269a6b5…`; `pyperformance-subset.py`, `PYTHONHASHSEED=0`):
- both arms rc 0 and `PYPERFORMANCE-OK 8`;
- both answers are `58211793…`, which equals the native reference.

**SQLite engine-repros** (`read-sqlite.py`, trap/eval's observe and judge; image manifests: malloc `fc4624193c28ccf5…`,
pools `0222760f5d5779a6…`, 33 images each):
- malloc: MISSED 29, NO-READING 4. All 33 cells equal case.json's virtual-malloc verdict.
- pools: CAUGHT 27, MISSED 6. 31 of 33 cells equal the record. The other two are 29 (`sqlite3Get4byte`) and
  31 (`statDecodePage`): cause 28 CAUGHT in the record, cause 24 here in the same function, MISSED under the
  arm's `[25, 28]`.
- Prior art: this is the pair recorded above on 2026-11-23 (capstone-qemu 0542c5556c): with exact bounds, 29 and 31
  are cause 28 at the same pc and address; without, cause 24. It is the moved-capability rule R-60 (qemu-crv
  `op_helper.c` `capstone_set_moved_cap`): a CINCOFFSET to a cursor the compressed form cannot represent writes an
  untagged integer, and the load through it traps 24. kit-p1v, the P1 twin, read the same 6 pools MISSED.

**PHP engine-repros** (`read-php.py` and trap/eval's run-realphp.py `--from-record`; image `a26d572f…`, probes
putenv `fd2e1555…` and nested `36cfcece…`):

| arm | asan-blind CAUGHT / MISSED | temporal-asan-visible CAUGHT / MISSED |
|---|---|---|
| stock | 5 / 30 | 1 / 5 |
| block | 12 / 22 | 6 / 0 |
| object | 25 / 11 | 6 / 0 |
| exact | 29 / 5 | 6 / 0 |

Against kit-p1v's committed bundles (`results/2026-11-23-p1v-twin`), 271 of 280 cells are the same. The 9 others are
four cases:
- **25 (crash-084)**: out-of-denominator on kit-p1v, MISSED on four arms here.
  - The putenv probe printed the ABSENT pattern exactly.
  - But the judge counts a probe reading only from a transcript that names its image's digest
    (`transcript_image`).
  - trap-vexec.sh printed no `capstone-exec: exit image=… sha256=…` line; the log has none.
  - This is a gap in the trap route's tooling (fix below).
- **03 (crash-008), exact arm**: unattributed here, CAUGHT by chain on kit-p1v. The trap route's fault line has no
  frame chain (`chain: []`), so the climb from `__stpcpy` to `php_date` cannot be made. Recorded as
  "unattributed: the trap route's fault line has no chain".
- **21 (crash-075)**: CAUGHT on pools and exact here; kit-p1v read infra (cause-30 node-table exhaustion). Recorded
  as infra/timing.
- **30 (crash-092)**: infra on the nested arms (the per-run watchdog), as in run18. A timing case.

**memcached: not qualified, a route gap.**
- 0 of 6 repetitions qualify, stock and protected alike.
- The server accepts, reads `shutdown` and exits 0, but every response reads `<EOF>`.
- Its responses go through `tcp_sendmsg` → `sendmsg` (memcached.c:137). trapk's `sublet_load_embedded` (loader.c)
  admitted only the iovec calls, execve and pselect6, and gave sendmsg -ENOSYS ("msghdr … get ENOSYS until their
  loaders exist").
- Ruling (runtime): no change to the workload. The fix is a msghdr loader in the kernel, which runtime has written
  (Image `749fed26…`). Unpatched memcached is requalified on that Image once runtime's directed test reads.
- A build with sendmsg rewritten as writev (`out-w`, stock `c165e169…`, protected `5fd5093c…`) is a twin diagnostic
  only: a patched workload, not a measured arm.
- That diagnostic's twin reading (`ovl-w`, Image `749fed26…`, lock 22:47:06, done 00:17:33): 6 of 6 repetitions
  qualify, stock and protected. Each transcript equals the native reference (1,931,207 B), each server stops on
  `shutdown` with exit 0, and identity reads `STAT pointer_size 128`. So once responses leave by writev, nothing else
  on the route stands between memcached and its oracle. It does not measure the unpatched server and does not
  exercise the msghdr loader. The overlay predates the exit-line adapter (it carries the earlier trap-vexec).

**trap-vexec.sh's exit line** (for case 25):
- The C launcher's line is `capstone-exec: exit image=<path> sha256=<hex>` (runtime/virtual/exec.c:1307). trap-vexec.sh
  now prints it on a run that ends, under the same gate (stderr a terminal or `CAPSTONE_EXEC_DIAGNOSTICS`).
- tests/trap-platform/test-vexec.sh: 8 of 8 PASS. The new checks are the exact line with the image's own digest, no
  line without diagnostics, and no exit line on a fault. The previous script fails the two new positive checks.
- The diff is handed to runtime (`trap-vexec-exitline.diff`, `c6a2bb88…`; script `41fb8b58…`).
- The probe's positive control on the twin is queued (`ovl-r60off` stage N3). Its pass condition: run-realphp's
  probe_reading reads ABSENT (False) from that transcript.

### Pre-registration: cause 24 under compressed bounds (runtime's rule, as registered)

Silicon has no exact bounds, so the board reads cause 24 where the exact-bounds record reads 28. For this corpus run
and its board boots, a cause-24 cell scores CAUGHT only if all three hold:
- (a) its faulting pc is in the same function as the exact-bounds reference catch;
- (b) the exact-bounds twin run of the same case and image catches it with cause 28 there;
- (c) the faulting operand is the result of a CINCOFFSET or CINCOFFSETIMM in that function.
Otherwise the cell keeps the judge's verdict.

The rule applies to every cell. On the twin, exact bounds off, the cause-24 cells are 7, all SQLite. PHP has none:
the same search finds 134 cause-25 and 52 cause-28 cells there. (c), read from each image's disassembly at the
faulting pc:

| cell | function | faulting operand | (c) |
|---|---|---|---|
| pools 31 | `statDecodePage` | `cincoffset a0, a0, a1` right before `lbu a1, 0(a0)` | met |
| pools 29 | `sqlite3Get4byte` | its argument, spilled and reloaded with `ldc`; the cincoffset is in the caller | not met |
| malloc 22 | `editDist3FindLang` | `cincoffset a0, a0, a1` right before `lw a0, 0(a0)` | met |
| malloc 02 | `sqlite3PagerSetPagesize` | loaded with `ldc` from memory | not met |
| malloc 05 | `sqlite3DropTriggerPtr` | loaded with `ldc` | not met |
| malloc 19, 20 | `nodeRelease` | loaded with `ldc` | not met |

- 29 stays MISSED, recorded as "24 at an operand formed by a cincoffset in the caller: excluded by rule (c) as
  registered". If the call-chain form is the better rule, it is pre-registered for the next corpus run, with 29 as
  the case that motivates it.
- (b) is one exact-bounds twin of the 7 cells plus pools 30, which is a control because it reads cause 28 either way
  (`ovl-r60b`, `x-capstone-exact-bounds=true`).
- Its off half is the same 8 cases on the same kernel (`ovl-r60off`). Both run on the twin's current Image
  `749fed26…`: the twin's Image path was rebuilt in place, and `4fc06dde…` is not kept. The diff between the two
  Images is the msghdr loader, which none of these cases calls.
- The two halves' `capstone-vexec` differ by the exit line above, which prints only on a normal exit, never on a fault.
- **Prediction:** pools 31 moves, and it is the only cell that does. 22 meets (c), but it cannot meet (a): the
  malloc arm has no reference catch, and its record is MISSED on virtual-malloc at both bounds settings (the
  2026-11-23 readings above). So the rule moves one cell, pools 27 → 28 CAUGHT, if (b) reads 28 for 31.

### (b) read, and the rule applied (twin, Image `749fed26…`, 2026-11-25, both halves done by 22:47)

The same 8 cases, the same images, the same kernel: `ovl-r60b` with `x-capstone-exact-bounds=true` and `ovl-r60off`
without it. Fault lines, cause at pc:

| cell | exact bounds on | off | off in twin-all (`4fc06dde…`) |
|---|---|---|---|
| pools 29 `sqlite3Get4byte` | 28 at +0x57c88, address …1fafa | 24, same pc and address | 24, same |
| pools 30 (control) `vdbeRecordCompareInt` | 28 at +0xcc340 | 28 | 28 |
| pools 31 `statDecodePage` | 28 at +0x1f7df4, address …1dde0 | 24, same pc and address | 24, same |
| malloc 02, 05, 19, 20, 22 | 24 | 24, same pcs and addresses | 24 |

- In both halves the control reads 28, so the property alone moves 29 and 31 from 28 to 24, at the same instruction
  and address. This reproduces the 2026-11-23 pair on the trap route.
- The five malloc cells read 24 with exact bounds too, so the compressed-bounds window does not explain them. Their
  addresses (0x0, 0x17, …392b, an ASCII word) are untagged values loaded from memory.
- The kernel change from twin-all's Image to this one does not move any of the 8 cells.

**The rule (pre-registered above), applied to the 7 cause-24 cells:**
- pools 31: (a), (b) and (c) all met, so **CAUGHT**;
- pools 29: (a) and (b) met, (c) not, so it stays MISSED, recorded as "24 at an operand formed by a cincoffset in
  the caller: excluded by rule (c) as registered";
- malloc 22: (c) met, but not (b) (24 with exact bounds) and not (a);
- malloc 02, 05, 19 and 20: none of the three is met.

So the rule moves one cell, pools 31, and pools reads **CAUGHT 28, MISSED 5**, as predicted.

**The putenv probe's positive control** (`ovl-r60off` stage N3, the new `capstone-vexec`):
- The probe printed the ABSENT pattern and `capstone-exec: exit image=/tmp/nt/img/probe-putenv.dom sha256=fd2e1555…`.
  That is the probe's own digest.
- run-realphp's `transcript_image` reads the digest from this transcript and `putenv_probe_outcome` reads False
  (ABSENT).
- On twin-all's transcript of the same probe, the identity reads None. That is why case 25 was MISSED there.
- The control fires both ways. It shows the identity can now be read; case 25's judgement is taken from the
  board-overlay twin run below.

**The board kit's own twin run.** `ovl-board` is the corpus overlay of the board payload `b8882143…`:
- SQLite and PHP stages only;
- it differs from twin-all's corpus stages only in `capstone-vexec`, which is 081c21bd's;
- it is queued on the twin with wide grants, as twin-all was.

Its readings are the board's predictions.

### twin-board read, and a correction to the board kit (2026-11-26)

`twin-board` ran `ovl-board` on Image `749fed26…` with wide grants. The lock was asked 22:48:10 and released 00:37:06.
Every stage ended rc 0, and the log holds 70 `@@CASE-END` lines.

- **SQLite**: both pools match twin-all case for case in verdict, cause, pc and function. malloc reads 29 MISSED and
  4 NO-READING; pools reads 27 CAUGHT and 6 MISSED before the cause-24 rule. The rule moves 31 as registered, so
  pools reads 28 CAUGHT and 5 MISSED. The one difference is case 11 (malloc): it is NO-READING both times, with the
  reason now "infra" where twin-all gave "inconclusive".
- **PHP temporal**: the table is identical to twin-all's.
- **PHP asan-blind**: three rows differ, in all four arms.
  - **25 (crash-084)**: was MISSED, now out-of-denominator. This is the expected change. The exit line lets the putenv
    probe read ABSENT, so the case's declared platform precondition is not met.
  - **30 (crash-092)**: was infra, now not-reached. Both are NO-READING, and this is the timing case (§ above).
  - **32 (crash-097)**: was MISSED, now **setup-fault**. The stock arm faults with cause 28 in memchr, before the case's
    marker, after `stream_socket_server(): unable to connect to udp://127.0.0.1:0 (Address not available)`.
    - Cause: the loopback interface. /init never brings `lo` up. In twin-all, the memcached stages (M20/M21, whose
      run.sh starts `ifconfig lo 127.0.0.1 up`) ran before the PHP stages in stage order, and the board kit has no
      memcached stage.
    - So the earlier statement that ovl-board "differs from twin-all's corpus stages only in capstone-vexec" holds for
      the files and not for the boot. The boot's history differs.
    - With lo up, twin-all and the reference run18 read 32 MISSED on all four arms.
  - Tallies (CAUGHT/MISSED), stock, block, object, exact: 5/28, 12/20, 25/9, 29/3. twin-all read 5/30, 12/22, 25/11,
    29/5. The moved rows are 25 and 32, which leave the denominator, and 30, which was already outside it.
- The kernel change (the msghdr loader, which adds only sendmsg and recvmsg to the calls the route admits) is not
  needed to explain these rows. For 32 the loopback explanation is a hypothesis until `twin-board2` reads: same
  Image, lo up. Its prediction is 32 MISSED on all four arms.

**Correction to the board kit.**
- make-php-ovl.py now starts each PHP stage with `ifconfig lo 127.0.0.1 up`, as the memcached stage does.
- `ovl-board2` is `ovl-board` with that line added to the eight PHP run.sh files and no other difference
  (diff -r: 8 files, one added line each).
- Its payload `payload-board2` (DTB caplifive-b5b2-wide.dtb):
  - fw_payload.bin `cf97af22f3713f5c…`, 21,657,608 B;
  - Image `029712a8c62d1a69…`, relinked from the same tree with only `init/version-timestamp.o` recompiled;
  - overlay `8ca92ec3…`, initramfs `5f2c872d…`;
  - the initramfs carries the ifconfig line 8 times, and the withdrawn one carries it 0 times.
- Payload `b8882143…` is withdrawn before any board use (attic `payload-board-lo-down`).
- The board's predictions come from `twin-board2`, which is queued.

### memcached requalified, unpatched, on the msghdr-loader Image (2026-11-26)

- Requested: runtime's ruling (above). Runtime's msgt directed test read on twin Image `749fed26…` (msgt.dom
  `2c6c30f7…`, msgt.native `cba6b00a…`). Native and protected printed identical lines: sendmsg, recvmsg with the iov
  split, an SCM_RIGHTS fd, MSG_TRUNC, fails=0. Runtime's own caveat: no negative control on a loader-free kernel.
- `ovl-req`, the unpatched build of `out/`: stock `53538940…`, protected `4108e588…`, with trap-vexec `41fb8b58…`
  (= runtime/trap-route-tools 081c21bd1f0e). The images are byte-identical to twin-all's M20/M21.
- Run `twin-req`, Image `749fed2606e3ebca`, hashed before and after; lock 00:24:01, done 00:48:27.
- **6 of 6 repetitions qualify** (read-mc.py against the native reference):
  - every transcript equals native's 1,931,207 B;
  - every server stops on `shutdown` with exit 0;
  - identity reads `STAT pointer_size 128`;
  - each run's exit line names its image (`53538940…` stock, `4108e588…` protected).
- The pair: twin-all, on Image `4fc06dde…` without the loader, read 0 of 6 with the same images: 4,905 B transcripts and
  every response `<EOF>`. The two runs also differ in the adapter, whose only change is a line printed at exit. So
  the kernel change, which runtime describes as the msghdr loader, is what moves memcached from 0/6 to 6/6 on the twin. This is the negative control runtime's
  msgt lacked, taken at the workload.
- The board kernel carries the same loader (`trapk/fpga/linux-6.1`), unvalidated on silicon. A memcached board boot is
  not pre-registered here.

### CPython on the trimmed stdlib tree (2026-11-26)

- The board stage was too big with the whole stdlib. `trim-tree.py` keeps `bench/` whole and, for each module and
  parent package in a `-X importtime` run's list, the module's source and `.pyc`, byte-identical:
  - the list comes from `twin-imp`, stock rc 0, answer `58211793…`;
  - the result is 150 modules, 190 files;
  - `ovl-trim` is 7,461,893 B, of which tree.tar.xz is 1,325,852 B.
- The images are unchanged: the overlay's app.xz files decompress to stock `e47ab197…` and prot `a269a6b5…`.
- Run `twin-trim`, Image `749fed26…`, lock 22:54:25, done 00:47:54:
  - both arms rc 0 and `PYPERFORMANCE-OK 8`;
  - both answers are `58211793977f706c…`, the native reference;
  - no Error or Traceback line;
  - twin wall time, from uptime: stock 263.4 s, prot 354.4 s.
- Board payload `payload-trim` (DTB caplifive-b5b2-wide.dtb, `sublet.dom_wide_grants=1` present in the bytes):
  - fw_payload.bin `ddee6d29e26b4d73…`, 23,754,760 B, under the 26,000,000 B cap;
  - Image `33ce8624…`, the same tree relinked, with only `version-timestamp.o` recompiled;
  - overlay `180aacaf…`, initramfs `bc3b311b…`.

### twin-board2: the loopback explanation holds (2026-11-26)

- `twin-board2` ran `ovl-board2` on Image `749fed26…`. Lock asked 00:40:19, done 01:28:30. Graded with grade-corpus.py
  against twin-all's 346 cells.
- 341 cells equal twin-all's. **32 (crash-097) is MISSED on all four arms**, as in twin-all and run18, so the loopback
  was the cause. 30 is back to infra, as in twin-all.
- Five cells differ:
  - 25 on the four asan-blind arms reads out-of-denominator, the exit line's expected effect;
  - SQLite malloc 11 is NO-READING both times, with the reason "infra" here and "inconclusive" there.

### Pre-registration: B5b-2 board boot "nested-cpymem", CPython then memcached (written 2026-11-26 02:40 host time (+08), 18:40Z on Nov 25, before the boot)

**What it is.** It covers runtime's B5b-2 items 2 and 3 on silicon in one boot:
- CPython's pyperformance subset on the trap route;
- then memcached as a server, which is also the msghdr loader's first silicon reading.

Both fit one payload under the 26,000,000 B cap, so runtime's rule puts them in one boot, CPython first and memcached
last (the newest kernel path). The board lane runs it.

**Platform.**
- Board 2's b42c8c644 session, right after misscost, as runtime allocated it; board 3's offer after ldcexc2 is the
  fallback. The boot runs on exactly one board, and its readings carry the board ID. The seal is the **full** B5b-2
  seal **b42c8c644**, never its spatial-only variant (runtime: on that variant every cause-25
  catch becomes MISSED by design).
- Runner: the board lane's run-one-e.sh (runner 5: incremental capture, the silence stop with the wedge snapshot, the
  STOP file), with the board's own booking and readbacks before and after.
- Image `<hdd>/nested/b5b2/cm/payload-cm3/fw_payload.bin`, sha256
  `ccecad78bfcb78b0634fd229859845eb81b0ab418326cbef4dbe2b4f715ce655`, 25,851,912 B:
  - firmware: sbi-keep trap-route-keep-tags;
  - DTB: caplifive-b5b2-wide.dtb `3d223e80…` (`sublet.dom_wide_grants=1`, present in the bytes);
  - **board kernel** = series at 6d4504f8b073 + the runtime lane's msghdr loader (unvalidated at registration),
    Image `f97b98f1…`, relinked from trapk/fpga/linux-6.1 with only the initramfs changed;
  - initramfs `acf4a766…` = base-rootfs `dd2dd8ff…` + overlay `ovl-cm3` (`093e5b6c…`).
- `ovl-cm3` is the twin-qualified overlays with only time limits changed, plus one instrument change:
  - CPython: `ovl-trim`'s stages C30/C31, with images stock `e47ab197…` and prot `a269a6b5…` and the 150-module stdlib
    tree;
  - memcached: `ovl-req`'s stages M20/M21, unpatched, with images stock `53538940…` and protected `4108e588…`;
  - trap-vexec `41fb8b58…` = runtime/trap-route-tools 081c21bd1f0e.
- `make-board-kit.py` (`ce218337…`) with `cm2.spec` (`985f1cab…`) changes:
  - /init → trapk/fpga/init-ordered-bound (`6462764b…`): per-stage bounds from /stages.bound;
  - CPython's per-run watchdog: 3300 → 9600 s;
  - an `@@UP <uptime>` line after each run's end line.
- Stage bounds: C30 and C31 10,800 s each, M20 and M21 3,600 s each. Each is at least 15× the stage's twin time
  (CPython 263 s stock and 354 s prot per run, prot stage 631 s by uptime; memcached 3.5-11 s per repetition on twin-cm3, about 30 s per stage).
- The instrument change: mc-harness is rebuilt from ports/memcached/app/host/mc-harness/mc-harness.c with its two
  wall-clock limits raised, the same on all arms. The rebuild without the change reproduces the shipped binary
  `90b558d8…`, so only the change differs: listen 120 → 600 s; no byte 60 → 300 s; binary `fde5d3c7…`, source
  `d016390a…`. Both limits stay under the 900 s per-repetition watchdog, so a hung server is still caught. Twin
  check: on twin-cm3 a whole repetition takes 3.5-5.0 s (the first of each stage, which includes the stage's unpacking, 11 s stock and 6.5 s protected). Listen time and the longest gap are each at most that, so at 15× they are 53-165 s: within the new limits, and over the old 60 s one.
- **Boot bound: 30,000 s.** That is the sum of the stage bounds (28,800 s), plus 1,200 s for upload, boot and the smoke
  control.

**Predictions** (the twin, `twin-cm3`, this exact overlay on Image `749fed26…`; lock granted after a 5,203.5 s wait, done 02:40:07; console log sha256 `1badfd76be9deb68…`):
- **CPython, both arms PASS:**
  - rc 0, answer `58211793977f706c…` = native's, `PYPERFORMANCE-OK 8`;
  - each exit line names its image (`e47ab197…`, `a269a6b5…`);
  - twin wall time: stock 253.4 s, prot 329.0 s.
- **memcached, 6 of 6 repetitions qualify:** every transcript equals native's 1,931,207 B, each server stops on
  `shutdown` with exit 0, identity reads `STAT pointer_size 128`, and each exit line names its image.
- **grade-cm.py on the twin log: exit 0, BOTH PASS.** The prediction is the same exit on the board's capture.
- The board is predicted slower, but by an unknown factor. That factor is a reading, below.

**Grader.** `python3 <hdd>/nested/b5b2/cm/grade-cm.py <UART capture> <new out dir>` (`90c39ded…`). It runs
cpython/read-cpy.py (`dc685836…`) and memcached/read-mc.py (`15d33f37…`), the latter against memcached/out, the native
reference.
- Exit 0 means both pass: CPython's both arms rc 0, answer = native's `58211793…`, `PYPERFORMANCE-OK 8`, no timeout
  or fault, and an exit line naming the arm's image; memcached 6 of 6 repetitions qualify.
- Exit 1: something was read and failed.
- Exit 2: no reading, meaning a stage or run is missing.
- Controls on recorded logs: twin logs pass (0). A changed answer, a dropped OK line and a wrong image each fail
  (read-cpy 1). memcached at 0/6 fails (1). A log without a stage gives 2. And the twin-cm3 log itself passes (0).

**How each outcome reads.**
- **CPython PASS:** the protected and stock images both run Fig. 10's workload correctly on silicon's trap route. The
  @@UP and PYPERF-TIME lines give wall times, which are a cost reading and not a verdict.
- **CPython FAIL:**
  - a wrong answer or a fault is a correctness finding, reported with the arm and the fault line;
  - a `@@CPY-TIMEOUT` at 9600 s is a budget reading, not a verdict.
- **memcached 6/6:** the msghdr loader serves sendmsg on silicon, on both arms.
- **memcached all `<EOF>`** (4,905 B transcripts, twin-all's shape): the loader does not serve sendmsg on silicon. That
  is a kernel finding, routed to runtime.
- **memcached `<TIMEOUT>` in a transcript:** a harness timeout at 300 s idle, not a protection verdict.
- **A stage cut at its bound, or a silence stop:** no reading for that stage. The wedge snapshot goes to runtime.
- **The board/twin ratio** is a reading of this boot: per run, from the @@UP lines and CPython's up0/up1, against the
  twin's. It is not a prediction.

### Result: nested-cpymem, CPython PASS on both arms; memcached 0 of 6 by the registered criterion, every transcript the same and off native's only in the two answers that read the wall clock (board 2, 2026-11-26 05:18-06:52 host time, read against 506d11206c40)

**Platform, as read back.**
- Board 2, booked by the board lane's runner and released (UNBOOK 200).
- Read back as b42c8c644 by name before and after the boot, `…-b5b2n-crev4b-wap-flu-r80-r77_b42c8c644.bit`: the full B5b-2 seal, as registered. Both readbacks give rc 3, which means the name matched and the console exposes no sha256.
- Image `ccecad78bfcb78b0…`, as registered.
- **UNGATED** by preflight-board-run.sh (runner 5 does not invoke it). The board lane's own prechecks ran: the image sha256 against the arming file, the seal's sha256, and the readback by name before and after.
- Capture: uart.txt sha256 `b1043a2b6bcb3ebe…`, 78,673 B. The grader's boot.txt (CR dropped) is sha256 `ffb832f8715f1a63…`.

**Grader, as registered** (`grade-cm.py`, `90c39ded…`): **exit 1.**
- cpython: rc 0, "both arms pass".
- memcached: rc 1. read-mc.py stopped with a base64 error after printing stock reps 1-3 as FAILS.
  - The error's cause: a console heartbeat line, `TRAPTEST: tick 5212.96`, landed inside protected rep 1's `server.out` block.
  - The reader takes the block whole, so it cannot decode it. The twin's log has 5 tick lines and none inside a block; the board's has 43.
  - The registered exit is 1 either way: stock reps 1-3 had already failed on their transcripts.

**CPython: PASS on both arms**, which is the prediction.
- Both arms: rc 0, answer `58211793977f706c…` (397 B) = native's, `PYPERFORMANCE-OK 8`, no timeout or fault.
- Each exit line names its arm's image: stock `e47ab197…`, prot `a269a6b5…`.

| arm | board wall (s) | twin wall (s) | board / twin |
|---|---|---|---|
| stock | 2,246.3 | 253.4 | 8.9× |
| prot | 2,398.0 | 329.0 | 7.3× |

- On the board, prot is +6.8 % over stock in wall time; on the twin it is +29.8 %. The twin's emulation cost is not silicon's, so only the board's figure is a silicon reading.
- Per benchmark, prot over stock on the board (PYPERF-TIME, ns):

| benchmark | change |
|---|---|
| json_dumps | +8.6 % |
| json_loads | +11.6 % |
| richards | +2.6 % |
| deltablue | +2.4 % |
| chaos | +3.8 % |
| go | +5.5 % |
| nbody | +8.1 % |
| gc_collect | +16.4 % |

- This is one run per arm (N = 1): a reading, not an estimate with a spread.

**memcached: 0 of 6 repetitions qualify**, against a prediction of 6 of 6.
- Every repetition on both arms:
  - harness rc 0;
  - `stop=shutdown exit=0`;
  - identity `STAT pointer_size 128`;
  - an exit line naming its image (stock `53538940…`, protected `4108e588…`);
  - a transcript of 1,931,245 B.
- All six transcripts are byte-identical to each other (sha256 `028c99b24131e5b7…`), and each differs from native's (1,931,207 B, `ee1522c8e0eceec9…`) in exactly two places:
  - `get gone` after `touch gone -1`, and `get past` after `set past 0 -1 1`, return the value. Native's do not. These are mc-harness.c:207-208, "expired at once" and "stored expired".
  - As a result, `STAT get_hits` reads 337 against 335 and `get_misses` 5 against 7.
- The twin of this exact overlay (twin-cm3) gives native's transcript in all six repetitions.
- None of the failure shapes the pre-registration named appears: no `<EOF>` transcript, no `<TIMEOUT>`, no cut stage. The msghdr loader delivered every transcript whole on both arms.

**Why the two answers differ: the board has no RTC.**
- In memcached 1.6.45, a negative exptime becomes `REALTIME_MAXDELTA + 1` = 2,592,001 (memcached.h:1064, `EXPTIME_TO_POSITIVE_TIME`).
- realtime() (memcached.c:174) reads a value above the 30-day delta as absolute Unix time. It gives 1, "expired at once", only when `2,592,001 <= process_started`, where `process_started = time(0) − 62` (memcached.c:203).
- So the item expires at once only if the wall clock is more than 30 days past the epoch.
  - The twin's kernel sets its clock from goldfish_rtc: `setting system clock to 2026-11-25T18:29:26 UTC`.
  - The board's log has no RTC line, so the board's wall clock counts from the epoch. At these commands the uptime was ~5,000 s. realtime() then returns `2,592,001 − process_started`, an expiry about 30 days ahead.
- Both arms read the same, because the cause is in time(0), not in protection.
- **Checked on the twin, one variable changed** (2026-11-26 08:59-09:00, `<hdd>/nested/b5b2/cm/epoch/run-epoch.sh`):
  - The setup: ovl-cm3's memcached stages only, Image `749fed26…`, and one initramfs (`5f351a47…`) for both arms. The arms differ only in QEMU's `-rtc base`.
  - **epoch** (`-rtc base=1970-01-01T00:00:00`): the kernel reads `setting system clock to 1970-01-01T00:00:02 UTC`, and all six transcripts are sha256 `028c99b2…`, the board's exactly.
  - **real** (no `-rtc`): the kernel reads `2026-11-26T01:00:25 UTC`, and all six are `ee1522c8…`, native's.
  - So the board's whole memcached difference is reproduced by the clock alone.

**Reading.**
- CPython: the stock and protected images both run Fig. 10's workload correctly on silicon's trap route, at +6.8 % wall time for prot (N = 1).
- memcached: protection does not change the server's behaviour on silicon. Both arms give the same 1,931,245 B transcript, and its two differences from native are the board's wall clock.
- It does not qualify by the registered criterion, and that verdict stands as registered.
  - A requalification needs native's reference taken at the board's clock, or the board's clock set before the server starts. Either is a new pre-registration.
  - The two `-1` fixtures cannot pass on a board without an RTC.
- The cpymem board/twin ratio for corpus case 21: CPython reads 8.9× stock and 7.3× prot. memcached's repetitions read ~10× in uptime per repetition (board 49-50 s, twin 3.5-6.5 s).

### Pre-registration: memcached's requalification on b42c8c644, the clock set before the server starts (written 2026-11-26 13:50 host time (+08), before the boot)

**Question.**
- nested-cpymem's memcached qualified 0 of 6 by its criterion.
- Every transcript was the same on both arms, and each differed from native's only in the two answers that read the
  wall clock. The cause is the board's missing RTC.
- The twin reproduced the whole difference by the clock alone (`cm/epoch`, result above).
- Does memcached qualify 6 of 6 on silicon when the clock is set before the server starts? Nothing else changes.

**What changes, and what does not.**
- The stage scripts, images and harness are nested-cpymem's, byte for byte:
  - M20/M21 of `ovl-cm3`;
  - stock `53538940…`, protected `4108e588…`;
  - mc-harness `fde5d3c7…`;
  - stage bounds 3,600 s each.
- runtime's rule applies: the test and its workload are not edited.
- The change is in the boot environment only. `/init` (`cm/clock/ovl-clk/init`, `5dc641dd…`) is init-ordered-bound with
  two lines before the smoke:
  - when `/clock.epoch` exists, the clock is set from it (`date -s @<seconds>`);
  - a `TRAPTEST: clock <UTC> epoch-s <seconds>` line is printed in every case.
- The board image's `/clock.epoch` reads `1795663474` (2026-11-26T03:24:34Z, the host time when the overlay was made).
  The two fixtures need only a clock more than 30 days past the epoch (`REALTIME_MAXDELTA`, memcached.h:1064) that runs
  forward. Both hold for any such value.

**Platform.**
- Board: b42c8c644, the full B5b-2 seal. The board lane runs it, on the board runtime allocates, with its runner as for
  nested-cpymem (booking and readbacks before and after, the silence stop, the STOP file).
- Boot bound: 9,000 s. That is the two stage bounds (7,200 s) plus 1,800 s for upload, boot and the smoke; on the twin
  the two stages took under 4 min.
- If the runner's logs carry no preflight-gate line, the boot is reported **UNGATED**.
- Payload `<hdd>/nested/b5b2/cm/clock/payload-board/fw_payload.bin`, sha256 `194a54623cd3056a…`, 19,565,064 B:
  - runtime's fpga-final make-payload.sh (the campaign's final Full kernel: NO_HZ_IDLE, TRAP_TEST off, the msghdr
    loader in), Image `c87952a9…`;
  - keep-tags firmware with `FW_FAST_RDTIME=y` (2 fast-path references);
  - `caplifive-b5b2-wide.dtb` (`3d223e80…`, `dom_wide_grants=1`), the DTB nested-cpymem and the twin used;
  - initramfs = base-rootfs + `ovl-board` (manifest equal to the payload's `overlay-files.sha256`).
- **Disclosed:** the kernel is the campaign's final Full kernel, not nested-cpymem's (6d4504f8b073 + the msghdr loader,
  `f97b98f1…`). So this boot differs from nested-cpymem in the kernel and the payload as well as the clock. Its reading
  is the server row on the campaign kernel, not a one-variable rerun.

**Twin, as predicted and as read** (`cm/clock/run-clock.sh`, `c45cacdab8950eb0`). Both arms run with QEMU's RTC at the
epoch, the board's clock, on the same `/init`; they differ only in `/clock.epoch`:
QEMU is qemu-crv (`ea5c05f3…`), with Image `749fed26…`, the keep-tags fw_jump.elf and `sublet.dom_wide_grants=1`, as in
twin-cm3. The two overlays' manifests differ in that one file and nowhere else.
- **set** (lock granted after a wait, done 13:44:01; initramfs `3795d28c…`; console log `fc95df40…`):
  - the kernel reads `setting system clock to 1970-01-01T00:00:02 UTC`;
  - the /init line reads `TRAPTEST: clock set rc=0 from /clock.epoch 1795660718` and
    `TRAPTEST: clock 2026-11-26T02:38:38Z`;
  - **all six transcripts are native's**, `ee1522c8…`;
  - read-mc2.py: 6 of 6, exit 0.
- **unset** (done 13:47:24; initramfs `3ce605b5…`; console log `e6c35de3…`):
  - `TRAPTEST: clock 1970-01-01T00:00:02Z epoch-s 2`;
  - all six are `028c99b2…`, nested-cpymem's board transcript, with `get_hits` 337 and `get_misses` 5;
  - read-mc2.py: 0 of 6, exit 1.
- So on the twin, setting the clock in /init is enough to give native's transcript. The control shows the check fires.

**Predictions on the board.**
1. A `TRAPTEST: clock 2026-11-26T…Z` line before the smoke.
2. memcached 6 of 6 qualify: every transcript equals native's 1,931,207 B (`ee1522c8…`), each server stops on
   `shutdown` with exit 0, identity reads `STAT pointer_size 128`, and each exit line names its image.

**Grader.**
- `python3 <hdd>/nested/b5b2/memcached/read-mc2.py <UART capture> <hdd>/nested/b5b2/memcached/out` (`c76d4138…`).
- read-mc2.py is read-mc.py (`15d33f37…`) with one change. Inside an RG_FILE block, whole `TRAPTEST: tick <uptime>`
  lines are dropped and counted, and a block that still does not decode is reported UNREADABLE and fails.
- Controls:
  - nested-cpymem's board capture: 0 of 6, 1 tick line dropped, no crash (read-mc.py had stopped there);
  - twin-cm3's log: 6 of 6;
  - twin-cm3's log with a tick line inserted inside a transcript block: 6 of 6, 1 dropped;
  - the same with a non-base64 line added as well: that repetition is UNREADABLE and FAILS, exit 1.
- Exit 0 means 6 of 6; exit 1 means read and failed; exit 2 means no stage.

**How each outcome reads.**
- **6 of 6:** memcached is served on silicon by the msghdr loader on both arms with native's transcript. The server row
  qualifies on the campaign kernel.
- **The two clock answers differ again with the clock line present:** the clock did not hold, and the clock line's value
  is read. That is not a protection verdict.
- **Any other difference:** reported as itself, with its arm.
- **No clock line, or a 1970 clock line:** the set failed. That is no reading for this question.

### Result: memcached's requalification. 6 of 6 qualify, every transcript native's, with the clock set from /init (board 2, 2026-11-26 15:26-15:37 host time, read against 740970700451)

**Platform, as read back.**
- Board 2 (booking 1585), booked by the board lane's runner and released (UNBOOK 200).
- Payload `fw_payload.bin` sha256 `194a54623cd3056a…`, 19,565,064 B, as registered.
- The readback names `…-r80-r77_b42c8c644.bit` before and after the boot: the full seal, as registered.
- The runner's logs carry no preflight-gate line, so the boot is reported **UNGATED**.
- Capture `uart.txt`: sha256 `a67782b907173a74…`, 74,563 B. It reaches `TRAPTEST: done`.

**Readings.**
- **Prediction 1, met.** Before the smoke, the console reads `TRAPTEST: clock set rc=0 from /clock.epoch 1795663474`
  and `TRAPTEST: clock 2026-11-26T03:24:34Z epoch-s 1795663474`.
- **Prediction 2, met.** The registered grader (`read-mc2.py` `c76d4138…`, against `<hdd>/nested/b5b2/memcached/out`)
  exits 0: **6 of 6 repetitions qualify**. Two tick lines were dropped inside transcript blocks.
  - For every repetition, on both arms:
    - harness rc 0;
    - `stop=shutdown exit=0`, at 10.75-11.53 s;
    - a transcript equal to native's (1,931,207 B);
    - identity `STAT pointer_size 128`.
  - Each exit line names its image: stock `535389400f49…`, protected `4108e588f238…`.
- **Repetition 3 of each arm has no `job.json`. That is by design.**
  - The stage's repetition 3 runs `capstone-vexec` directly as root (`-u root`), without the `capstone-job` wrapper
    that writes `job.json` (`run.sh`, the `RG_BEGIN:stock:3` block).
  - The loop that prints the files skips a missing one, so the capture has no `RG_FILE:job.json` block there.
  - The twin's set arm (`cm/clock/set/read-mc2.txt`) reads the same empty `job` on repetition 3 of both arms.
  - `job.json` is not in the qualify rule.
- **The in-run grade.** The board lane's appended plan step passed read-mc2.py a directory of its own as the native
  reference, and the grader stopped on a missing file (rc 1). The grade above is the board lane's rerun of the
  registered command, repeated here with the same output.

**Reading.**
- On silicon, memcached is served by the msghdr loader on both arms with native's transcript, once the clock is set.
  The server row qualifies on the campaign's final Full kernel. It stands as the Full arm's memcached row.
- nested-cpymem's 0 of 6 fits the board's missing RTC, which the twin reproduced by the clock alone (`cm/epoch`,
  `cm/clock`). This boot cannot separate the clock from its kernel change (below). nested-cpymem's own verdict, 0 of 6
  by its registered criterion, stands as registered.
- **Disclosed again:** this boot also changed the kernel and the payload from nested-cpymem's. So it is the server row on
  the campaign kernel, not a one-variable rerun.
- **Board / twin times** (the harness's `stop_seconds`, readings): board 10.75-11.53 s, against 1.11-1.37 s on the
  twin's set arm.

**Limits.**
- Three repetitions per arm, one boot, ungated.
- The clock value is the overlay's build time, so the boot's wall clock started at 03:24:34Z. Any value over 30 days
  past the epoch serves.

## Line B campaign arm "Full + ports": gcc's and perlbench's ports as B5TL1-paired stages (2026-11-26)

runtime's Line B plan (dev b4aae7e416e8, `docs/plans/2026-11-25-line-b-final-campaign.md`) adds an arm "Full + ports":
- B5TL1 on b42c8c644, with this lane's gcc port (all-abc-bm-ls-tfl-phi) and perlbench port in place of the stock
  stages.
- The plan's audit found that neither port was in any campaign arm.
- runtime's rule for the pair: the port stages use B5TL1's compiler identity and flags, and differ only in the port's
  own sources and shim. They are checked against B5TL1's replayed compile lines, not only run.sh.

### The images: B5TL1's own build trees, recompiled with the port sources

`<hdd>/nested/b5b2/ports/build-ports.sh` (`7c40ced6…`) copies the compiler lane's B5TL1 build trees
(`<hdd>/perf/cclever/arms/B5TL1/spec/<bench>`, read-only) and rebuilds each with that tree's own `compile.txt` and
`link.txt`.
- The compiler is B5TL1's `sdk/capstone-cc` (`6efe2f77…`), which runs llvm-c9 clang 29027126f39b (`ecd6d8c4…`), with
  the traplean1 SDK.
- The flags are B5TL1's own compile lines. B5TL1's `extra-flags` file is byte-equal to FRCDKMSa3's, and the base set is
  the same.
- Each image is stripped as build-arm.sh strips (llvm-c4 objcopy, keeping `.capstone_virtual`,
  `.capstone_application` and `.capstone_domreq`).
- Each image is scanned with `scan_illegal.py`, and `trap_altstack` is checked present (the trap runtime).

| arm | what differs from B5TL1's tree | strip.dom sha256 | scan | CDERIVE | CREVOKE | CRENEW |
|---|---|---|---|---:|---:|---:|
| perl-stock | nothing: the replay control | `f3c8c80d71844564` = B5TL1's | ILLEGAL=0 | 6 | 5 | 1 |
| perl-full | heads + bodies + HE + op slabs (rev 4), defines, shim, header dir | `b4eb3e6f267f5f34` | ILLEGAL=0 | 96 | 15 | 1 |
| perl-fullnr | perl-full + `_BODIES_NOREVOKE`, `_OPS_NOREVOKE` | `1f00aa5238b003c0` | ILLEGAL=0 | 96 | 8 | 1 |
| gcc-stock | nothing: the replay control | `f800e015d161c911` = B5TL1's | ILLEGAL=0 | 6 | 5 | 1 |
| gcc-port | 11 sources and `sublet-port.h` | `cff54b2a361867f2` | ILLEGAL=0 | 81 | 9 | 44 |

- **Both replay controls reproduce B5TL1's images byte for byte.** So the replay is B5TL1's build, and the port arms
  differ from it only where the table says.
- **perl-full** is the twin's full port (section "perlbench item 3: the full port", image `ce3a48e5` on the vm route):
  - the committed `port-sv-heads.py`, `fix-lookup.py`, `fix-lookup-pages.py`, `port-sv-bodies.py`, `port-he.py` and
    `port-op-slabs.py` (revision 4), run on the copy;
  - the patched sources are asserted equal to the twin's full tree (`perlspec/fx2/full`), apart from Makefile.spec and
    the build products;
  - each of the 69 compile lines is B5TL1's, plus `-DPERL_CAPSTONE_SUBLET -DPERL_CAPSTONE_SUBLET_BODIES
    -DPERL_CAPSTONE_SUBLET_HE -DPERL_CAPSTONE_SUBLET_OPS`, plus `-include oneop-shim.h` (`335b499c…`, Fig. 10's
    one-operand revision: `capstone_cap_revoke_child(parent, child)` becomes `capstone_cap_release(child)`), plus `-I`
    to a frozen copy of the B5TL1 runtime tree's `capability.h` (`a502b2a6…`) and `capability-slot.h` (`a00a47fc…`).
  - The CREVOKE census fires: the control has 7 fewer, the three allocator ports' 7, as on the twin's full-nr.
- **gcc-port:**
  - each of the 11 port files (`alloc-pool.c bitmap.c c-decl.c df-scan.c emit-rtl.c ggc-page.c gimple.c
    include/obstack.h lists.c obstack.c tree-phinodes.c`) is first checked byte-equal between B5TL1's tree and oneop's
    stock tree, then replaced by `oneop/gcc-arms/all-abc-bm-ls-tfl-phi`'s;
  - `sublet-port.h` is added;
  - the 387 compile lines are unchanged, as in oneop, where the port changes no compile line.
- **Fixtures** (`build-fixtures.sh`, `5ee38100…`): the committed `make-{bodies,he,ops}-fixture.py`, each generated from
  the tree's own `perlmain.c`, compiled with its `perlmain.o` line and linked in its place, on perl-full and
  perl-fullnr. All six scan ILLEGAL=0.

### The stages and payloads: the campaign's own overlays with the image replaced

The campaign pairs gcc's t1.c in overlay `SP-B5TL1-g3` and perlbench's makerand.pl in `SP-B5TL1-g4mk` (runtime's
`trapk/fpga/spec`, relinked into CF-* by `trapk/build-campaign.sh`).
- B5TL1's own stages were made by replacing the image in an existing stage (plan.json `native.arm_replaced`).
- `make-port-stages.py` (`374b3cc4…`) does the same:
  - it copies `SP-B5TL1-<g>/ovl` whole to `board/SP-B5TL1P-<g>/ovl`;
  - it replaces only the trap-route image;
  - it records a `port` entry in plan.json.
- `diff -rq` against the source overlay is asserted to give exactly plan.json and the image. So /init (`d6b55a97…`),
  smoke.dom, probes.list, run.sh and in/ are byte-identical.
- g4mk's plan.json `executables` entry names an older arm's image (`371553e3…`, B5T), inherited from its source stage.
  The shipped binary is B5TL1's `f3c8c80d…`, checked by hash. The label is stale, not the content; runtime records it in
  spectest Amendment 12. The port stage leaves that field as it is.
- runtime built the payloads on the final Full kernel. This lane rechecked both hashes:
  - fpga-final, NO_HZ_IDLE, TRAP_TEST off, msghdr loader in;
  - keep-tags firmware with `FW_FAST_RDTIME=y` (2 fast-path references);
  - `caplifive-b5b2.dtb`.

| payload | stage | fw_payload.bin sha256 | bytes |
|---|---|---|---:|
| CF-B5TL1-g3 (stock pair) | gcc stock | `997e2fd6fdbdecdf…` | 21,662,216 |
| CFP-B5TL1-g3 | gcc-port | `6616eecf0c797071…` | 21,662,216 |
| CF-B5TL1-g4mk (stock pair) | perl stock | `e82d42ebe558cb18…` | 19,565,064 |
| CFP-B5TL1-g4mk | perl-full | `432e6ae27d5aa062…` | 19,565,064 |

- runtime checked that each built overlay's manifest equals `SP-B5TL1P-<g>/ovl` byte for byte.

### The twin, at the board's memory: protection fixture and the stage pair (2026-11-26 13:57-14:00)

`twin-ports.sh` (`064197f8…`) runs both boots:
- qemu-crv, Image `749fed26…`, the keep-tags fw_jump.elf, `sublet.dom_wide_grants=1`;
- runtime's `trapk/boot-long.py` with `BOOT_MEM=962M` and `BOOT_SILENCE_S=21600`;
- /init = `init-twin` (`66815b00…`): the campaign overlays' /init plus a per-stage sampler of the lowest MemAvailable.
  It is twin-only and the same on every stage, and is not in the board overlays.
- **Disclosed:**
  - The first fixture boot (13:43) had a non-executable /init. The kernel fell back to buildroot's init and waited
    at a login prompt. It was stopped after 13 min, and nothing was read from it.
  - The runner now refuses an /init that is not 0755, tested both ways. The first grading of the stage pair had no
    grading root, because the root was never created, so every stage read MISS on a missing plan.json. The same
    capture was regraded once the root was made, and that grade is the one below.

**Fixture** (35 commands, `fixture/stage-fx`, console log `4d4dd517…`): the bodies, HE and ops fixtures on perl-full and on
perl-fullnr.

| build | bodies (modes 0-4) | HE (modes 0-4) | ops (modes 0-7 full, 0-6 control) |
|---|---|---|---|
| perl-full | 0 reads (`cur=5`); 1-4 fault, `sublet: fault cause=25` | 0 reads; 1-4 fault, cause 25 | 0 reads; 1-4 fault, cause 25; 5 `errors=3`; 6 `errors=0`; **7 `booted`, rc 0** |
| perl-fullnr | 0-3 read (1 and 3 the freed body, 2 the old body); 4 faults, cause 25 | 0-4 read (1-2 `read_after_delete`, 3-4 `read_after_free`) | 0-3 read (1-3 `read_after_free`); 4 faults, cause 25; 5 and 6 as on full |

- **34 of 35 read as the vm-route twin's full-port fixture did** (`perlfxfull-ce3a48e5`): every fault, every control
  read, and the live reads before each mode-3 and mode-4 fault: bodies 4,999 and 3,000, ops 2,999, all good, 0 bad.
- So each of the three ports' protection holds on the trap route, and removing the seven CREVOKEs lets the stale reads
  through.
- **Mode 7 differs, and the cause is the route.** Mode 7 is the boot's positive control. It parses `-e 0`, whose script
  perl opens as /dev/null, and the vm-route twin's sandbox refuses that (EACCES), so perl_parse fails. The trap route
  runs a plain Linux process, which opens /dev/null. This log has no `Permission denied` line, where the vm route's
  carries one, so the boot succeeds and prints `booted`.
  - So on this route the boot check's failure path is not exercised. Modes 0-6 boot from `fx-boot.pl` and return or
    fault as predicted.

**Stage pair** (one boot, console log `9d019eca…`), graded by `grade-spec-chain.py` (`4fa95eaf…`) with
`SPEC_STAGE_ROOT=ports/twinroot`: `C=ok`, **all four PASS**, specdiff 1/1 each.

| stage | image | run (s, twin) | lowest MemAvailable (kB, of 943,568) |
|---|---|---:|---:|
| B5TL1 perl stock (g4mk, makerand.pl) | `f3c8c80d…` | 59.81 | 859,276 |
| B5TL1P perl-full | `b4eb3e6f…` | 57.27 | 859,356 |
| B5TL1 gcc stock (g3, t1.c) | `f800e015…` | 1.25 | 856,400 |
| B5TL1P gcc-port | `cff54b2a…` | 1.47 | 857,760 |

- **Memory:** neither port lowers the system's lowest MemAvailable below stock's.
  - perl reads +80 kB, and gcc +1,360 kB.
  - The sampler reads once a second. gcc's 1.25-1.47 s run gets one or two samples, so its figure is weak.
- The twin's run times are not silicon's. They are given, and they are not predictions.

### Pre-registration: the "Full + ports" boots CFP-B5TL1-g3 (gcc) and CFP-B5TL1-g4mk (perlbench) on b42c8c644 (written 2026-11-26 14:03 host time (+08), before either boot)

**Question.** Do gcc's port (all-abc-bm-ls-tfl-phi) and perlbench's full port (heads, bodies, hash entries, op slabs) run
on B5b-2 silicon with B5TL1's answers, in the campaign's own form? What does each cost there in wall time, against its
stock pair on the same seal and kernel?

**The boots.**

| boot | payload (above) | pairs with |
|---|---|---|
| CFP-B5TL1-g3 | `6616eecf0c797071…` | CF-B5TL1-g3 `997e2fd6…` |
| CFP-B5TL1-g4mk | `432e6ae27d5aa062…` | CF-B5TL1-g4mk `e82d42eb…` |

- Seal: the full **b42c8c644**, read back by name before and after each boot.
- runtime allocates the board and slot, and the board lane runs both boots with the same runner, bounds and procedure as
  the campaign's CF-B5TL1-<g> boots.
- A pair is read only when both of its boots ran on the same board and seal.
- If the runner's logs carry no preflight-gate line, the boot is reported **UNGATED**.

**Predictions.**
1. Each port boot: smoke `C=ok`, its one stage `PASS` (QSTEP rc 0, specdiff 1/1), as on the twin.
2. **No time prediction.** Each port boot's run time (`up1 − up0`) against its stock pair's is the reading. N = 1 per
   arm. The twin's +17.6 % (gcc) and −4.2 % (perl) are an emulator's, and predict nothing on silicon.

**Grader.**
- `SPEC_STAGE_ROOT=<hdd>/nested/b5b2/ports/boardroot python3 <hdd>/b5b2linux/grade-spec-chain.py <UART> <out>
  B5TL1-<bench>` (`4fa95eaf…`). `boardroot/B5TL1/<bench>` is the port stage (`ports/stages/<bench>`).
  run-stage.py's reader does not read plan.json's image hashes, so the port stage's plan.json and the stock one read
  the same.
- Controls in board form: the twin capture's port stages, renamed to their board names (`ports/grade-ctl/`):
  - `pos-g3` and `pos-g4mk` read PASS, exit 0;
  - `neg-g3` and `neg-g4mk`, each with one base64 character flipped in its output's in-band file, read MISS, exit 1.

**How each outcome reads.**
- **PASS:** the port runs on silicon with B5TL1's answer. Its time against the stock pair is the port's silicon cost on
  this stage, N = 1.
  - It joins Line B's Full + ports row. The share is the runtime and rtl lanes' to compute, with the per-event prices
    named by their source.
- **MISS with QSTEP rc 0:** a wrong answer, reported with the stage's read lines. That is a correctness finding about
  the port on silicon.
- **A fault line (`sublet: fault`)** in a port boot and not in its stock pair: reported with its pc, against the port's
  sources.
- **A silence stop or a stage cut:** no reading for that boot. The wedge snapshot goes to runtime.

**Amendment 1 (2026-11-26 14:27 host time (+08), before any of the four boots): the grader becomes grade-spec-chain-v2.py, with v1 recorded beside it.**
- **Why.** nested-corpus showed that on silicon a kernel printk can land inside a program's own console output
  (result above, 0e8089933116). The board lane then found that v1 (`4fa95eaf…`) is wrong on two shapes:
  - a foreign line inside an `@@SPEC-FILE` block reads MISS, which protection did not cause;
  - a mid-line splice inside an `@@SPEC-HASH` block silently truncates a hash.
- **v2** is `<hdd>/b5b2linux/grade-spec-chain-v2.py`, sha256 `b976d2884c8c42d5…`. It is the board lane's, pre-registered at
  llvm-capstone board/meas-pw `5106037bcc89`.
  - It is v1 plus one step: inside those blocks only, it removes every `[ <t>] …` and `TRAPTEST: …` line with its
    newline, so that a split line rejoins.
  - A line still foreign afterwards makes the stage NO READING, never MISS. gzip's CRC32 checks every decoded file.
- **Checked here:**
  - v2 on `ports/grade-ctl/`: pos-g3 and pos-g4mk PASS, neg-g3 and neg-g4mk MISS, exit codes 0, 0, 1, 1, as under v1;
  - a copy of pos-g3 with a kernel fault line spliced mid-line into the t1.c output's SPEC-FILE block: v1 MISS, exit 1;
    v2 PASS with `INBAND … 1 interleaved line(s) removed`, exit 0;
  - the twin capture of the stage pair: v2's output is identical to v1's (all four PASS).
- **Binding:** v2's verdict, with the same stage roots as registered (`ports/boardroot` for the CFP boots). v1's line is
  recorded beside it. A stage where the two differ is reported with both readings and v2's INBAND line.
- Nothing else changes: the boots, the payloads, the predictions and the outcome readings.

### Result: "Full + ports". Both ports PASS on silicon with B5TL1's answers, and so do both stock pairs; v2 and v1 agree on every row (board 2, b42c8c644, 2026-11-26 15:37-16:14 host time, read against c4732c42a52b with Amendment 1 1ab5179e06d3)

**Platform, as read back.** One board, one seal, one kernel per pair, so both pairs are readable.

| boot | booking | payload | host time | uart sha256 | preflight |
|---|---:|---|---|---|---|
| CF-B5TL1-g3 (stock) | 1586 | `997e2fd6fdbdecdf…` | 15:37:23-15:42:58 | `ef564f145a309141…` | UNGATED; read afterwards, BLOCK by C10's 20 MB budget |
| CFP-B5TL1-g3 (port) | 1587 | `6616eecf0c797071…` | 15:43:00-15:48:36 | `5e4b0123351f21a8…` | UNGATED; read afterwards, BLOCK by C10's 20 MB budget (21.66 MB) |
| CF-B5TL1-g4mk (stock) | 1588 | `e82d42ebe558cb18…` | 15:48:42-16:01:34 | `2522348b57230d99…` | GO |
| CFP-B5TL1-g4mk (port) | 1589 | `432e6ae27d5aa062…` | 16:01:41-16:14:34 | `4c1171058609dd33…` | BLOCK by C10's FDT count; runtime ruled past it |

- The board lane ran the boots and preflight-board-run.sh. Every boot's readback names `…-r80-r77_b42c8c644.bit`
  before and after, and every booking was released.
- Preflight ran once, at 07:44:09Z, on all four payloads. The two g3 boots had started before it (07:37:22Z and
  07:42:59Z, their chain.log START lines), so both are UNGATED and their BLOCK is an after-the-fact reading. The g4mk
  boots started after it.
- **C10's BLOCKs are not about the payloads' contents.**
  - The g3 size BLOCK hits runtime's own B5TL1-g3 payload identically, and that payload ran on this seal.
  - The FDT BLOCK on CFP-g4mk is a false positive. The payload has the one real FDT at `0x14000` (3,518 B, version 17),
    as CF-g4mk has. Its second `d00dfeed` sits at `0xd73e9a` inside compressed data, with a totalsize of
    1,180,622,853, and is no FDT (checked here on the payload). The fixed check (dev `6daffca3824c`) reads GO.
- **The port images are in the port payloads,** by each payload's overlay manifest. CFP-g3 carries `cpugcc_r_base.native`
  `cff54b2a361867f2…` and CFP-g4mk carries `perlbench_r_base.native` `b4eb3e6f267f5f34…`, the port images built above.
  The captures print no image hash, so the identity rests on the payload hash.

**Grades.** Binding: v2 (`grade-spec-chain-v2.py` `b976d2884c8c42d5…`), with the registered stage roots; v1
(`4fa95eaf…`) beside it. The board lane graded the boots, and every row was rerun here with the same output.

| boot | stage root | v2 | v1 | v2 INBAND | `sublet: fault` lines |
|---|---|---|---|---|---:|
| CF-g3 | `c4/b5b2spec/board-stages` | `C=ok`, gcc PASS, rc 0 | the same | none | 0 |
| CFP-g3 | `nested/b5b2/ports/boardroot` | `C=ok`, gcc PASS, rc 0 | the same | none | 0 |
| CF-g4mk | `b5b2linux/g4mkroot` | `C=ok`, perlbench PASS, rc 0 | the same | 1 line removed | 0 |
| CFP-g4mk | `nested/b5b2/ports/boardroot` | `C=ok`, perlbench PASS, rc 0 | the same | 1 line removed | 0 |

- The two INBAND removals are whole `TRAPTEST: tick` lines (396.88 and 396.87) inside the perl hash block. v1's own
  filter also dropped them, so the two graders agree.

**Against the predictions.**
1. **Met, on both port boots:** smoke `C=ok`, and the one stage PASS.
2. **The time readings** (`up1 − up0` of the stage's `@@SPEC-END`, 0.01 s resolution, N = 1 per arm):

| stage | stock | port | port / stock | twin port / stock |
|---|---:|---|---:|---:|
| 502.gcc_r (t1.c) | 5.55 s | 5.83 s | 1.050 | 1.176 |
| 500.perlbench_r (makerand.pl) | 275.70 s | 279.75 s | 1.015 | 0.958 |

- gcc's difference, 0.28 s, is 28 resolution steps over a 5.5 s run.
- These are readings, one boot per arm, not cost estimates. The twin's ratios are an emulator's, and they did not
  predict these.

**Reading.**
- Both ports run on B5b-2 silicon in the campaign's form with B5TL1's answers, with no fault line in either. That is the
  first board run of each:
  - gcc's all-abc-bm-ls-tfl-phi port;
  - perlbench's full port (heads, bodies, hash entries and op slabs).
- They join Line B's Full + ports row. The share is the runtime and rtl lanes' to compute, with the per-event prices
  named by their source.

**Limits.**
- One boot per arm. Test-size inputs only: gcc's t1.c and perl's makerand.pl.
- The port's image identity is read from the payload, not from the capture.
- Both g3 boots are ungated. CFP-g4mk was preflight-BLOCKED by a false positive and ran on runtime's ruling.

### Pre-registration: CFP-B5TL1-g3 v2, gcc's port with a fail-stop CRENEW, on b42c8c644 (written 2026-11-26 19:59 host time (+08), 11:59Z, before any board boot of v2)

**Why v2.**
- On Line B a CRENEW can return an untagged 0, and then the old lifetime is left live. This happens when the slot is at
  its last generation and the table cannot grow. RTL at b42c8c644: `core/capstone_sublet_life.sv:487-488`, then
  `S_K_DEC0` at `:595-598`, with decision 1 at `:319-324`. The model: `target/riscv/op_helper.c:1190-1209` and
  `:1224-1228`.
- gcc's port (v1, `oneop/gcc-arms/all-abc-bm-ls-tfl-phi`) stores through the result at once at every renew site, and
  it cannot re-derive there. In `bitmap_elt_clear_from` a 0 would end the free chain silently, with the old lifetime
  still live.
- runtime asked for a fail-stop instead (agreed with this lane). v2's `sublet_renew` (`gcc-v2/sublet-port.h`,
  `31ca7f89…`) does three things on 0:
  - it ends the old lifetime with `sublet_release` (CREVOKE);
  - it writes `SUBLET-RENEW-FAILED` to stderr;
  - it calls `__builtin_abort`.
- **That release cannot itself fail.** CREVOKE checks the tag and the lifetime and takes no allocation decision:
  - in the model, `helper_cskill` (`op_helper.c:1169-1189`) never calls `capstone_alloc_slot`;
  - in the RTL, `:488` sends a CREVOKE, or a CRENEW with a generation to spare, straight to `S_K_PV_R`, and only an
    out-of-place CRENEW reaches `S_K_DEC0`;
  - after a failed renew the old lifetime is live, as above, so the release finds it.
- v2 differs from v1 only in `sublet-port.h` (`diff` against oneop's: that function and its comment).

**The CFP-g3 v1 row stays as recorded.**
- For its run to have met a 0, one slot would need a renewal out of place inside gcc's own process: at least 2^32 − 2
  releases of that slot (`GEN_LIM` = `32'hffff_ffff`, `capstone_sublet_life.sv:116`, b42c8c644's 2b bound; step 2a's
  2^30 − 1 does not apply to this seal), with the table also unable to grow.
- The trap kernel keeps one table per address space (`sublet-table.c:2`), and a fresh slot starts at generation 0
  (`cap_rev_table_abi.h:24`).
- The run took 5.83 s, about 1.46·10^8 cycles at 25 MHz. That is fewer cycles than the releases a 0 would need.

**The images** (`gcc-v2/build-gcc-v2.sh` `9a132f3b…`, built as `build-ports.sh` builds gcc-port: B5TL1's own
502.gcc_r tree, its 387 compile lines and `link.txt`, llvm-c9, the B5TL1 SDK; each of the 11 port sources is first
checked byte-equal to oneop's stock source):

| arm | what differs | strip.dom sha256 | scan | CDERIVE | CREVOKE | CRENEW |
|---|---|---|---|---:|---:|---:|
| gcc-port (v1, above) | — | `cff54b2a361867f2` | ILLEGAL=0 | 81 | 9 | 44 |
| gcc-port2 (v2) | `sublet-port.h` | `bd308d5ab7187be8` | ILLEGAL=0 | 81 | 52 | 44 |
| gcc-port2ff (control) | v2 + `-DSUBLET_RENEW_FORCE_FAIL` on every compile line | `1a9b370a6298307e` | ILLEGAL=0 | 81 | 16 | 1 |

- v2's 43 extra CREVOKEs are the fail-stop's release, inlined at the renew sites.
- In gcc-port2ff every renewal reads as failed without executing CRENEW. Its one CRENEW is the runtime's.
- The stages `gcc-v2/stages/v2` and `stages/v2ff` are the port stage (`ports/stages/502.gcc_r`) with only the image
  and plan.json replaced (`diff -rq`).

**Fixture** (`gcc-v2/build-bmfx.sh` `65348165…`): the port's bitmap fixture (`bmfx`), built three ways (v1, v2, v2ff),
all ILLEGAL=0. Modes:
- 0: a live element is read;
- 1: a freed element is read (a stale pointer);
- 2: an element's old pointer is read after it was reused;
- 3: a reused element is read through its new pointer.

**The twin requalification** (`gcc-v2/twin-gcc2.sh` `d62bdcc5…`, 2026-11-26, console log `932a4b6b…`):
- qemu-crv, Image `749fed26…`, at the board's memory, as `twin-ports.sh`;
- one boot runs the fixture (12 runs), then B5TL1F (v2ff's stage), then B5TL1P (v2's).

| build | mode 0 | mode 1 | mode 2 | mode 3 |
|---|---|---|---|---|
| bmfx-v1 | reads, rc 0 | faults, cause 25, rc 139 | faults, cause 25, rc 139 | reads, rc 0 |
| bmfx-v2 | reads, rc 0 | faults, cause 25, rc 139 | faults, cause 25, rc 139 | reads, rc 0 |
| bmfx-v2ff | reads, rc 0 | `SUBLET-RENEW-FAILED`, `Aborted`, rc 134 | the same, rc 134 | reads, rc 0 |

- v2 protects exactly as v1 does. v2ff stops at the first renewal with the line, in exactly the modes that renew.
- **Stages,** graded by `grade-spec-chain-v2.py` (`b976d288…`) with `SPEC_STAGE_ROOT=gcc-v2/twinroot`:
  - B5TL1P (v2): **PASS**, QSTEP rc 0, specdiff 1/1. The twin's run took 1.54 s (v1 1.47 s, stock 1.25 s, above).
  - B5TL1F (v2ff): the command's rc is **134**. Its in-band `.err` decodes to `SUBLET-RENEW-FAILED`, `Aborted`, and
    the kernel line reads `signal 6 has a handler, caught signals are not supported: killed`. The grader reads
    **MISS** (`rc!=0: 1`, specdiff 0/1), not NO READING. So a fail-stop on the board reads as a failed stage.
  - No `sublet: fault` line in either stage. The four in the capture are the fixture's modes 1 and 2.
- **Disclosed (an instrument slip):** the grade as scripted exits 2, `ERROR stage A0FX-gccbm is not in the expected
  list`. The grader refuses a stage it was not given, and the fixture stage was in the boot. The stubbed dry run had not
  reached the grader with the fixture stage in the log. The stage pair was graded on a copy of the capture with the
  fixture's QSTEP block removed (6,264 characters, `boot-nofx.log` `ba3a9f1c…`), and that is the grade above. The
  fixture's lines are read from the full capture.

**The board boot.**
- **CFP-B5TL1-g3v2:** the CFP-B5TL1-g3 payload with `stages/v2` in place of the port stage. runtime builds it on the
  same kernel and firmware, and checks its overlay against the stage. Its hash is recorded in an addendum before the
  boot.
- Optional, if runtime grants it: **CFP-B5TL1-g3v2ff**, the same with `stages/v2ff`. It checks that the fail-stop's
  line and rc reach a silicon capture.
- Seal: b42c8c644, read back by name before and after. runtime allocates board and slot. It pairs with CF-B5TL1-g3
  (`997e2fd6…`) only on the same board and seal.
- If the logs carry no preflight-gate line, the boot is reported **UNGATED**. The g3 payloads exceed C10's 20 MB
  budget (21.66 MB), as above.

**Predictions.**
1. CFP-g3v2: smoke `C=ok`, the stage **PASS** (QSTEP rc 0, specdiff 1/1), no `SUBLET-RENEW-FAILED` and no
   `sublet: fault` line.
2. CFP-g3v2ff, if run: the command's rc is 134, its `.err` holds `SUBLET-RENEW-FAILED`, and the grader reads MISS.
3. **No time prediction.** v2's `up1 − up0` against the stock pair's 5.55 s and v1's 5.83 s is a reading, N = 1.

**Grader.** `SPEC_STAGE_ROOT=<hdd>/nested/b5b2/ports/gcc-v2/boardroot python3 <hdd>/b5b2linux/grade-spec-chain-v2.py
<UART> <out> B5TL1-502.gcc_r`, with `boardroot/B5TL1/502.gcc_r` the boot's stage (v2 or v2ff). The boot holds one stage,
so the expected list is that stage alone.
- Controls in board form (`gcc-v2/grade-ctl/`), from the twin capture's stages renamed to the board name:
  - `pos` (B5TL1P) with `boardroot` reads PASS, exit 0;
  - `neg`, `pos` with one base64 character flipped in the `.s` output's in-band file, reads MISS, exit 1;
  - `ff` (B5TL1F) with `boardroot-ff` reads MISS (`rc!=0: 1`), exit 1.

**How each outcome reads.**
- **PASS:** v2 replaces v1 as the CFP-g3 row's image, and v1's reading stays beside it.
- **MISS with `SUBLET-RENEW-FAILED`:** a renewal failed on silicon. That is a finding about the table's growth, reported
  to runtime with the capture.
- **MISS without the line, or a fault line:** reported as itself, against the port's sources.
- **A silence stop:** no reading. The wedge snapshot goes to runtime.

**Limits.** gcc's t1.c only, one boot per arm. The fail-stop path is shown by the forced-failure control, not by a real
failure: a real one needs 2^32 − 2 releases of one slot, which no test input reaches.

**Addendum 1 (2026-11-26 20:03 host time (+08), 12:03Z, before either boot): the payloads.**
- runtime built both (`trapk/build-cfp-g3v2.sh` `9dc7d9eb…`, record `build-cfp-g3v2.done`). Each is CFP-B5TL1-g3's
  overlay with the gcc stage directory replaced, relinked as the campaign's port payloads were (fpga-final, keep-tags
  firmware with `FW_FAST_RDTIME=y`, 2 fast-path references).

| payload | fw_payload.bin sha256 | bytes | native in the overlay | plan.json |
|---|---|---:|---|---|
| CFP-B5TL1-g3v2 | `724f0027fdf5fe59…` | 21,662,216 | `bd308d5ab7187be8…` (gcc-port2) | `330efbb9…` |
| CFP-B5TL1-g3v2ff | `8fad069606cbb548…` | 21,662,216 | `1a9b370a6298307e…` (gcc-port2ff) | `3a580bb7…` |

- **Checked here.**
  - Both `fw_payload.bin` hashes and sizes.
  - Each overlay manifest against CFP-B5TL1-g3's differs in exactly two lines: the stage's `plan.json` and its native.
  - The native and `plan.json` extracted from each `overlay.cpio` have the hashes above.
- The boot's stage is named `B5TL1-502.gcc_r`, as the grader's expected list says.
- No twin replay of the payloads: the stages are twin-qualified above, and the kernel and firmware are CFP-g3's, which
  ran on this seal.
- The board lane slots both boots on board 2 after M-16's hbB. g3v2 runs first. Nothing else changes.

### Pre-registration: B5b-2 board boot "nested-corpus", SQLite then PHP engine-repros (written 2026-11-26 04:17 host time (+08), 20:17Z on Nov 25, before the boot)

**What it is.** Runtime's B5b-2 item 1 on silicon: the engine-repro corpus on the trap route, in one boot.
- SQLite: stages N1 (malloc) and N2 (pools).
- PHP: P10-P13 (asan-blind; arms stock, block, object, exact) and P14-P17 (temporal-asan-visible; the same arms).

It runs after `nested-cpymem`, whose @@UP lines give the first board/twin ratio. The board lane runs it.

**Platform.**
- Board 2's b42c8c644 session after nested-cpymem, as runtime allocated it. Board 3's offer is the fallback. The boot
  runs on exactly one board, and its readings carry the board ID.
- **The seal is the full B5b-2 seal b42c8c644, never its spatial-only variant.** On that variant every cause-25 catch
  is MISSED by design, so a corpus boot there would only be an explicit, labelled control. It is not this run.
- Runner: the board lane's run-one-e.sh (runner 5), with the board's own booking and readbacks.
- **The preflight state is part of the reading.** If the driver's first line reads `PREFLIGHT SKIPPED`, every verdict is
  reported as UNGATED.
- Image `<hdd>/nested/b5b2/corpus/payload-board4/fw_payload.bin`, sha256
  `540fec67c0a30b9e6cf53e856faceb1dfbf83629a2753e4c1294a580392dd600`, 21,657,608 B:
  - firmware: sbi-keep trap-route-keep-tags;
  - DTB: caplifive-b5b2-wide.dtb `3d223e80…` (`sublet.dom_wide_grants=1`, present in the bytes; memory 0x3c2d2000 B);
  - **board kernel** = the series at 6d4504f8b073 + the runtime lane's msghdr loader (unvalidated at registration),
    Image `d0142665…`. It is relinked from trapk/fpga/linux-6.1 with only `init/version-timestamp.o` recompiled
    (build.log CC lines);
  - initramfs `b0e469ff…` = base-rootfs `dd2dd8ff…` + overlay `ovl-board4` (overlay.cpio `ec03aeec…`);
  - trap-vexec = runtime/trap-route-tools 081c21bd1f0e.
- `ovl-board4` is `ovl-board2` (the twin-qualified overlay with lo up in the PHP stages) changed only in time limits and
  stamps, by `make-board-kit.py` (`ce218337…`) with `board4.spec` (`f008744e…`):
  - /init → trapk/fpga/init-ordered-bound (`6462764b…`), with per-stage bounds from /stages.bound;
  - the per-run watchdog goes from 300 to 900 s, the same on every arm;
  - an `@@UP <uptime>` line follows each run's end line, outside every reader's run text.
- **Stage bounds:**
  - N1 and N2, 7,800 s each;
  - P10-P13, 3,600 s each;
  - P14-P17, 4,500 s each.
- Each bound is at least 15× the stage's twin time without its hung runs, plus one full 900 s watchdog per hung run.
  Twin spans on twin-board4 are:
  - N1, 941 s, of which 900 s is case 11's watchdog;
  - N2, 41 s;
  - P10-P13, 939-946 s each, of which 900 s is case 30's watchdog;
  - P14-P17, 123-142 s.
  - So the worst bound need is P14-P17's 15 × 142 = 2,130 s against 4,500 s.
- **Boot bound: 49,200 s.** That is the sum of the stage bounds (48,000 s), plus 1,200 s for upload, boot and the
  smoke control.

**Twin reference** (`twin-board4`, this exact overlay on Image `749fed26…`, wide grants, `boot-long.py`):
- The lock was asked 02:30:20 and granted after 813.5 s; the run was done 04:13:28.
- Every one of the ten stages ended rc 0. Console log sha256 `9c38c8e375717550…`.
- `grade-corpus.py --emit` wrote 346 cells, `predictions.json` `0f613f2c3ffeec41…`.
- twin-board2 (the same overlay with the old limits) graded against these predictions reads 346 of 346 equal, exit 0.
- Twin per-run times, from @@UP:
  - the longest run that does not hang is case 70 (crash-161) at 58-78 s. It is an OOM kill of php-real-nested at the
    twin's 512 MB, and its cell is out-of-denominator from the corpus's pristine record (`SIGNAL:11`), whatever the
    run does;
  - next is case 21 (crash-075) at 30-50 s;
  - the hung runs are SQLite malloc 11 (`@@CASE-TIMEOUT`, rc 137) and PHP asan-blind 30 (`VM_EXIT_TIMEOUT`, four
    arms), each cut at 900.1 s.

**Predictions: every cell equals twin-board4's** (346 cells, `predictions.json`). Tallies:
- **SQLite:**
  - malloc 29 MISSED, 4 NO-READING (02 and 20 unattributed, 06 inconclusive, 11 infra);
  - pools **28 CAUGHT, 5 MISSED**, after the cause-24 rule moves 31 (statDecodePage), as registered at 21:42 on 11-25.
- **PHP asan-blind** (CAUGHT/MISSED/NO-READING):
  - stock 5/29/5, block 12/21/6, object 25/10/4, exact 29/4/6.
  - Across the four arms, the NO-READING cells are:
    - 15, 25 and 44 on every arm (out-of-denominator; 25 by its putenv precondition);
    - 30 on every arm (infra);
    - 03, 19 and 42, unattributed, three cells in all;
    - 37, not-reached, two cells.
- **PHP temporal-asan-visible:** stock 1/5/25, block, object and exact 6/0/25 each. The 25 NO-READING cells are
  out-of-denominator by the corpus record.
- **grade-corpus.py on the board's capture: exit 0.**

**Grader.** `python3 <hdd>/nested/b5b2/corpus/grade-corpus.py <UART capture> <new out dir>
<hdd>/nested/b5b2/corpus/platform-board.json <hdd>/nested/b5b2/corpus/twin-board4/predictions.json`
- Script hashes: grade-corpus.py `e2d80260…`, read-sqlite.py `8aac31a3…`, read-php.py `a2d71056…`,
  platform-board.json `0b5cfd85…`.
- It runs the readers with their own trap/eval judges, applies the cause-24 rule, and compares [verdict, reason] per
  cell.
- Exit codes:
  - 0: every cell equals its prediction;
  - 1: a cell differs;
  - 2: a reader found no reading, or a predicted cell has none (a boot that stopped part-way).
- Controls, run today:
  - twin-board4's own log gives 0 (346/346);
  - twin-board2's log gives 0;
  - the same log cut at P16's begin gives 2 (222 cells read, 124 with no reading);
  - twin-board vs twin-all gives 1, and a trimmed CPython log gives 2, as before.
- One control was void and was redone: a 4 MB cut of a 378 KB log is the whole log.

**How each outcome reads.**
- **Exit 0:** silicon's trap route, under the full seal, reads the corpus as the twin does, cell for cell.
- **A CAUGHT ↔ MISSED change in a cell** is a protection finding. It is reported with its fault line: cause, pc and
  symbol.
- **A cause-24 reading:**
  - outside pools 31, or in 31 outside statDecodePage, is listed apart, and its verdict stays the judge's (the rule as
    registered);
  - in a cell the twin never studied, it is flagged by the grader.
- **A change of reason only, inside NO-READING, on 30 (infra ↔ not-reached) or SQLite 11 (infra ↔ inconclusive):** these
  are the timing cases. Both pairs have been read on the twin across runs (twin-board, twin-all). It is a timing
  difference, reported as such, not a protection difference.
- **Case 21 (crash-075) cut by its 900 s watchdog:** at a board/twin ratio above about 18 it overruns. Its cell would
  then read a budget reading, not a verdict. The ratio is read from nested-cpymem first.
- **Case 70 running longer than on the twin:** the board has about 963 MB to the twin's 512 MB, so its OOM may come
  later or not at all. Its cell is out-of-denominator by the corpus record either way. Only its time can change.
- **Case 25 reading anything but out-of-denominator:** the putenv probe's platform reading differs from the twin's. It
  is reported as a platform difference.
- **32 (crash-097) not MISSED:** the loopback state differs from the twin's, read against the stage's `ifconfig` line.
- **A stage cut at its bound, or a silence stop:** that stage gives no reading (grader 2). The wedge snapshot goes to
  runtime, and the other stages' cells stand.
- **The board/twin ratio** is a reading of this boot, per run from the @@UP lines, against twin-board4's. It is not a
  prediction.

### Result: nested-corpus. The registered grader exits 1: all 114 differing cells are NO-READING "control-failed" in the three nested arms. The cause is the console, and with the console's splices undone every one of the 346 cells reads as predicted (board 2, 2026-11-26 06:57-11:08 host time, read against this pre-registration)

**Platform, as read back.**
- Board 2, booked by the board lane's runner and released (UNBOOK 200).
- The readback names `…-r80-r77_b42c8c644.bit` before and after the boot (rc 3 each, a name match): the full seal, as
  registered.
- Image `540fec67c0a30b9e…`, as registered.
- The runner's logs carry no preflight-gate line, so the boot is reported **UNGATED**.
- Capture `uart.txt`: sha256 `bee5a02abbbd8982…`, 339,350 B. All ten stages ended rc 0, and the console reached
  `TRAPTEST: done`.

**Grader, as registered** (`grade-corpus.py` `e2d80260…`, against `twin-board4/predictions.json`): **exit 1**. 346 cells
were read, 346 predicted, and 114 differ.
- Every differing cell reads `NO-READING`, reason `control-failed`, in the three arms whose controls must fault:
  - SQLite pools: 32 cells, predicted 27 CAUGHT and 5 MISSED;
  - PHP asan-blind nested-exact: 35 cells;
  - PHP asan-blind nested-pools: 35 cells;
  - PHP temporal nested-exact: 6 cells;
  - PHP temporal nested-pools: 6 cells.
- **No other cell differs.** SQLite malloc, PHP stock, PHP block (bundle-virtual-malloc), and every out-of-denominator,
  infra and unattributed cell read as predicted.

**Why the controls "failed": the board's console splices the kernel's fault line into the program's own line.**
- **SQLite pools.**
  - The control `uaf-mem5` faults on the board, exactly as on the twin: `sublet: fault cause=25 pc=0x3ff7c01f00
    address=0x3ff7f15e00`. bounds-mem5 likewise faults with cause 28 at `0x3ff7c01f38`, on the same pc as the twin.
  - But the console shows `CONTROL uaf-mem5[ 2243.335305] sublet: fault …` and then ` mark` on the next line.
  - The reader's `control_seen()` (wt-trap `engine-repros/shared/run-virtual.py:80`) requires the substring
    `CONTROL uaf-mem5 mark`. So it returns `none`, and the arm's every cell becomes control-failed.
- **PHP nested-exact and nested-pools.** The nested-bound probe faults as declared, again exactly as on the twin:
  `cause=28 pc=0x3ff7fe1cb4 address=0x3ff7fdb089`. On the console, the kernel line lands after `bounds=[…,0x3ff7fdb089) `,
  before the program's `tag=1`, so the judge reads `nested-bound-probe: DID NOT`.
- **The mechanism.** printk writes the serial console directly, while the program's own output still drains from the
  tty buffer through the slow UART. So a fault line can overtake the text the program printed just before it.
  - On the twin the UART drains at once, and twin-board4's log has no such splice.
  - Only fault lines move, so only arms whose controls or probes are meant to fault are hit.

**An unregistered re-read, with the splices undone** (`<hdd>/nested/b5b2/corpus/resplice/`).
- `resplice.py` (`fa6cf62e…`) finds a kernel fault line (`[ <secs>] sublet: fault …`) that starts after a non-empty
  prefix on its line. It excludes the `TRAPTEST: dmesg` and `@@PROBE-DMESG` echoes, which carry such lines by design.
  It moves that line after the rest of the interrupted line.
- It made **15 repairs** in the board capture, each printed:
  - the two SQLite pools controls;
  - the four nested-bound probes;
  - nine PHP case outputs: crash-140, crash-151 and crash-030, each in more than one arm.
- It made **0 repairs** in twin-board4's log, whose bytes it leaves unchanged.
- Registered grader, unchanged, on the repaired capture (`board-resplice.txt` `2acad9a4…`): **exit 0, 346 of 346 cells
  equal to the predictions**. The tallies:
  - SQLite: malloc 29 MISSED / 4 NO-READING; pools 28 CAUGHT / 5 MISSED;
  - PHP asan-blind: stock 5/29/5, block 12/21/6, object 25/10/4, exact 29/4/6;
  - PHP temporal: stock 1/5/25, and block, object and exact 6/0/25 each.
- **Negative control:** the repaired capture with the uaf-mem5 control's two fault lines removed reads that control as
  `none`. The 32 SQLite pools cells become control-failed again, exit 1. So the repair cannot make a control pass
  that did not fault. It only moves a fault line that is there.
- **This re-read is not the registered verdict, which stays exit 1.** It shows what the registered verdict measured: the
  console, not the protection.

**Reading.**
- Silicon's trap route, under the full seal, reads the corpus as the twin does, cell for cell, once the console's
  ordering is undone. Every CAUGHT ↔ MISSED cell, every cause and every NO-READING reason matches.
- The registered grader cannot say this on a board capture whose controls fault. A board reader has to tolerate a
  spliced kernel line, and a twin-qualified reader has never seen one.
  - Next instrument change, for the next board corpus boot: a resplice step before the readers, with this boot as
    its positive control. It goes in as a pre-registered instrument change.
- **Board / twin times** (from @@UP, readings):
  - whole boot 14,815 s against 5,374 s;
  - case 21 (crash-075) 208-256 s against 30-50 s per arm, inside its 900 s watchdog;
  - case 70 (crash-161) 803-860 s against 58-78 s. It is out-of-denominator either way.
  - The two hung runs, SQLite malloc 11 and PHP 30, read infra on both.

## perlbench item 3 on Line B: CRENEW for heads, bodies and hash entries, failing closed, on the trap-route twin (2026-11-26)

### Pre-registration: perl-fullren against perl-full, makerand.pl on the trap-route twin (written 2026-11-26 21:04 host time (+08), before the variant's runs)

**Question.**
- runtime asks for the CRENEW half of item 3 on the trap route.
- oneop measured all three variants' program side on 24 Nov (the sections above):
  - heads (5.36): 13.51 image instructions per removed CDERIVE;
  - bodies: 10.04;
  - hash entries: a net cost on suns.pl, because the bounds test runs at every issue and suns.pl frees few entries.
- oneop's table has no generations, so a renew there was a release plus an allocation. B5b-2's table has generations, so
  a renew normally stays in place. And the trap-route twin decodes CRENEW (qemu-crv `3a9fc39167`, `helper_csrenew`).
- What does renewing heads, bodies and hash entries at the free do to perlbench's instructions on Line B, against
  the current port, in the campaign's own build and stage?
- And is protection unchanged, including on the path where a renewal fails?

**A renewal can fail on Line B, so the variant fails closed.**
- A CRENEW at a slot's last generation needs a new identity, and the allocation decision comes before any lifetime ends
  (RTL at b42c8c644, `core/capstone_sublet_life.sv:487-488`, then `S_K_DEC0` at `:595-598`). The model works the
  same way (`target/riscv/op_helper.c:1190-1209`).
- In a table that can grow, the result is cause 30, and the kernel grows the table. In one that cannot, the result is
  an untagged 0 (`:319-324`, decision 1), with no record written and **the old lifetime left live**
  (`op_helper.c:1224-1228`).
- The trap kernel's table grows on demand. It stops growing only at the format's ceiling or when a page allocation
  fails (runtime: `sublet-table.c:54-58`, `:110`).
- So each of the three renew sites checks the result. On 0 it takes the port's own path:
  - heads: revoke and clear the issued entry;
  - bodies and hash entries: revoke, and return the segment's own pointer, so the next issue derives.
- **That revoke cannot itself fail there.**
  - CREVOKE checks the tag and the lifetime only, and it takes no allocation decision. In the model, `helper_cskill`
    (`op_helper.c:1169-1189`) never calls `capstone_alloc_slot`. In the RTL, line 488 sends CREVOKE straight to
    `S_K_PV_R`.
  - The old lifetime is live after a failed renew, as above.

**The arms** (`ports/build-renew.sh`, built as `build-ports.sh` builds perl-full: B5TL1's own tree, its `compile.txt`
and `link.txt`, llvm-c9 `29027126f39b`, the B5TL1 SDK, the one-operand shim):
- **perl-fullren** (`d5cb32d6f0bd75b0…`): perl-full plus three experiment scripts, never port scripts.
  - `renew/heads-renew.py` (``f0616af5…``): the 5.36 variant (oneop `0003`) in the 5.22 port's shape. The issued table tells
    a reissue apart, so no bounds test is needed.
  - `renew/bodies-renew-lb.py` (``5857aeec…``) and `renew/he-renew-lb.py` (``3bef5674…``): oneop's `bodies-renew.py` and
    `he-renew.py`, failing closed.
  - The defines are `-DPERL_CAPSTONE_SUBLET_{HEADS,BODIES,HE}_RENEW`, and `renew/renew-shim.h` (``378aa5b4…``) is
    force-included.
  - Op slabs are not renewed: oneop read that as a cost, and runtime did not ask for them.
- **perl-fullren-ff** (`c30ba1d7a9129481…`): perl-fullren plus `-DPERL_CAPSTONE_RENEW_FORCE_FAIL`.
  - Every renewal reads as failed (0, with the old lifetime left live) without executing CRENEW.
  - So every site takes its fall-back. This arm is the fixture's check of that path.
- **perl-fullren0**: the three scripts without the new defines. **It is byte-identical to perl-full** (`07738b77…`
  unstripped, `b4eb3e6f…` stripped), so the edits change nothing unless asked.
- Every image reads `ILLEGAL=0`.
- Static census (`scan_illegal.py`):
  - perl-full: CRENEW 1 (the runtime's), CREVOKE 15, CDERIVE 96;
  - perl-fullren: CRENEW 10, CREVOKE 15, CDERIVE 96;
  - perl-fullren-ff: CRENEW 1, CREVOKE 15, CDERIVE 96.
- **Lines of patch** (`diff -ruN`, build products excluded):
  - the port against B5TL1's stock tree: 639 added and 8 removed over 11 files (568 and 8 that are neither blank nor
    comment);
  - the variant against the port: 45 added and 0 removed in `sv.c` and `capstone_sv_segs.h` (40 code lines). The heads
    edit keeps the port's issue body under `#else`, which inflates the added count. The shim adds 11 lines (5 code).
- **These images never go to the board here.**

**Fixtures** (`renew/build-renew-fixtures.sh`, built as `build-fixtures.sh` builds perl-full's: the port's committed
`make-{heads,bodies,he}-fixture.py`, the arm's `perlmain.o` line, linked in `perlmain.o`'s place). One twin boot runs
`renew/stage-fx`, 56 runs:
- heads (modes 0-3): full, fullren and fullrenff, with stock-heads (perl-stock, whose heads carry no lifetimes) as the
  control;
- bodies and hash entries (modes 0-4): full, fullren and fullrenff, with fixture/bin's fullnr-bodies and fullnr-he
  (NOREVOKE) as the controls.

**The instrument, calibrated before this pre-registration.**
- oneop's `opcount.so` (`1144e04f…`, source `6decfe33…`) on qemu-crv. Both are QEMU 8.1.1.
- The boot script is `renew/boot-plugin.py`, runtime's `boot-long.py` with two changes:
  - it passes `-plugin`;
  - after `TRAPTEST: done` it waits for QEMU to exit by init's poweroff, because the plugin writes its file at exit.
- The runner is `renew/twin-renew.sh`: twin-ports.sh's twin, at the board's memory, with init-twin and one perlbench
  stage per boot.
- Calibration (2026-11-26, `twin-renew.sh` `db8eb818…`, `boot-plugin.py` `1d6c39c7…`): `cal-stock` (B5TL1's own g4mk
  stage) and `cal-full` (perl-full's port stage), one boot each. Both stages PASS under `grade-spec-chain-v2.py`.
  Image-region counts (opcount):

| boot | image instructions | CDERIVE | CREVOKE | CRENEW | twin run (s) |
|---|---:|---:|---:|---:|---:|
| cal-stock | 3,094,621,929 | 793 | 130 | 3 | 56.02 |
| cal-full | 3,209,053,644 | 1,943 | 311 | 6 | 56.28 |

  - The port costs 114,431,715 image instructions on makerand.pl (+3.70 %), and adds 1,150 CDERIVEs and 181 CREVOKEs.
  - runtime read 3,094,621,929 for the campaign's CF image on its own twin run, equal to cal-stock.
  - **So the saving is small.** Each renewal replaces at most one CREVOKE and one later CDERIVE. With at most 181 port
    releases to renew, oneop's 10-14 instructions per removed CDERIVE would make at most about 2,500 instructions:
    under 10^-6 of the image's count. It is readable only if the twin repeats exactly (prediction 5).

**Runs** (`renew/twin-renew.sh`, after this pre-registration):
- `fx`, the fixture boot;
- `ren1` and `ren2`, perl-fullren's stage (`renew/stages/fullren`, the port stage with the image replaced);
- `full2`, perl-full again, the same-day control.

**Predictions.**
1. **Answers.** Each stage PASSes under grade-spec-chain-v2.py (`b976d288…`) against `ports/twinroot`. A boot that does
   not PASS is not read.
2. **Fixtures.**
   - fullren and fullrenff fault in exactly the modes perl-full faults in, run by run.
   - bodies and HE: mode 0 reads, and modes 1-4 fault, as perl-full's did on this twin (`twin-fx`).
   - heads: mode 0 reads, and modes 1-3 fault.
   - Controls:
     - fullnr-bodies reads in modes 0-3 and faults in mode 4;
     - fullnr-he reads in modes 0-4;
     - stock-heads reads in modes 0-2, and its mode 3 is recorded.
   - If a control does not read where predicted, its fixture's protected modes are not read.
3. **Lifetime instructions** (image region, opcount; ren1 against full2):
   - CRENEW appears;
   - CREVOKE falls by the CRENEW count to within 0.1 % of it;
   - CDERIVE falls by at most the CRENEW count.
4. **Image instructions: ren − full between −2 and −14 per removed CDERIVE.** oneop read heads 13.51 and bodies 10.04,
   and hash entries cost.
   - Removed CDERIVEs are full2's count minus ren1's.
   - **Caveat: hash entries can flip the sign.** Their bounds test runs at every issue, and on suns.pl it outweighed
     the saving. If it does so here, ren − full is positive, and prediction 4 fails. opcount counts by region and
     opcode, not by site, so it cannot split the three sites. A positive difference is reported as the net of the three.
5. **Repeats.** ren1 and ren2 agree in image instructions to within 0.01 %, and full2 and cal-full likewise.
   - **Readability rule for prediction 4:** its saving is read only if both repeat differences, |ren1 − ren2| and
     |full2 − cal-full|, are under a tenth of |ren1 − full2|. Otherwise the twin's repeat noise covers the saving.
     Prediction 4 is then reported as not resolved, and only the lifetime counts of prediction 3 are read.
6. **Recorded, not predicted:**
   - kernel and firmware instructions, where a renewal's table work would show if the trap route charged it to
     software;
   - the stage's lowest MemAvailable, against perl-full's and stock's;
   - the run time (`up1 − up0`), a twin's.

**What is read.**
- The saving per removed CDERIVE, and its share of the port's own image instructions (full − stock, from the
  calibration).
- Whether any renewal failed cannot be read from opcount alone. A failure executes a CRENEW and a CREVOKE where a
  success executes only the CRENEW. So it would show as CREVOKE falling by less than CRENEW, and so would the
  untraced double renews oneop read (73 and 63). Prediction 3's band covers both.
- The fall-back path itself is shown by the fixtures of fullrenff.
- Not read: cycles. This is the twin.

**Limits.**
- Test-size makerand.pl only, the campaign's g4mk stage.
- One twin. The arms are experiments, not port scripts.
