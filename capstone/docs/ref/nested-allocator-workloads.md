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
