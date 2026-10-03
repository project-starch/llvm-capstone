# Supervised resume after a capability store-to-load: a bare directed test, pre-registered 2026-10-03, pushed before the boot

**Why.** Board boots supmon-c5q and supmon-c5u (`../supmon-2026-10-03/`) ran the SQLite speedtest under the FPGA monitor's
supervised CALL. In C5u the resume worked 212 times. Then the resume CALL after preemption 213 never completed:
- the re-arm stood (SUPA 0);
- no return and no escape came for 600 s;
- the commit pc stayed frozen at the instruction before the CALL, identical to C5q.
The escape before it landed directly after a capability store-to-load on the domain stack. In order: STC to granule
G, SB into the neighbouring granule, LDC of G, LBU through it. That is N = 1, and nothing else distinguishes it yet.

**The test.** `tests/sup-capstl.S` (built by `build.sh` with the ladder's committed harness; no compressed code).
- A supervised domain loops ITER = 4096 times on exactly that sequence over a 64-byte buffer.
- A small quantum makes the escapes land at every offset of the loop, and the monitor resumes every preemption.
- `-DINTLOOP` is the matched control. It is the same loop with STC/LDC replaced by sd/ld: only those two words and the
  lbu base differ, and the bytes stored and read back are the same.
- The readings in CAPPRINT order are below. A hang prints only SUPTEST BEGIN, SB1 and a dot per 256 escapes.

| # | reading | expected |
|---|---|---|
| 1 | csnodefree at the start | reported |
| 2 | csupstatus after the last CALL | 1 (kind 0: the domain returned) |
| 3 | iterations stored by the domain | 0x1000 (exactly once) |
| 4 | checksum, sum of (i & 0xff) | 0x7F800 |
| 5 | escapes | reported, > 0 |
| 6 | csnodefree at the end | reported |
| 7 | mcause | 0 |
| 8 | end marker | 0x5E5E |

Images (`images/SHA256SUMS`):
- intloop-q64: 480113b58ddb41aa
- capstl-q64: 092ec82d1ae818a1
- capstl-q47: 4097d317ec0e6a16
- capstl-q16: 2e28b7dc134bcad6

All four are 19,432 B, with board_rec at 0x80003c40.

**The session** (`run-capstl.sh`; one power cycle per image, so a wedge costs nothing in the next):
- the control `call-retpc` (a05ca464, PASS exact N = 4);
- intloop-q64;
- capstl-q64, capstl-q47 and capstl-q16.

`compare.py` gives the verdicts (checked on synthetic complete, miss, hang and no-data transcripts):
- COMPLETED: every value as above;
- MISS: a wrong value;
- HANG: the dot count says how far it got;
- NO-RESULT: the test never started.

**Readings and what they mean:**
- **intloop completes and capstl hangs.** The capability store-to-load before an escape breaks the resume. It is a
  silicon defect in the supervised switch, and its own repro folder goes to the RTL lane. Simulation of capstl-q16 has
  the trace.
- **All arms complete.** The loop does not create the condition. The candidates left are:
  - the real monitor's side of the resume: `__domcallsaves`' CPMP and S-CSR swap, cscratch := sp, sp := 0;
  - state accumulated over hundreds of resumes. Reading 6 against reading 1 shows a node leak, if any.
  The next variant adds the monitor's swap sequence to this loop.
- **All arms hang at a similar escape count.** Accumulated state, not the instruction sequence.
- **intloop hangs too.** The resume path breaks under a small quantum regardless of the domain's memory traffic. The
  quantum is the variable, and the ladder's q16/q64 resumes (74 each) are the comparison.

## RESULT, the hot-cache session (2026-10-03 22:12-22:17): every arm COMPLETED, so the loop does not create the condition
- Control: call-retpc a05ca464 PASS exact (N = 5).
- Every run: count 0x1000 exact, checksum 0x7F800 exact, the last event kind 0, mcause 0, csnodefree 0xFFFC at both
  ends (no node leak).

| arm | image | escapes |
|---|---|---|
| intloop-q64 | 480113b5 | 824 |
| capstl-q64 | 092ec82d | 1,540 |
| capstl-q47 | 4097d317 | 2,052 |
| capstl-q16 | 2e28b7dc | 6,148 |

- About 10,500 resumes, with not one hang.

**Why, and the next arm.** The RTL lane read the walks' path in the RTL (message, 2026-10-03):
- Every walk slot is a dcache LOAD.
- In wt_axi_adapter, a load's shadow-tag read waits in TAG_WAIT until every outstanding tag write has its B-response
  (`tag_wr_pend == 0`) and no tag read is in flight. A stuck counter hangs it forever, and the commit then holds the
  CALL at the head: the signature of C5q and C5u.
- That tag read happens only on an L1 MISS. In the hot loop the seal (944 B) and the 1 KiB save area never leave L1.
  In SQLite, 2M cycles of a multi-MB working set evict them before every walk.

**The eviction set** (`CAPSTL_SET=evict`; the hot images reproduce byte-identically from this source). Images:

| arm | image | what |
|---|---|---|
| evict-noploop-q100k | 7ef6603b | the domain sweeps 64 KiB one line per iteration, under a 100k-cycle quantum; the four memory instructions are addi/andi (the RTL lane's control) |
| evict-intloop-q100k | dddfaf24 | the same, with sd/sb/ld/lbu |
| evict-capstl-q100k | f2de4c8e | the same, with STC/SB/LDC/LBU |
| evict-capstl-q20k | fd6ea632 | a 20k quantum: more escapes, less turnover per quantum |
| mevict-noploop-q64 | 006e81fc | the MONITOR sweeps 64 KiB before every resume, so every RESTORE read misses; addi body |
| mevict-capstl-q64 | 125b7ab2 | the same, with the STC/SB/LDC/LBU body |

- EVICT runs use ITER = 2^20, so the checksum is 0x7F80000. MEVICT runs use ITER = 4096.
- Predicted escapes, reported only: about 600 at q100k, about 3,000 at q20k, about 1,500 at q64.
- No arm needs the hang to be "expected". All are predicted to COMPLETE under the null hypothesis.

**What the readings mean:**
- **A miss arm hangs and its noploop control completes.** Walk misses together with the domain's stores reproduce it.
  The dot count says after how many escapes. It goes to the RTL lane, whose TAG_WAIT detectors name the counter.
- **The noploop arms hang as well.** Misses alone suffice; the domain's stores are not needed.
- **Every arm completes again.** The board's real-monitor sequence is the remaining difference (CPMP and S-CSR swap,
  cscratch := sp, sp := 0), or a DDR/AXI timing the bare harness does not reach. The next step is the monitor's
  `__domcallsaves` sequence in this monitor loop.

## RESULT, the cache-miss session (2026-10-03 22:19-22:26): every arm COMPLETED again
- Control: call-retpc PASS exact (N = 6).
- Every run: the count and the checksum exact (0x100000 / 0x7F80000 for EVICT, 0x1000 / 0x7F800 for MEVICT), the
  last event kind 0, mcause 0, csnodefree 0xFFFC at both ends.

| arm | escapes |
|---|---|
| evict-noploop-q100k | 488 |
| evict-intloop-q100k | 891 |
| evict-capstl-q100k | 914 |
| evict-capstl-q20k | 4,575 |
| mevict-noploop-q64 | 537 |
| mevict-capstl-q64 | 3,950 |

- In all, about 22,000 bare resumes on silicon (hot plus eviction), and none hung.
- **Not yet shown:** that the sweeps EVICT the walks' lines. The D-cache is 8-way with 16-byte lines, and its
  replacement may be random. A 64 KiB sweep then leaves about (7/8)^16 = 12 % of any set's earlier lines in place.
  Until a latency probe shows the misses, this null is about "after a 64 KiB sweep", not "every walk read missed".
- From the RTL lane's simulation (memory delay 12): an LDC two instructions after an STC to the same granule reads the
  stale tag (S-07 in a three-instruction window), so the lbu through it faults with cause 24, with or without
  supervision. On the board the loop never faulted, so the window is shorter on silicon.

**The MSWAP set, pre-registered.** Every CALL, the first and every resume, is wrapped in the FPGA monitor's generated
`__domcallsaves` sequence:
- CPMP0..15 read, cleared and stored with STC;
- 8 more tag-setting STCs;
- mcause, mtval, stvec, scause, stval, sepc, sscratch, satp, 0x803 and cepc swapped out;
- cscratch := sp, sp := 0;
- after the CALL, all of it swapped back.
So each armed CALL's walks start behind a burst of tag writes, as in the real monitor. The expansion was checked:
CPMP 0x10..0x1f in order, STC offsets 0..240 / 256..368 / 464.

Images (board_rec 0x80015000):
- mswap-capstl-q64: 65ede31e
- mswap-evict-noploop-q100k: 662b0dc1
- mswap-evict-capstl-q100k: 1a039140
- mswap-mevict-capstl-q64: 11abd706

**Predicted:** COMPLETE under the null, with the same values as their non-MSWAP twins. If a MSWAP arm hangs while its
twin completed, the monitor's swap sequence is part of the trigger, and the dots say how far it got.
