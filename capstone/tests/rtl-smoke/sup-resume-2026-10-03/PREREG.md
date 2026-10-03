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

**The eviction positive control (LATPROBE), pre-registered.** At the FIRST escape (after any MEVICT sweep, before
the resume) the monitor times one load of a seal line (+512, inside the walks' slot area) twice, with `cycle` around
a dependent load, through a NONLIN alias of the seal region. The readings are 9 (the first load) and 10 (the same line
again).

Images:
- latprobe-capstl-q64 (hot): b144d117
- latprobe-mevict-capstl-q64: 78167180
- latprobe-evict-capstl-q100k: 5469eb78

Predicted:
- **hot:** 9 and 10 both small and about equal, an L1 hit (single-digit cycles). This is the NEGATIVE control: without
  a sweep the line stays in L1.
- **mevict and evict:** reading 9 well above reading 10, a miss to DDR, then a hit. A random-replacement survivor would
  show as a hit, so one probe line samples it; it is not a proof for all 59 lines.
- **If both swept arms read a hit:** the sweeps do not evict, and the cache-miss null above says nothing about misses.
- Readings 1-8 as for the twins.

## RESULT, the MSWAP session (2026-10-03 22:27-22:37): every arm HUNG, the addi control included, all before 256 escapes
- Control: call-retpc PASS exact (N = 7).
- mswap-capstl-q64, mswap-evict-noploop-q100k, mswap-evict-capstl-q100k and mswap-mevict-capstl-q64 each printed
  SUPTEST BEGIN and SB1, then 0 progress dots, and timed out at 90 s.
- **Not yet attributable.** The swap code is new and untested in this harness. A hang with nothing after SB1 fits
  "the swap sequence makes the switch hang" and fits equally "the swap macro breaks the bare monitor": for example a
  trap with no reachable handler in capability mode, which also hangs silently. The dots cannot place it either:
  first CALL or the 200th resume.

**MSWAP debug set, pre-registered** (TRACE_CHARS: 'A' after each arm, 'C' just before each CALL, 'R' after each
swap-in, for the first CALL and the first 16 resumes):
- **mswapdbg-plain-noploop** (225b963a). The SAME swap macro around ONE plain, un-armed CALL. This controls the
  instrument. Predicted: it COMPLETES, printing `AC` then `R`, with reading 2 = 0 (no supervision event), and 3/4 =
  0x1000/0x7F800. If it hangs, the swap macro is the bug and the MSWAP hangs say nothing about the hardware.
- **mswapdbg-noploop-q64** (4faa2899) and **mswapdbg-capstl-q64** (3d76477c), armed. The last characters place the
  hang:
  - `ACR` repeating, then a stop after `C`: a resume CALL never returns (C5u's signature);
  - a stop after `A`: the swap-out itself;
  - no `R` after the first `C`: the first armed CALL never returns or escapes.
- Before them in the same session, the pending LATPROBE set (latprobe-capstl-q64 hot, latprobe-mevict, latprobe-evict)
  as pre-registered above.

## RESULT, the debug session (2026-10-03 22:39-22:56): the swap macro is VOID as an instrument; the latency probe corrects a premise
- Control: call-retpc PASS exact (N = 8).
- **MSWAP is VOID.** Its own control hung. mswapdbg-plain-noploop puts the swap macro around ONE plain, UN-ARMED CALL,
  and it printed `AC` and never `R`, exactly like the two armed arms (mswapdbg-noploop-q64, mswapdbg-capstl-q64).
  - So with this macro the first CALL never returns, supervision or not. The macro breaks a plain CALL in the bare
    harness, and the four MSWAP hangs above say nothing about the supervised switch.
  - Which part breaks it, and why the real monitor's generated `__domcallsaves` (the same CPMP / CSR / cscratch / sp
    steps on paper) does not, is OPEN. Candidates: the bare harness's CPMP or 0x803 contents, which differ from the
    FPGA monitor's.
- **The latency probe.** First load / second load of seal line +512 at the first escape:

  | arm | readings 1-8 | probe (cycles) |
  |---|---|---|
  | latprobe-capstl-q64 (hot) | exact | 27 / 7 |
  | latprobe-mevict-capstl-q64 | exact | 83 / 6 |

  - latprobe-evict-capstl-q100k: NO-RESULT (infrastructure: JTAG load_image stalled for 6 minutes, nothing ran).
  - **Correction: the hot negative control did not read two hits.** Even without a sweep, the walk-written seal line
    is not in L1 at the first escape: 27 cycles against a 7-cycle hit. The premise above, "in the hot loop the walks
    never miss", is WITHDRAWN. The hot runs also had walk misses, at least for this line, and the sweep only makes
    them dearer (83).
  - For the null this means the ~22,000 clean bare resumes INCLUDED walk misses behind the walks' own tag writes. The
    adapter's miss path under real DDR alone does not reproduce C5u's hang.

## CORRECTION to the debug-session result, and the clash-fixed pair pre-registered
- **"MSWAP is VOID" had the wrong reason.** The RTL lane ported sup-capstl.S unchanged into simulation (2026-10-03).
  - MSWAP_PLAIN does not hang in the macro. It TRAPS, cause 24, at SWAP_IN's second instruction, the LDC through s10.
  - The cause is my register clash. The domain's checksum register x26 IS s10, the swap-area capability, and an
    UN-armed CALL/RETURN restores only the eight exchanged registers.
  - On the board the trap read as "AC, no R".
  - So the plain control was broken by the TEST, not by the swap macro.
  - The ARMED arms are not touched by the clash: the escape's RESTORE brings the monitor's s10 back from the private
    area.
- **The armed MSWAP arms hang deterministically in simulation too.**
  - The armed CALL's SAVE and exchange complete.
  - The escape's SAVE, exchange and RESTORE of the monitor's 3..66 complete.
  - The monitor retires SWAP_IN's `ccsrrw sp, cscratch`, and then its LDC through the restored, tagged s10 never
    retires (~168 instructions, 4M cycles).
  - No adapter detector fires (TAG_WAIT, starved switcher request): the load is stuck in the LSU. The RTL lane is
    tracing the load unit and the store-buffer head.
  - This is a DIFFERENT point from C5u: there the commit pc stayed before the CALL. The two are not yet the same
    defect.
- **Fix:** under MSWAP the checksum lives in x29. Non-MSWAP images are byte-identical. The earlier mswap* images were
  built from the pre-fix source (45e3c7681d66).

**Pre-registered, the clash-fixed pair** (call-retpc control first):
- **mswapfix-plain-noploop** (ffd6f6db): the swap around one plain CALL. Predicted to COMPLETE and print `ACR`, with
  reading 2 = 0 (no event), 3 = 0x1000, 4 = 0x7F800. This is the instrument's control. If it fails, nothing else here
  stands.
- **mswapfix-noploop-q64** (6b3641c1): armed. Predicted: `AC`, then no `R` (the first escape's SWAP_IN LDC hangs, as
  in simulation), then the timeout. If it COMPLETES, silicon differs from simulation at this point.

## RESULT, the clash-fixed pair (23:01-23:07), and the swap bisect pre-registered
- Control: call-retpc PASS exact (N = 9).
- **mswapfix-plain-noploop STILL hangs on silicon:** `AC`, no `R`, timeout. So does mswapfix-noploop-q64. With the
  clash removed, the swap macro plus ONE plain, un-armed CALL never returns on silicon. The armed MSWAP hangs are
  therefore still not attributable to supervision.
- 0x803 is `offsetmmu` (reset 0, switch id 8), so zeroing it in the bare harness is a no-op.

**Bisect** (plain control, NOPLOOP, TRACE_CHARS; `SWAP_PARTS` selects the parts; the full macro, 15, reproduces the
mswapfix images byte-identically):

| image | parts | content |
|---|---|---|
| swappart0-plain (3def813d) | none | the plain CALL, nothing swapped. The control of the control: it must COMPLETE and print `ACR`. |
| swappart1-plain (22865b7b) | 1 | CPMP0..15 read, cleared, STC'd out; LDC'd and written back after |
| swappart2-plain (d0248af5) | 2 | 8 tag-setting STCs of s11 |
| swappart4-plain (beebecd8) | 4 | the 9 CSRs and cepc swapped out and back |
| swappart8-plain (a8e17393) | 8 | cscratch := sp, sp := 0, and sp back from cscratch |

- The disassembly was checked per image: 64 / 0 / 0 / 0 / 0 CPMP CCSRRWs, 0 / 16 / 0 / 0 / 0 STCs of s11, extra
  mcause accesses only in part 4, cscratch CCSRRWs only in part 8.
- Predicted: part 0 completes. The part that hangs (`AC`, no `R`) names what breaks a plain CALL on silicon. If parts
  1-8 all complete, the hang needs a combination, and the next step is pairs.

## RESULT, the single-part bisect (23:09-23:15): every single part COMPLETES; three-part combinations pre-registered
- Control: call-retpc PASS exact (N = 10).
- swappart0 (nothing swapped), 1 (CPMP), 2 (8 tag-setting STCs), 4 (CSRs + cepc) and 8 (cscratch/sp) each printed
  `ACR`, with count 0x1000, checksum 0x7F800, status 0 (a plain call has no event, as predicted), mcause 0.
  `compare.py` prints MISS only because it expects the armed status 1.
- So no single part breaks a plain CALL. The full macro (15) does.
- **From the RTL lane's simulation of the ARMED arm** (previous build): the instruction stuck at issue is SWAP_IN's
  `CCSRRW x0 <- cepc, t1`, waiting on t1. The LDC that writes t1 issued and never returned, so the hang is in the LSU's
  LDC path, not at issue and not in the adapter.

**Three-part combinations** (plain control, each omits one part):

| image | parts | omits |
|---|---|---|
| swappart7 (56f3d413) | 1+2+4 | cscratch/sp |
| swappart11 (113772e0) | 1+2+8 | CSRs + cepc |
| swappart13 (bfcbf47c) | 1+4+8 | the tag-setting STCs |
| swappart14 (0c266a6c) | 2+4+8 | CPMP |

Each image that hangs proves its omitted part is not needed. Each that completes proves its omitted part is needed.

## RESULT, the three-part combinations (23:15-23:24): part 8 (cscratch/sp) is NEEDED, and no single other part is
- Control: call-retpc PASS exact (N = 11).
- **swappart7 (1+2+4, no cscratch/sp) COMPLETES:** `ACR`, everything exact.
- **swappart11 (1+2+8), swappart13 (1+4+8) and swappart14 (2+4+8) HANG:** `AC`, no `R`.
- Part 8 alone completed in the single-part run, so the hang needs part 8 (`cscratch := sp; sp := 0` before the CALL,
  `sp := cscratch` after it) TOGETHER with at least one of the other parts: CPMP STC/LDC, the tag STCs, or CSRs plus
  cepc STC/LDC.
- **From the RTL lane's simulation:** the fixed plain control (the full macro, one plain CALL) COMPLETES there. So the
  plain hang is board-only, i.e. in the board's memory path. Their ARMED hang is a different, sim-visible state: the
  load unit is held in WAIT_FLUSH by a flush asserted continuously after the escape, and they are tracing which flush
  source holds it.

**Pairs, pre-registered.** Each puts part 8 together with one other part, plus the full macro. All carry 'K' printed
immediately after the CALL returns, before the swap-in:

| image | parts |
|---|---|
| swappart9k (5e50d977) | 8+1 |
| swappart10k (7676e303) | 8+2 |
| swappart12k (6eb6f7ac) | 8+4 |
| swappart15k (d618c78b) | all |

- `ACKR` means it completed.
- `ACK` and no `R`: the CALL returned, and the swap-in hangs.
- `AC` alone: the CALL never returns.

## RESULT, the pairs with 'K' (23:24-23:29): a print right after the CALL removes the hang, even for the full macro
- Control: call-retpc PASS exact (N = 12).
- swappart9k (8+1), swappart10k (8+2), swappart12k (8+4) and **swappart15k (ALL parts)** each printed `ACKR` with
  everything exact.
- The full macro without the 'K' (mswapfix-plain-noploop, and swappart11/13/14) hung every time.
- So the one difference between hanging and completing is a few instructions (the UART poll and store) between the
  CALL's return and SWAP_IN's first instruction.
- **Together with the combinations, the hang needs** the CALL returning, IMMEDIATELY followed by
  `ccsrrw sp <- cscratch` (part 8), followed by an LDC from DRAM (the swap area):
  - without part 8, SWAP_IN starts with the LDC and completes;
  - with part 8 alone, the next memory op is the UART poll, and it completes;
  - with a short delay after the CALL, it completes.
- **This is the compiler-generated `__domcallsaves` sequence in the real FPGA monitor after EVERY domcall:**
  `domcall(t0, t0); ccsrrw(sp, cscratch, x0); ldc(ra, sp, -16)`. It is the leading candidate for C5q/C5u. It is NOT
  yet shown to be their cause.
- The RTL lane's simulated ARMED hang has the same shape: the CCSRRW retires, then the LDC is held in WAIT_FLUSH by a
  flush that stays asserted.

**Pre-registered, no prints between the CALL and the swap-in** (call-retpc control first):

| image | build | predicted |
|---|---|---|
| mswapfix-plain-noploop (ffd6f6db) | the full macro, plain, nothing after the CALL | HANGS again (`AC`, no `R`), making the full-macro plain hang N = 2 |
| swap15-plain-fence (ed44c96c) | the same with one `fence` right after the CALL | completes if ordering (a pending store or a held flush) is what the print hid |
| swap15-plain-nop8 (df58e15a) | the same with 8 `nop`s right after the CALL | completes if a few cycles of delay suffice |
| swappart9-plain (31407f0f), swappart10-plain (48e130a4), swappart12-plain (f98db002) | part 8 with 1, 2 or 4, no K | swappart10 (8+2) has no LDC in its swap-in; the readings say which pairs are enough |
| swap15-armed-fence (12d21743) | the ARMED arm (q64) with the fence after every CALL | completes if the same workaround clears the supervised hang (`ACR` per resume, then COMPLETED) |

The disassembly was checked: CALL, then fence (or 8 nops), then `ccsrrw sp, cscratch`, then `ldc t1, 464(s10)`.

## RESULT, the post-CALL set (02:55-03:09): a fence clears the PLAIN hang and 8 nops do not; the ARMED arm hangs even with the fence
- Control: call-retpc PASS exact (N = 13).

| arm | result |
|---|---|
| mswapfix-plain-noploop (the full macro, plain) | HANG again (`AC`): N = 2 |
| swap15-plain-fence | COMPLETES (`ACR`, exact) |
| swap15-plain-nop8 | HANG. So it is not pure delay. The RTL lane's simulation: 4 nops do not clear it either. |
| swappart9-plain (8+1: ccsrrw, then the CPMP LDCs) | HANG |
| swappart12-plain (8+4: ccsrrw, then LDC t1, 464(s10)) | HANG |
| swappart10-plain (8+2: ccsrrw, then no DRAM load before the UART poll) | COMPLETES |
| swap15-armed-fence (armed, a fence after every CALL) | HANG at the first escape (`AC`) |

- **The plain hang needs** a CALL returning, then `ccsrrw sp <- cscratch`, then a capability/DRAM load. A fence or a
  UART print between the CALL and the ccsrrw masks it. Eight nops do not.
- **In the armed case the fence does not mask it.**
- **The RTL lane's simulation trace** of the armed hang: `flush_commit_o = flush_commit | dom_switch_busy_i` stays
  high because the switcher never leaves busy. Its last printed step is the set_ra write of the CALL's rd. The leading
  mechanism, still being traced, is that write and the CCSRRW's capability writeback landing on the same
  register-file port in the same cycle.

**The armed batch, pre-registered** (q64, NOPLOOP, ITER = 65,536 so that the checksum is 0x7F8000, and about 8,000
resumes per arm):

| image | build | expected |
|---|---|---|
| armdep-q64 (2c8acedc) | SWAP_DEP: sp holds a copy of the swap-area capability across the CALL, and the swap-in loads THROUGH the restored sp (`ccsrrw sp, cscratch; ldc t1, 464(sp)`). This is the real monitor's dependency shape. | if the dependency is why the real monitor survives ~200 resumes, this either completes or hangs after MANY escapes (the dots say how many): a bare reproduction of C5u's rare hang |
| armdep-fence-q64 (1e04098c) | the same + a fence after every CALL | the fence as a mask for the real-monitor shape |
| arm-print-q64 (e04f74db) | independent LDC, a UART print after every CALL | — |
| arm-csrr-q64 (aff41100) | independent LDC, a plain `csrr t1, cycle` after every CALL. This is what the ~22,000 clean bare resumes had after their CALL (`csrr t0, csupstatus`). | — |

The disassembly was checked: armdep's swap-in is `ccsrrw sp, cscratch` then `ldc t1, 464(sp)`; arm-csrr has
`rdcycle t1` right after the CALL; arm-print has the UART poll there.

## RESULT, the armed batch (03:13-03:22): the real monitor's dependent shape reproduces a RARE hang in bare
- Control: call-retpc PASS exact (N = 14).

| arm | trace | result |
|---|---|---|
| armdep-q64 (sp-dependent LDC after the post-CALL ccsrrw: the real monitor's shape) | `ACR` x16 (the trace covers only the first 16 resumes) | HANG before escape 256 |
| armdep-fence-q64 | the same | HANG before escape 256 |
| arm-print-q64 (a UART print after every CALL) | `ACkR` x16 | HANG before escape 256 |
| arm-csrr-q64 (a plain csrr after every CALL) | `AC` | HANG at the first escape |

- **The sp-dependent load survives tens of resumes, then hangs.** That is C5u's behaviour (212 good resumes) in a
  90 KB bare image. An independent LDC dies at the first escape.
- **A fence or a print after the CALL only delays the hang.** A plain csrr does not help at all.
- **The RTL lane's simulation trace (2026-10-04):**
  - the escape's switch finishes, busy drops, and the monitor's ccsrrw retires;
  - the LDC is then dispatched to the Capstone DYN unit and never comes back, with DYN ready low for the rest of the
    run;
  - there is no rev-node drain pending and no unacknowledged rev-node response, which rules out the R-27 class;
  - the load unit never receives a request.
  So the LDC is stuck inside the DYN unit, between the rev-node validity query, the load request to the LSU side,
  and the wait for the word. One handshake is never accepted. A scalar `ld` skips the DYN unit.

**The DYN-unit discriminator, pre-registered.** These are armed, parts 8+4, ITER 65,536. The images differ only in
the swap-in's cepc load, at both CALL sites: LDC `1d0d335b` against `ld` `1d0d3303`, plus the test ID character:
- arm12-ld-q64 (8027a661): `ld`. Predicted to COMPLETE if the stuck element is the DYN unit's LDC path.
- arm12-ldc-q64 (07de9fb7): LDC. Predicted to HANG at the first escape, as the independent LDC did.

**The dependent arm twice, with a dot every 16 escapes:** armdep-d16-q64 (d09491cd), two runs. The dot counts bound
the hang's escape number, so the two runs show whether it is deterministic or spread.

## RESULT, the DYN-unit discriminator (03:22-03:32): LDC hangs, `ld` completes; and C5u is a DIFFERENT signature
- Control: call-retpc PASS exact (N = 15).
- **arm12-ld-q64 COMPLETED:** 8,552 escapes, count 0x10000 and checksum 0x7F8000 exact, csnodefree flat.
- **arm12-ldc-q64 HANGS** at the first escape. The two images differ only in that one instruction, at both swap-in
  sites.
- So on silicon the capability load (LDC) after a supervised escape's switch is what hangs, and a scalar `ld` from
  the same address does not. This matches the RTL lane's simulation, where the DYN unit's load syncer waits for an
  LSU copy of the LDC that the LSU no longer has.
- **armdep-d16-q64, two runs:** each hung after 1 dot, i.e. between escapes 16 and 31. `compare.py`'s ">= 256"
  assumes the default mask. The first 16 resumes carry trace prints, and the hang comes soon after they stop.
- **C5q and C5u are NOT this signature.** Their stages-driver wedge reads, identical in both boots:
  - aperture 224 = 0x1f: excommit 0, **ldsync 0, stsync 0**, lsu/dyn/flu ready 1, **flush 1**, privM 1;
  - aperture 225 = 0x88: trace-buffer-empty 1, **domsw 1**, every wait flag 0;
  - commit pc frozen at `li sp, 0`, BEFORE the resume domcall.
  So the real monitor's hang is a domain switch that never finishes (dom_switch_busy holding the flush), inside the
  RESUME CALL, with no syncer waiting. In the bare LDC hang, the RTL lane's simulation shows the last retired
  instruction is the post-escape CCSRRW at CALL+4, so the switch had finished.
- **The two are siblings in the supervised switch, not one hang.** The C5u link to the bare repro is withdrawn.

**Pre-registered: the same apertures on the bare hangs** (`run_sup_bare_wedge.py` = the ladder's runner plus the
stages driver's post-run read of 255/224/225/230-237, switches parked at 0 afterwards):
- arm12-ldc-q64 and armdep-d16-q64: predicted the DYN signature. 224 shows ldsync = 1 and domsw (225 bit 3) = 0.
  The commit pc is CALL+4, the CCSRRW.
- mswapfix-plain-noploop (the plain-CALL hang): unknown. The reading itself says which class it belongs to.
- Instrument check: every read returns a byte (none UNREAD), and each commit pc lies inside its own image's text.
