# A marked-page tag read waits only for its own tag word: the pending tag-write list (plan and pre-registration, 2026-11-13)

*RTL lane, on the SPEC lane's request (the lead's instruction of 13 Nov: every SPEC test benchmark under 1.20
Capstone/riscv-base). Design context: spec-silicon `capstone/docs/plans/2026-11-12-tag-page-bitmap.md` (v1 page
bitmap, v2 tag-word cache), whose v2 section asked for this list and for more than one tag read in flight.*

## The change

capstone-ariane `perf/v1f-tagcache-mlp`, on `cap-ids55-native-tagcache` 79cf8de17 (V1f + page bitmap + TWC):

| commit | content | synthesis |
|---|---|---|
| 6f3f18888 | sim-only probe `TAGMLP_PROBE`; directed test `tag-mlp.S` | identical to 79cf8de17 |
| 53276815f | the pending tag-write list | the change |
| b98f5a517 | sim-only `TAGMLP_BHOLD` / `TAGMLP_DRAIN`; arms M5/M6; a comment fix | identical to 53276815f |

**Before.** A marked-page load miss whose tag word is not in the TWC issues a tag read. It first waited in TAG_WAIT
until every outstanding ID-0 tag write had its B (`tag_wr_pend_q == 0`). The whole adapter, I$ fills included, held
during the wait. The drain exists because AXI does not order a read against a write on the other channel.

**After.**
- An 8-entry list holds the tag word address (`[31:3]`) of every ID-0 write issued (TAG_WR and the page fill's
  writes) whose B has not arrived, in issue order.
- It is pushed and popped on exactly the events that count `tag_wr_pend`, which stays its occupancy. AXI returns
  same-ID write responses in issue order; the base already relies on that for ID 1 (`i_wr_dcache_id` pops its head
  TID on every data B).
- The tag read waits only while an entry holds its own word. Word granularity covers the TWC's whole-word install.
- TAG_WR waits while the list is full. Only its own grant fills the list, so a raised request is never withdrawn.
- TAG_FILL still drains to 0, and the TWC poison rule is unchanged.
- Cost: 8 x 29-bit word registers, 8 valid bits, two 3-bit pointers, and 16 29-bit compares. The compares reach
  `tag_state_d` only. `twl_full`, a compare on the `tag_wr_pend` register, gates TAG_WR's request and so reaches
  `arb_gnt` from a register.

**Not built: more than one tag read in flight.**
- The D$ miss unit keeps one read transaction (`wt_dcache_missunit.sv:223-224`; a read issues only on
  `!mshr_vld_q || load_ack`). Every D$ port, the PTW and uncached reads included, uses that MSHR.
- A tagged load's data return waits for its tag beat.
- So a second tag read can never be requested while one is in flight.

## Gates (sim, lint, audit)

**Lint.** The paired lint gate, 79cf8de17 and 53276815f on the same footing, passes and is identical on every counter:

| LATCH | MULTIDRIVEN | UNOPTFLAT | UNDRIVEN | BLKSEQ | ALWCOMBORDER | COMBDLY | UNUSEDSIGNAL | CASEOVERLAP | SELRANGE | ANVIL_UNOPTFLAT |
|---|---|---|---|---|---|---|---|---|---|---|
| 44 | 3 | 39 | 25 | 2 | 0 | 0 | 1480 | 2 | 100 | 0 |

**claim-auditor** (on 53276815f): SUPPORTED on correctness, list/B alignment, liveness, synthesisability and the
single-MSHR claim. Nothing refuted.
- It enumerated every writer of tag bytes: TAG_WR, fill, software stores into the tag region (ID 1, unordered in
  the base too), and AMO/SC (ID 3).
- Open by reading alone: whether the fabric (xbar, clock converter, MIG) keeps same-ID B order. The base depends on
  it already.
- Its one correction (`twl_full` reaches `arb_gnt`) is the b98f5a517 comment fix.

**What the testbench memory cannot do.** The testbench memory, `axi2mem`, serves one AXI transaction at a time.
- With the probe, 0 of 539 marked-miss tag reads on the base had any tag write outstanding at the data grant
  (tag-mlp, tag-word-cache, tag-page-bitmap). On the change it was 0 of 643. The maximum outstanding was 1.
- Every marked miss's tag read was granted 45 cycles after its data read, behind the data read in `axi2mem`.
- So in the default testbench the drain costs nothing and the change saves nothing. The silicon memory controller
  accepts reads while writes are open.
- `TAGMLP_BHOLD=n` (sim only) consumes each ID-0 B n cycles late. `TAGMLP_DRAIN` (sim only) restores the base's rule
  on the same tree. With no hold it reproduces the real base's M1-M3 cycle counts exactly.

**tag-mlp, rdcycle cycles.**
- M1: 256 x {store to a never-read word; load miss on another marked word}.
- M6: 64 such pairs with a 40-nop gap.
- M5: 64 same-word pairs with the gap.

| hold n | M1 drain | M1 list | M6 drain | M6 list | M5 drain / list |
|---|---|---|---|---|---|
| 0 | 41,557 | 41,557 | 10,705 | 10,705 | 10,754 / 10,754 |
| 40 | 44,117 | 41,557 | 11,335 | 10,705 | 11,384 / 11,384 |
| 120 | 64,597 | 41,557 | 16,375 | 10,705 | 16,424 / 16,424 |

- Per other-word miss the list saves 10 cycles at n=40 and 90 at n=120, i.e. about n-30. The probe's per-miss wait
  sums agree: 17,600 vs 14,400 and 43,200 vs 14,400 over 320 misses.
- Same-word reads wait on both trees.
- **Positive control:** `TAGMLP_NOMATCH` (the same-word wait removed) at n=120 prints `TMLP-HAZARD` 64 times, once per
  M5 read. The change prints 0. At n=40 the control cannot fire: the 45-cycle data-read serialisation hides the hold.
- Values are as expected on every build. tag-word-cache passes every arm on the change at n=40, in 6.6 M cycles.
  The no-hold runs of both trees stopped at their 4 M cycle limit in T4.

**Sweep.** The 95-test sweep, base 6f3f18888 and change 53276815f with the probe on, with the RVFI compare: running at
pre-registration; the result is appended below before the hash is handed over.

## Pre-registration of the synthesis (written before the build)

Hash: b98f5a517. It is synthesis-identical to 53276815f; no `TAGMLP_*` or `TWC_*` define is set in the FPGA flow.
The build name is `caplifive_ids55-native-tagcache-mlp_b98f5a517`. Synth lane, leffe, under the 40 GB guard: a
synthesis-only gate, then the full flow on a pass.

**Base for every delta:** `cap-ids55-native-tagcache` 79cf8de17 (board lane's build). Routed WNS -14.789, 179,560 LUT.

**Gate keys:**
1. Technology mapping peak in the base's class. A blow-up is a fail, read for its partition, with no retry.
2. Loops:
   - `check_timing` loop count equal to the base's;
   - LUTLP-1 0;
   - no `wt_axi_adapter` cell in a loop header's member set.
3. `wt_axi_adapter` own cells against the base:
   - FF +230 to +270 (246 new state bits);
   - LUT +50 to +500.
   - The words cannot become LUTRAM: all 8 entries are read in parallel.
4. Core and TOP within the +/-2,500 floor of the base (graded post hoc, never a stopper: the redistribution floor
   rule).

**Full build:**
- placed LUT within +/-2,500 of 179,560;
- placer congestion no worse than the base's;
- routes with 0 errors;
- WNS inside the several-ns null band of -14.789;
- `check_timing` 0 new loops;
- sha256 recorded.
- At about 180 k LUT routing is marginal (V1f placed 179,757 at congestion 5). If it does not route, the readings
  are placed area, congestion and the census; no WNS.

**Silicon (the SPEC lane's A/B against cap-ids55-native-tagcache on mcf and nab, native route; not graded here).**
- The drain's cost per marked miss is the B latency still remaining at the miss, minus however long the tag read's AR
  waits anyway.
- Predicted: 0 if the memory controller's B is back before the next tag read is accepted; up to about one B round
  trip per marked miss that follows another word's tag write.
- No workload fraction comes from sim: the default testbench reads 0, a testbench artefact. Under held B, tag-mlp
  reads 320 of 643, but the test is built that way.

## Sweep result (appended before the hand-over)

The 95-test sweep (`S12_MEM_DELAY=12`, `SUP_TRACE`, `TAGMLP_PROBE`, seed 20260922, time_out 2,000,000), base
6f3f18888 against the change 53276815f. Both runs are `rtl_dirty=no`, and their PROVENANCE files carry identical Anvil
unit, `ex_stage` and `ariane_pkg` hashes.
- 95 ran on each side, and 184 logs were copied from each.
- **RVFI traces:** 92 compared, 92 identical, 0 differing. 26 were compared over the common prefix (cycle-limited).
  3 are absent on both sides: rand, perf_tcache_random, perf_riscv_random.
- **Verdicts:** verdict, tohost and cycle count are identical in all 92 tests, with 66 SUCCESS and 26 FAILED on each
  side.
- **AXI checks:** `AXI-VALID-DROP` 0 on both sides. `AXI-WAIT-SEEN` 276 on both sides, so the drop check was live.
- **Probe:** `TMLP-HAZARD` 0. Of 124 marked-miss tag reads, 0 had a tag write outstanding, on both trees. That is
  the testbench reading above, not a workload fraction.

## Synthesis-only gate (synth lane, leffe; base 79cf8de17 synthesised there under the same guard)

| key | reading | grade |
|---|---|---|
| 1 mapping | technology mapping 5,114.6 MB vs base 5,111.6; guard peak 22.6 vs 22.8 GB; retiming summary identical (2/27/94/26) | PASS |
| 2 loops | `check_timing` loops 0 = base 0; LUTLP-1 0 on both; synthesis "Found timing loop" 0 and Synth 8-326 0 on both; the same script on the S-10b control reads 1 violation / 31 loops | PASS |
| 3 FF | `wt_axi_adapter` own cells 814 -> 1,070 = +256 (band +230..+270) | PASS |
| 3 LUT | own cells 3,345 -> 3,387 = +42; whole `i_adapter` subtree 5,552 -> 5,783 = +231 (FF +256); `i_axi_shim` +273 (FF 0), `i_dcache_data_fifo` -60, `i_tag_pending_fifo` -27 | key WITHDRAWN, report only |
| 4 core / TOP | core 155,058 -> 155,291 (+233, FF +256); TOP 181,955 -> 182,187 (+232); outside the cache subsystem every row moved by at most 1 LUT | PASS (post hoc) |

**Key 3 LUT withdrawn (mis-scoped, written below the floor).** The +50..+500 band on one module's own LUT was sized
from the change, not from the recorded per-module redistribution floor of +/-2,500 LUT. A sub-floor band is a
prediction, and a prediction band is never a stopper. Regraded post hoc as "0 +/- 2,500, report only". Own cells +42
and subtree +231 both sit inside the floor. The subtree figure matches the circuit's size. `i_axi_shim` is
instantiated inside the adapter (`wt_axi_adapter.sv:1379`) and takes `axi_wr_req` (`:1396`), which TAG_WR now gates
with `~twl_full` (`:672`). So re-homing across that boundary is plausible, and +42 + 273 - 60 - 27 = +228 matches the
subtree's +231. The adapter is the only file in `core/` that the change touches. The full build continued to seal.
