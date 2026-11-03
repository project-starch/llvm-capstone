# Taking the LDC/STC lifetime check off the critical path (2026-11-03)

*The RTL lane, on the lead's optimisation plan, direction 4, relayed by the runtime lane. This note is written and
pushed BEFORE any RTL is written, as the design's pre-registration.*

## What costs the cycles (bitstream 3, e7af48457)

The ObjBounds record (`2026-10-30-objbounds-variant.md`) measured the per-access lifetime check against its removal:
- LDC: 3.96 cycles per op on one resident node (s6).
- STC: 4.67 cycles per op on one resident node (s7).
- LDC on a 1,024-node chase: 14.5 per op (s3), where the record read itself misses.
- Plain loads and stores: 0 on a hit (s1). Their verdict is the LSU's R-43 cache. A miss costs a replay of about 33
  cycles under memory delay 12 (s5's passes).

All of the LDC/STC cost is the DYN unit's `get_node_query_validity` (`capstone_dyn_unit.anvil:173-180`, called at
`:685` and `:753`). Two facts read in the source decide the design:

1. **The memory half of an LDC/STC already runs in parallel, and it is speculative.**
   - The issue stage dispatches LDC and STC to BOTH the DYN unit and the LSU in the same cycle
     (`issue_read_operands.sv:1333-1337`). The LSU gets rs1's register metadata (`:1183`, `:1243`).
   - The load unit completes the LDC without waiting for the DYN unit's verdict (`load_unit.sv` WAIT_LDC_RESP).
   - The consuming LDC's clear enters the store buffer's speculative queue like any store (`store_unit.sv:453-476`),
     and an STC's store does too. Commit drops both when the instruction carries an exception.
   - What is serial is the DYN unit's own chain: its checks, THEN the query (send, then receive), THEN the request
     to the load or store syncer, THEN the result. The query delays the writeback, not the access.
2. **The handshake alone costs about half.** Option A (0af4307fc) answered the query in ex_stage at once, with no
   record read, and still paid 1.98 cycles per LDC and 2.95 per STC. So a cache answered through the same channel
   saves at most half. To save the whole, the query must OVERLAP the syncer exchange, not just get faster.

## Rejected first: route LDC/STC through the LSU's R-43 cache

The idea is to extend the LSU verdict (`load_store_unit.sv:1366-1403`, today `fu inside {LOAD, STORE}` only) to
operations LDC and STC, and drop the DYN query as ObjBounds does. On a hit it costs 0, but it is rejected:
- **A miss becomes a replay of about 33 cycles in place of a query of 4 to 15.** The cache has 256 entries (64 sets
  × 4 ways) and is emptied at every namespace flush. s3, and perlbench's 1,131 to 16,030 live nodes per run (census
  logs), would get SLOWER.
- **It changes architectural readings.**
  - The LSU's 25 beats the DYN unit's exception at the capability writeback (`cva6.sv:1540`). An out-of-bounds LDC
    through a revoked capability reads 28 today and would read 25.
  - The LSU's tval is the effective address, while the DYN unit's is rs1's cursor (`capstone_dyn_ftval`).
  - A revoked base on an unmapped page reads the page fault today and would read 25.

## The design: overlap the query with the syncer exchange (DYN unit only)

In LDC and STC, after the DYN unit's own checks pass:
- Start the node query and the syncer request IN PARALLEL (Anvil `;`).
- JOIN both answers, then decide.
  - Validity 0: raise INVALID_CAPABILITY, exactly as today, and discard the loaded value. The LSU's speculative
    effects (the clear, the store) die at commit with the exception.
  - Validity 1: write back as today.

Nothing else changes:
- the order of the DYN unit's checks;
- which exception wins at the capability writeback, and every tval;
- the LSU, commit, and the rev-node unit;
- every other DYN op.

A failed verdict is the same exception from the same unit. Only its cycle moves.

What must hold, and how each condition is checked:
- **The invalid path leaves the syncers consistent.** Today an invalid LDC/STC sends the syncer an exception packet
  (`abort_accumulation_load/store`). In the overlapped form the syncer has already received the real request. The
  generated unit is checked for exactly one request per syncer per op, on every path. Directed arms then run
  back-to-back LDC/STC after an invalid one.
- **No speculative effect survives a 25.** The arms cover a consuming LDC through a revoked base (memory must still
  hold the tagged capability afterwards) and an STC through a revoked base (memory must be unchanged).
- **The verdict still fires.** `stale-access` must read 0x19 ×4, 0, 0x19 ×2, 0x77, 6, and `objb-stale-w` its bitstream-3
  vector. This is the ObjBounds witness inverted.

## Prediction (simulation, memory delay 12; written before any RTL)

**Model.** Per op, new = variant + max(0, q − a) + j. Here q is the query's latency from its send, a is the syncer
exchange's latency from its send, and j is the join's own cost (0 or 1 cycle). The variant is the no-query tree
(24e588199 / 603528b8b).
- On a resident node the variant already fits the access, and q is at most about 4. So:
- s6 (1,024 LDC on one node): **5,205 ≤ new ≤ 6,229**. The control reads 9,264.
- s7 (1,024 STC): **6,592 ≤ new ≤ 7,616**. The control reads 11,378.
- s2 (64-node chase, 1,024 LDC): **5,215 ≤ new ≤ 6,239**. The control reads 9,251.
- hpm-3 / hpm-4 (256 ops, the kit images): **1,381 ≤ new ≤ 1,637** and **1,820 ≤ new ≤ 2,076** (bitstream 3
  2,394 / 3,088). The stall field drops by at least 75 % of bitstream 3's excess over ObjBounds.
- s3 (1,024-node chase, record reads that miss) is the UNCERTAIN shape. The new value is at most the control's
  174,952 and at least the variant's 145,274.
  - What recovers it is the access's own miss latency overlapping the record read.
  - **Pre-registered: at least 25 % of the 29,678-cycle gap is recovered** (new ≤ 167,533).
  - Below that, the overlap is not reaching the record-read case, and the record says so.
- s1, s4 and s5 are equal to the control within 1 % on plain-load cycles: s1 16,770; s4 656 / 708; s5's churn
  663,679. s5's passes and s4 contain LDCs, so they fall between control and variant.
- Values:
  - Every CAPPRINT value equals the control's.
  - The 95-sweep reads the same readings, and its RVFI traces are 92 of 92 identical against e7af48457
    (`sweep-compare-rvfi.py`). Only cycles move.
- **If any value or trace differs, the design is wrong. It is a FINDING, not a tuning question.**
- Failure branch: if s6 reads more than 7,253 (at least half the gain lost), the join or the Anvil thread structure
  costs more than the model allows. Then the generated unit is read before anything else is tried.

## Gates, in order

1. The Anvil change only: `capstone_dyn_unit.anvil`, LDC and STC. Branch from objbounds-off-b 1782d20bd (e7af48457's
   tokens).
2. Directed arms on the new tree and on the control: hpm-3/4, objb-perf s1-s7, stale-access, objb-stale-w, the
   consuming-LDC and STC-through-revoked arms, and back-to-back LDC/STC after a 25.
3. `rtl-lint-gate.sh`, run paired with the control.
4. 95-sweep with the classifier, `sweep-compare-rvfi.py` and `sweep-compare-cycles.py`.
5. claim-auditor over the diff. The soft spots to attack:
   - the syncers on the invalid path;
   - a younger load forwarding from the speculative STC before the 25;
   - the Anvil thread join's lifetime of the loaded value.
6. A synthesis-only gate through the synth lane (40 GB guard), keyed by module stem:
   - anvil_dyn_unit own cells;
   - ex+issue summed;
   - mapping peak in class;
   - loop headers equal.
7. The full build through the synth lane.

**The pair.** Check on (this design) against check off (the sealed `caplifive_vp2c_603528b8b.bit`). Its tree is
untouched, so only ONE new build is needed.

## Result (2026-11-03): REFUTED. The design is withdrawn and C.5a is preferred.

**Run.** capstone-ariane ovl-check 715b21b4a, clean tree; vp9 `.sup/ovl1`, memory delay 12, R43_TRACE. Control: the
recorded runs on 1782d20bd. Every value equals the control's (stale-access 25 everywhere, objb-pcc-ctrl, r43-evict-live).

| shape | control | ovl-check | pre-registered band |
|---|---|---|---|
| s6, 1,024 LDC | 9,264 | **9,264** | 5,205-6,229 |
| s2 | 9,251 | **9,251** | 5,215-6,239 |
| s7, 1,024 STC | 11,378 | **9,361** | 6,592-7,616 |
| s3 | 174,952 | **176,394** (worse) | at most 167,533 |
| hpm-3, 256 LDC | 2,395 | 2,392 | 1,381-1,637 |
| hpm-4, 256 STC | 2,905 | 2,392 | 1,820-2,076 |

LDC gains nothing. STC recovers about 43 % of its gap. Every band is missed, and the failure branch fires.

**Why the premise was wrong.** This note's own first fact refutes the design: the LSU half of an LDC/STC is
dual-issued, so on bitstream 3 it ALREADY overlapped the query.
- When the query returned, the load syncer already held the LSU's result, and the exchange was instant.
- The query's own latency was the cost. Starting the exchange alongside it cannot remove that.
- STC's 2 cycles per op came from the store syncer's handshake, which was serial behind the query.
- Only a faster verdict, one that does not wait for the record read, removes the cost.

**The rejection above is withdrawn too.**
- "A miss becomes a replay of about 33 cycles" was read off s5's plain-load passes, a mixed reading. The external
  collaborator's directed arm (perf/c5a-cached-ldc, `c5a-cost.S`) measures 12.1 per LDC miss, 13.0 per STC miss and
  17.4 on the s3 chase.
- The three reading changes listed there are avoided by that branch's CAP_WB merge order. Its order is the LSU's
  exception, then the DYN unit's own faults, then 25 with the latched rs1 cursor as tval, then the replay marker.
- The verdict does not ride the LSU's exception path, so a page fault still wins.

C.5a (fe5aa8440) is being reproduced on the same arms, together with `ovl-invalid.S` (capstone-ariane ovl-check
b22ee7afd), before it is recommended for synthesis.
