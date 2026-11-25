# B5b-2 4(b): interruptible CREVOKE in the RTL (parts 1 and 2), 2026-11-25

The design is llvm-capstone `capstone/docs/plans/2026-11-24-interruptible-crevoke.md` (spec-silicon d3a4a2259bd8);
the reference is capstone-qemu da49602d76 (`cap_rev_kill_step`). RTL: capstone-ariane branch `b5b2-crevoke4b`, on
904cd9c6b (the B5b-2 next bitstream, on boards 2 and 3).

## What was built

- **Part 1** (e3ba8aed0): CREVOKE finds the run's last record t with the forward scan, then releases the run from its
  END backward in chunks of KCHUNK = 64. Each chunk is checked walking back from t before anything in it is released:
  each member live, deeper than the operand, with a nonzero prev whose record names it as next. A failed check is 29,
  with earlier chunks left released and the header written. The chunk is then released as the walk reaches it and
  spliced out (p.next = stop, stop.prev = p, each re-read first as the reference re-gets them).
  - After a chunk, a pending interrupt ends the operation with status 2 (PARTIAL).
  - ex_stage shows the DYN unit status 0 and marks that CREVOKE's result with the R-43 replay cause, so commit
    re-fetches the same pc and the decoder takes the interrupt there (epc = the CREVOKE). The re-executed CREVOKE
    starts again from its checks.
  - The predicate (cva6.sv) is the decoder's own interrupt selection (without RVH), plus a debug request.
  - CRENEW is unchanged (the forward pass, list order, one execution), as in the reference.
  - Records leave in reverse list order, then the operand, so later CDERIVEs take slots in that order.
- **Part 2** (710998447): the saved position, option A under the semantics owner's ruling of 2026-11-25. The contract
  clause: between a stop and its restart, the operand's run changes only through lifetime instructions; a program
  that writes such a record any other way gets an unspecified result bounded by B1' (only records that pass the chunk
  check are released), B2 (it ends, causes 24-30) and B3 (writes only table words).
  - A PARTIAL keeps {operand id, namespace key, t, t's generation, stop}. A restart of the same id in the same
    namespace resumes from t if t re-reads live at that generation, deeper than the operand, with stop as next, and
    stop names t as prev; otherwise it scans afresh.
  - It is forgotten by CDERIVE, CRENEW, CSMINT's record, a CREVOKE of another id, the CREVOKE's end, an srevroot
    change, a CSTRAW dispatch, debug mode and a debug request.
  - No scan-phase stop (the ruling; a single kept id livelocks two contexts), so the first execution's scan is O(n).
- **Fixes:** ca1ea676d (the claim-auditor's corrupt-table case: the splice between chunks wrote stop from a stale copy;
  reproduced, then fixed by re-reading p and stop); bf8fc0cfb (the dead register that fix left; the lint's one name).

## Readings

- **Unit against the reference**: `sublet-life/crevoke-chunk-diff.py`, `sublet-life/crevoke-sv-diff.py` and
  `sublet-life-diff.py` (capstone-ariane, beside the unit testbench).
  - Chunk shapes (chain, fan, random trees at 1, 2, K-1, K, K+1, 2K+1, 3K+2 and 1,000 records; CDERIVE / CREVOKE between
    executions; a stop never restarted; broken links; the auditor's cross link): equal at K 64 and 256, and with a stop
    after every chunk at K 1, 7 and 64. The no-back-link mutant differs on the wrong-link case only.
  - Saved position, 12 in-contract cases at K 1, 7 and 64: equal, each restart on its expected path (resume; fresh
    after every item of the invalidation list; miss after a store into t). Out of contract, a depth store into the
    run: B1'-B3 hold.
  - Random streams: 82/82.
  - Latency at the unit (2-cycle memory, interrupt held, K 64): restarts 1,098 cycles on part 2 against 25,284 on part 1
    (4,096 records); the first execution scans 394,270 cycles at 65,532 records.
- **Core** (Verilator, memory delay 12, `+R43_TRACE`):
  - 50 in-tree Sublet tests on part 1, against 904cd9c6b's three core files: 47 identical in value (two ctxport arms
    +12 cycles), and b5b2-crevint's three arms as pre-registered in value and checksum. The base reads the list-order
    ids (the control).
  - Parts 2 and the fixes: every value identical to part 1's run.
  - b5b2-crevsv's five arms (what an S handler does between restarts): the pre-registered paths, two stops each,
    resumes none 2 / derive 0 / kill 1 / srev 0 / straw 0, no miss.
  - Stop latency: 4,042 cycles from an interrupt during a resumed chunk to the PARTIAL.
  - Instrument misses (my pre-registered relations, values unaffected, not widened after the fact):
    - crevint's timer-trap counts (7 against at most 4, 2 against 1): extra ticks taken at the re-fetched CREVOKE;
    - crevsv's checksum against crevint's: the page addresses move with the file's data;
    - crevsv's S traps against timer traps at the site.
- **Lint on bf8fc0cfb**: the Sublet gate passes, identical by name to 904cd9c6b's (LATCH 44, UNOPTFLAT 39,
  UNUSEDSIGNAL 831, 39 loop names). The ids55 build adds exactly two unused names, the new ex_stage ports.
- **claim-auditor** over e3ba8aed0 and 710998447: sound on the replay mark, the predicate, the saved position inside
  the contract, CRENEW and the header; one corrupt-table defect, fixed at ca1ea676d. Residuals it named, unchanged:
  - an older committed store into the table may not have drained before a lifetime op reads it (pre-existing);
  - ldc_lsu_ex outranks the replay mark at CAP_WB (pre-existing; LDC/STC bypass the DYN unit here).

## The store drain and the audit's residual 2 (2026-11-25, the semantics owner's rulings)

- **Residual 1, the ruling.** The reference reads the table as memory holds it, with no ordering clause for software
  writes, so the lifetime unit must see the hart's own committed stores. The trap-route kernel grows the table with
  plain stores (record page, directory entry, then CAPACITY), with no fence on its !SMP build.
- **904cd9c6b's exposure** (b5b2-sbdrain.S, five shapes, on 904cd9c6b's core files): no exposure found in five shapes
  at sim memory delay 12. In each, a CDERIVE or CSMINT retried after the growth stores saw the grown table, the
  kernel's sret path included.
- **The fix, capstone-ariane 609d7216c.** The unit accepts a lifetime operation only while the committed store queue
  is empty (cva6.sv no_st_pending_ex, registered in ex_stage). LSU probes are not held.
  - No deadlock: the stores waited for are older and committed, and drain without the unit.
  - A first version (36c71714c) also waited for the D$ write buffer. That cost 430 cycles a round of
    {CDERIVE, CREVOKE} even with no store, because the buffer holds the unit's own writes until DRAM acknowledges
    them. It was replaced, with the ruling's agreement.
  - Cost at sim delay 12: 16 rounds of {CDERIVE, CREVOKE} take 4,283 cycles against 904cd9c6b's 4,221; with four
    stores ahead of each round, 4,600 against 4,538.
- **The D$ property behind it** (b5b2-sbfwd.S, 88db40027, with sim-only D$ trace points). The D$ holds a 16-byte read
  while a plain store to its granule is in the write buffer. Shown on CAPACITY, the header's free list, a record, and
  a directory entry, each while its entry was resident, with the fence controls showing no hold. The walker's
  directory reads are 16-byte reads too.
  - The hold's view of the store buffer is the whole buffer (store_buffer.sv's shadow), so 904cd9c6b already read
    memory's state by the hold. The drain removes the dependence on the store-buffer half of it.
  - Word forwarding is not relied on and is unclaimed: its trace point never fired.
  - The lines relied on are named in the test's header.
- **Residual 2** (the LSU exception record outranking the replay mark): unreachable under CAPSTONE_DYN_BYPASS. LDC/STC
  results are forwarded as normal results, so the record is never captured: 0 captures and 0 merges over 62 tests,
  faulting LDCs included. The guard is in anyway: the record merges only into the result of the LDC/STC the DYN unit
  dispatched.
- **Core, 609d7216c:** 62 tests identical in value to bf8fc0cfb's (cycle readings move). Lint: the Sublet gate
  identical by name to 904cd9c6b's; ids55 adds the three new ex_stage ports as unused names.

## The synthesis gate on 88db40027 (4(b) + the drain + the guard): pre-registered (2026-11-25, before the request)

88db40027's core differs from 609d7216c only in SUP_TRACE blocks (sim only). The keys are bf8fc0cfb's below. The
reported bands add, for the drain: the unit's dyn acks gated by one more AND, and ex_stage's one flop and the guard's
AND, so the lifetime unit and ex_stage stay inside their bands below. bf8fc0cfb's synthesis-only gate (4(b) alone)
is the sizing read for 4(b); the difference between the two gates is the drain and the guard.

## The synthesis gate on bf8fc0cfb: pre-registered (2026-11-25, before the request)

GATE KEYS (base: step 9's gate on 904cd9c6b, dcp sha16 1eebdc9602d3f0bf):
| key | pass if |
|---|---|
| 1 memory | technology-mapping peak within +-505 MB of 4,954.8 MB, under the 40 GB guard |
| 2 loops | 'Found timing loop' 0, Synth 8-326 0, check_timing 0 (toy fires), LUTLP-1 0 (toy fires) |
| 3 core LUT | at most 122,911 + 2,500 = 125,411 |
Reported, not graded (predicted before the run):
- the lifetime unit (rev_node, capstone_sublet_life): +600 to +1,500 LUT and +450 to +700 FF. The new state is about
  575 flops: t, the chunk cursor, the back-link expectation, p and the release cursor (31 bits each), the chunk count
  (16), p's word 1 (64), the interrupt flop, and the saved position {v, 63-bit id, two 57-bit keys, t, stop, 32-bit
  generation};
- ex_stage: under +200 LUT (the replay mark, the predicate's flop, the straw dispatch OR);
- cva6 top: under +50 LUT (the predicate);
- core FF: +450 to +750;
- every other module: within the per-module floor.
If key 1 or 2 fails: the partition / loop members by stem, and no full build.

FULL BUILD on a pass (base: 904cd9c6b's seal, sha256 36e70855...1b08):
| key | prediction | failure branch |
|---|---|---|
| placed LUT | 148,038 + 1,000 +- 3,000 | outside: per-module placed rows against the seal |
| placed FF | 68,231 + 600 +- 2,000 | the same |
| routed | 0 errors, 0 unrouted, 0 partial | routing failure: placed area, congestion windows by module; NO WNS |
| congestion | <= 5, placed and router | above: the windows' modules |
| loops | check_timing 0, LUTLP-1 0, TIMING-23 0, each toy firing | any: members by stem |
| WNS clk_out1 | inside 904cd9c6b's null band [-1.47, +6.01] (+2.266 +- 3.74) | bad side: worst-path family and depth before any board use; good side: reported |
Reported: design WNS/TNS/WHS, slices, RAMB36 (115), worst-20 family, the count of rev_node cells on the worst 500 (POS
control: a known-cell query), sha256.

Flash: none until the seal and its kit. The silicon arms (a bare kit of b5b2-crevint / crevsv with the stop latency
pre-registered) follow from me before any flash. Launch is under your synth-guard rules.

## The synthesis gate on 88db40027: read (board lane, 2026-11-25)

PASS 3/3 on the keys above.
- Key 1: technology mapping 4,968.7 MB (+13.9).
- Key 2: 0 loops everywhere, each toy firing, the in-flight check PASS.
- Key 3: core LUT 124,041 (+1,130, limit 125,411).
- dcp sha16 7ae4662a01e6e3f5. The grade is at <synth>/b5b2c4f-88db40027-gate-out/grade-b5b2c4f.txt.

The full build launched on that PASS at 07:10:53Z.

Reported bands (ungraded), two outside their prediction:
- **core FF 47,080 (+1,303) against +450..+750.** The board lane read the two kept dcps by cell name (bf8fc0cfb 2869522d825285e0, 88db40027 7ae4662a01e6e3f5; totals equal each grade's row):
  - i_scoreboard's +516 FF is on unchanged RTL;
  - ex_stage_i outside rev_node and dyn_unit is 20,329 -> 20,599, all of it 277 new retiming copies (`_fret`/`_bret`): dyn_child_repr_q 1 -> 211 and dyn_child_q 128 -> 192;
  - plain FFs moved -5;
  - the drain adds exactly st_drained_q (+1), and ldc_lsu_ex* is flat (75).
  - So the miss is the flow's retiming, not the drain or the guard. The name lists are at <synth>/twc/rc/ffstem/{bf8,c4f}.ff.txt.
- **rev_node LUT 5,428 (-2,891), with dyn_unit +2,478 beside it.** This is the same re-homing as bf8fc0cfb's. The cone sum dyn_unit + rev_node + i_scoreboard + i_issue_read_operands is 60,493 (-471, inside +-2,500).
- ex_stage's own LUT row is -316 (inside "under +200"). rev_node FF is +500 (inside).

## The B5T mcf / M-16 wedge: observation-only apertures, pre-registered (2026-11-25, before any synthesis)

**The wedges.** B5b-2 Linux boot 5 (SPEC mcf on 904cd9c6b, board 3) went silent at uptime 2,799 s. P1's M-16 shows the same LSU head. Snapshots were read twice, 0 unstable:
- **Both:** sw=219 0xd7 (the LSU head's op, 215 = LDC) and sw=220 0x8c (lsu_ctrl.valid 1, lsu_ready 0, fu CAPSTONE_DYN).
- **mcf only:** sw=221 0xf8 (the commit queue 4/4 valid, store-port data_req 1, data_gnt 0, no_st_pending 0). M-16 reads 0x02 (no store pending).

The LED path holds each bit high for 2^20 cycles after any pulse (ariane_xilinx.sv's stretcher). So a 0 was 0 for that whole window: the LDC sat at the head with lsu_ready 0 throughout, and mcf's store port was never granted.

**RETRACTED the same day (rtl lane, to runtime, board and nested):** "load_state = 0 means the load unit was in IDLE with a full load buffer, so the LDC was never issued". sw=195 (load_state) is undriven on both lines: load_unit.sv declares `state_o` and never assigns it (904cd9c6b :109, 715bdd1fe :102). The lint baseline has carried it as UNDRIVEN. sw=229's high nibble derives from it, so it is dead too.

What still stands:
- The write buffer's grant (wt_dcache_wbuffer.sv p_buffer: rdy, !ni_conflict, !gran_hazard) reads no term of the load controller.
- A controller in WBUF_HOLD drives no rd_req (p_fsm's default).
- The LSU, load unit and D$ have no synthesized diff between 904cd9c6b and 88db40027.

**The write buffer's drain, by source.** A store is refused when the buffer is full with no hit, on S-07 gran_hazard, or on ni_conflict. An entry then drains in four steps:
1. **Tag lookup.** This is the only low-priority port of wt_dcache_mem's strict arbiter.
2. **Memory write.** The miss unit serves the lowest-index requesting port first. A read miss stalls on a colliding in-flight write.
3. **The write's response.**
4. **The cache update.** It is acked only when the selected high-priority read is tag-only or misses the bank, and a capability read always collides.

A port that requests every cycle can therefore starve step 1 or step 4.

**The patch.** capstone-ariane b5b2-wedgeap 47d749153, on 438a9c70b (core = 88db40027):
- every new bit is a registered copy, taken at its source module;
- nothing new reaches a path to commit, and the old bits keep their places.

| byte | content |
|---|---|
| 195 | {load buffer flushed[1:0], valid[1:0], load unit state[3:0]} (driven) |
| 209 | {load-port (1) D$ controller state, PTW-port (0) state} |
| 223 | {CAPSTONE-port (2) state, wbuf full, S-07 gran_hazard, wbuf tag lookup req, its ack} |
| 192 [7:1] | {wbuf return pending, cache-update req, its ack, write in flight, memory write req, its ack, ni_conflict} |
| 229 [3:1] | D$ rd_req of ports 2, 1, 0 |

**Predicted readings, per candidate** (written before any synthesis; byte layout above; the LED stretcher's rule: a 0 was 0 for 2^20 cycles, a 1 was 1 at least once):

| | candidate | 195 | 209 hi / 223 hi | 223 lo {full s07 lreq lack} | 192 [7:1] {rtrn wrq wra txv mrq mra ni} | 229 lo {rq2 rq1 rq0} |
|---|---|---|---|---|---|---|
| A | load-port controller parked in READ/REPLAY (no tag, no kill); lookups starved | any | 1/6/7 / 0 | 1 0/1 1 0 | lookups never acked | rq1 1 |
| B | LDC in R-29's WBUF_HOLD behind a store the buffer refuses | state 10 or 9 | 8 / 0 | full 1 or s07 1 | names the stuck drain step | |
| C | leaked load-buffer entries; LDC refused in IDLE | state 0, valid 11 | 0 / 0 | any | any | |
| D | port-2 controller parked, starving the lookups | any | 0 / 1/6/7 | 1 x 1 0 | | rq2 1 |
| E | S-07 refusal while the resident entry cannot drain | any | any | s07 1 | names the stuck step | |
| F | memory never acks the write | any | any | full 1, lreq 0 | mrq 1, mra 0, txv 1 | |
| H | a written-back entry's cache update never acked (a high-priority read on the bank each cycle) | any | any | full 1, lreq 0 | rtrn 1, wrq 1, wra 0 | the reading port's rq 1 |
| G | LDC waiting on a store-buffer page-offset match, store side stuck | state 3 | 0 / 0 | per the store side | per the store side | rq1 0 |
| I | PTW port parked | any | 209 lo 1/6/7 | lookups starved | | rq0 1 |

**Sufficiency check.** Each pair of candidates differs in at least one field:

| pair | split by |
|---|---|
| A / D / I | which controller nibble is parked, and which rq bit is set |
| B / C / G | 195 |
| F / H | 192: mrq 1, mra 0 vs rtrn 1, wrq 1, wra 0 |
| E | s07 1, with the drain step beside it |

**The one confound.** A port that requests every cycle reads rq 1, exactly as an occasional request does. The ack bit, lack 0 or wra 0, is what shows starvation.

**M-16** (P1, 715bdd1fe) reads the same 219/220 and an empty store queue (221 = 0x02). On this set it would land in A, C, D, G or I.


**Sim readings on 47d749153** (memory delay 12):
- Lint, Sublet gate: PASS (LATCH 44, UNOPTFLAT 39 = reference names, UNUSEDSIGNAL 831, UNDRIVEN 24 against 88db40027's 25, the one name gone being state_o). ids55: identical by name to 88db40027's.
- Observation-only check: sbfwd x6, sbdrain-fill and crevint-lat read the same values and cycle counts as their reference runs.
- Positive controls through the real LED mux (SUP_AP_SWEEP):
  - SL_SIM_WEDGE_LEAK: 195 = 0x30, every other byte 0. That is row C.
  - SL_SIM_WEDGE_NOTAG: 209 = 0x10 and 229 = 0x04. That is row A's controller half. The test had no store pending, so 223/192 read 0 and row A's write-buffer half is not exercised.
- b5b2-ldchold (the semantics owner's directed arm, for candidate H): 576 load-port holds, every one ended, 0 hold-request lines. With the M timer: 918 interrupts and 571 holds, all ended.
  - The longest hold, 3,398 cycles, waited on one write's response from memory, with no read port requesting.
  - That is consistent, by source, with the adapter's TAG_FILL of a fresh page.
  - H with a competing high-priority reader that collides every cycle was NOT constructed: no test drives a port that way.
- No stuck path was found.

**Synthesis-only gate** (base: 88db40027's gate, dcp sha16 7ae4662a01e6e3f5):

| key | pass if |
|---|---|
| 1 memory | technology-mapping peak within +-505 MB of 4,968.7 MB, under the 40 GB guard |
| 2 loops | 'Found timing loop' 0, Synth 8-326 0, check_timing 0 (toy fires), LUTLP-1 0 (toy fires) |
| 3 core LUT | at most 124,041 + 2,500 = 126,541 |

Reported, not graded (predicted before the run):
- **FF:** +34 new flops (195's 8, three controller copies 12, the write buffer's 11, rd_req's 3), each inside its own module. Core FF inside the floor; retiming moves more.
- **LUT:** under +150 in the core (the two LED mux banks gain 2 inputs and 3 wider bytes, plus 14 flop-input LUTs). Every module stays within the per-module floor.
- **FSM extraction:**
  - read the .vds for each wt_dcache_ctrl instance, the load unit's state_q and the write buffer;
  - a re-encoded state machine is recorded as a cost (the 715bdd1fe precedent: tapping a state register changed its extraction by +1 FF), not as observation-only.
- **Timing:**
  - the post-synthesis estimate on clk_out1, and on the led_stretch endpoints, against 88db40027's gate, as an ESTIMATE (ungraded);
  - routed WNS exists only with a full build. There it is graded like 88db40027's: inside 904cd9c6b's null band [-1.47, +6.01] on clk_out1, with the led_stretch worst slack reported beside it.
- **Failure branches:** if key 1 or 2 fails, the partition and loop members by stem, and no full build.
