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

## 88db40027's full build: sealed and graded (2026-11-25)

The board lane sealed `caplifive_ids55-native-tagcache-mlp-fpieee-hpmreplay-dynbypass3-dtlb64-shrinkfix-ic32k-r72-stcnoclaim-twc16k-merge-f2-hpmtag-lrscfix-tagoverlap6c-pagedtbl-r76-r75d-b5b2n-crev4b_88db40027.bit`.
- Size: 11,443,722 B.
- **sha256 `bdff45d0a96f68ad1ea3f2077c7ca350960047a1f394d21849bd4a003da58d9a`**, recomputed here from the copy in bitstreams/ and equal to the build's verdict.
- The build tree is at 88db40027, with no core change. Its synthesis rows equal 88db40027's own gate on every module (+0).

**Graded here: PASS on every pre-registered full-build key** (the bf8fc0cfb section's table; 904cd9c6b's seal in brackets):

| key | band | read |
|---|---|---|
| placed LUT | 149,038 +-3,000 | 149,188 (+1,150) |
| placed FF | 68,831 +-2,000 | 69,534 (+1,303) |
| routing | 0 errors, 0 unrouted, 0 partial | routed clean, 199,232/199,232 |
| congestion | <= 5 | placed level 4 (3); router: no hotspot, with the control log parsing to 5 |
| loops | 0, each toy firing | check_timing 0 in the report and on the routed dcp, TIMING-23 0, LUTLP-1 0, each toy 1 |
| clk_out1 WNS | [-1.47, +6.01] | +0.510 (+2.266), inside the band |

Reported:
- design WNS +0.177, TNS 0, WHS +0.054, 0 of 212,404 endpoints failing;
- slices 44,301 (86.95 %), RAMB36 115;
- rev_node on the worst 500: 223 paths / 211 cells (904cd9c6b 258 / 309; control issue_read_operands 480 / 480);
- the worst-20 family is fu_data_q cap_metadata -> fu_data_q / the scoreboard's cap_result, unchanged;
- routed rows beyond +-2,500: dyn_unit/_spawn_0 +2,786 and rev_node -2,878, the same re-homing as both gates.

The board lane's grade is <synth>/b5b2c4ff-88db40027-full-out/grade-board/grade-c4f-full.txt. This seal is the target of the crevoke4b-bare-2026-11-25 kit (PREREG 77f0e191b97f). It is the 4(b) candidate, not final.

**The aperture gate on 47d749153: read (board lane, 2026-11-25).** PASS 3/3:
- technology mapping 4,968.4 MB (-0.3);
- loops 0, each toy firing;
- core LUT 124,167 (+126, limit 126,541).

Sizing against the predictions:
- core LUT +126 (pred under +150) and FF +27 (pred about +34);
- load_unit FF +8 (pred +8);
- each wt_dcache_ctrl FF +3 (pred +4: one bit of each 4-bit copy merged with an equivalent register; Vivado merges only equivalent registers);
- the write buffer FF +11 (pred +11);
- wt_dcache's own row FF +3 (pred +3);
- 0 rows beyond +-2,500.

FSMs: state_q in wt_dcache_ctrl and __parameterized0 is inferred and encoded sequential in both .vds, so nothing was re-encoded. The post-synthesis timing estimate is UNAVAILABLE: the synth dcps carry no clocks. The led_stretch slack comes from the routed build. dcp sha16 47ed72426378903d.

The full build of 47d749153 alone was disarmed before it started. The semantics owner ruled the FLU's untagged-x[rs1] rule ungated (reference-exact) in the same next build. That combined hash takes its own synthesis-only gate.

## The combined hash: apertures + the FLU's untagged-x[rs1] rule (capstone-ariane b5b2-wedgeap-flu 28fcbb646), pre-registered before its gate (2026-11-25)

**The ruling.** The semantics owner ruled (2026-11-25): CINCOFFSET, CINCOFFSETIMM and SCC take the reference's CHERI rule on an untagged x[rs1] in every mode (capstone-qemu 9289114e06 op_helper.c :762-779).
- The FLU applied it only under the CSRUNV-era virtual-context flag, which is constant 0 under the Sublet format.
- So silicon raised 24 where the reference adds: B5T-g2's x264 and imagick faults.

**The change.** 28fcbb646 = 47d749153 + capstone_flu_unit.anvil's three gates dropped, plus the FLU's vctx channel field and sampling loop (capstone_unit.anvilh, ex_stage's hookup).

**The vctx audit.** It is in the commit message. The DYN unit's vctx is the live table-namespace flag. Its gated CDERIVE sites (:277, :286) are unreachable under Sublet: CDERIVE decodes to CDERIVE_S, whose refusals are unconditional.
- **CORRECTION:** my first audit table called them live, with an M-mode exposure. Withdrawn the same hour, before any edit.
- This is measured, not only read: b5b2-untagged's plain-M inexact child reads 29 on both trees.

**Readings** (memory delay 12):
- **b5b2-untagged:** all nineteen values as pre-registered on both trees. Its first cut's M arm was VOID (the child was exact) and was rewritten before the rerun.
- **Regression:** the B5b-2 / B5b-1 lists, 56 other tests, are identical to 47d749153 in values and cycles.
- **Lint:** Sublet gate PASS. By name identical to 47d749153's after Anvil's thread renumbering; ids55 the same.

**Synthesis-only gate** (base: 47d749153's gate, dcp sha16 47ed72426378903d):

| key | pass if |
|---|---|
| 1 memory | technology-mapping peak within +-505 MB of 4,968.4 MB, under the 40 GB guard |
| 2 loops | 'Found timing loop' 0, Synth 8-326 0, check_timing 0 (toy fires), LUTLP-1 0 (toy fires) |
| 3 core LUT | at most 124,167 + 2,500 = 126,667 |

Reported, not graded (predicted before the run):
- **The FLU** (flu_unit, homed under ex_stage_i or with its egress): the revived integer arms are an imm add, a two-operand add that may share CINCOFFSET's existing adder, and a result mux. Predicted under +300 LUT; FF -1 (vctx_q gone).
- **ex+issue** (the FLU egress lands in issue_read_operands / scoreboard, as with item 4): the stem sum within +-2,500 of 47d749153's. Per-module rows beyond +-2,500 are reported by stem.
- **Item 4's October history** (the same egress drove technology mapping to 94 GB): any single Vivado process above 9 GB is reported with the partition it was mapping.
- **Failure branch:** if key 1 or 2 fails, the partition and loop members by stem, and no full build.

**Full build on a pass.** 88db40027's keys (the bf8fc0cfb table), centred on 88db40027's seal:
- placed LUT 149,188 +-3,000;
- placed FF 69,534 +-2,000;
- routed clean;
- congestion <= 5;
- loops 0, each toy firing;
- clk_out1 WNS inside 904cd9c6b's null band [-1.47, +6.01].

Reported beside them: the led_stretch endpoints' worst slack (the apertures' cost, since the synth dcp has no clocks), the rev_node worst-500 count, and sha256.

This bitstream is diagnostic (the wedge apertures) plus 4(b) plus the FLU rule. It is a final candidate only if the wedge is also resolved (the semantics owner's ruling 1).

Its kit:
- **crevoke4b-bare-2026-11-25**, whose 17 vectors read identically on 47d749153 in sim (all 17, every position);
- **b5b2-untagged as a bare image** (to build). That is the B5b-2 kit's first exercise of this rule in U.

## 4(b) on silicon: AS PREDICTED, board 2, N = 1 (2026-11-25)

The crevoke4b-bare-2026-11-25 kit (PREREG 77f0e191b97f) ran in the semantics owner's granted sequence on board 2. The board lane's records are board/meas-pw 6e07527a8a98.
- **The control, 904cd9c6b resident, no flash:** 17/17 exact on the base column.
- **The flash of 88db40027's seal** (sha256 bdff45d0...8d9a, checked first, its name read back after the flash and after the loads).
- **The 17 on the seal: 17/17 exact on every E position.** That is 16 arms plus the control. Eight of the arms discriminate the
  two bitstreams: the crevint and crevsv arms other than kill.
  - No VOID arm.
  - Stop latency 1,130 cycles at each resumed-chunk stop on silicon (565 ticks at 2.0 cycles/tick), against the 15,000 bound. Sim read 3,692.
    The first stop, in the fresh O(n) scan, is 1,222 ticks = 2,444 cycles: recorded, not bounded.
  - Every recorded id and checksum equals the sim.
  - No sbdrain exposure in the five shapes.

Both runs were graded twice: by the board lane's grader, and here from the raw captures with an independent script whose positive control fires. The kit README has the full readings. Board 2 stays on 88db40027.

## S-35's mechanism in simulation, and the pattern pre-registered for the diagnostic seal 28fcbb646 (2026-11-25, before that seal boots)

**Reproduced.** b5b2-ldcexc2 (capstone-ariane b5b2-ldcfix, +SUP_AP_SWEEP, memory delay 12; predictions in its header before the run):
- **The setup:** protected U, a backlog of 16 + 1 stores through a capability, then at once an LDC through an untagged base (24 from the LSU's capability check).
- **The arm:** it runs to the cycle limit; WEDGE-WBUF reads "store refused: full=1, lookups done, no memory write pending".
- **The control** (the same LDC through the valid base): completes, 0x0 with 0 traps, in 33,579 cycles.
- **The misaligned variant** (b5b2-ldcexc, integer M) completes with cause 4 and one trap. That exception is registered once on the MMU path and the LDC stops re-presenting.

**The mechanism, read in the trace:**
1. The faulting LDC is not popped (load_unit SEND_TAG_LDC: kill, then IDLE), so it is re-presented every cycle as a capability-width read on load port 1 (RQDBG rd_req 010 held; 195/229 cycling IDLE <-> SEND_TAG_LDC).
2. Each such read collides with the write buffer's cache update (wt_dcache_mem bank_collision's rd_cap term), so the update is never acked (192: return pending 1, update req 1, update ack 0), and no entry evicts.
3. The buffer stays full (223: full 1, lookups done), the store port is refused, and the commit queue fills.
4. The older store at the commit head waits for room (commit_stage :627-630), so the LDC never becomes the head and its exception is never taken. No timeout applies.

**The silicon question.** Whether mcf's wedge (S-35) IS this mechanism is the diagnostic seal's question: B5T mcf re-run on 28fcbb646, with the snapshot tool reading the new bytes.

**Pre-registered pattern** (stretched reading: a 0 was 0 for 2^20 cycles; a cycling state reads the OR of its codes; IDLE 0 | SEND_TAG_LDC 9 = 9):

| switch | layout at 28fcbb646 | THIS mechanism reads | decisive |
|---|---|---|---|
| 192 | {wbuf return pending, cache-update req, its ack, write in flight, memory write req, its ack, ni_conflict, commit slot 0 valid} | 0xd1: bit 7 = 1, bit 6 = 1, bit 5 = 0 (the ack never), bit 4 = 1, bits 3-1 = 0, bit 0 = 1 | bit 6 = 1 with bit 5 = 0 |
| 223 | {CAPSTONE port (2) controller state[3:0], wbuf full, S-07 gran_hazard, lookup req, its ack} | 0x08 (port 2 idle, full, lookups done) | full 1, lookup req 0 |
| 209 | {load port (1) controller state[3:0], PTW port (0) state[3:0]} | 0x10 (READ \| IDLE = 1; PTW idle) | high nibble 1 |
| 229 | {load_state[3:0], rd_req port 2, port 1, port 0, 0} | 0x94 (load state 9; port 1 requesting) | rq1 = 1, state nibble 9 |
| 195 | {ldbuf flushed[1:0], ldbuf valid[1:0], load unit state[3:0]} | state nibble 9; the upper nibble recorded | state 9 |
| 219 / 220 / 221 | as on 904cd9c6b | the LDC's op (0xd7) / 0x8c / 0xf8, as S-35 read them | |
| 225 / 222 | as on 904cd9c6b | 0x80 / 0x00 | |

**Grading:**
- **AS PREDICTED:** 192 bits 6/5 = 1/0, 223 = 0x08, 209's high nibble = 1, and 229's rq1 = 1.
- **A DIFFERENT SHAPE:** any of those four differs. The table's other candidates then name it.
- **No wedge:** mcf completes. Then the trigger did not occur on that run, which is no evidence either way. If mcf passes, the FLU ungate may have removed the upstream fault, as the semantics owner predicts.

**Read order if a snapshot gets cut short:** 192, 223, 209, 229, 195, then 221, 220, 219, 225, 222.

**Prediction for the fixed seal** (the semantics owner's): mcf completes, or faults with a named cause and pc. A silent wedge there refutes the fix.

## R-80's fix (capstone-ariane b5b2-ldcfix a5c45a46d, on 28fcbb646): pre-registered before its gate (2026-11-25)

**The change:** one term in load_unit.sv's exception block, `if (state_q == SEND_TAG_LDC) pop_ld_o = 1'b1;`. A faulting LDC leaves the LSU as a faulting plain load does. Under CAPSTONE_DYN_BYPASS, the R-43 replay marker and the probe's timeout 25 take the same path. The commit message has the mechanism and readings.

**Readings:**
- b5b2-ldcexc2 wedges on 28fcbb646 and completes on the fix with one cause-24 trap. Its control is unchanged.
- 56 regression tests have identical values. b5b2-srvcost's measured costs move -2 and -1 cycles, its end -3 over 198 probes; sref and sref-gen move -2 cycles each.
- Lint: at baseline, identical by name.
- The claim-auditor found no lost writeback, no double pop, no new loop term, and one misplaced comment, which was fixed.
- UNRESOLVED: the NO_DYN_BYPASS control build does not elaborate at 28fcbb646 at all (issue_read_operands.sv:2109).

**Synthesis-only gate** (base: 28fcbb646's gate, dcp sha16 cf3055c8baa1251a):

| key | pass if |
|---|---|
| 1 memory | technology-mapping peak within +-505 MB of 4,969.8 MB, under the 40 GB guard |
| 2 loops | 'Found timing loop' 0, Synth 8-326 0, check_timing 0 (toy fires), LUTLP-1 0 (toy fires) |
| 3 core LUT | at most 126,258 + 2,500 = 128,758 |

Reported, not graded:
- load_unit: LUT within a few cells, FF +0 (one pop term, no new state);
- the load unit's FSM: not re-encoded;
- rows beyond +-2,500 by stem.

If key 1 or 2 fails: the partition and loop members, and no full build.

**Full build on a pass.** The keys of 88db40027's table, centred on 28fcbb646's seal when it exists, otherwise on 88db40027's:
- placed LUT +-3,000;
- FF +-2,000;
- routed clean;
- congestion <= 5;
- loops 0, each toy firing;
- clk_out1 inside [-1.47, +6.01].

Reported: the led_stretch worst slack and the rev_node worst-500 count.

**Silicon** (the semantics owner's predictions):
- mcf on the fixed seal completes, or faults with a named cause and pc. A silent wedge refutes the fix as S-35's remedy.
- The matched SPEC pair (the same images, 28fcbb646 against the fixed seal, one board): each row within 1 %. A row beyond that makes the registered-only park the follow-up, with its own gate.
- Kits: crevoke4b and wapflu, whose vectors are to be re-simulated on a5c45a46d before that seal is graded with them; b5b2-ldcexc2 as a bare image (to build).

## The diagnostic seal 28fcbb646: sealed and graded (2026-11-25)

The board lane sealed `caplifive_ids55-native-tagcache-mlp-fpieee-hpmreplay-dynbypass3-dtlb64-shrinkfix-ic32k-r72-stcnoclaim-twc16k-merge-f2-hpmtag-lrscfix-tagoverlap6c-pagedtbl-r76-r75d-b5b2n-crev4b-wap-flu_28fcbb646.bit`.
- Size: 11,443,722 B.
- **sha256 `925607df163bd8c8b96a2ccef84126478251124eb02fe31d301cfee18cc0d32e`**, recomputed here from the copy in bitstreams/ and equal to the build's verdict.
- The build tree is at 28fcbb646 with no core change.
- Its gate PASS 3/3 is recorded above, in the board lane's grade.

**Graded here: PASS on every pre-registered full-build key** (the combined-hash section's keys, centred on 88db40027's seal):

| key | band | read |
|---|---|---|
| placed LUT | 149,188 +-3,000 | 151,405 (+2,217) |
| placed FF | 69,534 +-2,000 | 69,497 (-37) |
| routing | 0 errors, 0 unrouted, 0 partial | routed clean, 200,863/200,863 |
| congestion | <= 5 | placed level 4; router no hotspot, its control parsing to 5 |
| loops | 0, each toy firing | check_timing 0 (report and routed dcp), TIMING-23 0, LUTLP-1 0, each toy 1 |
| clk_out1 WNS | [-1.47, +6.01] | +1.516 |

Reported:
- design WNS +0.177, TNS 0, WHS +0.024; slices 44,753 (87.84 %); RAMB36 115;
- rev_node on the worst 500: 299 paths / 229 cells;
- led_stretch worst setup slack +8.556 (88db40027's seal: +8.744), the apertures' only timing cost.

The grade is <synth>/b5b2waff-28fcbb646-full-out/grade-board/grade-waf-full.txt.

**Its session** (board 2, in the semantics owner's order): the flash, then the wapflu kit, crevoke4b's 17, the three g2, and B5T-g1b with mcf last. If mcf wedges, its snapshot is graded against the pattern pre-registered above (c7877d008be7).
- On this seal, crevoke4b cannot detect a missed flash: board 2 comes from 88db40027, whose column equals 28fcbb646's.
- The flash checks are the readback and wapflu's b5b2-untagged (the board lane's note).

## The final B5b-2 candidate: R-77's PRE fix on a5c45a46d (capstone-ariane b5b2-prefix b42c8c644, core = 7861abcfa), pre-registered before its gate (2026-11-25)

**What it adds to a5c45a46d.** The verdict probe's timeout counts only after the lifetime unit ACCEPTS the probe: `rvc_timeout = rvc_pend_q && rvc_sent_q && rvc_wait_q == RVC_PROBE_WAIT_POST`. The ~1M-cycle PRE bound is removed. The semantics owner ruled a spurious 25 on a live object a correctness defect, and accepted the trade: a unit that never accepts now hangs, with the access at the commit head.

**The evidence, b5b2-crevint -DCI_YLDC** (each cut pre-registered on the branch before its run; the readings are in the test header):

| tree | condition | reading |
|---|---|---|
| 28fcbb646 (no R-80) | a verdict-miss LDC during a CREVOKE walk | WEDGES (see below) |
| a5c45a46d (b0) | as is | completes, no 25; the LDC presents ONCE (948,921 times on 28fcbb646) |
| a5c45a46d (b) | sim-only 4,000-cycle PRE bound | the probe times out unaccepted, and the LDC faults 25 on the LIVE object |
| 7861abcfa (c) | the same bound, inert here | as (b0), with no timeout |

The 28fcbb646 wedge, in order:
- the replay-marked LDC re-presents on load port 1;
- the walk's record writes cannot drain: WEDGE-WBUF at cycle 106,085, and the CREVOKE never retires;
- the probe is never accepted, and its timeout about 1.05M cycles later does not end the wedge.
This is R-80's second trigger. It needs no faulting access, and it is demonstrated in RTL simulation; it is not an attribution of the mcf instance. Its LED bytes are 192 0xd0, 223 0x0a, 229 0x04, 195 0x99, read beside c7877d008be7's pattern and not instead of it.

Two cuts before each reading were VOID and are recorded:
- the cache hit because p's own CDERIVE fills it;
- the walk's forward scan read p, so its tap resolved the probe.
My first void rule was also mis-stated: probe-sent prints on ACCEPTANCE.

**Gates on 7861abcfa** (memory delay 12):
- the ldcexc trio as on a5c45a46d;
- the B5b-2 (40) and B5b-1 (17) lists identical in values AND cycles to a5c45a46d's, 57/57. The comparator's positive control against the 28fcbb646 run reads 4 differences.
- Lint: Sublet gate PASS with every counter at baseline, and both configurations identical by name to a5c45a46d's (1,628 and 2,498 warnings; the same diff against an older hash shows 229 lines).

**The claim-auditor** (read-only): PLAUSIBLE-BUT-UNPROVEN.
- (1) the false 25 removed, and (3) nothing else changed (POST, refusal record, counter wrap, acceptance cycle): SUPPORTED.
- (2) no reachable hang: SUPPORTED by a CORRECTED argument. My cited one was wrong: LCC sends a lifetime-unit query and is not in the empty-scoreboard gate.
  - The bound that holds: the unit never sits idle refusing the probe.
  - Nothing commits past the waiting access, so the 8-entry scoreboard bounds younger queries, with no speculative issue.
- The auditor's LCC-race discriminator (-DCI_YLCC) could not construct the race in two cuts, both VOID by their condition and recorded:
  - with the unit idle, the probe is accepted the cycle after the miss;
  - during a walk, LCC cannot dispatch.
- Hang classes that survive, each a pre-existing defect that used to end in a 25 at about 1M cycles:
  - H1, a unit read with no rvalid;
  - H2, the unit's write port starved;
  - H3, CSMINT's create response unacked after a flush, which the auditor argues is unreachable without proving it exhaustively.

**R-46 and the pending list** (closed with the semantics owner):
- R-46's write side IS reachable: a taken CBNZ with a tagged rs2 (branch_unit.sv :100-103; CBNZ is legal, decoder.sv:1563-1569).
- Every reader is unreachable under Sublet:
  - pc_cap_check and the PC probe need capmode (commit_stage.sv:423, :518), and capmode_d = 0 (csr_regfile.sv:349);
  - the switcher's pc_next is read only by CALL, RETURN and CSRUNV, all illegal (decoder.sv:1456, :1472);
  - the escape needs sup_active.
- CJALR is illegal (decoder.sv:1557), so pcc_tmo's acceptance wait is unreachable here and is inherited by any line with CJALR.
- The fixes for R-70, R-57 and R-79 are ancestors of a5c45a46d.
- No other RTL-ready item.

**Synthesis-only gate** (base: a5c45a46d's gate, B5B2R80-GATE PASS):

| key | pass if |
|---|---|
| 1 memory | technology-mapping peak within +-505 MB of 4,968.3 MB, under the 40 GB guard |
| 2 loops | 'Found timing loop' 0, Synth 8-326 0, check_timing 0 (toy fires), LUTLP-1 0 (toy fires) |
| 3 core LUT | at most 126,264 + 2,500 = 128,764 |

Reported, not graded:
- load_store_unit LUT within a few tens of cells of a5c45a46d's (one 20-bit compare loses its PRE arm), FF +0;
- no FSM re-encoded;
- rows beyond +-2,500 by stem.
On a key-1 or key-2 failure: the partition and loop members, and no full build.

**Full build on a pass.** 88db40027's key table, centred on a5c45a46d's seal if it has graded PASS by then, else on 28fcbb646's (151,405 / 69,497):
- placed LUT +-3,000;
- FF +-2,000;
- routed clean;
- congestion <= 5;
- loops 0 with each toy firing;
- clk_out1 inside [-1.47, +6.01].
Reported: the led_stretch worst slack and the rev_node worst-500 count.

**Silicon.** The a5c45a46d sessions stand as R-80's and ldcexc2's evidence. On this seal:
- the same kits, whose vectors are to be simulated on b42c8c644 before it is graded with them;
- mcf, by exposure (the board lane's column): a wedge refutes the fix; a fault with 25 is now itself a defect reading.

## R-80's seal a5c45a46d: sealed and graded (2026-11-25)

The board lane sealed `caplifive_ids55-native-tagcache-mlp-fpieee-hpmreplay-dynbypass3-dtlb64-shrinkfix-ic32k-r72-stcnoclaim-twc16k-merge-f2-hpmtag-lrscfix-tagoverlap6c-pagedtbl-r76-r75d-b5b2n-crev4b-wap-flu-r80_a5c45a46d.bit`.
- Size: 11,443,722 B.
- **sha256 `9b46b14fb6252873696acc9762d7aecbc53271567447460207737c2366f1e36e`**, recomputed here from bitstreams/.
- The build tree is at a5c45a46d with no core change.
- Its gate read B5B2R80-GATE PASS 3/3:
  - mapping 4,968.3 MB;
  - core LUT 126,264 (+6);
  - load_unit 417 LUT (+4) and 101 FF (+0), no FSM inferred or re-encoded.

**Graded here: PASS on every pre-registered full-build key** (the R-80 section's keys, centred on 28fcbb646's seal):

| key | band | read |
|---|---|---|
| placed LUT | 151,405 +-3,000 | 151,406 (+1) |
| placed FF | 69,497 +-2,000 | 69,500 (+3) |
| routing | 0 errors, 0 unrouted, 0 partial | routed clean, 200,931/200,931 |
| congestion | <= 5 | placed level 4; router no hotspot |
| loops | 0, each toy firing | check_timing 0 (report and routed dcp), TIMING-23 0, LUTLP-1 0, each toy 1 |
| clk_out1 WNS | [-1.47, +6.01] | +1.778 |

Reported:
- rev_node on the worst 500: 340 paths / 199 cells;
- led_stretch +9.412.

The grade is at <synth>/b5b2r80f-a5c45a46d-full-out/grade-board/grade-r80-full.txt.

Its sessions are R-80's and ldcexc2's silicon evidence, graded from the kits' `expected-vectors-a5c45a46d.json` and the ldcfix PREREG:
- board 3: flash, the kits, the Linux boot-3 image, the SPEC chain, then ldcexc2 last;
- board 2: the matched pair after its 28fcbb646 session and the ldcexc2 pre-mode control.

mcf is read by exposure (the board lane's column). A wedge snapshot is read by grade-mcfpat.py, read-walkvar.py and read-r8x.py; an R-80 pattern on this seal is "R-80's fix not holding".

## mcf on the fixed seals: graded by exposure, not per run (2026-11-25, with the board lane)

On silicon the mcf wedge is intermittent. On 904cd9c6b and 88db40027 together there are three runs, each cut at the 3,600 s stage bound: boot 5 wedged at about 2,766 s, and the other two ran clean.
- That is about 9,966 s of unfixed mcf exposure with ONE wedge: a rate of about 1 per 10,000 s, with a 90 % Poisson interval from 1 per 194,000 s to 1 per 2,100 s (computed by both lanes).
- Pooling the two bitstreams treats their LDC paths as one, which is unproven.
- The model exp(-T/tau) assumes a constant hazard across mcf's phases, which is unknown. Both clean runs passed 2,766 s, so the wedge is not tied deterministically to that point.

On a5c45a46d and b42c8c644:
- a WEDGE refutes the fix as mcf's remedy;
- a clean full-length run of T seconds weighs about exp(-T/tau) against "not fixed", and only cumulatively, across boards and re-runs. It is never a pass/fail on one run.

The board lane records each run's exposure: board, seal, mcf entry, end (completion, watchdog or silence), seconds, and outcome. Exposure is taken from each capture, not rounded. The diagnostic full-length B5T-g1b on 28fcbb646 is the first full-length unfixed calibration point. The discriminating pair is ldcexc2 (b5b2-ldcfix-bare-2026-11-25), but only on a board where the pre-fix seal reproduces its wedge.
