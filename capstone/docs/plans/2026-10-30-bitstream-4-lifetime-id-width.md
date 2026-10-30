# Bitstream 4: the lifetime-id width, option B (55-bit ids) with option A (20-bit index) as the fallback

*RTL lane, 2026-10-30. The lead's ruling (via runtime): B has priority. A is the fallback if B fails its size gate, its full
build or the RTL grade. WNS is not an acceptance criterion. Both options sit on bitstream 3 (capstone-ariane e7af48457).
Synthesis runs in the synth lane on its user's approval. The flash is runtime's, on the lead's word in runtime's session,
after the RTL lane's PASS on the sealed build.*

## Option B: ids55, capstone-ariane 325966ee4 (origin/rtl-ids55; afd35e449 on top adds tests only, core/ identical)

The id is {generation[29:0], slot[24:0]}. In memory lo = slot << 39 | addr[38:0]; in a register the cursor is sext39.
Load rule (`core/load_unit.sv:831-841`): an integer or FP load of any width (LD/LW/LWU/LH/LHU/LB/LBU, FLD/FLW/FLH/FLB)
of bytes in a tagged granule's FIRST doubleword reads those bytes of sext39. LDC (decoded in ex_stage), the switcher's
rows, LR and AMOs read raw, and so does an untagged granule. This is the ISA's proposed rule 1
(`design/virtual-capstone/isa.md`), with one difference: its "misaligned included" has no reading on this core, which
traps misaligned loads (Q-34). The record is word1 = {flags[62:60], gen[59:30], parent[29:5], depth[18:14]@4:0},
word0 = {depth[13:0]@63:50, next[49:25], prev[24:0]}. rev_node_t is 137 bits.

**Taken over from runtime at 567e6ee02. Four defects were found and fixed before any synthesis:**
1. ex_stage unpacked every tagged load, so a plain `ld`/`fld` of a granule's SECOND doubleword was projected. The -DGEN_HI
   arm read 9 where 41 is right. Fixed in c62ddda04: load_unit gains `result_ldc_o`, the LSU pipe carries the tag for LDC
   results only, and ex_stage's data_wtag is 0.
2. The rev unit's node writes were TAGGED, so a plain `ld` of table word 0 or of a header was projected. Fixed in c62ddda04.
3. CSNODE's status was read from cr[60+:2] of a 112-bit create_res. Fixed in d22b130c1 (cr[110+:2]); arms -DCSNODE_HI
   (ancestor 262176) and -DCSNODE_FULL.
4. ariane_pkg had forward references, which Verilator does not see. Fixed in d22b130c1 by moving the CAP_* localparams and
   typedefs above the first struct.

**Runtime's two rulings in 3e3f58244, and one guard left out:**
- A physical slot >= 65,536 reads DEAD: the address uses slot[15:0], and VALID is forced from flops (arm -DPOOL_PROBE).
- A context scalar read returns raw. RETRACTED in 325966ee4. copy_step must carry the REGISTER cursor, as QEMU does
  (capstone_supervisor.c:427/:454). The corrected rule: unpack if the read is a capability row OR a first doubleword
  (arm va-runv2 -DCAP_A1).
- R3, the virtual-arm out-of-range guard (the RTL lane's), stays OUT: the only capacity latch is loaded on allocations.

**Gates on 325966ee4:**
- Lint s12 PASS at REF: LATCH 44, MULTIDRIVEN 3, UNOPTFLAT 39 (names identical), UNUSEDSIGNAL 1479 (REF 1480; +48
  attributed by name), every other counter at baseline.
- 95-sweep vs e7af48457: 92 identical, 0 differ, 3 missing on both sides.
- 16 wide-record arms are EQUAL to their narrow builds on e7af48457, value by value.
- The va-runv family (23 builds) is EQUAL.
- GEN_HI 41, CSNODE_HI 0x40020, CSNODE_FULL 30, POOL_PROBE 25 and CAP_A1 read as pre-registered.
- The CSNODE and pool arms read the broken values on the pre-fix trees, so they are positive controls.
- Two claim-audits: a synthesis-risk audit of 567e6ee02 (refuted it: the four defects) and a delta audit of
  567e6ee02..3e3f58244 (found the copy_step break).

**Synthesis-only size gate (synth, vs b3's own synth checkpoint, graded by stem):**

| Key | Reading | Band | Verdict |
|---|---|---|---|
| Mapping peak | 5,093.4 MB (b3 5,004.9); run peak 22.87 GB | 4.8-6.5 GB; < 30 GB | PASS |
| Loops | 8-326 0, timing-loop headers 0, check_timing 0 (controls fire on vp3) | 0 | PASS |
| core | +6,962 LUT / +3,654 FF | +3,000..+12,000 / +2,000..+5,000 | PASS |
| ex+issue | +8,130 / +2,850 | +2,500..+10,000 / +1,500..+4,000 | PASS |
| pmp_data_if FF, csr_regfile FF, LSU LUTRAM | +389, +556, +400 | +400 +-50, +400..+700, +350..+450 | PASS |
| presence | 880 / 55 x3 / 25 / 25 / 14,080 bits / 90 | the widening survived | PASS |

- Two module rows are outside their bands. Both sit in stems that passed, and are attributed at pointer level.
- capstone_rev_node: +3,762 LUT against a +500..+1,500 band. A structural proxy on the generated unit puts the unit's own
  growth at about +20 % (mux bits 9,071 → 10,846; register bits +167, near synth's FF +211). The remainder is consistent
  with redistribution inside ex_stage, whose own total is +22.
- LSU, excluding load_unit: 94 over its band. In source, i_pipe_reg_load is a bare shift_reg, so its +1,455 is logic
  homed or retimed into it.
- Routing-risk branch: "above ~+7,000 → to the lead before the full flow". The reading, +6,962, is 38 under, so the branch
  did not fire by letter. It was reported to the lead anyway; the standing ruling covers a routing failure.

**Full build, sealed: PASS on every pre-registered key.**

The keys were pre-registered in the GO message to the synthesis lane before the build launched. They were not pushed to a
lane branch first. The readings are synth's; the grades are this lane's.

| Key | Reading | Pre-registered | Verdict |
|---|---|---|---|
| Placed LUT / FF | 175,692 / 67,270 (slices 48,831, b3 48,086; LUTRAM 2,734) | 176,000 ±2,000 / 67,300 ±500 | PASS |
| Congestion, placer final | max 16x16 at level 4, East only; 8x8 elsewhere | no window above level 5 | PASS |
| Congestion, router | hotspot max level 4; per-direction max ≤ 90.22 % | ≤ 5 (vp3 failed at 6 and 100-118 %) | PASS |
| Routing | 0 failed, 0 unrouted, 0 partial; 0 overlaps (117,118 → 54 → 1 → 0, 34 min) | routes, 0 errors | PASS |
| Loops, DRC | check_timing 0 loops; TIMING-23 0 (matcher reads TIMING-16 x1001); LUTLP-1 0 (PDRC-138 x290) | 0 / 0 / 0 | PASS |
| Item-4 cone | 280 LUT + 40 CARRY4 | 280 ±20 + 40 | PASS |
| Presence, routed netlist | 880 / 55 x3 / 25 / 25 / 880 RAMD64E = 14,080 bits / 90 | the widening survived | PASS |

- Flow: exit 0 in 2h51m, peak 22.80 GB. Technology mapping peaked at 5,094.4 MB; at the gate it was 5,093.4.
- Seal: ariane_xilinx.bit, 11,443,722 B, sha256 f0893218dcdb53f4ad0b9fe452571f9e5c3bc5fa7ffade0025657c662c4da8cc.
  Tarball synth-325966ee4-exit0.tar.gz, 389,603,704 B, sha256
  fe30f7465ab53e70ce453627b0c3a9ec42ea3ca7b0538d8124f7583d45239179; its inner .bit is the seal and its PROVENANCE names
  325966ee4.

**Reported, not graded:**
- WNS −11.964 against b3's −10.047. The difference, −1.917, is inside the ~6.5 ns per-path null.
- TNS and the failing-endpoint count roughly doubled: −103,708 against −44,011, and 25,147 of 202,243 endpoints against
  13,396 of 188,701. This is consistent with the cost pre-named as credible, the forwarding mux widened from 65 to 90 bits,
  for 3 operands and 13 sources. It is not attributed beyond that. WHS is +0.041.
- Widened compares on the worst 500: 0 for every matcher.
  - rvc_* (POS 1): worst through −6.522.
  - rd_ctag (POS 1): +5.968.
  - cpmp_tracked: +6.483. pc_revnode: +17.794. Both are bounded by slack; membership is POS 0, as on A and b3.
  - NEG is 0 for every matcher.
- **The worst-20 family is new.** All 20 start at dom_switcher `_init_0_reg`, at 119 levels. On b3 all 20 started at
  csr_regfile cpmp_q, at 111 levels. They end in b3's family: fu_data_q's cap_metadata_a/b and rd_metadata, and the
  scoreboard's cap_result.rs1_metadata.
  - By source (`core/capstone_dom_switcher.anvil.sv`), `_init_0` is set in reset (:7812) and cleared on every later clock
    (:9455, unconditional inside the else at :8138). So a path that starts at it carries one transition per reset.
  - It is ORed with `_thread_0_event_counter_870_1_q` into the same event (:7673, :6488), so the same downstream cone is
    live from that flop. That flop's slack has not been read. These paths are not called false or harmless here.

**Flash gate, the non-cacheable tag: closed.**
- The question: a refill's tag is the adapter's user lane (wt_dcache_mem.sv:355-358), and that lane is loaded from
  axi_rd_user on data beats (wt_axi_adapter.sv:752-757). The boot ROM's and the debug module's axi2mem leave user_i
  unconnected, and the CLINT never assigns r.user.
- The answer, from source: the FPGA configuration sets FetchUserEn = DataUserEn = 0
  (capstone_cv64a6_imafdc_sv39_config_pkg.sv:40/:42). So AXI_USER_EN = 0 (build_config_pkg.sv:158), and
  axi_shim.sv:272-276 sets rd_user_o to '0.
- A read outside the data region takes no shadow-tag beat (wt_axi_adapter.sv:142/:156). Boot ROM, debug ROM and CLINT
  loads therefore arrive untagged, and the load rule cannot fire on them.
- Synth's netlist agrees, identically on B and b3: no r_user field survives in the r spill, and the user lane's fan-in is
  data beats, the tag FSM and ids. The behaviour is inherited from b3, not introduced by B.

**The board monitor's CSMINT path under B (runtime's request).** The twin boots fw_jump, so silicon would otherwise be the
first test of csbi-batch d9da397's capstone_csmint.S.
- Source reading: compatible.
  - The header is narrow on both sides (ex_stage.sv:1535-1536 and :1701-1719 = csmint's +0/+8/+16).
  - CSNODE's refusals stay 30 and 29. rd is the 55-bit ancestor id, stored verbatim; CSRETIRE takes the same scalar.
  - Every plain load in the two files reads an untagged word.
  - LCC sel 4 is end and sel 2 is the cursor.
- Simulation: va-csmint-mon assembles the pinned include, byte-identical to the board's. It reads 81 of 81 values equal
  to e7af48457 (24,502 retired on both).
- va-csmint-emu reads 30 of 32 equal. The other two are the arm's raw record word 1, encoding-only and computed exactly.
  The RTL lane's "identical" prediction missed them because that arm's print list had not been read.

## Option A: ids20, capstone-ariane 741aff6a6 (tests 3f876a255)

The id is {gen[29:20], idx[19:0]}; word1 carries gen at [59:50] and parent at [49:30].
- Size gate PASS: core +90 LUT / +35 FF, mapping 5,003.7 MB, 0 loops, presence OK.
- vp3-idx20.S: three arms read as pre-registered on 741aff6a6, and fail on e7af48457.
- 95-sweep: 92 identical, 0 differ, 3 missing on both sides.
- Full build, sealed: PASS on every pre-registered key.
  - Placed LUT 168,935 against 169,061 ±2,000; FF 63,681 against 63,647 ±300.
  - Congestion: no window above level 5 (the placer's maximum is level 4, as on b3). That grade used the placer's reading
    only. A's router level has been asked of the synthesis lane so that both options carry the same reading.
  - 0 routing errors; check_timing 0 loops.
  - TIMING-23 0 and LUTLP-1 0. Both matchers fire on other rules.
  - Widened compares on the worst 500: 0. rvc_* and rd_ctag by membership with POS 1. cpmp_tracked and pc_revnode by
    slack bound (+5.805 and +19.649 against the -6.440 floor), since their membership test has POS 0, as on b3.
  - WNS -9.462 (b3 -10.047), reported, not graded. TNS -58,579 (b3 -44,011). The worst-20 come from csr_regfile cpmp_q,
    b3's family.
  - .bit 11,443,722 B, sha256 32322b704d79eb463de3c1910dcf7af666d6e32b4dfec09a4be4f33d3532c882.
  - The fallback: not flashed while B passes.

## Corrections made on the way

- RETRACTION: 3e3f58244's "context scalar reads raw" (above).
- WITHDRAWN: the RTL lane's "cheap R3 guard". vp_hdr0_q is loaded only on allocations.
- MIS-SCOPED: A's pre-registered ±500 band on ex_stage's own cells. Per-module own cells measure homing, so B's
  pre-registration grades by stem.

## Handoff (2026-10-31, ~00:1xZ)

- B's grade, sha256 and file name have gone to the runtime lane. The runtime lane flashes, on the lead's word in its own
  session, and confirms the board copy against the sha256 above.
- The first boots, and what they read, are the runtime lane's to choose.
- A (741aff6a6) is kept, sealed, as the fallback and is not flashed.
- Owed, none of it gating:
  - the synthesis lane's reading of `_thread_0_event_counter_870_1_q`'s worst slack;
  - A's router congestion level;
  - `design/virtual-capstone/isa.md`'s "not in the RTL" and "Silicon" paragraphs, which become untrue once B is flashed
    (the semantics owner's document).
