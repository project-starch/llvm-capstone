# Option B (55-bit lifetime ids) on the current lineages: V1 e5n-ids55, V2 e5c-ids55, V3 c5a-ids55

*RTL lane, 2026-11-09. The lead and the collaborator chose option B (ids55). Relayed by the runtime lane: prepare every
variant in parallel, so that every scenario has a sealed bitstream. Target: V1 sealed, with its first-boot vectors and
gate driver, by Tue 10 Nov 06:00 SG; native perlbench and mcf train move onto it on boards 3 and 2 from ~10:00.
Synthesis is the board lane's (apollo, V1) and the synth lane's (leffe2, V2 and V3), under synth-guard. Everything in
this section was written before any synthesis or simulation of these hashes.*

## The three trees (capstone-ariane, local branches of the stage-B / C.5a lineage; build hosts get bundles)

B itself is rtl-ids55 afd35e449 (core = 325966ee4), sealed as bitstream 4 on bitstream 3's base (record:
`2026-10-30-bitstream-4-lifetime-id-width.md`).

| Variant | Hash | Base | Construction |
|---|---|---|---|
| V1 e5n-ids55 | b91439bf7 | e5-native 14b0a57df | E5's 20-bit split reverted (d8626c14c, 443f67892, comment-only 274f670ad, option A's arms 35c224153), then afd35e449 merged |
| V2 e5c-ids55 | 35928968c | E5c 246e7abc2 | the same reverts (no 274f670ad there), then afd35e449 merged; no DETACH/ADOPT, CSRUNV legality stage-B's (action <= 2) |
| V3 c5a-ids55 | 026e6b3bd | C.5a 1fc2cd5aa | afd35e449 merged; none of stage-B's R-32, R-68, DYN handshake trim or LDC rs1 change |

- After the reverts, core/ under V1's merge differs from a481031ca (stage-B) only by the three native commits. Under
  V2's merge it differs only by the clamp's csr_regfile.sv.
- V2's core differs from V1's in exactly four files: dyn_unit.anvil (legality), commit_stage.sv, csr_regfile.sv
  (detach/adopt) and cva6.sv.
- Merge conflicts, all resolved keeping both sides, by one asserted script (scratchpad `ids55-resolve.py`):
  - cva6.sv, load_store_unit.sv: the CJALR fix's pcc nets and ports beside B's typed rvc_den_id;
  - ex_stage.sv: C.5a's dynlt nets beside B's 25-bit rvc_probe_id;
  - testlist_sup.yaml: V1 and V2 only.
- **Sites stage-B / C.5a added after e7af48457 that B could not see** (the pcc_inv_now class). Every line in each
  merged tree was matched against
  `\[29:0\]|30'd|30'h|\[29 *-|\+: *30\b|\[17:0\]|\[15:0\]|logic *\[29|\b30\b` and kept only if 325966ee4's version of
  the same file lacks it. That leaves 7 sites in every tree, all from the CJALR fix (1fc2cd5aa):
  - commit_stage pcc_id_o / pcc_verdict_id_i, ex_stage pcc_id_i / pcc_verdict_id_o, the LSU's pcc_id_q: [29:0] -> cap_id_t.
    At 30 bits the verdict id never equals the tracker's 55-bit pc_revnode_id_q once the generation is nonzero, so the
    head would wait for a verdict that never matches.
  - pcc_inv_now's two compares: [17:0] -> [24:0], the slot, as B's rvc_ok_kill / rvc_set_kill.
  - After the edit the listing is empty on all three trees.
  - The listing is a literal match, so a claim-auditor pass aimed at what it cannot see is part of the gate below:
    named widths, Anvil types, implicit truncation at ports, auto-merged hunks.

## Does B cost cycles? (asked by the lead; pre-registered before any cycle run)

B adds no pipeline register, FSM state or wait on any LDC/STC path. Every slot move is a combinational mux inside an
existing stage (quoted from e7af48457..325966ee4):
- LDC unpack, ex_stage's load result: `load_result_tag ? ids55_cursor(load_result) : load_result`;
- STC pack, load_store_unit's store data: `ariane_pkg::ids55_pack_lo(fu_data_i.cap_data.cap_metadata_b[...] ...)`;
- the load rule, load_unit before the existing shift:
  `ld_rdata = ids55_proj ? ariane_pkg::ids55_cursor(req_port_i.data_rdata) : req_port_i.data_rdata`;
- the switcher's context reads: `dom_switch_data_resp_o = (...) ? ids55_unpack_word(...) : {...}`;
- the one new load_unit output, `result_ldc_o`, uses the same LDC_CLEAR_WAIT mux as `result_tag_o`.

The rev unit keeps 16-byte nodes and the same accesses per walk. The revocation cache keeps its sets and ways; its
tags widen.

**Prediction:** per image, V1's cycles are IDENTICAL to e5-native's (delta 0) on all 32 first-boot images. The runtime
lane's threshold is > 1 % more cycles on the LDC/STC-heavy images (objb-perf-s1..s7, slot-1..9). B can cost timing
(WNS), not cycles.

The cycle changes B is known to make are reads of a de-tagged granule's raw word. Such a read returns a different value
and so makes a different access: in the 95-sweep, 3 of 92 tests, +2 / -67 / -69 cycles. Any nonzero delta is traced
before it is reported.

## First-boot vectors (the 32 C.5a-gate images)

Each variant's 32 image sims (same ELFs, same recorder area, S12_MEM_DELAY=12, time_out 4,000,000, model rebuilt)
become its vectors file and its board grade's sim column.
- Prediction: every image's vector equals its parent's: e5-native's column for V1, E5c's for V2, C.5a's own sims
  for V3. objb-pcc is empty on every tree.
- A differing VALUE must be one of B's known format effects, traced before the file is frozen:
  - a raw read of a stored capability's second doubleword reads the generation alone in the node field;
  - a de-tagged granule's first doubleword keeps slot << 39.

Gate drivers: `perf/v{1,2,3}gate/v{1,2,3}-gate.sh`, copies of e5n-gate.sh with the bitstream, vectors and grader
swapped. A FAIL flashes C.5a 1fc2cd5aa back. The grader rule is E0's.

## Functional gates per variant, all before any flash (synthesis starts after lint, by the runtime lane's ruling)

1. Lint gate with its numbers, against B's own lint on 325966ee4 (LATCH 44, MULTIDRIVEN 3, UNOPTFLAT 39 with names
   identical, UNUSEDSIGNAL 1479).
   - Any UNOPTFLAT name not on the parent's list is a stop.
2. B's acceptance arms: ids55-basic and its four defines, va-runv2-capa1, va-runv6-w, the 16 wide-record arms, the
   va-runv family of 23. Graded against B's own readings on 325966ee4 (rtl-ids55-wt .sup/final, .sup/runv6w).
   - V1 also runs the native arms (va-runv-xfer, va-runv-bare, -ctrl), graded against e5-native's readings.
   - V2 also runs va-runv-bare-satp.
3. The 32 first-boot image vectors, above.
4. The 95-sweep with the RVFI trace compare (capstone-ariane sweep-tools 48969d26a), against the parent's sweep.
   - Predicted differences: B's own 8 of 92 (the raw-read effects of 78be372d38df) and nothing else.
5. claim-auditor over the merge, aimed at missed id-width sites (the brief above).

## Synthesis pre-registration (keys graded by the synthesizing lane; floors ±2,000 TOP / ±2,500 per module)

The basis is B's own gate on bitstream 3. Against b3's synth checkpoint:
- core +6,962 LUT / +3,654 FF;
- ex+issue +8,130 / +2,850;
- pmp_data_if FF +389, csr_regfile FF +556, LSU LUTRAM +400;
- placed +6,631 LUT (169,061 -> 175,692) / +3,623 FF;
- WNS -11.964 against b3's -10.047 (-1.9, inside the per-path null).

The 7 widened CJALR sites add a few hundred LUT at most.

| Key | V1 vs e5-native | V2 vs E5c 246e7abc2 | V3 vs C.5a 1fc2cd5aa |
|---|---|---|---|
| post-synth core LUT | 144,615 -> 151,700 (+7,100 ±2,000) | 144,854 -> 151,950 (+7,100 ±2,000) | 145,391 -> 152,450 (+7,100 ±2,000) |
| core FF (reported) | 41,427 -> ~45,080 (+3,654 ±1,000) | 41,430 -> ~45,080 | 41,426 -> ~45,080 |
| ex+issue (reported; graded by stem) | +8,130 ±2,500 | +8,130 ±2,500 | 89,473 +8,130 ±2,500 |
| "Found timing loop" headers | 0 | 0 | 0 |
| placed LUT (reported) | ~176,000 ±1,500 | 169,322 -> ~176,000 ±1,500 | ~176,400 ±1,500 |
| placer / router congestion | <= 5 | <= 5 | <= 5 |
| routed | clean | clean | clean |
| check_timing loops | 0 | 0 | 0 |
| WNS (null band ±3.74 around the parent) | -13.953 -> ~-15.9 | -12.723 -> ~-14.6 | -13.758 -> ~-15.7 |

- **Routing margin, stated plainly.** Every variant lands at about 176k placed LUT. That is ~2.5k under the vp3 build
  that failed to route (178,543, 2026-10-24) and above B's own 175,692, which routed with congestion 4. A routing
  failure is possible and is not predicted.
- **Failure branch:** if route_design fails, there is no bitstream and no WNS is recorded. Placed area is reported
  from the post-phys_opt checkpoint. V3 is the smallest delta from what has routed.
- Presence on the routed netlist (B's own key): the widened id buses survive, read on the routed DCP after the seal.
