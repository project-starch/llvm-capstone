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

## R-70: B's load rule reads a stale tag past a buffered store (found 2026-11-09 by the native CSCHECK arm on V1)

**The defect.** A plain store to one doubleword of a tagged granule clears its tag when it drains. Until then the D-cache
line keeps the old tag. The store buffer's hazard compare is doubleword-granular (store_buffer.sv `address_checker`,
page offset [11:3]), and R-29's hold in wt_dcache_ctrl.sv treats a plain read of the low word as conflict-free, so a
younger plain load of doubleword 0 reads the stale tag and B projects it (sext39) where program order makes it raw.
- va-cscheck-mon-ids55 on V1 (capstone-sbi b366655): rows 3, 4, 5, 9 and 10 read 25 where 0 / 28 / 0 / 27 / 0 are
  registered. The retirement trace shows the scratch word read raw `0x000001808000d000` once (first touch, a cache
  miss) and projected `0x000000008000d000` six times.
- ids55-sbhazard.S (V1, predictions in its header before the run): A1 `0x80003040` (projected), A2 `0x10080003040`
  (raw), A3 projected. The prediction was projected on all three, so 2 of 3: timing-dependent even in simulation.
  F (fence) and B (same doubleword) raw. The B control is weak: this slot leaves R's bits 63:56 zero.
- The firmware workaround (capstone-sbi 4f51541, a fence before the raw load) reads every registered row on V1.

**The fix, V1f = e5-ids55-r70 d92abd930 (5fc07e4e1 + d92abd930; tests ba8bd0a96 below it).**
- In wt_dcache_ctrl.sv, on the load unit's port only (SpecHold), a plain read's data_rtag is cleared while a plain
  store to the same physical granule is pending in the write buffer or the store buffer, under R-29's
  speculative-entry rule. It is taken live in READ (the tag cycle) and held for a replay or miss response.
- No hold is added, and no state machine, store-buffer hazard compare or load-unit stall changes.
- Prior art, read before writing: S-10b (21-08-2026, the [11:3] -> [11:4] compare formed a LUTLP-1 loop through the
  load unit's stall cone); R-29's own comment (this controller).
- A first cut gated every port. Lint kept all 39 UNOPTFLAT names, but the standing rd_ctag ring's example path then
  showed the new assign on port 2 (the rev node's), so d92abd930 restricts it to port 1.

**Pre-registered for V1f (before any V1f simulation):**
- ids55-sbhazard: A1 = A2 = A3 = R = `0x10080003040` (raw), F = R, B = R, cursor `0x80003040`, mcause 0.
- va-cscheck-mon-ids55 WITHOUT the fence (b366655): every registered row, as the fence build reads on V1.
- The 32 first-boot images: every value and every end cycle identical to V1's. No image's dw0 load follows a plain
  store to its granule's other doubleword while that store is buffered, as far as is known; a difference is traced
  before it is reported.
- The native arms (va-runv-bare, -ctrl, -tag928, -xfer): identical to V1's.
- Lint: every counter at V1's; UNOPTFLAT names identical to V1's.
- Synthesis: core LUT within ±500 of V1's; check_timing 0 loops; DRC LUTLP-1 0 on the post-synth checkpoint (S-10b's
  failure mode, which no in-flight key catches).

**Addendum (2026-11-09, before any V1f/V2f/V3f synthesis; two corrections the board lane caught).**
- The V1f hash for synthesis is **0ef470559** (e5-ids55-r70), not d92abd930 as written above. 0ef470559 = d92abd930
  plus the restructure that moves the stale-tag logic inside the SpecHold generate block: d92abd930 failed the lint
  gate on UNUSEDSIGNAL 1481 (ceiling 1480); 0ef470559 passes with every counter equal to V1's. No function change.
  V2f = **c95525476** (e5c-ids55-r70), V3f = **596c36fc7** (c5a-ids55-r70): each is its unfixed variant plus the same
  wt_dcache_ctrl.sv, byte-identical to 0ef470559's.
- The synthesis key "core LUT within ±500 of V1's" was written BELOW the recorded redistribution floor (±2,000 at TOP,
  ±2,500 per module; source-identical builds move that much). Third instance of this mistake on this lane. The key
  is **core LUT 0 ±2,000** against the unfixed parent's post-synth report (V1 for V1f, V2 for V2f, V3 for V3f). The
  ±500 reading is printed beside it as REPORT only, never graded.
- LUTLP-1: the board lane runs it in flight on the post-synth AND post-opt checkpoints and stops on any count above 0;
  until the S-10b-tree control (eaa4e7984) shows whether such a loop is already visible post-synth, the post-synth 0
  is necessary, not sufficient, and post-opt is the stronger reading.

**Addendum 2 (2026-11-09 ~15:50, after V1's in-flight gate and before any fixed-variant synthesis reading).**
- V1's in-flight gate (board lane, apollo) FAIL-STOPPED on this lane's prediction key: post-synth core LUT +9,899 vs
  e5-native, against +7,100 ±2,000. Loops 0, LUTLP-1 post-synth 0. The other readings beside their report-only
  predictions: core FF +3,792 (+3,654 ±1,000), ex+issue +7,560 (+8,130 ±2,500), pmp_data_if FF +410 (~+389),
  csr_regfile FF +532 (+400..+700). The excess sits in modules that redistribute between builds: dyn_unit +2,874
  (+2,363 in _spawn_0), csr_buffer +2,081 (source unchanged, re-homed logic, cause UNRESOLVED), dom_switcher +1,093;
  iro +3,731 against scoreboard -2,303. The prediction was low by ~2,800; its basis (B on bitstream 3, +6,962) had no
  stage-B / native logic under it. The board lane relaunched V1 (e5nids55r, 15:43:54) stopping only on loops and LUTLP-1.
- **Routing risk, restated with this reading.** On B's own bitstream-3 ratio (placed growth ~0.95 of post-synth growth)
  V1 places near 168,933 + ~9,400 = ~178,300 LUT, essentially AT the vp3 build that failed to route (178,543,
  2026-10-24). Pre-registered failure branch, unchanged: route_design fails -> no bitstream and no WNS; placed area from
  post-phys_opt; the unfixed V2/V3 (and their fixed builds) are then the candidates, and what to cut for V1 is the
  lead's decision.
- **The fixed variants' area key, amended.** The board lane's TOP floor rose to ±2,500 (its retraction: a parent sums
  its children, so one module's bistable can land whole at TOP). The key is core LUT 0 ±2,500 vs the unfixed parent's
  post-synth report, GRADED POST HOC, never a stopper; the ±500 reading stays REPORT. This lane agrees with the board
  lane's rule from here on: in-flight stoppers are loop headers, LUTLP-1 and area past routability; prediction bands
  are graded after the run.

## V1s: B's memory format with a 20-bit CORE slot (a speculative area hedge for V1f's routing; NOT full B)

*Approved by the runtime lane 2026-11-09 ~16:00 as a hedge, synthesised only if V1f fails to route. Written before any
V1s simulation or synthesis.*

- capstone-ariane e5-ids55-s20: **eef29005e** (core; on V1f 0ef470559 + its tests), tests 171f1fbc9.
- Lanes only: the {slot, tag, metadata} lane's slot is 20 bits (CAP_LSLOT_W; CAP_MD_W 85, CAP_PCMD_W 84, CAP_WORD_W 149,
  Anvil raw_data / pc_next_metadata 84). Ids stay 55 bits in every unit (rev node, ex_stage, revocation cache,
  trackers); the memory format is B's (25-bit slot field). The virtual capacity / next-unused saturate at 2^20.
- Fail closed, never truncated: an id whose slot does not fit a lane enters it untagged (cap_lane, ra_data); a stored
  word whose slot field has bits [63:59] set is unpacked untagged and raw. **Twin difference:** a use then faults 24
  where QEMU (25-bit slots) answers 25. Unreachable while tables hold <= 2^20 records (the module's flat table is
  2^20 records; tags cannot be forged).
- **Semantics claim:** equivalent to B (V1f) while every table holds <= 2^20 records.

**Pre-registered readings.**
- ids55-basic -DSLOT20 (free head 2^20 - 2, next-unused 2^20 - 1): CSNODE's rd 1048575 (live), the stored child's raw
  slot 1048574, every other reading as the default build; identical on V1s and V1f.
- ids55-basic -DSLOT20_FULL (next-unused exactly 2^20): V1s refuses the ancestor bump, srevroot 0 then the trap
  handler's 30; V1f (full B) allocates slot 2^20, CSNODE's rd 1048576. The discriminator.
- B's acceptance arms (the 46) and the native arms: identical to V1f's. The 32 first-boot images: identical to V1f's in
  values and end cycles. ids55-sbhazard: as V1f.
- Lint: UNOPTFLAT names = V1f's; UNUSEDSIGNAL may move by the narrowed fields (attributed by name before hand-off).
- Synthesis (only if V1f fails to route; graded post hoc, stoppers = loops and LUTLP-1): post-synth core LUT
  **-1,500 to -2,500 vs V1f** (5 of B's 25 lane slot bits in the register file, operand muxes, scoreboard, pipeline and
  PC/CSR lanes); core FF about -700. A reading inside ±2,500 of V1f is NOT distinguishable from noise and is reported as
  such.

## Readings (2026-11-09, graded against the predictions above)

**Functional, simulation** (S12_MEM_DELAY=12 unless stated; every run dir carries a PROVENANCE with its Anvil unit
hashes, and every copied unit set equals a from-source regeneration at the same Anvil sources):
- B's 46 acceptance arms on V1 (c5a-wt .sup/v1arms): 46 of 46 equal B's own readings on 325966ee4.
- Native arms on V1's core with rd made visible (va-runv-bare, -ctrl, -xfer): equal e5-native's in every value.
  va-runv-bare -DTAG928 (the claim-auditor's SUSPECT): the satp the next escape saved equals the original exactly, as
  pre-registered; the context then escaped with cause 25 on the garbage srevroot (not predicted).
- R-70 (see ISSUES R-70): on V1 the hazard arms read projected in 2 of 3; on V1f raw on all three, with the younger-store
  and same-set other-granule arms projected, at memory delays 12 and 40. The native CSCHECK emulation reads every
  registered row on V1f with and without its fence (capstone-sbi b366655 / 4f51541), and on V1 only with it.
- V1s boundary: -DSLOT20 equal on V1s and V1f (rd 0xfffff, live); -DSLOT20_FULL refused on V1s (srevroot 0, then 30),
  allocated on V1f (rd 0x100000).
- 32 first-boot images on V1f: all 32 vectors equal e5-native's (perf/v1fgate/v1f-vectors.txt, sha256 ea7e1b65...).
  **Cycles, V1f vs e5-native: delta 0 on all 32** (end cycle and committed count), the LDC/STC-heavy objb-perf-s1..s7
  and slot-1..9 included. The prediction (no added cycle on any LDC/STC path) holds exactly.
- 95-sweeps, RVFI trace compare: V1 vs its 18-bit base, V3 vs C.5a's own sweep, and B vs bitstream 3 each differ in
  exactly the same 8 of 92 tests (identical line counts and first retirements; 78be372d38df's stored-word raw reads);
  V2 equals V1 on all 92.

**Synthesis** (board lane on apollo, synth lane on leffe2; stoppers loops and LUTLP-1, area post hoc):
- LUTLP-1 positive control (synth): S-10b eaa4e7984's post-synth checkpoint reads 1 violation (62 LUT cells through
  csr_regfile, the board lane's own reading), 31 check_timing loops and 179 "Found timing loop" headers, so the
  post-synth check fires on that shape. The board lane retracted its earlier statement that S-10b emitted no headers.
- V1 b91439bf7: post-synth core 154,514 (+9,899 vs e5-native; the prediction +7,100 ±2,000 was low, attributed to the
  redistribution modules), TOP 181,042; LUTLP-1 0; rebuilt as the comparator.
- **V1f 0ef470559 SEALED** (board lane, 17:22): caplifive_e5nids55f_0ef470559.bit, sha256 06f6be7e3cf977908d0f103526929d97643c8241e30be6df3133d653867d86bf.
  4/4 vs e5-native: WNS -13.873, 0 loops, routed clean, LUTLP-1 0 at synth, opt and routed DRC; placer congestion 5;
  placed LUT 179,757 (slices 98.67 %) -- routed ABOVE the vp3 failure point (178,543), so total placed LUT is not the
  cliff by itself. Post hoc vs V1: core +1,095; the R-70 controller 145 -> 416 LUT.
- V2 35928968c gate: core 151,925 (prediction 151,950), LUTLP-1 0. V3 026e6b3bd gate: core 151,822, LUTLP-1 0.
  V2f c95525476 gate: core 152,220 (V2 +295), LUTLP-1 0 post-synth and post-opt. ex+issue rows outside their bands
  were ruled report-only (the totals land on prediction).
- **V1f's sweep (18:26): RVFI traces identical to V1's on all 92**, as predicted; V1f's last functional gate. Its board
  gate (perf/v1fgate/v1f-gate.sh, dry-run on every branch and against the real store) is armed by the runtime lane on
  board 2 after the e5-native quick tests.
- V2 35928968c SEALED (synth): sha256 afcdb31ee691db0f2446b1edd2710ee3700969ddc1eb306ab6ca4af99eb3296d; every key PASS
  except routed WNS -17.301, outside its -12.723 ±3.74 band by 0.84 ns by letter: ruled reported and non-gating (the
  user's 2026-11-04 decision; the lead's bitstream-4 ruling; inside the ~6.5 ns per-path null), carried to the lead.
- V3 026e6b3bd SEALED (synth): sha256 aaffcfd3d5150e4b54c35763f767dd56a956f9e1061e8efba4f4529e4fff1b04; every key PASS;
  WNS -10.449; placed 176,090.
- V2f c95525476 SEALED (synth): sha256 1c2d4e9143247ab5ec620cc84a3aa25d960abaa60286397c3bd9575ce19e01d7; LUTLP-1 0 at
  synth, opt (control fires post-opt too: S-10b after opt_design reads 1 / 31) and routed; WNS -12.129.
- **RETRACTION (this lane's estimate, 2026-11-09):** V3 was predicted ~0.8k ABOVE V1 (~178.5-179.5k placed), inferred
  from C.5a's seal being +776 / +805 above e5-native's. It placed 176,090, 3.7k BELOW V1f, and its post-synth core is
  2.7k below V1's. An inference smaller than the ±2,500 floor carries no information; it should have been stated as
  "indistinguishable from V1".
- Functional, fixed hedges: V2f and V3f each read B's 46 arms exactly; V2f's native arms = E5c's. Vectors: V2f = E5c's on
  all 32, V3f = C.5a's sims on all 32 (perf/v2fgate, perf/v3fgate, shas 4c1cba25... / de7e0067...). Cycles: V2f vs E5c
  delta 0 on 32 of 32; V3f vs C.5a delta 0 on the 18 images whose C.5a commit traces exist (the other 14 by vector
  only). V3f's grader passes C.5a's own silicon run and fails the stage-B-timing control.
- Still running at this writing: V1s and V3f synthesis (leffe2); V2 / V3 unfixed arms and vectors; V1s's images.
