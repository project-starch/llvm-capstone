# Bitstream C.5a: LDC/STC take the LSU's cached lifetime verdict, and the PC tracker checks a new PC capability (2026-11-03)

*The RTL lane. The lead's ruling of 2026-11-03, relayed by the runtime lane: "batch". 1fc2cd5aa (C.5a and the PC-tracker
fix, R-67) goes to synthesis as one bitstream once the sweep with its RVFI and cycle comparators, the claim-auditor on
both commits, and lint have passed. The check-off partner is the sealed ObjBounds bitstream (`caplifive_vp2c_603528b8b`).
This record is pushed BEFORE the hand-off to the synth lane, as the build's pre-registration.*

## The change

capstone-ariane `1fc2cd5aa`. Its parent is `e71b236f2` and its base is `e7af48457` (bitstream 3). Both commits are the
external collaborator's (branch perf/c5a-cached-ldc, documentation tip fe5aa8440; delivered as a verified bundle, sha256
e17fd3c982e388c9…). Core files:
- **e71b236f2 (C.5a):** LDC and STC take the LSU's revocation-cache verdict for rs1's node in place of the DYN unit's
  uncached query. The DYN unit's query becomes `1'd1`, and its generated unit is byte-identical to ObjBounds'
  (`capstone_dyn_unit.anvil.sv` sha256 131818fb5bfda03c, read here in both trees).
  - Merge order at CAP_WB: the LSU's exception, then the DYN unit's own 24/26/27/28/29/4/6, then 25 (tval = the latched
    rs1 cursor), then the replay marker, then success.
- **1fc2cd5aa (R-67 fix):** a new PC-capability id is adopted UNKNOWN, and the head is held until the LSU answers from
  positive evidence. A dead PCC raises 25.

## Gates (simulation, memory delay 12)

Trees: the C.5a tree is capstone-ariane `c5a-eval` a642f4891 (core = 1fc2cd5aa's; sweep PROVENANCE `rtl_dirty=no`, lint
run on the same tree after that commit, whose `git status` shows only the submodule symlinks). For the regression it is `c5a-assert`
26b34ffd3, which is the same core plus a sim-only `ifndef SYNTHESIS` counter block in ex_stage.sv. The control is
`c5a-eval-ctl` 4dcb67e4e (bitstream 3's RTL at 87368cc69 plus the same tests). Every comparison below pairs the two by
test name and runs at delay 12. Both regression PROVENANCE files read `rtl_dirty=no`.

- **Lint** (`verif/sim/rtl-lint-gate.sh` on the C.5a tree): PASS. Every counter is at baseline. UNUSEDSIGNAL reads 1429
  against REF 1431; the −2 is the DYN unit's query inputs, as on ObjBounds. The UNOPTFLAT names are unchanged.
- **95-sweep against bitstream 3.** The control is vp9 `sweep-b-off`: the ObjBounds build with its define off,
  objbounds-off-b 1782d20bd, `rtl_dirty=no`. It is the same control ObjBounds' own sweep used.
  - Classifier: 89 identical, 3 differ, 3 missing on both sides (rand, perf_*_random, as on every sweep).
  - RVFI traces: 91 identical and 1 differing, with 27 of the 92 parked and compared by prefix. That one is cap-tag-cache's T4
    timing control, a register derived from cycles; its values otherwise agree.
  - Cycles: 11 of 92 differ, all faster (s07-wbuf-tag-reorder −32, stc-ldc-sbpressure −75, …). These are the same 11
    tests the author's document lists.
- **Paired regression groups:**
  - Scope: sup, r26, r12cost/mint/partB/probe/recl/s1deep/s1wit/s1, r43, r34_coldmiss, s1gap and stale; 413 listed,
    408 distinct, all run on both trees.
  - RVFI traces: 363 identical and 45 differing, with 103 of the 408 parked and compared by prefix. A taint comparator classified
    every differing line; it was controlled both ways, reading 0 on an identical pair and 1 UNEXPLAINED on a fabricated
    value change.
    - Its source decoding for compressed instructions over-taints, which errs toward calling a line timing. So the
      mcycle group was also checked by raw diff on all 32 tests. Every one of the 246 differing lines is one of:
      - a cycle read: `csrr mcycle` ×118 or `rdcycle` ×62;
      - a `sub` of two cycle reads ×32;
      - a `c.add` that accumulates such a difference ×34. The 45 sort into four groups:
    - **32 differ only in values derived from mcycle.** These are r12-mint ×6, r12-revoke-walk-cost, r12-s1 ×22,
      s1gap-close, s1gap-far and m1-take-cost, with 0 unexplained values. The cost brackets move by a few tens of
      cycles at most and do not scale with n: r12-s1 nets −60, −7 or 0, r12-mint −6/−7 or 0, m1-take-cost −5 per
      take. The one larger move is
      s1gap-close (GAP 0): the first revoke after the last mrev costs +151 (229 → 380) and the ALU gap after it +50.
      s1gap-far (GAP 1000) reads only the +7 constant shift. The mechanism, read from both traces: the LDC before the
      bracket commits 120 cycles earlier on C.5a (cycle 2255 against 2375), because it no longer waits for the DYN
      query. The revoke still completes at about the same absolute cycle (2606 against 2575), so the bracket opens
      earlier and absorbs the wait. This is the effect the test's own header describes for step 6, the bracket
      growing while the total stays put, and the program ends 105 cycles SOONER. On silicon the k = 1 revoke bracket
      is therefore predicted to read longer on C.5a than on bitstream 3, and its total not longer.
    - **Interrupt and quantum landing shifts.** The only readings that move are escape and resume counts. The final
      values (0x1234, mie 0x888, …) are identical. The tests: sup-mtip, va-runv-quantum, vb-trans-plain,
      sup-stcsb-hot-ctl (16 → 18) and sup-s16-esc-n32 (resumes 20 → 23). sup-quantum-q16 escapes 84 times instead of
      74, because each re-entry now pays the PCC probe; it still ends on the exact ladder value, 0x1234, with mie
      0x888.
    - **Parked tests** whose paths part after a timing shift: cap-tag-cache, sup-guards, sup-mswap-capstl and
      sup-stcsb-ev ×3.
    - **One behaviour change, vb-trans (the no-table build).** Every arm is refused 25 on both trees. On C.5a the
      refusal moves from the first data access (stub+8, tval the data address) to the first fetch (stub+0, tval the
      code address), and arm 5's 28 becomes 25. This is 1fc2cd5aa's dead-PCC check firing before the access does.
  - Classifier: 362 identical and 46 differ, which is 45 + 2 − 1:
    - classifier only: r26-diag-csfence-noolder and r30-fill-init-r60. Both are parked, their retired count moved and
      their traces are prefix-identical;
    - RVFI only: r12-s1-n8192. It is parked, has one differing line derived from mcycle, and is classifier-identical.
  - The detector fired 0 times on both sides.
  - Cycles (`sweep-compare-cycles.py`, end line on both sides for all 408): 127 differ, 106 slower and 21 faster.
    - 59 differ by at most 10 cycles, mostly +7, which is the PC tracker's constant shift.
    - 63 differ by 11-200, for example +46 on the six r12-stale-site arms and +51 on stale-access. These are the
      DEAD-record and miss path. The largest relative move in this band is s1gap-close at −2.2 %.
    - Three tests are slower by 11-13 %: sup-quantum-q16 +140,690, sup-s16-esc-n32 +43,234 and sup-stcsb-hot-ctl
      +28,080. In all three the added cycles equal the added escapes (10, 3 and 2) times the test's own cost per escape,
      about 14,000 cycles each. So the cost is the count of escapes, and on these tests' tiny quanta each re-entry's
      PCC probe pushes the work into more quanta. With a production quantum the same probe should be a per-entry cost of
      the order of the miss path. That is inferred, not measured; the SPEC readings measure it.
- **Fail-open scan** (scratchpad audit-scan.py over all 408 C.5a logs; it was controlled to fire both ways):
  - 9,670 LDC/STC writebacks on CAP_WB. 0 of them are a SUCCESS without a lifetime verdict, 0 are fail-closed results
    with have=0, and 0 are no-verdict errors.
  - 18 writebacks without a verdict carry an exception. These are the DYN unit's own faults raised in the dispatch
    cycle, which is the auditor's predicted fast-reject case.
  - **R-27 (b)'s premise is false as written:** an LDC/STC CAN dispatch in a flush cycle. This happened 8 times, in
    r27-ccsr-n0, r27-fence-st, r27-fencei-st, r27-ldf-n2, r27-ldf-n2-legal, sup-arm, sup-mswap-postnop and
    tag-half-store. In every instance the DYN unit dropped the op, and 0 results appeared for a flush-dispatched op
    before its re-dispatch. So the auditor's UNRESOLVED edge, a flush-cycle dispatch that still answers, is exercised
    8 times and does not occur.
  - The no-verdict `$error` is compiled in. It sits in the same `SUP_TRACE` always_ff as the dyn-result `$display`,
    which printed 9,652 lines.
  - The counter block's `final` totals did not print in any of the 408 runs. Why has not been diagnosed. 26b34ffd3's
    message says they print, and they do not. So the flush-cycle count above is the 8
    `C5A-AUDIT-FIRE` lines, which are its positive evidence, and it is not a total. The missing-record REPLAY livelock
    the auditor named has the same `$error` as its only witness, and that witness reads 0.
- **Directed arms:**
  - ovl-invalid (LDC/STC through a revoked base, memory read afterwards) reads the pre-registered vector on bitstream 3,
    on C.5a and on ObjBounds: 0x19, LINEAR, 0, 0x19, 0x18, 0x19, 0x55, 0, 4 on b3/C.5a, and 0, NOT_CAP, 0x18, 0, 0, 0,
    0x55, 0, 1 on ObjBounds.
  - R-67's lv-t1-fire reads FIXED on C.5a (25 at the target), while its three controls are unchanged.
- **claim-auditor**, on both commits, read-only:
  - Verdict: PLAUSIBLE-BUT-UNPROVEN, with no fail-open path found in source.
  - Its two residual risks are now measured: the flush-cycle dispatch edge (above), and the missing-record REPLAY
    livelock, whose witness never fires.
  - It left the namespace of the monitor's PCC probe after an escape UNRESOLVED "until the sup group runs". The group
    has now run, and every escape and resume in it took that path (sup-quantum-q16 alone escapes 84 times).
    - No completed test reads a 25 or an exception count the control does not. The exception: vb-trans, whose dead
      PCC is refused by design.
    - No test targets the namespace specifically.
  - Behaviour changes it named, carried into the silicon readings below:
    - a live LDC/STC takes 25 after a probe timeout (about 1.07M cycles; base waited without bound);
    - a PCC hold blocks interrupts and the quantum until the probe answers;
    - every PCC-id change costs a probe;
    - the R43_PROBE_TIEOFF build refuses every new PCC id;
    - in a supervised domain, a held head with a CSR violation escapes with the CSR cause where base gave 25.
  - Synthesis risk: it found no new loop. The new combinational paths are two TRANS_ID compares at the DYN result
    endpoint and a 30-bit flop-to-flop compare into `pcc_hold` → `commit_ack`. None of their inputs is on the standing
    UNOPTFLAT list, but `commit_ack` heads the standing commit→csr loop. So the synthesis-only gate runs first.
- **lc-model hit rates** (the runtime lane, spec-silicon 03d5686bbff0): perlbench 0.11 % misses, mcf 1.88 %, hit_dead 0,
  disagree 0. They are an input to the SPEC readings, not a gate.

## Pre-registered synthesis keys (graded by the RTL lane; per module stem, not core totals)

The reference for every key is bitstream 3's own build of e7af48457. ObjBounds' build of 603528b8b is the second
reference for the DYN unit, because its generated unit is the same file.

1. **Mapping.** The technology-mapping peak at the marker is in bitstream 3's class, 4.8-5.6 GB (b3 5,004.9 MB,
   ObjBounds 4,951.7 MB). No single process exceeds 9 GB.
2. **Loops.** "Found timing loop" 0, Synth 8-326 0, check_timing 0 loops, each with the matcher's control firing.
3. **The change reached synthesis.**
   - The generated `capstone_dyn_unit.anvil.sv` in synth's tree has sha256 131818fb5bfda03c.
   - Cells named `*dynlt*` are > 0 under the LSU and under ex_stage, with b3's checkpoint reading 0 as the NEG.
   - `*pcc_verdict*` / `*pcc_hold*` cells are > 0 under commit_stage or the LSU, NEG 0 on b3.
   - The revocation-cache cells (`*lsu*rvc*` LUTs, 1,980 on b3) are PRESENT, within ±10 % of b3. This is the opposite
     of ObjBounds, whose key read 0 here.
4. **Stems** (synth checkpoint, own cells, against b3):
   - `anvil_dyn_unit`: within ±200 LUT of ObjBounds' own row. It is the same file, so a difference is homing.
   - `load_store_unit` (lsu_i): LUT +0 to +1,500; FF +20 to +80. The new state is the dynlt and pcc records and the
     PCC probe request, about 43 flops in all by the author's count.
   - `commit_stage`: FF +0 to +40 (the UNKNOWN state and the held id); LUT within ±500.
   - ex+issue summed: LUT within ±2,500, because re-homing between them is expected (the batch and ObjBounds both
     showed it).
   - Core LUT within ±2,000 of b3's 144,584, with LUTRAM equal to b3's 870 ± 50 (the cache stays).
   - The core FF total is REPORTED, not graded: ObjBounds' core-FF clause failed on a retiming flip of the FPU.
5. **Placed area.** LUT ≤ 169,061 + 2,000; FF within 63,647 + 43 ± 300.
6. **Congestion.** No placer or router window above level 5.
7. **Routing.** 0 failed, unrouted and partially routed nets.
8. **Routed checks.** check_timing 0 loops; TIMING-23 0; LUTLP-1 0; each with its b3 control.
9. **Routed names.** Key 3's `*dynlt*` and `*pcc_*` names are present on the routed dcp.
10. **WNS** is reported, not graded. b3 is in service at −10.047, and ObjBounds sealed at −9.433. The synth lane flags
    any WNS < 0 as not flashable; that is its standing rule, and the flash is the lead's decision.
11. **The bitstream:** .bit sha256, size and path.

Failure branches:
- A mapping blowup (one process > 9 GB, or the 34 GB watchdog) is graded by genome and Synth 8-326 targets. There is no
  retry.
- A routing failure gives the placed area, keys 1-5 and a congestion census, and NO WNS.

## Synthesis-only gate: readings and ruling (2026-11-04)

The synth lane ran the gate on 1fc2cd5aa under synth-guard (40 GB) with the 34 GB watchdog and the 9 GB per-process
alarm. It stopped as planned after synthesis (1h46m). Keys 5-11 above stand as written, except key 9's names, which
are corrected below. Three clauses failed as written. Each one was a key written against something it could not
measure; none of them reads growth. The corrections are mine, made after the reading, and are named as corrections.

| Key | Reading | Grade |
|---|---|---|
| 1 | mapping peak 5,082.0 MB (b3 5,004.9); largest single process 4.48 GB (the 9 GB alarm did not fire); run peak 21.77 GB | PASS |
| 2 | "Found timing loop" 0, Synth 8-326 0 (control vp3g 7 / 73); check_timing 0 loops (control: 1 on an earlier dcp) | PASS |
| 3 | dyn unit sha256 131818fb5bfda03c; `*dynlt*` 307 cells (293 lsu_i, 9 ex_stage, 5 other; b3 0); `*lsu*rvc*` 1,833 LUT (b3 1,980, −7.4 %) | PASS |
| 3 | `*pcc_verdict*` / `*pcc_hold*` cells 0 (b3 0); `*pcc_verdict*` nets 37 (b3 0), id bits 0-17 each FDCE-driven | FAIL as written; key mis-named |
| 4 | load_store_unit LUT +838 / FF +27; commit_stage FF +1 / LUT +86; core LUT 145,391 (+807); LUTRAM 870 | PASS |
| 4 | dyn_unit 12,692 LUT vs ObjBounds 6,540 (+6,152) and b3 13,237 (−545); all of it in `_spawn_0` (b3 12,226, ObjBounds 5,367, C.5a 11,703) | FAIL as written; wrong reference |
| 4 | ex+issue +2,869 (ex −292; issue +3,161 = issue_read_operands +3,425, scoreboard −264) vs ±2,500 | FAIL as written; band below the floor |

- **`*pcc_*` (key 3).** `pcc_verdict` is the LABEL of the `always_ff` block (load_store_unit.sv:1562), and `pcc_hold` is a
  combinational net in commit_stage (:234, :382, :402). Neither can be a cell name. The registers are `pcc_v_q`,
  `pcc_live_q` and `pcc_id_q[29:0]` under lsu_i, and `pc_revnode_unk_q` under commit_stage_i. b3's source contains none of
  those four names. The reading already in hand shows the register: its id bits are FDCE-driven nets that b3 does not
  have. The commit-stage FF delta, +1, is exactly `pc_revnode_unk_q`, because the held id `pc_revnode_id_q` was already in
  b3. Graded PASS on that evidence. The confirming cell reading by the `_q` names is requested, and it does not gate.
- **dyn_unit (key 4).** The pre-registration said "It is the same file, so a difference is homing". That was wrong.
  A module's synthesised size also depends on what its environment consumes. ObjBounds' build pruned the LSU's whole
  lifetime verdict (key 3's `*lsu*rvc*` read 0 there and 1,833 here), so its 5,367 is a pruned reading of the same source.
  C.5a's environment is b3's, so the right reference is b3. Against b3 the unit reads −545, and −523 of that is in
  `_spawn_0`. That is the direction the only source change predicts: the two `get_node_query_validity` calls are gone.
  Which cells ObjBounds pruned has not been read. Graded PASS against b3.
- **ex+issue (key 4).** The band was the single-module floor (±2,500) applied to a two-module sum. The core total is
  +807, so modules outside ex, issue and commit moved by about −2,148, and none of them has a source change. That is
  re-homing into issue_read_operands, but it is an inference until the rows name those modules. The full flow is gated
  by the placed total (key 5, ≤ 171,061); about +807 at synthesis puts it near 169.9k. Graded PASS with that stated.
  Requested, not gating: the hierarchical rows of every module outside ex, issue and commit with |Δ| > 500 against b3.

**Ruling: the gate passes, and the full flow goes.** Key 9 now reads: `*dynlt*` present under lsu_i and ex_stage_i,
and `pcc_v_q`, `pcc_live_q`, `pcc_id_q*` (lsu_i) and `pc_revnode_unk_q` (commit_stage_i) present on the routed dcp.
The POS control is `pc_revnode_valid_q`, present on both trees; the NEG is b3's routed dcp, which reads 0 for the four
new names.

**The two confirming readings (synth, gate checkpoint, C.5a / b3), both as ruled:**
- (a) Primitive cells. `*dynlt*` 307 / 0; `*pcc_v_q*` 3 (1 FF) / 0; `*pcc_live_q*` 59 (1 FF) / 0; `*pcc_id_q*` 30 FF, all
  under lsu_i / 0; `*pc_revnode_unk_q*` 30 (1 FF) / 0. The POS control `*pc_revnode_valid_q*` reads 3 / 14. Every
  register the change adds is present, and none is present on b3.
- (b) LUT rows outside ex, issue and commit with |delta| > 500: dom_switcher 10,358 -> 8,065 (-2,293), csr_regfile_i
  27,862 -> 28,966 (+1,104), nothing else. The remaining -959 is spread across modules each under 500. Neither file is
  in the change: `git diff --stat e7af48457 1fc2cd5aa` lists six files, none of them the switcher or the CSR file. Both
  moves are inside the per-module floor, so this is re-homing, now named.

## Full build: the seal and the grade (2026-11-04)

The synth lane's run exited 0 after about 2h31m. Run peak 21.59 GB, largest single process 5.92 GB, collection
14.04 GB. **.bit sha256 5b7b1edcd9bcd2521bd4bd59107747e268487056f1bfcd96403fd612eeb6d764**, 11,443,722 B; the copy
in the tarball is byte-identical. Tarball synth-1fc2cd5aa-exit0.tar.gz, sha256 83cdc0e1..., and its PROVENANCE reads
1fc2cd5aac78.

| Key | Reading | Grade |
|---|---|---|
| 5 | placed LUT 169,738 (<= 171,061); FF 63,642 (63,690 ± 300); slices 48,694, 95.57 % (b3 94.38 %, ObjBounds 93.94 %) | PASS |
| 6 | placer max East global 32x32 at level 5; router effective congestion 5 (2 readings at 4, 2 at 5); one Route 35-447 (congestion prevents routing all nets), which neither b3 nor ObjBounds had | PASS, at the limit |
| 7 | failed, unrouted, partially routed and overlapping nets 0; overlaps 223,569 -> 63,775 -> 20,121 -> 246 -> 0 | PASS |
| 8 | check_timing 0 loops (control 1); LUTLP-1 0; TIMING-23 0 | PASS |
| 9 | on the routed dcp: `*dynlt*` 307; `pcc_v_q`, `pcc_live_q` 1 FF each and `pcc_id_q` 30 FF, all in lsu_i; `pc_revnode_unk_q` 1 FF in commit_stage_i; POS `pc_revnode_valid_q` 3 (b3 14); NEG b3 0 for all five | PASS |
| 10 | WNS −13.758 (b3 −10.047, Δ −3.711); TNS −218,814 (b3 −44,011); failing endpoints 38,719 / 188,738 (b3 13,396); WHS +0.053 | reported, not graded |
| 11 | .bit sha256 as above | PASS |

- **The grade: a flash candidate on every graded key.** Flashing is the lead's decision. The synth lane rules any
  negative WNS not flashable, which is its standing rule, so it calls this NOT FLASHABLE.
- **WNS** moved −3.711 ns. That is inside the one-sample null band of 3.74 ns, at its edge. The worst path is bitstream
  3's own family (csr_regfile `cpmp_q` -> issue_read_operands `fu_data_q`), and C.5a's source does not touch it.
- **TNS and failing endpoints** grew about 5x and 2.9x. The failures are concentrated where C.5a's CAP_WB merge feeds:
  issue_read_operands 11,137, scoreboard 10,580, lsu_i 4,324 and dyn_unit 2,073. The synth lane could not diff these
  by module, because b3's table covers only 4,206 of its 13,396. This growth is NOT ATTRIBUTED.
- **The reading that attributes it**, requested before the flash decision: C.5a's own cells on the worst 500 paths,
  with a POS control. Those cells are `*dynlt*`, `*dyn_ldst*`, `pcc_*` and `pc_revnode_unk_q`. Also requested: the
  worst slack through any of them, and the same-family slack of `cpmp_q` and `cur_idx_q` against b3.
- **Congestion** is at the pre-registered limit, with a router warning that b3 lacked. Slices are at 95.57 %, the
  highest of the three builds. This is consistent with the routability memory: around 170k LUT is marginal.

**The attribution reading** (synth, read-only, the same script on both routed dcps). The worst 500 setup paths span
−13.758..−11.048 on C.5a and −10.047..−7.098 on b3.
- **C.5a's cells on the worst 500:** `pc_revnode_unk_q`'s cells are on 109 paths, one per path (the name matches 24
  LUT, 5 CARRY4 and 1 FF). `*dynlt*`, `*dyn_ldst*`, `pcc_v_q`, `pcc_live_q` and `pcc_id_q` are on 0. POS `*fu_data_q*`
  is on 137 paths (b3 81).
- **The worst slack through each:** `pc_revnode_unk_q` −11.545 at 98 levels; `*dynlt*` −9.626 at 100; `*dyn_ldst*` −8.560
  at 77; `pcc_v_q`, `pcc_live_q` and `pcc_id_q` are positive (+7.9, +7.8 and +16.9).
- **The five worst paths through C.5a's cells** all start at csr_regfile `cpmp_q_reg[4]` (bE[1]) and end at dom_switcher
  `cur_base_q`, at −11.3 to −11.5 and 98 levels. The source shows the arc (commit_stage.sv:379-407, :532):
  - `pcc_hold` is set from `pc_revnode_unk_d`, the same-cycle next state.
  - That next state comes from the 30-bit compare of the head's PC-capability id against the tracked id.
  - The hold gates the commit acknowledge, which reaches the domain switcher.
  - b3 had the same compare, but it fed only the 25 exception (`!pc_revnode_valid_d`). The new arc is that compare
    into the commit acknowledge.
- **The same family against b3:** from `cpmp_q`, the worst is −13.758 at 114 levels against −10.047 at 111, with the same
  endpoint, fu_data_q[0] cap_metadata_b. 38,712 of the 38,719 failing endpoints are reachable from `cpmp_q`. From
  `cur_idx_q`: −13.545 at 113 against −9.351 at 112.

**Reading.**
- The WNS path is NOT through C.5a's cells. b3's own family moved about −3.7 ns at +3 levels, which is delay per level,
  consistent with the denser placement (95.57 % slices, congestion at level 5).
- Nearly every failing endpoint hangs off that one family. So the 2.9x endpoint count and the 5x TNS are what a
  uniform ~3.7 ns shift of the cpmp_q cone would give. That is consistent with the shift, and it is not proof.
- C.5a adds ONE new path family inside the worst 500, cpmp_q -> the PC-id compare -> `pcc_hold` -> commit acknowledge
  -> dom_switcher, at −11.5. It is 2.2 ns better than the WNS.
- One build cannot separate placement density from the new logic. This is the second time the cpmp_q family has led a
  build: the batch led on it too.
- **A lever, if a respin is wanted (not proposed here; that is the lead's call, and the code is the external
  collaborator's):** drive `pcc_hold` from flops by holding on the registered unknown state plus a registered id-change.
  That costs one cycle per PC-capability change and takes the 30-bit compare off the commit acknowledge.

## Pre-registered silicon readings (first boots; the runtime and board lanes run them)

- **hpm kit** (`hpm-bare-2026-11-03`): hpm-3 and hpm-4 read ObjBounds' stall and cycle pattern, not bitstream 3's.
  Silicon cycles are not the simulator's.
- **R-67 kit:** t1-fire reads FIXED (25 at the target, no marker). The three controls are unchanged.
- **The ObjBounds kit** (`objb-bare-2026-10-30`):
  - stale-access reads bitstream 3's vector (25 ×4, 0, 25 ×2, 0x77, 6), and objb-stale-w bitstream 3's.
  - s6/s7 read ObjBounds' cycles to within the PC-tracker's constant shift.
  - s3 reads ABOVE bitstream 3's: about +3 per LDC, all misses.
  - s4 pass 1 reads ABOVE bitstream 3's (sim 1,364 against 656: every node re-misses after the flush), and pass 2
    reads ObjBounds'.
- **The constant shift.** 1fc2cd5aa moves every arm by a constant few cycles against its siblings: s1 +7, the hpm arms
  −7, s6 +7 over ObjBounds. The SPEC readings carry it as a per-context-entry cost, not a per-access one.
- **SPEC.** The re-miss after every escape (about 11 cycles per node) and perlbench's 1,131-16,030 live nodes are what
  the runtime lane's lc-model hit rates measure. A perlbench regression there goes to the lead before the flash.
