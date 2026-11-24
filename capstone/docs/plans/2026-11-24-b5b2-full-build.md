# B5b-2's full build: the board lane's instruments, written before the build reads (2026-11-24)

The rtl lane requested this build and pre-registered its keys on llvm-capstone `land/rtl-b5b2-plan` 1707da4be58a
(`capstone/docs/plans/2026-11-24-b5b2-trap-route.md`, "B5b-2's full build: pre-registration"). The keys, the predictions
and the verdict are the rtl lane's. This file records how the board lane builds and reads it, written before any reading
exists. The gates before it are in `2026-11-24-b5b2-gates.md`.

## The build

- **Source:** capstone-ariane b5b2-step6 0c04581493e3, which is the branch tip.
  - Checked here: core/ differs from gate 6b's 0c9992bc6 only in load_store_unit.sv's two-line sim-only
    `ifndef NS_SIM_NOFLUSH`. The synthesised RTL is therefore gate 6b's.
- **Launch:** 13:46:47Z on apollo, gated on `mem-headroom.sh 44 16` (107 GB available).
  - `synth/twc/full-chain.sh` (sha16 7f500c1301f2b153) runs the build under vivado-guard at 40 GB.
  - Attached: the loops-only in-flight gate against gate 6b and the LUTLP-1 stop.
  - The guard was confirmed sampling at 13:48:11Z.
- **Seal name:** `…-pagedtbl-r76-r75d-b5b2_0c0458149.bit`, which is B5b-1's name with b5b1 replaced by b5b2.
- **Base:** B5b-1's seal on d0afdb40f (sha256 38491cf4…), graded from its grade-board tree.

## The grader and the readings it uses

The grader is `synth/twc/b5b2full/grade-b5b2-full.py` (sha16 8fecb23c138f6cd8). The post chain is `post-b5b2-full.sh`
(fba0494a98597393). Once the build ends, the chain runs four instruments on the routed checkpoint, each under its own
guard, and then grades.
- **loops.tcl** (cc0530a30591968b): check_timing loops and TIMING-23. The toy loop checkpoint is the control and must
  read at least 1 on each.
- **lutlp.tcl** (unchanged): LUTLP-1, again with the toy as its control.
- **b5b2-paths-core.tcl** (6929e38dbf9afcf7): the clk_out1 path readings.
- **grade_full.py** (unchanged), loaded with one `need()` relaxed for a router log that has no hotspot block, as in
  B5b-1's grader v2.

**Keys as encoded:**
- placed LUT 173,100 ± 3,000;
- placed FF 78,033 ± 2,000;
- routed clean: 0 errors, 0 unrouted, 0 partial;
- placed congestion ≤ 5;
- router congestion ≤ 5. A log with no hotspot block reads "no hotspot reported" only if its router Phase 2 ran. B5b-1's
  own log has no such block, so R-75 (d)'s log is the block's control.
- loops 0, in four readings:
  - check_timing in the build's report;
  - check_timing on the routed checkpoint;
  - TIMING-23;
  - LUTLP-1 on the routed checkpoint.
- clk_out1 WNS inside [-1.89, +5.59] (B5b-1's +1.851 ± 3.74), one-sided as pre-registered: a reading above the band
  is reported, not graded.
  - It is read from the routed timing summary's Intra Clock Table.
  - The path tool's clk_out1 worst must equal it within 0.001 ns, otherwise ERROR.

**Reported, not graded:**
- design-wide WNS, TNS, WHS and failing endpoints; clk_out1 TNS; slices;
- RAMB36 against B5b-1's 113. Gate 6b's +2 in twc_ep_mem should carry.
- the clk_out1 worst 20;
- the worst setup path through each timing point the gates could not read. A pattern with no nets prints NONETS, never
  a zero:
  - (a) the LSU's cap_exception cone, nets `*ex_stage_i/lsu_i/*cap_exception*`;
  - (b) the CPTAG walk, `*i_adapter/cptag_*`;
  - (c) commit's cap_we, `*cap_we_id_ex*`;
    - That net is commit's output: commit_stage's cap_we_o drives cap_we_id_ex (cva6.sv:2170), which enters issue_stage
      as cap_we_i (:1764). Gate 6b's checkpoint keeps the net by that name only at i_scoreboard's input, and
      commit_stage_i there has no cap_we_o pin.
    - Under Sublet, cap_we_o[0] is gated by `rs1_tag_i[trans_id]` (commit_stage.sv:600-603, fix 2), so the path the
      rtl lane asked for is the one from rs1_tag_q_reg through that net.
    - Also timed: from rs1_tag_q_reg, and from rs1_tag_q_reg through that net.
  - (d) `*csr_scr_*`;
- the sha256.

## Controls, run before the build reads

**Names on gate 6b's synthesis checkpoint.** It has no clocks, so the path Tcl fails closed there with NO-SLACK.
`names.tcl` counted each pattern instead:
- lsu_cap_exception 71 nets, cptag 250, cap_we_id_ex 1, csr_scr 100;
- rs1_tag_q_reg 8 cells;
- the nonsense pattern 0.

**On B5b-1's routed checkpoint:**
- loops.tcl reads check_timing 0 and TIMING-23 0; on the toy it reads 1 and 1.
- The path Tcl reproduces B5b-1's clk_out1 worst exactly: +1.851, 103 levels, i_issue_read_operands fu_data_q
  cap_metadata_b[14] to cap_metadata_a[17].
- The base's timing points:
  - lsu_cap_exception +13.535 (33 levels);
  - cap_we_id_ex +11.561 (40 levels);
  - cptag, csr_scr and rs1_tag_q absent, as they should be before steps 4-6.

**The grader:**
- B5b-1 against itself reads PASS 10/10. The DRY chain wrote only under `b5b2full/dry/`.
- Fixtures copied from B5b-1's reports and doctored each FAIL their own key:
  - placed LUT 176,101 (176,100 passes) and FF 80,034;
  - one unrouted net;
  - placed congestion level 6 and router level 6;
  - check_timing 1, TIMING-23 1, LUTLP-1 1;
  - clk_out1 -1.900 (-1.889 passes).
- clk_out1 +6.000 passes as "outside on the good side, reported".
- These read ERROR, never a grade: the path tool disagreeing with the table, a toy that does not fire, and a loops log
  without its end marker.
