# Spatial-only core (steps A + B): full build, board lane pre-registration

Status: pre-registered 2026-11-20, before the launch. The keys are the rtl lane's (land/rtl-spatial-prereg 9dacb7d2658e,
`capstone/docs/plans/2026-11-20-spatial-only-core.md`), adopted unchanged. Brief: programme line A row 2.

**Tree, read at source.** capstone-ariane spatial-only-b 1f1ed11f9; P1 d83d3a633 and step A cf7a00ce6 are both
ancestors.
- Against P1, `core/` changes in seven files, +149 lines with none removed.
- Against step A, it adds `core/decoder.sv` +9 only: REVOKE, MREV and DROP decode illegal under `CAPSTONE_SPATIAL_ONLY`.
- The submodule pins equal P1's.
- Step A's synthesis-only gate on this flow (board/meas-pw 2f6d63a7e6eb): core -7,774 LUT, -3,696 FF post-synth.

**Keys** (`graders/grade_full.py` against P1's seal, plus the rtl lane's bands):
- placed LUT [177,068, 182,068] (P1 - 7,800 ± 2,500); placed FF [70,843, 74,843];
- routed clean; placed congestion ≤ 5; check_timing loops 0;
- WNS against P1's null band [-16.913, -9.433]. Above the band is reported as such; below it goes to the worst-path
  families before any flash.

**Memory.** Step A's gate peaked at 32.02 GB guard RSS, against P1's 27.35, unexplained. This run is under the same 40 GB
ceiling, and the guard's log is read at launch and at the end. A kill at the ceiling is recorded with its phase, and
there is no retry with a higher ceiling.

**Failure branch.** On a placement or routing failure, placed area and congestion only are recorded; no WNS is read and
nothing is flashed. There is no retry and no flow change. The same holds for a guard kill.

**Name.** `caplifive_ids55-native-tagcache-mlp-fpieee-hpmreplay-dynbypass3-dtlb64-shrinkfix-ic32k-r72-stcnoclaim-twc16k-merge-f2-hpmtag-lrscfix-tagoverlap6c-pagedtbl-spatialonly_1f1ed11f9.bit`.
It is sealed and handed to the runtime lane. Its board arm boots the CSCHECK-emulation firmware variant
(`-DCSC_SPATIAL_ONLY`), the rtl lane's stated second variable.

## Reading: routed and sealed; every rtl-lane key passes, and WNS lands ABOVE P1's band at +0.170 ns (apollo, tag spo-1f1ed11f9, 2026-11-20)

Launched 15:13:34Z, sealed 16:56:44Z as
`caplifive_ids55-native-tagcache-mlp-fpieee-hpmreplay-dynbypass3-dtlb64-shrinkfix-ic32k-r72-stcnoclaim-twc16k-merge-f2-hpmtag-lrscfix-tagoverlap6c-pagedtbl-spatialonly_1f1ed11f9.bit`.
- sha256 d16c21977962e3d6ad2ee03adeb7eff4b4843b3c21c76a8e58e305201c48140e; the stored copy hashes equal;
- tarball 235c4ce2745b…; PROVENANCE HEAD 1f1ed11f9;
- guard exit 0 at a 32.02 GB run peak, the same as step A's gate (P1's full build 27.39 GB). It is inside the ceiling and
  stays unexplained;
- in-flight loops-only PASS.

| Key | Reading | Band | |
|---|---|---|---|
| placed LUT | 179,661 (88.16 %, P1 -7,707) | [177,068, 182,068] | in |
| placed FF | 72,847 (-3,696) | [70,843, 74,843] | in |
| placed congestion ≤ 5 | 4 (P1 4) | | PASS |
| check_timing loops | 0 | | PASS |
| routed clean | 0 errors, 232,712 / 232,712 | | PASS |
| WNS against P1's null band [-16.913, -9.433] | **+0.170 ns, TNS 0.000, 0 of 214,985 endpoints failing** | above the band | reported as such |

Slices are 50,422 (98.96 %), 378 fewer than P1.

**Timing, read at the routed report.**
- "All user specified timing constraints are met."
- The core clock is `clk_out1_xlnx_clk_gen` at 40.000 ns (25 MHz), the same as P1's, and `corev_apu/` is unchanged
  against P1.
- That clock's own WNS is +0.523 ns, against P1's -13.173; the design's +0.170 is another group's.
- What carries P1's 13 ns is not established by this reading. The worst-path families of the two builds are the next
  question, and it is the rtl lane's.

**The instrument.**
- grade_full.py printed NO DATA: the router printed no "Effective congestion level" block, which it prints only when it
  finds congested clusters. Its routing task and all its phases are in the log.
- The grade above is from a local copy, `grade-board/grade_full_nocongline.py`. It differs in one reading: with no block
  printed, router congestion reads "none printed", but only if the routing task ran.
- On P1 against itself, the copy reads 4/4, as the original does.
- With it, grade_full.py's own key list reads 3/4. Its WNS key FAILs for leaving the band upward, which the
  pre-registration reports and does not count as a failure.

The seal goes to the runtime lane. Its board arm boots the `-DCSC_SPATIAL_ONLY` firmware variant, the stated second
variable.

## Reading: where P1's 13 ns sits (read-only on both routed checkpoints; rtl and runtime lanes' asks, 2026-11-20)

Scripts: `<mnt-hdd>/synth/twc/lattr/lifetime-attr.tcl` and `modgroups.tcl`. Logs: `lattr/{P1,SPO,P1x,MG-P1,MG-SPO}.log` and
`*.w20.rpt`. Each run was under a 16 GB guard and peaked at about 3.5 GB. The design-wide counts were taken first, so a
zero on the paths means something.

**(b) Lifetime names on the worst 500 paths.** A path matches if any of its cells OR nets carries the name.

| Pattern | P1 design cells / nets | P1 worst-500 paths | Spatial worst-500 paths |
|---|---|---|---|
| `*/ex_stage_i/rev_node/*` | 8,291 / 10,855 | **500** | 0 (the stub: 260 / 357) |
| `*pc_revnode*` | 260 / 349 | 39 | 0 |
| `*revnode_*` | 4,137 / 10,907 | 39 | 0 |
| `*pcc_*` | 224 / 681 | 39 | 0 |
| `*rvc_*` | 4,253 / 12,255 | 0 | 0 |
| `*cpmp_tracked_revnode*` | 1,141 / 6,201 | 0 | 0 |
| `*i_rev_node*` | **0 / 0: no such name** | 0 | 0 |
| POS `*fu_data_q*` | 19,758 / 23,769 | 389 | 5 |
| NEG | 0 / 0 | 0 | 0 |

P1's worst 500 span -13.173 to -11.136 ns. Spatial-only's span +0.170 to +0.990.

**(a) The worst paths.**
- P1's five worst have 109-111 logic levels, and **31 of those logic cells are inside `ex_stage_i/rev_node/`** (by the
  cell lines of `P1.w20.rpt`). They start at `dom_switcher/cur_idx_q` and end at
  `issue_read_operands/fu_data_q[cap_metadata_a]`.
- Spatial-only's twenty worst are DDR controller paths at 0 levels.

**(c) The same families on spatial-only**, the start and end stems of P1's worst 20:

| Family | P1 slack / levels | Spatial-only slack / levels |
|---|---|---|
| `dom_switcher/cur_idx_q` → `fu_data_q[cap_metadata_a]` | -13.173 / 111 | +11.126 / 37 |
| `csr_regfile_i/cpmp_q[bE]` → `fu_data_q[cap_metadata_a]` | -13.146 / 121 (from any cpmp_q) | +11.728 / 35 |
| `dom_switcher/cur_idx_q` → `i_scoreboard/mem_q[rs1_metadata]` | | +7.373 / 46 |
| from any `cpmp_q` / `cur_idx_q` / dom_switcher `_init_0` / `event` | -13.146 / 121, -13.173 / 111, -11.478 / 110, -12.021 / 110 | +3.993 / 49, +3.525 / 48, +3.477 / 46, +3.257 / 46 |

**Module groups, worst 2,000 paths** (runtime lane's ask):
- P1 has 14 groups. All of its top ten start at `dom_switcher/cur_idx_q` or `csr_regfile_i/cpmp_q`. Their worst slacks
  run from -13.173 (42 paths, to issue_read_operands) and -12.695 (1,455 paths, to the scoreboard) down to -10.6. They
  have 76-120 levels and end in issue_read_operands, the scoreboard, the LED stretch counter, the dom_switcher, lsu_i,
  the frontend and id_stage.
- Spatial-only has 5 groups. `i_ddr → i_ddr` holds 1,973 paths at +0.170 and 0 levels. The worst core-clock group is
  `issue_read_operands → issue_read_operands`, at +0.523 over 98 levels.

**Against the rtl lane's stated test** ("if P1's worst 500 are mostly through the lifetime names, the 13 ns is that cone;
if not and the same families simply gained at equal depth, it is placement"):
- all 500 cross the rev_node unit's hierarchy;
- the families did not gain at equal depth: the two same-endpoint families lost 74 (111 → 37) and 86 (121 → 35)
  logic levels, and from the same start registers to any endpoint the loss is 62-72.

By that test the reading is the lifetime cone, not placement. This is N = 1 build per tree, and the share of the 74 lost
levels that lies outside the 31 cells inside rev_node is not attributed here.

The spatial core's own deepest path, issue_read_operands → itself at 98 levels, meets 40 ns by 0.523 ns.

## Reading: where P1's worst paths cross the rev_node hierarchy (rtl lane's register-cut ask, 2026-11-20)

Script: `<mnt-hdd>/synth/twc/lattr/cone-ports.tcl` on P1's routed checkpoint. Log `lattr/CONE.log`; full path reports
in `lattr/CONE.full.rpt` (`report_timing -path_type full -input_pins`). The guard peaked at 3.42 GB.

- **The port lookup did not resolve.** Every crossing reads `?noport(...)`: no hierarchical port pin of rev_node was
  found on the net segments of the routed netlist. The names below are the nets at the boundary pins, not port names.
- The three named paths cross identically. They are the worst path (-13.173, 111 levels; cur_idx_q[3] →
  fu_data_q[0][cap_metadata_a][19]), the cur_idx_q family, and the cpmp_q family (-13.146, 121 levels). Each crosses:
  1. IN at `ex_stage_i/rev_node/rd_data[5]`;
  2. OUT at `issue_read_operands/_ep_res_valid_selector_q[1]_i_8__0_2`;
  3. IN again at `ex_stage_i/rev_node/mem_q[6][sbe][cap_result][rs1_metadata][26]_i_28_0`;
  4. OUT at `ex_stage_i/dyn_unit/_spawn_0/compress_bounds0[t]_return__1[2]`.
- **Histogram over the worst 500.** All 500 have cells under the rev_node hierarchy, and none was VOID.
  - 461 enter at `rev_node/rd_data[5]` and leave into `issue_read_operands/_ep_res_valid_selector_q[1]_i_8__0_2`.
    Of those, 72 cross a second time via `_ep_res_valid_selector_q_reg[5]_21` / `[4]_15`, and 94 re-enter at the
    `mem_q[6]...rs1_metadata` net (93 leave into `dyn_unit/_spawn_0/compress_bounds0[t]_return__1[0..8]`, 1 into a
    dyn_unit `mem_q_reg`). 295 + 72 + 94 = 461.
  - 39 enter at `rev_node/rv_rd_res[data_rvalid]` and leave into `i_scoreboard/pc_revnode_unk_d145_out`.
- **Caveat on every hierarchy-based count in this document, the "31 cells inside rev_node" included.**
  - The flow synthesizes with the default hierarchy mode and `RETIMING true` (`run.tcl:87`), so logic and registers
    move across module boundaries.
  - A net under `rev_node/` that is named after the scoreboard's `mem_q` shows it has happened here.
  - Cell and net hierarchy in the routed netlist is therefore approximate RTL ownership. The spatial-only comparison
    (the same families 74-86 levels shorter) does not depend on it.

## Reading: the split at the rd_data crossing, and the next family once the rev_node cone is cut (rtl lane's ask, 2026-11-20)

Read-only on P1's routed checkpoint: `<mnt-hdd>/synth/twc/lattr/CONE.full.rpt` (the split) and `next-family.tcl` →
`NEXT.log` / `NEXT.next.rpt`. The period is 40.000 ns.

**(1) Arrival at the crossing.** The three named paths, with the cumulative time in the report:

| Path | Launch clock pin | `rev_node/rd_data[5]` | Endpoint arrival | Required | Before / after the crossing |
|---|---|---|---|---|---|
| cur_idx_q[3] → fu_data_q[cap_metadata_a][19] (worst, 111 levels) | 1.281 | 26.625 | 54.537 | 41.364 | 25.344 / 27.912 ns |
| cpmp_q[5][bE][1] → the same endpoint (121 levels) | 1.588 | 26.658 | 54.570 | 41.424 | 25.070 / 27.912 ns |

- Before the crossing, the path runs through the D$ data banks, the adapter's `data_rdata_q` logic and the missunit
  into `rd_data[5]`.
- A naive split on P1's placement, as arithmetic and not a prediction: a flop taking the data at that pin leaves about
  +14.7 ns before it, and launching from it leaves about +11.7 to +11.9 ns after it.

**(2) The worst path that does not cross rev_node.**
- Scan: none of P1's worst 5,000 paths (down to -9.999 ns) is free of cells under the rev_node hierarchy.
- With timing disabled in memory through all 8,291 primitive cells under rev_node, the worst path is **-4.631 ns, 56
  levels, `dom_switcher/cur_idx_q[3]` → `lsu_i/lsu_bypass_i/mem_q[0][data]/CE`**. All of the worst 20 are that
  family.
- Controls: before disabling, the worst path carries rev_node cells; after it, none of the reported 20 does.
- On spatial-only, the same start to the same lsu_bypass endpoints reads +3.525 ns over 48 levels.
- Limits:
  - disabling every cell under the hierarchy is wider than a cut on `rv_rd_res`, which leaves the unit's own internal
    paths and the post-cut side;
  - retiming makes the hierarchy approximate (see above);
  - re-placement after an RTL change moves all of these numbers.
