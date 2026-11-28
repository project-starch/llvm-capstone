# The hardware-cost table rebuilt on Line B (2026-11-26; campaign plan item 13)

*RTL lane, at the runtime lane's request. Every number below is read from a sealed build's own routed reports and
none is an estimate. The paper (Table hw-cost, section 7.4, macros/results.tex at ebc1700a) is not touched; this
gives its rows on Line B by the rule the paper's own comment states.*

## The three builds and their reports

| arm | seal | bitstream sha256 | build tree (apollo) | routed |
|---|---|---|---|---|
| Plain (matched CVA6) | 8db8f9cda | efc3ab8abd5aa194... | /mnt/hdd/<user>/synth/rvbm-8db8f9cda-full | 108,786 nets, 0 routing errors |
| Full | b42c8c644 | 6a09f34d50842e01... | /mnt/hdd/<user>/synth/b5b2r77f-b42c8c644-full | 198,616 nets, 0 routing errors |
| +Spatial (spatial-only) | 14f2ef81a | df9de4d49b21b475... | /mnt/hdd/<user>/synth/b5b2spf-14f2ef81a-full | 188,208 nets, 0 routing errors |

- **Report:** `corev_apu/fpga/reports/ariane.utilization.rpt` in each tree, `report_utilization -hierarchical`, Design
  State: Routed.
- **Totals:** its whole-design rows equal each build's `work-fpga/ariane_xilinx_utilization_placed.rpt` (LUT 71,991 /
  150,303 / 141,016; FF 42,706 / 68,506 / 63,917).
- **Seals and sha256:** each tree's HEAD is its seal. The bitstream sha256 is recomputed from
  `/mnt/hdd/<user>/capstone/bitstreams/`.
- **Local changes:** the only ones in the trees are the build-environment scripts (env.sh, env_cap.sh, fpga-env.sh,
  synth-guard.sh), none in RTL.
- **8db8f9cda's artifact collection failed** (`collect.exit=127`). Its routed reports and bitstream are intact, and it
  is the same matched base the paper already uses.
- **Parser:** the rtl lane's scratch hier.py rebuilds instance paths from the report's indentation; every number in
  this note is a row of those reports.

## Core totals (the cva6 instance)

| | Plain 8db8f9cda | Full b42c8c644 | +Spatial 14f2ef81a |
|---|---|---|---|
| LUT | 46,968 | 124,710 (**2.655x**) | 115,401 (**2.457x**) |
| FF | 20,684 | 46,047 (2.226x) | 41,461 (2.004x) |
| RAMB36 / RAMB18 | 50 / 0 | 100 / 2 | 100 / 2 |

- For comparison, P1 (d83d3a633) in the paper is 161,602 LUT, 3.44x (\evalHwCoreRatio).
- **The spatial-only bound becomes a measurement: 14f2ef81a's core is 2.457x CVA6's LUTs.** The paper's bound for P1 was
  2.3-3.0x (\evalHwSpatialOnlyRange), and the measurement lies inside it.
- Full minus +Spatial, the lifetime machinery as the two seals differ: core LUT 9,309, FF 4,586, no BRAM.

## Per-module breakdown, the paper's rule

Every row is a module's routed LUTs on the arm minus on 8db8f9cda. Each module is assigned by the list in the paper's
comment (macros/results.tex at ebc1700a, the rtl lane's 2026-11-17 breakdown), applied unchanged:
- **spatial datapath and bounds:** issue_read_operands, scoreboard, wt_dcache, store_unit, csr_buffer, pipe_reg_load,
  lsu_bypass, id_stage, load_unit, commit_stage;
- **lifetimes:** rev_node, plus ex_stage's own logic (the paged walker and the record TLB);
- **mixed LSU + dyn_unit:** lsu_i's own logic and dyn_unit;
- **the rest:** each other row by its own name.

| row (paper macro) | P1 (paper) | Full b42c8c644 | +Spatial 14f2ef81a |
|---|---|---|---|
| spatial datapath and bounds (\evalHwSpatialLut) | 55,363 | **58,074** (FF +14,173) | 59,061 (FF +13,532) |
| lifetimes: rev_node + ex_stage own (\evalHwLifetimeLut) | 7,943 | **8,217** (FF +6,554) | 1,499 (FF +3,484) |
| mixed LSU check + dyn_unit (\evalHwLsuMixedLut) | 10,612 | **5,588** (FF +3,140) | 2,109 (FF +2,250) |
| dom_switcher (\evalHwSwitcherLut) | 11,077 | **0: no instance** | 0: no instance |
| mixed csr_regfile (\evalHwCsrMixedLut) | 22,676 | **364** (FF +591) | 185 (FF +593) |
| tag path: wt_axi_adapter (\evalHwTagPathLut) | 4,325, +48 RAMB36 +2 RAMB18 | **4,231**, +50 RAMB36 +2 RAMB18 | 4,235, +50 RAMB36 +2 RAMB18 |
| counters / debug | 480 | 28 | 29 |
| stock units (the rest: FPU +706, MMU +532, ...) | 3,128 | 1,456 | 1,479 |
| rows sum (the core delta) | 115,604 (114,634) | 77,958 (77,742) | 68,597 (68,433) |

Reading the rows:
- **The record cache row has no hardware on Line B.** The paper's \evalHwRecordCacheLut is a B5a-minus-P1 difference;
  no Line B seal builds a record cache. The cache model decides it (tools/record-cache-model/README.md).
- **The switcher is gone.**
  - Neither b42c8c644's nor 14f2ef81a's routed hierarchy has a dom_switcher instance or any module of that name,
    while P1 carries 11,077 LUT of it.
  - It is not flattened into the core's top: the core's own cells are 34 LUT / 136 FF on both.
  - Its requests come only from CSRUNV and the supervised CALL/RETURN, all illegal under the Sublet format (decoder.sv
    at b42c8c644), so the inputs are constant and synthesis trims it.
  - The trim itself is not read from a log: the synthesis.log kept in the seal's tarball lists no per-module trimming.
  - What is measured is that the routed design has no switcher logic left under its name.
- **csr_regfile is no longer a mixed cost:** +364 LUT against P1's 22,676. The supervisor and context CSRs went with
  CSRUNV (step 8a).
- **Lifetimes, Full against +Spatial:**
  - rev_node is 5,750 LUT on Full and 91 on +Spatial, the stub.
  - ex_stage's own logic is 2,467 and 1,408. The walker and record TLB leave with the lifetime table, and the
    remainder is ex_stage's other own logic.
  - lsu_i's own logic is 2,846 and 512: the verdict cache.
  - dyn_unit's load syncer is 1,821 and 658.
- **Inside the floor:** the spatial row moves +987 from Full to +Spatial. That is inside the per-module redistribution
  floor of +-2,500, so it is placement and synthesis re-homing, not added logic. issue_read_operands is +1,348 and the
  scoreboard +1,491.
- **Overlap:** the module rows overlap Vivado's hierarchical totals by about 200 LUT, as in the paper's own breakdown.
