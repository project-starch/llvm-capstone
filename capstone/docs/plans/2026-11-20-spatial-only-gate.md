# Spatial-only core, step A: synthesis-only gate as a measurement (board lane copy of the pre-registration)

Status: pre-registered 2026-11-20, before the launch. The run is the rtl lane's request, and the bands are its own,
copied unchanged. Brief: post-submission programme, line A row 2 and section 7.2 item 3. No full build and no flash follow
from this run.

**Tree.** capstone-ariane spatial-only cf7a00ce6; P1 d83d3a633 is an ancestor. Against P1, `core/` changes in six
files, +140 lines with none removed:
- the config package defines `CAPSTONE_SPATIAL_ONLY` unless `CAPSTONE_LIFETIME_ON`, which the FPGA flow never sets;
- `core/capstone_rev_node_stub.sv` (112 lines) replaces the rev_node unit in `ex_stage.sv` (:1799). It is listed in
  `core/Flist.cva6` (:120), which the FPGA flow reads through `util/flist_flattener.py`;
- `commit_stage.sv`, `load_store_unit.sv`: every verdict reads live, and the PC tracker never holds.

Nothing else is removed. The question is how much of the now-dead lifetime logic synthesis removes by itself.

**Flow.** P1's synthesis-only gate (`launch-job.sh` gate), with the same watchers and probes as the I-cache gate:
- source-list watch and the in-flight loops-only gate (ref P1's gate);
- the kept checkpoint, LUTLP-1 with the toy control, and check_timing loops (rc-keys.tcl loops-only) with the toy control;
- grader `twc/spatial/grade-sp.py`. It was dry-run on B5a's gate, where it reproduced ex_stage +5,647 and FAILed the
  mapping key as recorded. It exits 2 on missing data.

**Graded:**
- mapping in P1's class: the Technology Mapping peak within ±500 MB of P1's 5,041.7 MB;
- loops: 0 "Found timing loop" headers, check_timing 0 and LUTLP-1 0, each control firing.

**The rtl lane's bands** (reported in or out, against P1's gate: ex_stage 55,982 LUT, core 162,403, design 189,693):
- rev_node instance under 200 LUT;
- ex_stage -5,000 to -9,000 LUT;
- load_store_unit's own cells -300 to -2,500;
- core -5,500 to -11,000.

If ex_stage lands above -3,000, the tool keeps the dead logic, and step B's explicit removals carry the area. Below
-9,000 means something uncounted went, and the per-module rows go to the rtl lane.

**Reported rows, both runs:** ex_stage and its children, lsu_i with its own cells and children, commit_stage_i,
csr_regfile_i, i_pmp_data_if, core and design. Latch lines (`[Synth 8-327]`) are reported against P1's 2.

## Reading: PASS on both graded keys; ex_stage -11,239 LUT, below the rtl lane's band (apollo, 2026-11-20, 12:21-13:11Z)

Record: `<mnt-hdd>/synth/sp-cf7a00ce6-gate-out/grade-sp.txt`; P1's rows come from `pw-d83d3a633-gate-out`; kept
checkpoint `twc/gate-synth-sp-cf7a00ce6.dcp` (df0473d0bab1…).
- Graded:
  - mapping: the Technology Mapping peak is 4,914.4 MB (P1 5,041.7, -127.3), and the whole run peaks at 4,982.5 MB;
  - loops: "Found timing loop" 0, check_timing 0 (toy 1), LUTLP-1 0 (toy 1), in-flight PASS, source-list watch OK.
- The rev_node row's module is `capstone_rev_node_stub`. add_sources.tcl lists both files, and only the stub elaborates.
- Guard: stopped after synth_design by design (rc 143), at a **peak RSS of 32.02 GB**. That is inside the 40 GB ceiling
  and 4.7 GB above P1's gate (27.35), although the tool's own mapping peak is lower. Reported, not explained.
- Latch lines: `[Synth 8-327]` 2, as on P1.

| Row | LUT (Δ vs P1) | FF (Δ) | LUTRAM (Δ) | Band | |
|---|---|---|---|---|---|
| design | 181,926 (-7,767) | 74,437 (-3,696) | 1,600 (-880) | | |
| core | 154,629 (-7,774) | 50,399 (-3,696) | 396 (-880) | -11,000..-5,500 | in |
| ex_stage | 44,743 (-11,239) | 19,778 (-3,616) | 0 (-880) | -9,000..-5,000 | **OUT (below)** |
| ex_stage own | 1,872 (-756) | 3,126 (-1,530) | | | |
| csr_buffer_i | 2,849 (-2,188) | 72 (+25) | | | |
| dyn_unit | 9,720 (+1,722) | 1,234 (-5) | | | |
| fpu | 7,950 (-589) | 2,210 (+284) | | | |
| i_mult | 686 (-382) | 388 (0) | | | |
| rev_node (stub) | 14 (-5,353) | 235 (-1,408) | | < 200 | in |
| lsu_i | 21,649 (-3,767) | 12,502 (-982) | 0 (-880) | | |
| lsu own | 0 (-2,474) | 379 (-923) | 0 (-880) | -2,500..-300 | in |
| lsu i_cva6_mmu | 9,725 (-783) | 8,104 (-3) | | | |
| lsu i_store_unit | 4,922 (-368) | 1,908 (-5) | | | |
| lsu i_pmp_data_if | 427 (-67) | 915 (0) | | | |
| commit_stage_i | 131 (-20) | 236 (-57) | | | |
| csr_regfile_i | 28,393 (+1,420) | 6,286 (-16) | | | |

Why ex_stage went below -9,000 is the rtl lane's to read from these rows. By the numbers:
- the stub (-5,353), lsu (-3,767) and csr_buffer_i (-2,188) account for -11,308 between them;
- dyn_unit +1,722 and csr_regfile +1,420 move the other way. They are inside the ±2,500-per-module redistribution floor,
  so they are not attributed.

The lsu's own LUTRAM is all gone (-880), which is the verdict arrays. No full build or flash follows from this run.
