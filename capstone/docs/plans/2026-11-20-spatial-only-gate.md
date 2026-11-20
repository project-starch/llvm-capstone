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
