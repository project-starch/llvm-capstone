# B5b-1 step 2a (the plain-SV lifetime unit): synthesis-only gate, board lane copy of the pre-registration

Status: pre-registered 2026-11-23, before the launch. The keys are the rtl lane's (land/rtl-b5b1 4d7144c160bf,
`docs/plans/2026-11-22-b5b1-format.md`, "Step 2a's synthesis-only gate"), copied unchanged.

**Tree, read at source.** capstone-ariane b5b1-step2-fix a5b2a0d1f, on origin.
- Its base 4c576521d (the 1b tree) is an ancestor.
- Outside `verif/`, eleven files change, +867 / -11:
  - the new `core/capstone_sublet_life.sv` (614 lines), listed in `core/Flist.cva6`, which the FPGA flow reads;
  - capstone_dyn_unit.anvil and capstone_unit.anvilh;
  - commit_stage.sv, decoder.sv, ex_stage.sv, ariane_pkg.sv, issue_read_operands.sv, pmp_data_if.sv and tracer.sv.
- No submodule pin moves. The default build is Sublet, so no extra defines.
- The instance under ex_stage_i keeps the name `rev_node`; its module becomes capstone_sublet_life.
- Lint is PASS by the rtl lane's numbers. UNOPTFLAT is 39, with the names equal. UNUSEDSIGNAL is 1565, down from 1662
  because the old unit leaves.

**Gate** (`twc/b5b2a/post-b5b2a.sh`, graded by `twc/b5b2a/grade-b5b2a.py` against the 1b gate,
`<mnt-hdd>/synth/b5b1-4c576521d-gate-out`, board/meas-pw 287b64c07d86). The grader was dry-run three times:
- the base against itself: PASS 3/3, every delta +0;
- B5a's gate as trial: FAIL on mapping (+1,812.6 MB) and on core LUT (+3,746);
- the R-75 (d) gate as trial: PASS 3/3 (core LUT +942).
- Its module check printed the NOTE "not capstone_sublet_life" on all three, each of which synthesised capstone_rev_node.

| key | prediction | if missed |
|---|---|---|
| 1 mapping | technology-mapping peak within ±505 MB of 5,053.0 MB; the whole run under the 40 GB guard | a blowup dies at the guard: record the genome and the 8-326 targets, no retry |
| 2 loops | 0 by every probe, each control firing | any loop: its members by stem; a rev_node or capstone_sublet_life member is a FAIL |
| 3 core LUT | at most 162,040 + 2,500 = 164,540 | above: per-module attribution before anything else |
| 4 the unit | rev_node's LUT (module capstone_sublet_life) reported beside the 1b gate's capstone_rev_node, 5,884; expected smaller | larger: reported, not a kill |

Reported, not graded:
- core FF, against 53,987;
- dyn_unit, ex_stage_i, issue_stage_i, id_stage_i and i_pmp_data_if;
- RAMB36 and LUTRAM.

**A synthesis risk the rtl lane named.** The new unit uses automatic tasks that make non-blocking assignments inside an
`always_ff` case, plus `unique case` with a default. Verilator accepts both; whether Vivado does is UNRESOLVED.
`twc/b5b2a/unit-record.py` records it:
- the module of the rev_node instance, from the utilization report;
- every .vds WARNING, CRITICAL WARNING or ERROR line that names capstone_sublet_life, by message id.

It was checked on the 1b gate. There it read module capstone_rev_node, and its control counter (load_store_unit
warnings) read 5.

The .vds cannot be used for the module check. Vivado keeps only the first 100 messages of each id, so the 1b gate's
.vds has no "synthesizing module 'capstone_rev_node'" line at all, although the module was synthesised.
