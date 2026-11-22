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

## Reading (2026-11-23): B5B2A-GATE PASS (3/3 keys)

Tag b5b2a-a5b2a0d1f, launched 17:53:55Z, gate.done 18:48:29Z. The run's exit 143 after gate.done is the gate's normal
end. The guard peak was 32.81 GB under the 40 GB ceiling (the 1b gate: 31.44); the host's MemAvailable never fell
below 79 GB. Synthesis checkpoint sha256 a387e7ab04492fea…; srcwatch OK (vendored 0, fix 1).

| key | reading | verdict |
|---|---|---|
| 1 mapping | technology-mapping peak 5,025.4 MB (-27.6 against 5,053.0; limit ±505) | PASS |
| 2 loops | "Found timing loop" 0, Synth 8-326 0, check_timing loops 0 (toy 1), LUTLP-1 0 (toy 1), in-flight gate PASS | PASS |
| 3 core LUT | 141,732 (-20,308; limit 164,540) | PASS |
| 4 the unit | rev_node is module capstone_sublet_life: LUT 3,054, FF 1,390, beside capstone_rev_node's 5,884 / 1,642 (-2,830) | reported: smaller, as predicted |

Core FF 52,931 (-1,056). RAMB36 113 and LUTRAM 2,480 are unchanged.

**The -20,308 is not the unit's, and is attributed.**
- csr_regfile_i reads 6,910 LUTs against 28,245, a fall of 21,335. Its FFs are unchanged (6,341 against 6,337).
- Every one of the 38 earlier capstone builds read on this host (gate and full) reads csr_regfile at 26,825-28,997, so this reading is the
  only outlier.
- csr_regfile.sv itself is unchanged by the tree. The change is in pmp_data_if.sv. Under `CAPSTONE_SUBLET_FMT` it
  ties `cpmp_revnode_valid_q` to 0, the runtime lane's ruling of 2026-11-22: "fail closed: no CPMP entry allows".
  - Both CPMP allows AND that bit per entry: the data check, `pmp_data_if.sv:233-241`, and the fetch check's fast and
    slow paths, `:258-280`.
  - So `cpmp_allow` and `cpmp_if_allow` are constant 0. The 16 `decompress_cap_metadata` calls (`cpmp_fat`, `:228`)
    and their bound compares become dead logic.
- The netlist shows it, from the boundary pin census of both kept checkpoints (`twc/csrprune/ports.tcl`). Vivado had
  moved that decompressor cone across the boundary into csr_regfile:
  - csr_regfile_i's output pins go from 10,373 to 8,584;
  - its `cpmp_fat` pins go from 54 to 0;
  - each CPMP entry's cursor pins go from 56 to 0, and its `revnode_id` pins from 30 to 0;
  - pmp_data_if's 16 `cpmp_tracked_revnode_id_q` registers leave (-910 FF).
- So the area and timing of any bitstream from this tree are not like-for-like with bitstream 1 or the 1b gate:
  about 21k LUTs of CPMP checking are absent by construction. An access that a CPMP entry used to allow now faults,
  which is the ruling's stated intent.

**The unit's registers** (`twc/csrprune/unitregs.tcl` on the trial checkpoint):
- 1,390 sequential cells, against about 1,512 declared bits plus the state and pend registers.
- Three registers are absent entirely: `vctx_q`, `from_free_q` and `fail_q`. In capstone_sublet_life.sv all three are
  written and never read; `fail_q`'s only reader is the simulation `$display` at `:609`. They are dead in the source,
  not lost in synthesis.
- Trimmed: cap_q 31/64, h0w0_q 31/64, aw1_q 33/64, pw0_q 31/64, pw1_q 63/64, nw0_q 61/64, link0_q 31/32,
  rd_idx_q 26/30 and wr_q 155/167. Not attributed bit by bit.

**rtl's open risk** (automatic tasks with non-blocking assignments in `always_ff`; `unique case` with a default):
- Vivado accepted the unit with ERROR 0 and CRITICAL WARNING 0 naming it. FSM st_q was inferred and encoded
  (8-802 and 8-3354).
- "WARNING 0" is complete only for message ids under the .vds's cap of 100. Five warning ids reach that cap, and so do
  the INFO ids for merged and constant registers (8-3886, 8-3333):
  - 8-7137;
  - 8-7129, port with no load;
  - 8-6014, unused sequential element removed;
  - 8-3917, port driven by a constant;
  - 8-3332, sequential element removed.

  So for those ids the count is blind, not zero. The register census above is the substitute.
- Whether Vivado gives those tasks' assignments the simulated semantics is not something a synthesis-only gate can see.
  That needs a run of the netlist or the board.

**Reported rows** (against the 1b gate):
- ex_stage_i -4,684;
- dyn_unit -712;
- id_stage_i -464;
- i_pmp_data_if LUT -46 and FF -910;
- lsu_i -868;
- dom_switcher -2,424.

**Not attributed: issue_stage_i +8,791 (42,758 to 51,549).**
- Its parts: i_issue_read_operands +7,225, of which +4,145 is its own logic. The capability-metadata register file
  rises +2,784, from 2,430 to 5,214, against 2,430-2,475 in every earlier gate. The scoreboard rises +1,566.
- The tree's issue_read_operands.sv change is an op list and a simulation-only `ifdef`.
- Across seven earlier gates, issue_stage_i read 42,586-49,204, so this reading is about 2.3k above the highest. The
  1b gate had moved it -5,885.
- It is beyond the ±2,500 floor and left unattributed. The metadata register file's doubling is the row that stands
  outside its own history.

**The CPMP cut is dead logic, read at source in the gate tree (a5b2a0d1f).** The rtl lane cited this chain, and the
runtime lane accepted it as releasing its hold on a 2a full build. I read every link myself:
- Both CPMP faults fire only under `capmode_i`: fetch at `pmp_data_if.sv:331-335`, load and store at `:396-399`.
- capmode is sticky. It resets to 0 and is set only by `capmode_set_i`: `csr_regfile.sv:342`, `:3365` and `:3461`.
- `capmode_set_i` is driven by `capenter_commit` alone (`cva6.sv:2202`). That is a committed CAPENTER without an
  exception (`commit_stage.sv:359-364`).
- Under `CAPSTONE_SUBLET_FMT` the decoder makes CAPENTER illegal before any other check (`decoder.sv:1441-1443`).

So the check is never armed on a Sublet core, and the roughly 21k LUTs are a Sublet saving, not a loss of function.
They stay not like-for-like with bitstream 1. The rtl lane's condition stands: if any capmode setter becomes legal
under Sublet, the tie-off becomes a pass-through first. The rtl lane's Sublet-build simulations on capstone-ariane
b5b1-step2-fix (sublet-ns, -f4, -edge, -edgem, va-runv-snap) have S fetching and loading ordinary code under the
tie-off. The runtime lane relayed them; they were not re-run here.

The rtl lane's explanation of the trimmed register widths is the lane's claim, not checked here:
- cap_q and h0w0_q are duplicate flops, merged;
- the constant-zero bits are wr_q [166:160], [30] and [28:25], rd_idx_q [28:25] and link0_q [31];
- aw1_q, pw0_q, pw1_q and nw0_q keep only the fields that are read.

**Next: splitting the issue_stage_i growth.** It runs as a reported-only gate, same chain and guard, tag
b5b2s-8cbf393d4, on capstone-ariane 8cbf393d4.
- That tree is the 1b tree plus the CPMP tie-off, a simulation-only `ifdef` in issue_read_operands.sv, and verif files.
  It is an ancestor of a5b2a0d1f, and 2a adds nothing more to pmp_data_if.sv.
- If issue_stage_i already grows here, the growth is the tie-off's redistribution.
- If it does not, it belongs to 2a's op lists.
- Its grader verdict line is not a gate.

## Reading (2026-11-23): the 8cbf393d4 split gate, reported only

Tag b5b2s-8cbf393d4, gate.done 19:59:28Z, same chain. The guard peak was 33.76 GB under the 40 GB
ceiling; the host's MemAvailable never fell below 78 GB. Checkpoint 082abdd55bd0b1e3…. Loops 0 with every control
firing. Mapping 5,049.8 MB. rev_node is module capstone_rev_node, as it must be on this tree. The grader printed PASS
3/3 against the 1b gate; that line is not a gate here.

| row | 1b 4c576521d | split 8cbf393d4 (1b + tie-off) | 2a a5b2a0d1f |
|---|---|---|---|
| core LUT | 162,040 | 141,017 | 141,732 |
| csr_regfile_i | 28,245 | 7,219 | 6,910 |
| issue_stage_i | 42,758 | 44,527 | 51,549 |
| i_issue_read_operands | 29,737 | 29,294 | 36,962 |
| capability-metadata register file | 2,430 | 2,430 | 5,214 |
| dyn_unit | 11,319 | 14,004 | 10,607 |
| dom_switcher | 11,545 | 11,527 | 9,121 |
| rev_node | 5,884 | 4,062 (same module) | 3,054 |

- **The CPMP cut is the tie-off's, by a single-variable split.** The tie-off alone takes csr_regfile down 21,026 and
  the core down 21,023. pmp_data_if's FF falls -910 on the split too.
- **The issue_stage_i growth belongs to 2a, not to the tie-off.** On the split it is +1,769, inside the ±2,500
  floor, and the metadata register file is unchanged at 2,430. Between the split and 2a it is +7,022, and the register
  file doubles to 5,214.
- **Against the split, 2a's whole core is only +715.** The issue_stage_i growth sits beside dyn_unit -3,397,
  dom_switcher -2,406 and rev_node -1,008. In 2a's report branch_unit_i has no row of its own (142 LUT on the split).
  That pattern fits logic moving across module boundaries more than new logic, but it is not shown: the cause of the
  register file's doubling inside 2a stays unattributed. The rtl lane has no source reason for it.
- Even with no source change to the old unit, rev_node moves -1,822 on the split. That is the size of null-change
  movement at module level here.

**An instrument for 2b's gate, dry-run on the split and 2a checkpoints.** The rtl lane asked for it in place of a 2a
attribution, which 2b's widened register file (the lane grows from 90 to 98 bits) would overtake.
- `twc/b5b2b/pins.tcl` is a boundary-pin census of three cells, each pin's far-side leaf pins grouped by owner, with
  ex_stage_i and issue_stage_i split one level deeper:
  - the capability-metadata register file;
  - i_issue_read_operands;
  - i_scoreboard.
- Both dry runs finished with bad=0, under an 8 GB guard, peak 1.59 GB.
- What the dry pair shows. These are leaf-pin counts, not LUTs:
  - the register file's inputs from dom_switcher fall 2,799 -> 150, beside the file's +2,784 LUT and dom_switcher's
    -2,406;
  - i_issue_read_operands' outputs into dyn_unit rise 3,259 -> 8,576, beside dyn_unit's -3,397 and the operand
    stage's own +4,005;
  - branch_unit_i's boundary is gone in 2a: 0 pins against the split's 131 in and 624 out.
- So the boundaries themselves moved, which fits logic being absorbed across them. It is still not a LUT
  attribution, and the doubling stays NOT ATTRIBUTED for 2a. The census runs on both of 2b's checkpoints in 2b's
  gate.

## Reading (2026-11-23): step 2b's gate on 174e7d04a, B5B2B-GATE PASS (3/3 keys)

Tag b5b2b-174e7d04a (capstone-ariane b5b1-step2b 174e7d04a; keys land/rtl-b5b1 5ac2be9540d1). The base is the 2a gate
above. gate.done 23:32:29Z. The guard peak was 28.97 GB under the 40 GB ceiling; the host's MemAvailable never fell
below 81 GB. Checkpoint 8a497bafaa21eb6b…. srcwatch OK.

The chain is `twc/b5b2b/post-b5b2b.sh`, which is the 2a chain with the base and grader swapped and the pin census added.
The grader `twc/b5b2b/grade-b5b2b.py` was checked before the launch:
- dry-run on the split -> 2a pair, it reproduces the reading above (core +715, issue_stage_i +7,022, register file
  +2,784, 6 rows beyond the floor);
- with the core limit set to 0 it reads KEY3 FAIL, so that key can fire.

| key | reading | verdict |
|---|---|---|
| 1 mapping | TM peak 5,158.2 MB against 2a's 5,025.4 (+132.8; limit ±505); whole run 5,226.3 MB | PASS |
| 2 loops | 'Found timing loop' 0, Synth 8-326 0, check_timing 0 (toy 1), LUTLP-1 0 (toy 1), in-flight gate PASS | PASS |
| 3 core LUT | 144,151 (+2,419 against 141,732; limit 146,732) | PASS |

**Reported, not graded:**
- **Core FF** 54,347 (+1,416; the prediction was at least +248). The metadata register file's own FF is +248, which is
  the lane's 8 bits x 31 registers exactly.
- **LUTRAM** +128 at core and at top. The rvc tag array is 64 x 63 per way (RAM64M x 63, four ways), against 2a's
  64 x 55.
- **The unit** is still module capstone_sublet_life. Its LUT is 7,882 against 2a's 3,054 (+4,828); its FF is 1,410
  (+20). The .vds names it in no ERROR, CRITICAL or WARNING line; the load_store_unit control reads WARNING 5. That
  zero is blind on the five warning ids at the .vds's 100-per-id cap (8-3332, 8-3917, 8-6014, 8-7129, 8-7137), as for
  2a.
- **Module rows beyond ±2,500:**
  - rev_node +4,828;
  - dyn_unit -3,012;
  - i_issue_read_operands -3,268 (33,694);
  - the metadata register file -2,566 (5,214 -> 2,648);
  - i_scoreboard +2,602 (17,189).
  - Also: dom_switcher +1,822, issue_stage_i -666.

**The pin census** (`pins.tcl` addeb3a5808de4eb). It ran on both checkpoints with bad=0; the 2a base reads as in its
dry run. These are far-side leaf pins, 2a -> 2b:
- the register file's inputs from dom_switcher 150 -> 2,575: back near the split's 2,799. Its LUT is back near the
  split's 2,430 too;
- i_scoreboard's inputs from rev_node 4 -> 5,330;
- i_scoreboard's outputs into i_issue_read_operands 27,196 -> 18,557;
- i_issue_read_operands' outputs into rev_node 21 -> 1,877;
- i_scoreboard's inputs from i_issue_read_operands 4,607 -> 1,365, and from dyn_unit 1,801 -> 210.

**What this does and does not show:**
- 2a's register-file doubling went away in 2b, together with the dom_switcher boundary it had absorbed. That is the
  pattern the rtl lane read as redistribution, now seen in both directions. It is still a pin count, not a LUT
  attribution.
- rev_node's +4,828 sits beside a new boundary of some 5,000 pins into the scoreboard and the falls in dyn_unit and the
  operand stage. That fits logic absorbed into the unit across its boundary more than a growth of the unit itself.
  It is NOT ATTRIBUTED: the 55 -> 63-bit widening also lands in the unit, and nothing here separates the two.
- Routing is not measured by this gate.

No full build was requested for 2b.
