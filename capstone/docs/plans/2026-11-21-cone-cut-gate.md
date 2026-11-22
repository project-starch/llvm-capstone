# The rev-node cone cut on P1: synthesis-only gate (board lane copy of the pre-registration)

Status: pre-registered 2026-11-21, before the launch. The keys are the rtl lane's (land/rtl-cone-cut 5e3a51014bc9,
`capstone/docs/plans/2026-11-21-rev-node-cone-cut.md`), copied unchanged.

**Tree, read at source.** capstone-ariane p1-revrd-cut 5155e8e02, parent P1 d83d3a633. `git diff d83d3a633 5155e8e02`
touches `core/ex_stage.sv` only (+48/-17).
- It registers `rv_rd_res` (data, user and rvalid), with the read index and namespace flag it is decoded under.
- The grant stays combinational.
- Lint is at P1's baseline on every counter (the rtl lane's numbers).
- It cuts the crossing this lane read on P1's routed checkpoint: `rev_node/rd_data[5]` for 461 of the worst 500 paths,
  `rv_rd_res[data_rvalid]` for 39 (board/meas-pw 00cc2382f1eb, 70c6aafff17c).

**Gate.** P1's synthesis-only flow, with the same watchers, the kept checkpoint, LUTLP-1 and check_timing loops, each
with its toy control. Chain `twc/cut/post-cut.sh`, grader `twc/cut/grade-cut.py`. The grader was dry-run on B5a's gate,
where it FAILs mapping, LUT and FF as it must for that tree.

| Key | Limit |
|---|---|
| mapping | Technology Mapping peak within 10 % of P1's 5,041.7 MB; whole run inside the 40 GB guard |
| loops | 0 by every probe, each control firing |
| core LUT | P1 ± 2,500 |
| core FF | P1 + 150 to + 400 |

The full build waits for the rtl lane's pairing arms and its 95-test sweep. Its keys are in the rtl lane's file:
- WNS about -4.6 ns, inside [-8.3, -0.9]; a WNS inside P1's null band is the kill;
- a routing-failure branch;
- congestion ≤ 5.

## Reading: the gate FAILS one key: core FF +101 against the pre-registered +150 to +400 (apollo, tag cut-5155e8e02, 2026-11-21)

Record: `<mnt-hdd>/synth/cut-5155e8e02-gate-out/grade-cut.txt`. The guard stopped after synth_design by design, at a
27.49 GB peak.

| Key | Reading | |
|---|---|---|
| mapping | TM peak 5,050.2 MB (P1 5,041.7, +8.5; limit ±504) | PASS |
| loops | none by any probe, each control firing | PASS |
| core LUT ±2,500 | 162,874 (+471) | PASS |
| core FF +150..+400 | 54,196 (**+101**) | **FAIL** |

The modules move well beyond the ±2,500-per-module floor, though the core nets +471:
- ex_stage +5,039 (FF +106);
- dyn_unit +7,421;
- issue_stage -4,218;
- lsu_i -774;
- rev_node -1,094.

**By its own keys the full build does not follow.** Whether the FF shortfall means registers trimmed or merged, or a
mis-sized key, is the rtl lane's to read.

## Re-key after the reading, and its count (2026-11-21)

**The rtl lane re-keyed both FF keys after the FAIL above, in writing, before any full launch** (land/rtl-cone-cut
319c1f5b9d25). This record treats it as a re-key made after the reading, not as the key first registered.
- Its stated reason: the +150 floor was the cut's own 155 bits with no redistribution allowance.
- New gate key 1: a name count of the cut's flops on the gate checkpoint, at least 150 to pass, with P1's gate
  checkpoint reading 0 as the control.
- New gate key 2: core FF within P1 ± 2,000. +101 passes.
- Full-build FF key: placed FF within P1 ± 2,000. The other full-build keys are unchanged.

**The count** (`<mnt-hdd>/synth/twc/cut/cutflops.tcl`, logs `twc/cut/CF-{CUT,P1}.log`; 8 GB guard, 1.6 GB peak):

| Name | Cut gate dcp | P1 gate dcp (control) |
|---|---|---|
| `rv_rsp_q_reg*` | 129: data_rdata 64, data_ruser 64, data_rvalid 1 | 0 |
| `rv_rsp_idx_q_reg*` | 25 | 0 |
| `rv_rsp_vctx_q_reg*` | **0** | 0 |
| any other sequential cell named `*rv_rsp*` | 0 | 0 |

That is 154 FDCE in all, against the rtl lane's 155: no flop carries the namespace flag's name. **Key 1 reads 154 ≥ 150:
PASS.** By the rtl lane's written rule the full build may launch. It waits for a free guarded slot.

**The 155th bit (the rtl lane's reading, then a read-only check).**
- The rtl lane reads `rv_rsp_vctx_q` (ex_stage.sv:2008) as an equivalent register of `vctx_rev_qq` (:1712): same D,
  clock, async reset and reset value. Vivado's equivalent-register removal keeps one flop, under `vctx_rev_qq`'s name.
- Checked on both gate checkpoints (`twc/cut/vctx.tcl`, logs `VX-{CUT,P1}.log`). The script errored after its first
  pattern (exit 1, on an empty-list property read), so only the `vctx_rev_qq_reg*` rows below were read.
  - Cut: 4 FDCE, `vctx_rev_qq_reg` plus three fanout replicas `_rep`, `_rep__0`, `_rep__1`, with a total Q fanout of
    351 (68 + 92 + 99 + 92). One load is named for the cut's consumers (`vp_hdr0_q[63]_i_1`).
  - P1: 1 FDCE, Q fanout 14.
- So the merge reading fits: the register gained about 25 times its fanout and was replicated. The rtl lane's "1 FDCE in
  both" does not hold for the cut, which carries 4.

## Reading: the full build routes, but its WNS is the pre-registered kill (apollo, tag cut-5155e8e02, 2026-11-22)

Sealed 07:18:31Z as `caplifive_…-pagedtbl-revrdcut_5155e8e02.bit`:
- sha256 a1981c3c854106e941001a7056b5b0abe7dd4948c26f49aad6479f6aa838dbe6;
- guard exit 0 at 27.48 GB.

grade_full.py against P1's seal reads 4/4 (`grade-board/grade-full.txt`): routed clean 242,960 / 242,960, placer congestion
5, loops 0, LUT 187,801 (+433), FF 76,644 (+101). Against the rtl lane's keys (319c1f5b9d25):

| Key | Reading | |
|---|---|---|
| LUT ± 2,500, FF ± 2,000, routed, congestion ≤ 5, loops 0 | +433, +101, clean, 5, 0 | PASS |
| WNS inside [-8.3, -0.9]; inside P1's null band [-16.913, -9.433] is the kill | **-10.130 ns** (Δ +3.043) | **KILL** |
| 0 worst-500 paths through the old rd_data crossing | 0 through `rev_node/rd_data*`, 0 through `rv_rd_res*` | PASS |

**The seal is not handed over for a flash.**

**What limits it now** (read-only on the routed checkpoint: `twc/lattr/cut-paths.tcl` → `CUTPATHS.log`, and
`cone-ports.tcl` → `CUTCONE.log`):
- The worst 20 all start at `csr_regfile_i/cpmp_q[8][bE][1]` and end at `issue_read_operands/fu_data_q[0][cap_data]`,
  with 99-108 levels (-10.130 down to -9.734).
- **All 500 still cross the rev_node hierarchy, at a different entry:**
  - the net `rev_node/_thread_3_event_counter_531_1_q_reg_1`, with 12 rev_node cells per path (31 on P1);
  - 448 leave into `dyn_unit/EVENTS3[1084].event_current`, and 52 into `i_scoreboard/pc_revnode_unk_d145_out`.
- The cut's own nets are on none of the worst 500: `vctx_rev_qq*` 0 (the register and its three replicas, fanout 68 / 92
  / 99 / 92) and `rv_rsp_*` 0 (`rv_rsp_q[data_rvalid]` fanout 133).
- The port lookup again returned `?noport`, so these are boundary nets. Retiming makes the hierarchy approximate.
- Which RTL path the new entry is, is the rtl lane's to read.

**The second entry, path by path** (read-only on the cut's routed checkpoint; `twc/lattr/entry2.tcl` → `E2.log`, and
`entry3.tcl` → `E3.log`, for the rtl lane's mapping):
- The worst path and the first three of each exit family (dyn_unit, the scoreboard's pc_revnode) all enter rev_node at
  one pin: `rev_node/_thread_3_event_syncstate_546_q_i_2/I2`, on the net named `_thread_3_event_counter_531_1_q_reg_1`.
  - Inside rev_node the cells are plain LUTs: 6 on the dyn_unit family, 13 on the pc_revnode family (through
    `pw_wr_q`, `_mem_ch_write_req_valid_selector_q`, `revnode_fill_*`, `cpmp_revnode_valid_q`, `pc_revnode_unk_q`). None is
    named `_bret`/`_fret`/`_rep`.
  - No net named for a grant is on these paths.
- **The cell that drives the entry is a LUT6 named for rev_node logic, `_thread_3_event_syncstate_546_q_i_4`, which sits
  in the D$ tag SRAM's hierarchy** (`i_wt_dcache_mem/gen_tag_srams[4].i_tag_sram/.../i_ram/`). Optimization has moved it
  across the boundary.
  - Its net's segments run from `i_wt_dcache_mem ... enable_q_reg` through `i_cache_subsystem_n_1913` into
    `ex_stage_i/rev_node`, with one more segment in `gen_rd_ports[2].i_wt_dcache_ctrl`.
  - Its six inputs are: port 2's `i_wt_dcache_ctrl` FSM state (I0), the D$ `cache_en` flop (I1), and four LUTs named
    `gnt_rd_q_i_*` in the D$ tag SRAMs (I2-I5).
- **Reading:** the second entry is D$ read port 2's grant, from the D$ read arbitration and that port's controller state,
  feeding the rev_node unit's events combinationally. That is the grant the cut left live by design
  (`rv_rd_res.data_gnt`). The names are approximate under cross-boundary optimization and retiming.
