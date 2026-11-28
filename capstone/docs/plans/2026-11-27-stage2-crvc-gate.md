# Stage 2 (CRVC decode mode) on Line B: the synthesis gate and full build, pre-registered (2026-11-27)

*RTL lane. Pre-registered before the request to the board lane, which runs apollo builds. The cycle band and go/no-go
for the board pair are the compiler lane's (llvm-capstone 4d0f6b69116f, origin/compiler/stage2-trap-spec,
tests/stage2-crvc-board/PREREG.md). This note covers only whether the RTL builds, and what the build should read.*

## The hash

- **The hash:** capstone-ariane `796af23d9`, on origin/stage2-crvc, based on b42c8c644 (the Line B Full seal).
- **The change outside verif/** (`git diff --stat b42c8c644 796af23d9`):
  - core/compressed_decoder.sv +46: the eight CRVC forms under `crvc_i`, each a fixed bit permutation into an existing
    32-bit custom-2 encoding;
  - core/csr_regfile.sv: a `crvc_q` flop loaded from bit 0 of the srevroot CSR write (which already flushes), read
    back in bit 0, and cleared on switcher id 67; `crvc_o = CAPSTONE_EXT && crvc_q && cap_data_ctx_o[0]`;
  - core/cva6.sv +3 and core/id_stage.sv +2: the wire.
- **What is sim-only:** the SUP_TRACE monitor and the CRVC_SIM_NOFLUSH control.
- **Decoder instances:** one compressed decoder instance (id_stage.sv:167, NrIssuePorts = 1, SuperscalarEn 0).
- **Lint:** on be20216e7, whose core/ equals 796af23d9's (be20216e7..796af23d9 changes two test files only), equal to
  b42c8c644 on every counter, on the same footing (s2pre-wt .sup/lint-crvc-{b42c8c644,be20216e7}.log):
  - LATCH 44, MULTIDRIVEN 3, ALWCOMBORDER 0, COMBDLY 0, UNOPTFLAT 39, BLKSEQ 2, UNDRIVEN 24, UNUSEDSIGNAL 831,
    CASEOVERLAP 2, SELRANGE 100, ANVIL_UNOPTFLAT 0;
  - LINT GATE PASS, loop names 39 = reference.
- **Why a gate first:** the decoder output feeds `id_stage_i.genblk2[0].decoder_i.instruction_o`, on the standing
  UNOPTFLAT list.

## Synthesis-only gate (base: b42c8c644's gate, /mnt/hdd/<user>/synth/b5b2r77-b42c8c644-gate-out)

| key | pass if |
|---|---|
| 1 memory | technology-mapping peak within +-505 MB of 4,968.2 MB, under the 40 GB guard |
| 2 loops | 'Found timing loop' 0, Synth 8-326 0, check_timing 0 (toy fires), LUTLP-1 0 (toy fires) |
| 3 core LUT | at most 125,279 + 2,500 = 127,779 |

Reported, not graded (predicted before the run):
- **id_stage_i** (base 1,430 LUT / 304 FF):
  - LUT +30 to +250: one extra 32-bit alternative per CRVC slot in the decoder's output mux, plus seven slot
    compares;
  - FF +0.
- **csr_regfile_i** (base 4,491 / 2,767):
  - FF +1 (crvc_q);
  - LUT under +30 (the read-back OR, the write enable, the AND with cap_data_ctx_o[0]).
- **Core FF:** inside the floor; retiming moves more than one flop.
- **Every other module:** within the per-module floor (+-2,500). A move larger than the change in a module the diff
  does not touch is re-homing, read by stem rather than graded.
- **Failure branches:** if key 1 or 2 fails, record the partition and loop members by stem, and run no full build.

## Full build on a pass (base: b42c8c644's seal, sha256 6a09f34d...b190a)

| key | prediction | failure branch |
|---|---|---|
| placed LUT | 150,303 + 100 +- 3,000 | outside: per-module placed rows against the seal |
| placed FF | 68,506 + 1 +- 2,000 | the same |
| routed | 0 errors, 0 unrouted, 0 partial | routing failure: placed area, congestion windows by module; NO WNS |
| congestion | <= 5, placed and router | above: the windows' modules |
| loops | check_timing 0, LUTLP-1 0, TIMING-23 0, each toy firing | any: members by stem |
| WNS clk_out1 | inside b42c8c644's null band [-3.135, +4.345] (+0.605 +- 3.74) | bad side: worst-path family and depth before any board use; good side: reported |

Reported:
- design WNS / TNS / WHS, slices, RAMB36 (115), the worst-20 family (b42c8c644's: issue_read_operands fu_data_q);
- the count of compressed_decoder / id_stage cells on the worst 500, with a POS control (issue_read_operands) and a
  NEG control;
- sha256 and the resident name the board reads back.

Flash: per the compiler lane's PREREG design (seal control on b42c8c644 first, then the reflash to this seal), after
the placement RESULTS gives P_s. The seal's readback name goes into the compiler lane's amendment before the first
boot. Launch follows the board lane's synth-guard rules (40 GB for the gate).

## Gate result (board lane, apollo, 2026-11-26 20:20-21:09Z; graded independently by the rtl lane from the grade file)

**PASS on all three keys.** Output: `/mnt/hdd/<user>/synth/b5b2crvc-796af23d9-gate-out`. The run stopped after
synthesis by design (run.exit 143, the same as the base gate). The largest Vivado process was 8,012 MB, with no alarm.

| key | read | grade |
|---|---|---|
| 1 memory | TM peak 4,970.3 MB (base 4,968.2, +2.1); whole-run peak 5,101 MB | PASS |
| 2 loops | 'Found timing loop' 0, Synth 8-326 0, check_timing 0 (toy 1), LUTLP-1 0 (toy 1) | PASS |
| 3 core LUT | 124,832 (-447 vs 125,279) | PASS |

Predictions against readings:
- **id_stage_i:**
  - LUT 1,620 (+190), inside +30..+250.
  - FF 314 (+10) against a predicted +0: **missed**. The 796af23d9 diff adds no flop to id_stage.sv or
    compressed_decoder.sv. Core FF is +8 in all, with issue_stage_i -5, so this reads as retiming re-homing. It is not
    attributed.
- **csr_regfile_i:**
  - FF 2,768 (+1, crvc_q), as predicted.
  - LUT 4,438 (-53) against "under +30": inside, though the sign is re-homing.
- **Every other module:** within +-2,500. The largest moves are i_issue_read_operands -1,046 and ex_stage_i +601;
  RAMB36 is unchanged.

## Full build result (board lane, apollo, 2026-11-26 21:58-23:14Z; graded by the rtl lane from the routed reports)

Run: `/mnt/hdd/<user>/synth/b5b2crvcf-796af23d9-full`, exit 0, 1h15m05s, peak RSS 24.07 GB, guard REASON=ok.

| key | read | grade |
|---|---|---|
| placed LUT | 149,891 (seal 150,303, -412), band 150,403 +- 3,000 | PASS |
| placed FF | 68,514 (seal 68,506, +8), band 68,507 +- 2,000 | PASS |
| routed | 0 routing errors; 197,909 of 197,909 routable nets fully routed | PASS |
| congestion | placer region sizes N 8x8/8x8, S 1x1/4x4, E 8x8/4x4, W 2x2/4x4 (level <= 3); router: no hotspot or initial-estimate line | PASS |
| loops | check_timing loops 0 and latch_loops 0 (build report); LUTLP-1 0 at synth and opt (in flight); routed methodology DRC 0 TIMING-23 / LUTLP-1 rows | PASS; on the routed dcp too (board lane, twc/b5b2crvc/post-full.log): CHECKLOOPS 0 and T23 COUNT=0 (toy 1/1), LUTLP-1 COUNT=0 (toy 1) |
| WNS clk_out1 | +1.796 (seal +0.605, +1.191), band [-3.135, +4.345] | PASS |

Reported, not graded:
- design WNS +0.176 on a MIG path (seal +0.177); TNS 0; WHS +0.031 (seal +0.043); BRAM tiles 117 = the seal.
- **Sealed** 2026-11-26 23:16:40Z (board lane) as caplifive_ids55-native-tagcache-mlp-fpieee-hpmreplay-dynbypass3-dtlb64-
  shrinkfix-ic32k-r72-stcnoclaim-twc16k-merge-f2-hpmtag-lrscfix-tagoverlap6c-pagedtbl-r76-r75d-b5b2n-crev4b-wap-flu-r80-
  r77-crvc_796af23d9.bit, 11,443,722 B, sha256 0149657ee39f6ff8cf744d9081f9e8ed55362f81400174de4af9c523dd56790b,
  recomputed here from `<hdd>/capstone/bitstreams/`. Tarball synth-796af23d9-exit0.tar.gz sha256 3f136eb7555fe80e...;
  its .bit is identical, and its PROVENANCE HEAD is 796af23d9. Collector peak 3.57 GB.
- No readback name yet. The flash follows the compiler lane's PREREG (the seal control on b42c8c644 first, after the
  placement RESULTS).
- **Worst 20 on clk_out1:** all from i_issue_read_operands fu_data_q_reg[0][cap_data][cap_metadata_a][14], the seal's
  family, +1.796 to +2.554 at 93-94 levels (the seal's +0.605 at 84). They end in fu_data_q cap_metadata/rd_metadata or
  i_scoreboard mem_q cap_result rs1_metadata.
- **Worst 500** (n = 500, +1.796 to +6.846; each path's cells matched by name):
  - compressed decoder (`*/id_stage_i/*decoder*`): 0 paths;
  - id_stage_i: 244 paths, 35 cells;
  - POS i_issue_read_operands 480 paths (480 on the seal), NEG 0.
  - **The id_stage prediction MISSED:** I predicted near 0. Read against the seal (board lane, crvc-paths2.tcl on both routed dcps, twc/b5b2full/bp/{crvcf,seal-b42}-paths2.log):
    - b42c8c644's seal: id_stage_i is on 137 paths over 36 cells, decoder 0, POS 480, NEG 0.
    - Three of this build's four heavy cells (213 paths each) are the seal's three heavy cells by exact name (106 each):
      fu_data_q[0][cap_data][rs1][4]_i_2 (LUT5), fu_data_q[0][cap_data][rs1][4]_i_5 (LUT6), issue_q[0][valid]_i_2 (LUT6).
      The fourth is mem_q[6][sbe][cap_result][dom_switch_req][base_addr][63]_i_3 (LUT4, 4 paths on the seal); the
      seal's fourth is mem_q[7][issued]_i_2. The other cells on both are one-path fu_data_q rd_metadata LUT6s.
    - So the rise from 137 to 244 is more paths through cells the seal already has. It is not a cell the change added.
    - The limit: a name match cannot see flattened compressed_decoder logic named after its output nets, so
      "decoder 0" is a name count, not a cone proof.
- **Final word to the compiler lane:** flashable from the rtl side, in their PREREG's order.
