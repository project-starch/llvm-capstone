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
