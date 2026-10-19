# capstone-sublet: the RTL side of the Sublet lifetime extension, and the FP-fixes bitstream

*RTL lane, 2026-10-13. Branches in capstone-ariane: `fp-fixes` (the next bitstream's candidates) and
`capstone-sublet` (the ISA extension, on top of fp-fixes). Hashes below are the lane branches' as pushed; the
landing on the shared branch squashes per logical change.*

## Why

The Sublet lifetime extension (llvm-capstone #216 / capstone-qemu #21, `docs/design/virtual-capstone/isa.md`
§"Sublet child lifetimes") is the part of the new processor model that an allocator needs on silicon and that does
not depend on the virtual profile's open questions (translation in C mode, the per-context table, exact bounds).
It is two instructions over the existing revocation-node list, a fourth permission bit, a parent link in the node
record, and two semantic alignments of existing instructions. The application ports and mallocng are migrating to
it (#220, #221, #222), so a bitstream carrying it is what lets the board run the migrated allocators.

Separately, three defects found on `caplifive_r51r52_0568f93a9` block every hard-float native process or shorten
capabilities, and go in the next bitstream regardless.

## Branch `fp-fixes` (off 0568f93a9, the resident bitstream's tree)

| Commit | What | Evidence |
|---|---|---|
| cc1159a51 | R-53 (was S-19): `commit_stage.sv` dirty_fp_state parenthesised as upstream CVA6 #2936; no FP instruction ever set mstatus.FS Dirty, so Linux skipped the FP save at every context switch | s19-fs-* arms read FS 2 -> 3 with SD 1 (Clean on the base tree); rv64ud 12/12 |
| ae21e121e | R-52 residual: `issue_read_operands.sv` third-operand forwarding compared the producer's rd against the CONSUMER's rd, so `fadd fa0,fa0,fa5` took fa5 from the in-flight fdiv that writes fa0 | r52-resid-a reads 0x406e0db8fc0a7d07 (= 240.42883112, the board lane's expected value); order B/D controls equal |
| 2a3a87dfc | R-54: `compress_bounds` takes E + 1 when the rounded-up span reaches 1024 granules, instead of wrapping to half the span | bounds-renorm reads 0x80faa000 covering (0x80f6a000 without); bounds-model.py 0 of 400,000 x 3 short (1,050 without); QEMU lockstep: capstone-qemu bounds-encoder-renormalise aad781e15acd |
| 4a4261bb4 | Option (b): the ENCODER stops using the cursorless scheme (bit 27 always 0 from the core; the decoder still reads cursorless words). Closes R-11's truncation at base; every at-base top now rounds OUTWARD as QEMU's | cursorless-off reads 0x80408000 / 5008 (0x80407ffc / 5003 on 2a3a87dfc, B4's H shape); lint UNUSEDSIGNAL 833, the one new name compress_cap's cap.cursor |

Gates on every commit: rtl-lint-gate.sh at baseline (per-commit lint for cc1159a51, ae21e121e, 8cda78a46 too),
the 95-test sweep identical to 0568f93a9 on 2a3a87dfc and 328f61761, the FP / bounds / sup arms as above.

**The lead's choice for the FP bitstream** (nothing is synthesised until the lead says so):

- **(a) = 2a3a87dfc**: R-53 + rs3 + renormalisation. R-11's truncation stays on the heap and outside the glue
  (B4's H reads [+0, +65532) at base when it straddles a 4 MiB boundary; covering once moved).
- **(b) = 4a4261bb4**: (a) plus the encoder change. No truncation anywhere; but where the cursorless scheme gave an
  EXACT top, a glue-carved global's capability now reaches into its upper neighbour by under a granule (the board
  lane's table: k800 64 B, s06agg 112 B, r1_slots_pools 2,560 / 496 B, live16 1,008 B, SQLite cells 4,096 B,
  b0-hello 384 B, b1-thread 2,048 B, memcached 32,768 B) and level0 objects of 4 KiB and up round outward.
  check-repr --renorm over the 11 acceptance images (the board lane, b0-silicon-delegated-runtime.md): under (b) no
  fault on any image; under (a) every verdict as on 0568f93a9, R-11's existing trap bases included (stale-full
  r1_slots_pools AT RISK at 7 of 255 bases, the SQLite cells cell5-memsys5-O2 and cell6-sublet-O2 at 14 of 253
  each), because the cursorless scheme truncates at base and the renormalisation does not reach it.
- r54-bounds as the before/after arm (the board lane's corrected prediction): 0568f93a9 39, (a) 0, (b) 64.
- Monitor side (the board lane owns): no change for the renormalisation (every monitor capability is exact under
  every scheme); under (b) the C-13 granule rule leaves the CODE region's top rounding into the seal region for
  code-heavy layouts (none of the 11 images), fix = size the granule from max(split_size's granule, repr_gran).
- QEMU side (the helper lane owns): aad781e15acd lands only with the flash that carries 2a3a87dfc.

## Synthesis of `fp-fixes` 4a4261bb4, pre-registered 2026-10-14 (before the handoff; the hash pinned)

The lead's decisions of 2026-10-14 in the RTL lane's conversation: the FP bitstream is built from option (b) =
4a4261bb4; two builds, this one now and the Sublet one when its gates pass; the build is authorised now, the flash
needs the lead's own word when the sealed hash exists. Name: `caplifive_fpb_4a4261bb4.bit` (tag fpb = FP fixes,
option b). The board's acceptance is pre-registered on b0-silicon-runtime 5e82c06d61f9 for both options; the
QEMU renormalisation aad781e15acd lands with this flash.

What the synth lane receives with the hash:
- Tree: capstone-ariane 4a4261bb4 = 0568f93a9 + cc1159a51 (R-53) + ae21e121e (rs3) + 2a3a87dfc (R-54) + 4a4261bb4
  (encoder without the cursorless scheme). Hand-written files touched: commit_stage.sv, issue_read_operands.sv,
  ariane_pkg.sv (compress_bounds / compress_cap). No Anvil unit changes; the generated units are byte-identical.
- Lint (rtl-lint-gate.sh in the pinned container, every commit on its own): LATCH 52, MULTIDRIVEN 3, ALWCOMBORDER 0,
  COMBDLY 0, UNOPTFLAT 40, ANVIL_UNOPTFLAT 0, BLKSEQ 2, UNDRIVEN 25, UNUSEDSIGNAL 833 (832 + compress_cap's cap.cursor,
  no longer read by the encoder), CASEOVERLAP 2, SELRANGE 100; the negative control reads BLOCKED.
- The 40 UNOPTFLAT names are IDENTICAL between 0568f93a9 and 4a4261bb4 (by-name diff of the two listings, 2026-10-14
  morning), so "loop membership by stem unchanged" is pre-build evidence here, not a prediction.
- Behaviour: the 95-test sweep's RC list identical to 0568f93a9's; rv64ud 12/12, the s18/s19/r52/r51 FP arms, the
  bounds arms (bounds-renorm, cursorless-off) and the supervised arms as recorded in the commits. RETRACTED
  2026-10-14, same day: "no log at the cycle limit" was read with the wrong number (4,000,013, the ladder runner's;
  the sweep's is 2,000,013). 26 of the sweep's 95 tests end at ITS limit on every tree, baseline included, because
  they are readings-only arms that never write tohost (capsbi-init, ccsrrw, cbnz, capenter, call-*, domain-save-*,
  r30-fill-init, r31-revoke-cursor, ...), and the harness printed that as SUCCESS. For those 26 the RC list compared
  nothing; the readings-level comparison in the RTL tree (`.sweep-compare.py`: per-test exceptions, CAPPRINT readings,
  retired instructions) was run over every gated tree against the baseline output (0568f93a9, seed 20260922, memory
  delay 12) on 2026-10-14: 2a3a87dfc and 4a4261bb4 each "95 tests: 92 identical (exceptions, readings, retired),
  0 differ, 3 missing" (rand, perf_tcache_random, perf_riscv_random: no log on either side, on every tree), the
  SUP_TRACE detector 0 on both sides. That is the FP bitstream's neutrality evidence; the RC lists were not.
  The harness fix (the limit now reads FAILED) rides the Sublet branch; the sweep baseline is re-taken on 0568f93a9
  under it, and the 26 arms get a tohost write in a later commit.
- run.tcl and the flow untouched.

Predictions, written before the build (the refutation keys in brackets):
- Routed WNS inside the standing band -13.335..-5.855 (715bdd1fe -9.949, 0568f93a9 -8.368 held it; a null RTL
  change has moved WNS by 3.7 ns). [outside the band: the failing endpoint's start and end points are read before
  anything else is concluded]
- Loop membership by stem unchanged; LUTLP-1 0. [a new stem names its cone; LUTLP > 0 is a stop]
- Area: the issue stage gains one 4-way select per scoreboard entry for the rs3 source (tens of LUTs); ex_stage's
  compress path LOSES the 21-bit cursorless scheme and its output mux and gains the renormalisation's 64-bit
  compare and add (net flat to down, under a hundred LUTs either way); commit_stage's dirty-FP term is one gate.
  No new flops. [a rise of hundreds of LUTs anywhere is a finding, not noise]
- The discriminator for a timing move: a failing endpoint whose path enters compress_bounds (the renormalisation's
  adder feeds the T field) names the encoder change; one inside issue_read_operands' operand muxes names the rs3
  fix; anything else is the band's noise.
- Route time: 0568f93a9's route took ~156 min against 20-31 for the three before it and was flagged; the same flag
  applies, and a repeat is a question for the synth lane, not this change.
- DTS: 65536 revocation nodes, unchanged (INIT_STAGE's head := 3; the pool and CAP_REVNODE_MEM_BASE untouched).

### Grade of `caplifive_fpb_4a4261bb4.bit` (sealed 2026-10-14 by the synth lane; graded here against the lines above)

- Seal: exit 0, write_bitstream completed; bitstream 11,443,722 B, sha256
  a63cbd815fe96957a979fc94c6c1bf29060428db9523d43485d392aeca10ff65; flow 4h33m (route time VOID: the host was shared
  at load 80-100 for the first hours); synthesis peak 22.05 GB; the tarball synth-4a4261bb4-exit0.tar.gz.
- **WNS: OUT of band.** Routed -15.360 (TNS -730,170; 99,793 of 255,583 endpoints failing; WHS +0.035), 2.025 ns
  below -13.335, against 0568f93a9's -8.368 with the same flow. The pre-seal placement estimate was -14.980, the
  router's figure -15.407; the sealed timing summary is the authority.
- Stems: HOLDS. Synthesis-stage membership the same by stem; routed loops 1, the TIMING-23 arc i_dtlb state_q[3]
  I1->O, identical to 0568f93a9. LUTLP-1: 0, HOLDS (the CFGBVS-1 control present).
- Area: **REFUTED.** TOP +1,739 LUT against a prediction of "net flat to down, under a hundred either way":
  issue_read_operands +2,446 LUT, scoreboard +207, ex_stage own +179, dom_switcher +193; lsu -326, dyn_unit -131,
  csr_regfile -69, rev_node -4, commit_stage 0. "No new flops" HOLDS (FF -92). The prediction modelled the rs3
  select (tens of LUTs, correct for that) and the encoder as "a 64-bit compare and add", and missed that the
  compare-and-add sits in the capability result path that the scoreboard and issue stage forward: the IRO cone's
  CARRY4 count went 63 -> 186.
- The discriminator (repaired before the seal, above): **(a), the renormalisation adder is named.** The worst ten
  paths all start at csr_regfile's cpmp bounds register and end at the scoreboard's rs1_metadata and the issue stage's
  rd/cap metadata registers, slacks -15.360..-15.154, 143 logic levels with 59 CARRY4 of which 47 in
  issue_read_operands (0568f93a9's worst path: 100 levels, 17 CARRY4, none in IRO); in the worst 500, IRO metadata
  endpoints 0 -> 98 and scoreboard rs1_metadata endpoints 40 -> 138. FP's worst endpoint reads -6.986 on 0568f93a9
  (8.374 ns better there); 0568f93a9's worst endpoint reads -11.856 on FP (3.488 ns worse). The compress-cone query:
  the matcher as first written FAILED its negative control (the dyn unit's outputs drive LSU requests whose responses
  return to the metadata registers, so the cone held the LSU/dcache loop; 26/5 and 30/5 on the control), result
  discarded; refined post hoc (LSU, cache, CSR, MMU and frontend cells excluded, disclosed as post hoc): FP's worst
  20 paths 20/20 in the cone (78-84 cells, 40-47 CARRY4 each), positive control 68/37, negative dcache 0/0, negative
  frontend 0/0. On 0568f93a9's netlist the refined query is NOT clean (its worst 20 and its frontend control both hit
  15/1 through one dom_switcher CARRY4), so for 0568f93a9 the endpoint slack is the datum, not the query. Assumption
  stated, not verified cell by cell: by the RTL the only arithmetic between the unit outputs and the scoreboard
  metadata registers is compress_cap -> compress_bounds.
- The rs3 arm: rs3_src and rs3_fpr did not survive optimisation (0 nets, 0 cells; reported as its own finding); anchored
  on the surviving rs3_fwd_req comparators, the worst 20 paths touch 0 of them (2 touch rs3_cap_metadata, the data
  side); the worst path constrained through rs3_fwd_req is -6.931 against -15.337 unconstrained. The rs3 fix is not
  the cause. Caveat: the inline rs3_src compares inside the rs3_data / rs3_cap_metadata assigns are not anchored.
- Reading: R-54's renormalisation as written (`tg = (base + ((1 << (E+3)) - 1)) >> (E+3); bg = start >> (E+3); if
  (tg - bg >= 1024) E = E + 1`, two barrel shifts, a 64-bit add, a subtract and a compare, evaluated inside
  compress_bounds on every capability result) costs about 2,400 LUT and about 2 ns of WNS on the forwarding path
  that was already the critical cone, and the Sublet tip, which carries the same adder, routed at -15.898 pre-seal.
  Whether a -15.360 bitstream runs on the board is empirical (every accepted bitstream had negative WNS, between -8.3
  and -12.9), and the flash is the lead's call; the fallback bitstream stays registered. The RTL-side follow-up, not
  started: a cheaper formulation of the renormalisation condition (decide E+1 from the length and the two low
  granule residues rather than from two full 64-bit shifted sums), synthesised alone before it goes anywhere.

## Synthesis of `fp-fixes` afcd19bea (the respin), pre-registered 2026-10-14 before the handoff

Build-go: the lead's word to the synth lane, not yet given for this hash. Name `caplifive_fpb_afcd19bea.bit`.

- Tree: 4a4261bb4 + afcd19bea, which changes compress_bounds alone (ariane_pkg.sv) and the bounds model: R-54's
  renormalisation condition in residue form. The decision E+1 is the same function of (start, len) as the first form
  MEANT, proven by derivation (the rounded span is q + ceil((sr+lr)/g), so it reaches 1024 exactly when q == 1023
  with a non-zero residue or q == 1022 with sr + lr > g) and checked in bounds-model.py over 8,259,510 regions
  (18,024 wrapping, 0 mismatches; the form without the q == 1022 term mismatches 182 of 197,607, so the check fires).
  CORRECTED the same day (the helper lane's cross-check against QEMU's cap_compress, 2,001,820 cases): not the same
  decision as the first form COMPUTED at the top of the address space. That form's end-address add was 64-bit in
  the RTL as in QEMU, so a region whose rounded top reaches 2^64 wrapped it and missed the step ([0, 2^64-1) at
  E = 51 decoded [0, 2^63), 43 of 44 such regions); the residue form takes E+1 there. CORRECTED AGAIN the same
  evening (the board lane's decode check, 200,000 regions): the respin does NOT fix that region, it changes how it
  fails. A rounded top of 2^64 is unrepresentable in a 64-bit top under either form: the old form decodes half the
  span, the residue form rounds the top to 2^64, which wraps to 0, and decodes an empty capability; 71,962 regions
  with a rounded top at or above 2^64 decode not covering under both, and every region with a representable rounded
  top covers under both, 0 misses. My model had hidden this too: its decoder built the top in unbounded integers and
  returned 2^64 for nine constructed regions, which the covering test accepted; it now masks to 64 bits as the RTL
  does (fp-fixes d00cee347) and asserts that the residue encoder decodes short on exactly the unrepresentable
  regions. Unreachable today (the genesis capabilities stop at 2^63); the forms differ only there and only in the
  failure mode. The model carries the 64-bit form as a control that must fire (comment and model only, the netlist
  of afcd19bea unchanged), and the flash pairing is afcd19bea with QEMU 4a21b485b01f. Measured directly afterwards (the helper
  lane): the codec test's exact 128-bit reference at 4a21b485b01f against the model's residue form, 0 mismatches on all
  2,001,820 encoder cases (1,970,205 ordinary, 31,571 wrapping, 44 top-of-range), the old masked sum form differing
  on 196 wrapping cases as the control; and ref_compress == cap_compress on all 6,014,628 checks there. The model had hidden it because its 'sum'
  form used unbounded integers after a 64-bit mask on the end: neither the hardware nor exact arithmetic. On the path it
  replaces the end-address add, two 64-bit barrel shifts, a subtract and a compare with a 10-bit window on len, two
  masks from E, one add of the residues and two reductions.
- Gates on the committed tree: lint 52/3/0/0/40/0/2/25/833/2/100 with every UNUSEDSIGNAL and UNOPTFLAT name identical
  to 4a4261bb4's, negative control BLOCKED; bounds-renorm and cursorless-off read 4a4261bb4's values; the 95-test
  sweep against 0568f93a9 (the old harness on both sides, every arm with a trace): 92 identical, 0 differ, 3 random
  generators without logs. No Anvil change; the generated units are byte-identical to 4a4261bb4's.
- The flow untouched; DTS 65536 unchanged; the bookkeeping prefixes (`_fpb_`) already cover the name.

Predictions, written before the build (refutation keys in brackets):
- Routed WNS INSIDE -13.335..-5.855, and better than 4a4261bb4's -15.360 by at least 1.5 ns. [WNS still near -15
  with the worst paths through the compress cone's arithmetic: the adder's form was not the cost, and what remains
  (the E selection feeding the window and the masks, or the T/B field shifts) is the next thing read; WNS in band
  with the worst path elsewhere: the FP bitstream is graded on that path like any other.]
- The worst 20 paths carry fewer compress-cone CARRY4 cells than FP's 40-47, by the synth lane's query with its
  controls (the CARRY4 metric, the same refinement as the sealed grades); the IRO cone's CARRY4 count drops from 186
  toward 0568f93a9's 63. [unchanged counts: the arithmetic is elsewhere in the cone, not in the condition]
- Area: issue_read_operands down by on the order of 2,000 LUT from 4a4261bb4 (+2,446 against 0568f93a9 there), the
  scoreboard and ex_stage own cells down by part of their +207 / +179; no new flops; everything else flat. [IRO flat:
  the condition's logic was not what the +2,446 was]
- Loop membership by stem unchanged; LUTLP-1 0.
- Route time: no prediction (host-dependent); the synthesis timing-optimisation phase is recorded for the spread.

### Grade of `caplifive_fpb_afcd19bea.bit` (the first respin; sealed 2026-10-14 by the synth lane)

- Seal: exit 0, write_bitstream completed; bitstream 11,443,722 B, sha256
  d8d2b641da70207c74d41a69ef60a561426fc573b6767fd03bd6d3c28a5fb3f8; provenance afcd19bea; the Anvil units
  regenerated and byte-identical to 4a4261bb4's; flow 1h36m; synthesis peak 21.36 GB; the synthesis
  timing-optimisation phase 8.0 min against FP's 164 (the spread 10-97 on the earlier builds).
- **WNS: REFUTED.** Routed -15.310 against FP's -15.360 (+0.050, not the predicted 1.5 ns; TNS 13.3% better,
  -633,024 against -730,170; failing endpoints 99,729). The refutation key fires: the condition's form was not the
  cost.
- The cone arm, with a blind spot disclosed: the sealed grades' cone counted only the scoreboard's rs1 and result
  metadata endpoints, and on this build it read 0 CARRY4 on the worst 20 because **the worst endpoints MOVED**: all
  top 10 are the scoreboard's dom_switch_req ra_data registers (-15.310..-15.089), written through
  compress_cap(dom_switch_ra_data) at ex_stage.sv L1484, every path starting at csr_regfile's cpmp bounds
  register; in the worst 500, ra_data endpoints 114 -> 216 and rs1_metadata 138 -> 73. With the cone extended to
  ra_data (post hoc, disclosed; the extension fails FP's frontend negative control 33/8, so FP keeps its narrow-cone
  figure) and re-run on all three netlists: 0568f93a9 worst 20 = 0 CARRY4 (controls 0/0, 14/0); the respin 20 on all
  20 (65-66 cells; controls 0/0 and 12/0 clean; positive 55/21) against FP's 40-47. "Fewer than 40-47": HOLDS.
  "The IRO cone's CARRY4 drops toward 63": REFUTED (214 narrow, 285 extended). The sealed worst path: 129 logic
  levels (CARRY4 31, of which IRO 20; LUT 98) against FP's 143 (CARRY4 59, IRO 47; LUT 84): the carry roughly
  halved, the LUT levels grew by 14 (the window shift and the masks are LUT depth), the slack unchanged. FP's worst
  endpoint reads -13.551 here (1.809 ns better than on FP); 0568f93a9's worst endpoint -11.028.
- Area against 4a4261bb4 (LUT / FF): **REFUTED**: issue_read_operands +2,494 (predicted about -2,000); scoreboard
  -1,049 (more than its +207), ex_stage own -77; NOT flat elsewhere: dyn_unit -2,180, dom_switcher -1,593,
  csr_regfile -1,461, lsu -242, rev_node +47; TOP -2,190 / +24 (new flops +24, small, not zero). Loops: HOLDS
  (membership the same, the dtlb arc, LUTLP 0). rs3: 0 of the worst 20 through its comparators.
- Reading: the three compress_cap instances (the flu result, the dyn result, the domain-switch return address) all
  carry the renormalisation, and the slack is set by whichever of them is slowest after each change; the cheaper
  arithmetic halved the carry on the path and the LUT depth of the window and masks took its place, so the serial
  stage between the exponent search and the field shifters, not the arithmetic inside it, is what the path waits
  on. The second respin (below) forms both candidate encodings in parallel so the decision overlaps the shifts; if
  that too leaves the slack near -15, the depth is the whole decompress-to-compress path through the CSR
  capability registers and no form of the condition reaches the band, which is a different decision (a pipeline
  stage, or a bitstream tested outside the band). Flashing any of the three sealed bitstreams at these numbers is
  the lead's call; the respin is the best of them on TNS.

## Synthesis of `fp-renorm-dual` 2859dfb67 (the second respin), pre-registered 2026-10-14 before the handoff

Build-go: the lead's word to the synth lane, not yet given. Name `caplifive_fpb_2859dfb67.bit`.

- Tree: 4a4261bb4 + afcd19bea (the residue-form condition) + 59e6bcce1 (comment and model) + 2859dfb67, which
  changes compress_bounds alone: the two candidate encodings at E and E + 1 are formed in parallel (their field
  shifts and 9-bit round-ups) and the renormalisation decision selects between them at the end, so the decision's
  latency overlaps the shifts instead of preceding them. The function of (start, len) is unchanged (the residue
  decision verbatim; the model is the oracle; bounds-renorm and cursorless-off read 4a4261bb4's values).
- Why: the first respin (afcd19bea) routed at -15.348 against -15.360 with TNS better by 13.5%: cheaper arithmetic,
  the same worst path. Sealed: -15.310, the worst endpoints moved to the domain-switch return-address
  compress instance at the same slack, the carry on the path halved and the LUT depth up by 14 levels (the grade
  above).
- Gates on the committed tree: lint 52/3/0/0/40/0/2/25/833/2/100 with every name identical to 4a4261bb4's, negative
  control BLOCKED; the 95-test sweep against 0568f93a9 92 identical; the two bounds arms unchanged. No Anvil change.
- The flow untouched; DTS 65536 unchanged; the `_fpb_` bookkeeping covers the name.

Predictions, written before the build (refutation keys in brackets):
- Routed WNS better than afcd19bea's -15.348 by at least 1 ns, the serial stage removed from the path; inside the
  band if the remaining chains (the length subtract and the residue add, both present on every path through the
  function) are not what sets it. [WNS within 0.5 ns of -15.348: the decision's depth is not what the worst path
  waits on; read the worst path's cells for which chain remains, and the next candidate takes the residue add off
  the path by comparing masked end and start instead of adding residues.]
- The worst path's compress-cone CARRY4 count drops below the respin's (the E + 1 candidate's shifts add LUTs, not
  carry cells); the IRO cone's CARRY4 count stays near the respin's. [a rise: the second candidate's round-up chains
  were placed in series after all]
- Area: issue_read_operands UP from the respin by the second candidate's shifters (hundreds of LUT), no new flops,
  everything else flat. [IRO flat: synthesis had already duplicated the shifters itself]
- Loop membership by stem unchanged; LUTLP-1 0. Route time: no prediction; the synthesis timing-optimisation phase
  recorded for the spread.

## Branch `capstone-sublet` (on top of fp-fixes)

| Commit | What |
|---|---|
| 124d38406 | The load-side S-06 sim invariant moves into `load_unit.sv`, scoped to LDC returns. A plain scalar load's dcache user lane is the testbench's preloaded ELF row (`ariane_tb.sv` USER_MEM), not the granule's metadata; the 28-bit format decoded that row as type 4 and the check never fired, the 27-bit format decodes it as NOT_CAP and it fired on the format tree's first sweep. Positive control: the inverted check aborts linear-clear-audit on its first LDC. |
| d34c26002 | The capability word becomes `{revnode_id 30 \| perm 4 \| cap_type 3 \| bounds 27}`, QEMU's widths and order; MANAGE = perm bit 3; the cursorless type and the decoder's branch are gone; TIGHTEN takes four rights bits, CAPPERM accepts 15, LCC field 5 reads four bits; `CAP_PERM_MANAGE` / `CAP_PERM_ALL` in asm_insn.h. perm4-roundtrip reads 15 / 7 / 15 / 8 / 27 / 5008. |
| b330433fe | Node record gains `parent` (16-bit bare index, MSB field, packed into the b word's spare bits [46:31]); rev-node unit gains `derive_req` (init_req's shape: child after the parent at depth + 1, never linear, parent recorded; status 2 at the depth maximum) and `kill_req` (child live under its generation, a direct non-linear child of the parent and not the parent; invalidated and pushed to the free list, then KILL_NODE, REVOKE_NODE line for line, walks the junior run and splices child and run out: QEMU's cap_rev_retire); decoder funct7 0x51 CDERIVE rd,rs1,rs2 and 0x52 CREVOKE rs1,rs2 (rd must be x0), both allowed under supervision; dyn unit CDERIVE / CREVOKE with QEMU's check order; the two ops join every op table (check_cap_op, check_op_cap_dyn, the issue serialisation set, the tracer's group 5). Arm sublet-derive-kill (19 readings). |

| 52ad6e5ea | A non-linear SPLIT keeps its node (no sibling). Load-bearing with CDERIVE on this RTL: a sibling is inserted right after the parent at the parent's depth, so it would sit between a NONLIN parent and its depth + 1 children and stop the parent's revoke walk; it would also outlive CREVOKE of the half it was cut from. Arm sublet-split-shares-node (the escape reads 0 on the old tree, 25 on the new). |
| 11c405304 | CREVOKE joins R-45's commit-flush list (`commit_stage.sv`, the fix that already orders younger loads behind REVOKE and DROP, on silicon since 2026-09-29), so a load right after CREVOKE refetches and is denied. Found by the port of QEMU's cases: the `ld` right after CREVOKE executed ~400 cycles before the kill finished and committed in order one cycle after it. A first version was an issue-side barrier over every tree mutation, withdrawn once the board lane pointed at R-45; SPLIT/DELIN/MREV/CDERIVE end no node and stay unordered. QEMU's sublet-lifetimes cases 0-34 and 37 ported as sublet-cases.S (36 of 36, plus the wrong-PC negative control that must fail). |

| 464abe2ce | MREV deepens its whole subtree (deepen_req: a read-only overflow check before the handle, a write walk after) and a LINEAR split's sibling goes BEFORE the parent (QEMU's cap_rev_mrev and dup_before). The claim-auditor refuted the earlier "deepening is unreachable" argument: this RTL duplicates LINEAR capabilities (R-21: TIGHTEN/SHRINKTO return rs1 unchanged), so a node can carry a LINEAR (MREV) and a NONLIN duplicate (DELIN, CDERIVE); mrev twice then revoke of the inner handle orphaned the child. The helper lane found the same duplicate in QEMU (SHRINKTO; probe capstone/runtime/tests/sublet-lifetimes/mrev-over-derived), so QEMU exercises its deepening walk too. Arm sublet-mrev-orphan reads 0 (the orphan) on the step-4 tree and 25 here. |
| cd5f35df2 | Arm sup-sublet: CDERIVE and CREVOKE inside a supervised domain (sup-escape.S's pattern, one seal and stub per arm, the capabilities inherited in GPRs, no mint inside). A derive, a store and load through the child, a CREVOKE and a RETURN read 0 \| 0xe 0x77 0 \| 1 0x77 0xe 0x5a5a; the four refusals escape as kind 2 with CSUPCAUSE 25 (a load through the dead child, R-45's refetch), 27 (no MANAGE), 29 (a grandchild), 26 (a LINEAR parent), each at offset 0 from its faulting instruction with the stub left SEALED (4); mcause 0. sup-static-audit.py reads 0 forbidden on every stub. |
| ac2968e10 | The testharness ends a run at its cycle limit as FAILED (`rvfi_tracer.sv`: `end_of_test_q <= 32'hffff_ffff` instead of `32'h1`, which `ariane_tb.cpp` printed as SUCCESS with tohost 0). Positive control hang-control.S (no tohost write) reads FAILED at the limit. The supervised ladder takes `SUP_TIME_OUT`; the three arms that legitimately exceed 4M get their budget in `testlist_sup.yaml` comments with the cycle they end at: sup-mswap-capstl 6M (4,632,088), sup-stcsb-ev-q5k 6M (5,017,611), sup-stcsb-ev-q2k 9M (7,134,138). |
| d93a8d66b | From the synthesis-oriented audit: KILL_NODE (a copy of REVOKE_NODE with other exits, which had raised the rev-node's memory write-request selector, R-35's -27.665 ns path, from 13 to 21 sites) is deleted; kill_req latches the same walk and sets kill_mode, REVOKE_NODE's exits answer kill_res or rev_res on it (18 sites). MREV's order becomes check, deepen-write, handle: a flush between mrev_res and the write pass used to leave the parent's depth bumped and its subtree not deepened, so a later REVOKE of the parent stopped at its first child; now a flush in either gap leaves the subtree one deeper than needed, which no walk can mis-read, and the replay deepens once more. The ex_stage record comment names the 110-bit record. Lint UNUSEDSIGNAL 902 -> 894 by name. Cost: MREV retires 8 cycles sooner and the next rev-node request completes when it did before (the pre-registration below has the measurement). |
| 4be92a85a | verif/sim/Makefile only: a FAILED simulation still writes its retirement trace. The recipe ran the simulator as its own line, so after ac2968e10 a limit-ending run (exit non-zero) stopped make before spike-dasm wrote the .log; the first tip sweep under the fixed harness compared 66 of 95 and reported the 26 readings-only arms MISSING on both sides. Negative control: that run (.log.iss only); positive: hang-control FAILED with its .log, the wrong-PC control FAILED with its .log, a passing case unchanged. core/ and corev_apu/ untouched. |

Divergences recorded (helper lane's Q-15 and this doc): no representability check in CDERIVE (the RTL rounds outward
at compression; QEMU's compact profile refuses with 29; the migrated allocators run on QEMU's exact-bounds profile
and never see the refusal); TIGHTEN asking for a right the source lacks raises 29 here and in the academic spec, 26
in QEMU (sublet case 26); the physical QEMU profile raises 2 for CDERIVE/CREVOKE and must flip in the same step as
the bitstream that carries them. The capability word itself needs no QEMU change: every QEMU profile has used
{revnode_id 30 | perm 4 | cap_type 3 | bounds 27} since capstone-qemu #21 (cap_compress.c is shared by all
profiles; bounds were always the 27-bit full scheme), so the format move brings the RTL to QEMU, not the other way
round. What is coupled to the Sublet bitstream is therefore the QEMU decode gate above and, on the SDK side, what
check-repr.py (c9dab70b5a15) lists: its default --format moves from cursorless to full, the globals carve may need
CAPSTONE_GLUE_CARVE_ALIGN or a globals sub-region (the OVERLAP under full encoding), and WIDEN must read 0.

Gates on the tip (RTL d93a8d66b, gate tree 4be92a85a; every claim under the FIXED harness with the trace kept, on a
snapshot proven equal to `git archive 4be92a85a` file by file, 2026-10-14): lint at baseline by name (the 40 loop
names identical to 0568f93a9's); the 95-test sweep by .sweep-compare.py against 0568f93a9 re-swept under the same
harness: 91 identical, 1 differ (r31-revoke-cursor's retired count at the cutoff, step 5's spin shift), 3 missing (the
random generators, no log on either side), the 26 limit-ending arms compared on readings for the first time; the 36
ported cases pass and the wrong-PC control fails; the four Sublet arms, sup-sublet and the supervised set at its
budgets read their recorded values; the 17 R-12 ladder arms against the step-5 tree: 8 identical, 9 carrying one
8-cycle move from MREV to the request after it, attributed below. The synthesis-oriented audit (rtl-oracle) answered:
see the pre-registration.

## Synthesis of `capstone-sublet` d93a8d66b, pre-registered 2026-10-14 (before the handoff; the hash pinned)

Build-go: the lead's word of 2026-10-14 ("go for the FP build now; the Sublet build when its gates pass"). The
flash waits for flash-go: the lead's own word once the sealed hash exists AND the lockstep list is green (the FP
acceptance boots complete; the QEMU decode gate on capstone-qemu physical-sublet-decode, now 256e2505b9ad on the new
virtual-capstone 70e98dd3903c (sublet-lifetimes 119/119, runtime 69/69), merged; for the FP flash the renormalisation
branch bounds-encoder-renormalise lands at 4a21b485b01f (QEMU's cap_compress in the residue form verbatim, the
codec test taking the step in exact arithmetic with four top-of-range regions, 6,014,628 checks, 0 failures), the
encoder that pairs with the residue-form RTL; ef143e7012e4 pairs with the sum-form RTL and is superseded (the helper
lane, 2026-10-14); check-repr
and build-b0-hello.sh defaulting to `--format full --renorm` with the OVERLAP interim `--allow-overlap` and the
figures recorded; the static-audit header naming 0x51/0x52 as allowed; the acceptance images rebuilt under the new
word). Name: `caplifive_sublet_d93a8d66b.bit`.

What the synth lane receives with the hash:
- Tree: capstone-ariane capstone-sublet 4be92a85a (since then c8d1ff214: two tests-only commits above it, the timed
  REVOKE pair and sweep-compare.py; RTL unchanged) = 4a4261bb4 (the FP bitstream's tree, a strict prefix) + 124d38406 (the load-side
  S-06 invariant scoped to LDC) + d34c26002 (the word {revnode_id 30 | perm 4 | cap_type 3 | bounds 27}) + b330433fe
  (CDERIVE / CREVOKE, the node parent link, derive_req / kill_req) + 52ad6e5ea (non-linear SPLIT keeps its node) +
  11c405304 (CREVOKE in R-45's commit flush; QEMU's 36 cases ported) + 464abe2ce (MREV deepens its subtree; LINEAR
  split siblings before the parent) + cd5f35df2 (the supervised arm) + ac2968e10 (the testharness ends a run at the
  cycle limit as FAILED) + d93a8d66b (the audit's consequences: one walk for revoke and kill, deepen before the
  handle, the record comment) + 4be92a85a (verif/sim/Makefile only: a FAILED simulation still writes its retirement
  trace). The synthesis target is 4be92a85a's parent d93a8d66b: `git diff --stat d93a8d66b 4be92a85a` lists one file, verif/sim/Makefile,
  so core/ and corev_apu/ are byte-identical between the hash that is built and the tree the gates ran on.
  Hand-written RTL touched beyond the FP tree: ariane_pkg.sv, commit_stage.sv,
  decoder.sv, ex_stage.sv, issue_read_operands.sv, load_unit.sv, tracer.sv; Anvil: capstone_unit.anvilh,
  capstone_rev_node.anvil, capstone_dyn_unit.anvil, capstone_flu_unit.anvil (the generated units differ from
  0568f93a9's).
- Lint (rtl-lint-gate.sh, the pinned container, from `git archive` of the tip): LATCH 52, MULTIDRIVEN 3, ALWCOMBORDER 0, COMBDLY 0, UNOPTFLAT 40, ANVIL_UNOPTFLAT 0,
  BLKSEQ 2, UNDRIVEN 25, UNUSEDSIGNAL 894, CASEOVERLAP 2, SELRANGE 100; PASS; negative control BLOCKED. The 40 UNOPTFLAT names are identical to 0568f93a9's (by-name diff, 2026-10-14). UNUSEDSIGNAL by file against 0568f93a9 (832):
  capstone_dyn_unit.anvil.sv 259 -> 296, capstone_rev_node.anvil.sv 34 -> 58 (generated thread wires, by name),
  ariane_pkg.sv 6 -> 7 (compress_cap's cap.cursor, the FP tree's 833); every other file unchanged. In the hand-written
  files the only by-name differences against 4a4261bb4 are the three metadata slices renumbered by the word
  (commit_stage.sv cap_result_metadata, load_unit.sv cap_metadata and rs1_cap_metadata).
- Behaviour, all under the FIXED harness (the limit reads FAILED) with the Makefile fix (a failed run keeps its
  trace), on both sides: the 95-test sweep compared by .sweep-compare.py against 0568f93a9 swept under the same
  harness: 95 tests: 91 identical (exceptions, readings, retired), 1 differ, 3 missing (rand, perf_tcache_random,
  perf_riscv_random: no log on either side); the SUP_TRACE detector 0 on both; the 26 limit-ending arms compared on
  their readings this time, all identical; the one difference is r31-revoke-cursor retired 999,229 vs 999,223 at the
  2,000,013-cycle cutoff with identical exceptions and readings, step 5's shift (MREV's check pass starts the arm's
  spin later), present on 464abe2ce already. The first attempt (before the Makefile fix) compared 66 and reported the 26 limit-ending
  arms MISSING on both sides: a run that exits non-zero stopped the recipe before spike-dasm wrote the trace. QEMU's 36 portable sublet-lifetimes cases pass and
  the wrong-PC control fails; sublet-derive-kill (19 readings), sublet-split-shares-node (7), sublet-mrev-orphan (8),
  sup-sublet (29) read their pre-registered values; sup-arm/escape/quantum and the three re-budgeted supervised arms
  reach CAP_PASS; the 17 R-12 ladder arms (reclaimer, pool-exhaust, delin-dead, the S1 ladder) against the
  step-5 tree 464abe2ce, the ten limit-ending arms included, by exceptions and readings: 8 identical; opcost's
  two-node REVOKE 0xfb -> 0x103 (+8) with its SPLIT and both MREV readings unchanged; the S1 ladder's timed
  revoke +8 at every N (n1 0x13d -> 0x145 ... n160 0x2557 -> 0x255f: the intercept, not the slope); freelist-n16386
  reaches 16,964 readings before the 2M-cycle limit against 16,933, every common reading identical. Attributed, not
  assumed: the per-instruction commit cycles in the SUP_TRACE show the cycles MOVED, not added. In s1-n8 the first
  MREV commits 8 cycles sooner on the tip and every later instruction runs 8 cycles early with identical gaps, until
  the timed REVOKE, which commits at the SAME absolute cycle on both trees (4284): its reading grows by the 8 cycles
  the program arrived early, and the program's total time is unchanged to the cycle. opcost and the n16386 loop show
  the same shape (MREV -8 / REVOKE +8, sum 0; the loop MREV -27, DROP +10, REVOKE +15, net -2 per iteration). A
  one-change variant (step 6's kill-mode exits alone on the step-5 tree, deepen bound and MREV order as in step 5)
  reads step 5's values exactly on opcost and s1-n8, and the rtl-oracle's read of the generated unit confirms the
  exits and the kill_mode set cost zero states, so the move belongs to MREV's reorder (check, deepen-write, handle):
  the handle's parent write is now MREV's last request, MREV retires sooner, and the next rev-node request completes
  when it did before. Which structure holds that completion point (the rev-node memory channel's write completion
  or the R-35 fill taps) is UNRESOLVED and does not change the reading of the move. A constant per-revoke offset that follows an
  MREV; the board lane's S1 release-cost slope is unaffected, its intercept moves by 8 cycles. RETRACTED here, lane
  branch only (d93a8d66b's message, never on dev): "r12-s1-n8: the same verdicts" was an exit-code comparison; under
  the fixed harness that arm had no retirement trace on either side (the Makefile gap above), so RC=1 on both meant
  only that both reached the limit.
- The synthesis-oriented audit (rtl-oracle, 2026-10-14): every record literal names all eight fields and every
  generated write drives the parent slice; the 110-bit packing and both taps' slices bit-exact; the 3-bit stage
  case full with a default; R-27's nine-channel drain order-consistent; the three new endpoints add no same-cycle
  request-to-response path and no new crossing into the wt_dcache_mem ring. Its two findings are already in the
  tree (d93a8d66b): the kill walk no longer duplicates REVOKE_NODE, and MREV deepens before it mints.
- run.tcl and the flow untouched. DTS: 65536 nodes unchanged (INIT_STAGE head := 3, the sentinel 65535).

Predictions, written before the build (refutation keys in brackets):
- Routed WNS inside -13.335..-5.855. [outside: read the failing endpoint's start and end before concluding]
- Loop membership by stem unchanged; LUTLP-1 0. [a new stem names its cone; LUTLP > 0 is a stop]
- The first failing endpoint, if timing moves, sits behind the rev-node's memory-channel write selector that R-35
  named as the -27.665 ns path: the selector has 18 write sites and 140-bit arms against 13 and 124 on 0568f93a9,
  so revnode_fill_id_q / revnode_rdfill_id_q (ex_stage.sv, the R-35 registers) or the consumers of node_update_a/b
  come first. [an endpoint elsewhere first is a different finding]
- Area: capstone_rev_node up by hundreds of LUTs (two walk modes, a deepen stage, three endpoint selectors, a
  16-bit parent field through every record register: four rev_node_t registers x 16 FFs = 64 FFs plus the
  deepen_mode / kill_mode / stage bits); capstone_dyn_unit up by hundreds (CDERIVE, CREVOKE, the MREV sequence, the
  NONLIN split branch); the decoder two cases; issue_read_operands two ops in a set; commit_stage one op in a list;
  ex_stage two more drain bits and 16 more request bits. Everything else flat. [a rise outside these units is a
  finding]
- Route time: flagged if it crosses the 3h30m mark again, as 0568f93a9's did.

Elaboration-only pass (Vivado `synth_design -rtl`, run.tcl lines 1-85 byte-identical to the repo's, a fresh worktree
at d93a8d66b, Anvil regenerated, the same 276 files as the FP build; the synth lane, 2026-10-14, on the lead's word):
exit 0, 0 ERROR, 0 CRITICAL WARNING, 484 WARNING over 20 ids against the FP build's elaboration stage 478 over 21;
every Synth-8 id count identical, the whole difference three tool ids (IP "moved" notices from the harness, two
run-property notices only the full flow emits). The three classes asked for, read with Vivado's per-id message cap
LIFTED so the rev-node unit was actually visible (capped, 8-6157 saturated before rev_node was elaborated and the
zero meant nothing): no struct-literal field-count mismatch; no width mismatch on the 140-bit node write request or
its slices in ex_stage.sv (the same six warnings on the same six lines as FP); no case-without-default and nothing on
the 3-bit stage register in the generated rev-node unit. Per module, uncapped: rev_node six removed registers (all
Anvil scaffolding present in FP's generated source), 29 no-load port bits of which the only new ones are
_ep_kill_req_0 bits 16-29, the parent id's generation field, unread there by design because CREVOKE's dyn-unit
handler has already checked both nodes through the generation-checked query and the kill handler re-checks the
child's generation and the parent's index only (rev_req's bits 16-29 are unread for the same reason); dyn_unit and
flu_unit removed-register names all present in FP's source. Limit: "present in FP" is a source-name comparison, not an
uncapped FP warning diff (the FP worktree was busy). The dyn unit's Anvil compile took 461 s against 153 s for the FP
tree's whole pre-Vivado phase; the build guard's 240 s window aborted the first Sublet attempt before Vivado started,
the second runs with a 20-minute window. Both builds ran on a host shared at load 80-100, so the route-time lines of
BOTH pre-registrations are VOID for these runs (they measure the host); WNS, stems, LUTLP and the area rows stand.

Synthesis timing optimisation, an observation outside both pre-registrations (the synth lane's wall-clock and cpu
figures, cpu/wall 0.99 on both, single-threaded): FP 4a4261bb4 163.9 min, of which at least 59 at host load <= 20;
the Sublet tip 41.4 min entirely at load 14-20. The host inflates FP's figure and cannot account for the gap. The
baseline is NOT a point: with compress_bounds and decompress_bounds byte-identical by function-body sha across
8f6a0af98, 715bdd1fe, 776d9d859 and 0568f93a9, the phase took 10, 48, 97 and 28 min, and the largest swing (776d9d859
97 -> 0568f93a9 28) came from a diff touching only ex_stage.sv and issue_read_operands.sv. So FP's 164 is the longest
observed, the Sublet tip's 41 sits inside the historical spread, and "neither the renormalisation nor the cursorless
decode alone is slow" rests on single samples from a 10-97 min spread. The combination hypothesis (FP alone has the
renormalised encoder output feeding the cursorless decoder's base reconstruction, a wider encode-to-decode cone)
stays consistent with every point and is not established by them; a 2a3a87dfc build would discriminate only with a
reading far outside 10-97 and is not asked for. What grades the FP bitstream is the routed WNS and, if it leaves the
band, the failing endpoint's cone. Before the seal the FP pre-registration's discriminator was repaired (the synth
lane, 2026-10-14): compress_bounds and decompress_bounds inline, so the check has to be on their call sites, and the
DECOMPRESS arm cannot fail: in 0568f93a9's routed worst-100 every path passes through the pmp data interface's
decompress_cap_metadata, and decompress_bounds is byte-identical between 0568f93a9 and FP, so a hit there says nothing
about the change. That arm is retired as unfalsifiable. The COMPRESS arm (FP's changed function) is graded by a
read-only netlist query on the routed checkpoint, whether the worst path's cells lie in the all_fanin cone of the
compress_cap outputs (ex_stage's flu/dyn rs1 and result metadata, dom_switch_ra_data, commit_stage's capenter pc
metadata), with a constructed path through that cone as the can-fail control, and a name heuristic only as the
pre-filter; the rs3 arm is graded the same way on the fanout of ae21e121e's rs3_src select. Correction to the FP
pre-registration's own description of that change (the synth lane read the diff, 2026-10-14): ae21e121e adds no
4-way per-scoreboard-entry select; its only new logic is a 5-bit 2:1 mux per ISSUE PORT (rs3_src = rs3_fpr ?
result[4:0] : rd), substituted for rd as the comparator input of the existing per-wb-port and per-entry rs3 matches
(rs3_fwd_req, rs3_data, rs3_cap_metadata). The area prediction (tens of LUTs) stands; the cone to query is that
mux's fanout through the comparators into the rs3 operand selects, and if the rs3_src name did not survive
optimisation the cone is anchored on the rs3_fpr-gated result bits and reported as such, never as a zero. FP's synthesis-stage loop indicator: 7 loops as on 0568f93a9, membership the same by stem, the
through-point differences all inside the existing LSU/dcache family, none in compress_bounds, issue_read_operands
or the scoreboard; the routed check_timing at seal is the grade.

### Grade of `caplifive_sublet_d93a8d66b.bit` (sealed 2026-10-14 by the synth lane; graded against the lines above)

- Seal: exit 0, write_bitstream completed; bitstream 11,443,722 B, sha256
  89ba6d816c23b1115585ff6bcc2d5ceeb4523967731a84baf4f8955256077430; provenance HEAD d93a8d66b; flow 2h37m (route
  time VOID); synthesis peak 21.71 GB; the tarball synth-d93a8d66b-exit0.tar.gz.
- **WNS: OUT of band.** Routed -15.856 (TNS -889,578; 100,895 of 255,793 endpoints failing; WHS +0.058), 2.521 ns
  below -13.335; 0.5 ns worse than the FP bitstream's -15.360 on a tree that is FP plus the Sublet commits.
- Stems: HOLDS with one note. Synthesis-stage membership the same (7 loops); routed loop count 1 as on FP and
  0568f93a9, but the arc left unbroken for routing MOVED: lsu_bypass's cap_clear_addr_q[55] I3->O here against the
  dtlb state_q[3] arc on the other two. The clear_addr loop is in all three synthesis logs' member lists (FP's
  synthesis put 18 8-326 exceptions on it, the Sublet tip's 0): the same loop set, a different member left for the
  router, not a new loop. LUTLP-1: 0, HOLDS.
- First failing endpoint: **not** the rev-node memory write selector, so the prediction's named endpoint does not
  fire; the worst 20 paths start at csr_regfile's cpmp bounds iE register and end at the issue stage's and the
  scoreboard's metadata registers (-15.856..-15.574), the same shape as FP's. revnode_fill_id_q (169 nets) and
  revnode_rdfill_id_q (58 nets) survive and sit on 0 of the 20; node_update_a/b did not survive optimisation (0 nets,
  reported as its own finding). The selector is close behind: the worst path through revnode_rdfill_id_q reads
  -14.914, 0.942 ns from the worst, ending at the issue stage's cap_metadata_b; through revnode_fill_id_q -13.248.
- Area against 4a4261bb4 (LUT / FF): **partly REFUTED.** rev_node +623 / +151, the predicted hundreds; dyn_unit
  +4,879 / +36 (own cells +3,075), thousands against "hundreds"; and "everything else flat" is wrong in the other
  direction: issue_read_operands -7,976, csr_regfile -7,500 (the cursorless decode that decompress_cap_tagged and
  decompress_cap_metadata inlined there is gone; an inference from where the drops land, not a cell-level
  measurement), scoreboard +716, lsu -664, dom_switcher +257, ex_stage own +120; TOP -6,976 / +184.
- Cross-build endpoint slacks: FP's worst endpoint reads -15.260 here; 0568f93a9's worst endpoint reads -12.998 here
  (-8.368 there, -11.856 on FP).
- The compress-cone query, a second post-hoc refinement applied to all three netlists (dom_switcher excluded too;
  its rd_req_q cells are loop members and polluted the negative controls; even then the frontend control hits 14
  non-arithmetic cells on 0568f93a9 and the Sublet tip, so the metric is CARRY4 count, chosen post hoc and disclosed):
  0568f93a9 cone CARRY4 108 (IRO 63, dyn 45), worst 20 paths 0 hits, controls 0/0, positive 18; FP cone 248 (IRO
  186, dyn 62), worst 20 all 40-47 hits, controls 0/0, positive 37; Sublet cone 252 (IRO 185, dyn 65, rev_node 2),
  31-43 hits on 19 of 20, controls 0/0, positive 43. With this version the 0568f93a9 side of the FP comparison is
  clean, which the FP grade above lacked: 0568f93a9's worst paths carry no compress-cone arithmetic, FP's and the
  Sublet tip's carry 31-47 carry cells of it.
- Reading: the same cause as FP's, R-54's renormalisation adder on the forwarding path, with the Sublet units adding
  0.5 ns on top and the rev-node write selector's read-fill path within a nanosecond of the worst. The respin
  candidate (the residue form of the condition, on fp-fixes first) applies to both trees; the Sublet branch takes it
  after the FP tree's result. Flashing either bitstream at these numbers is the lead's call.

## Lessons recorded this cycle (memory, not rules)

- A committed file must equal the tested file: 711d7b44b carried a half edit that does not parse while every gate
  had run on the working tree; after `commit -o`, `git diff HEAD -- <paths>` must show no hunk of the change.
- The scratch arm runner reports RC=1 for a test absent from its list exactly like a failure; "identical" across
  two trees from such arms is void. The sweep's list is the instrument for the bounds-regression tests.
- A sim-only check that fires after an ENCODING change: print its inputs on the old tree first.

## Tree B, pre-registered 2026-10-19 BEFORE any handoff: the satp flush alone, and why its own bitstream may not be worth a cycle yet

**The hash.** capstone-ariane `virtual-profile-2-c` `8de3c48cf`, whose `core/` is byte-identical to `d427e7391`'s;
everything after that commit is arm source and testlist only. Against tree A (`8522d72fe`) the whole RTL diff is
**36 insertions and 3 deletions across six files**: a new output and its default in `csr_regfile.sv`, a one-bit port
carried through `ex_stage.sv`, `load_store_unit.sv` and `cva6_mmu.sv`, an unqualified valid-clear in `cva6_tlb.sv`,
and one OR term at the shared TLB's instantiation. `run.tcl` untouched.

**Gates, all green and each with its control fired first.**
- Lint PASS, all eleven counters at baseline: 52 / 3 / 0 / 0 / 40 / 0 / 2 / 25 / 1395 / 2 / 100. **UNOPTFLAT
  membership identical BY NAME** to the base, 40 names each way, the comparison positive-controlled by an invented
  name. This closes the gap the earlier plan carried, that the count had held while the names were never diffed.
- 95-test sweep against the base: **92 identical, 0 differ, 0 missing**, three random generators excused by name
  because they produce no log on either side. Status 0 from the in-tree comparator, whose self-test fires in all
  three directions.
- 152-test supervised list against the base: **152 identical, 0 differ, 0 missing.** Status 0.
- The fix's own validation: four arms read the pre-registered set exactly, with the negative control unchanged in
  every reading, its retired count and its cycle count.

**Predictions, stated against the measured floor rather than tighter than it.**
- Routed WNS inside the standing band, and within the noise of tree A's −18.985. No tighter claim: a null change on
  this design has moved WNS by several nanoseconds.
- **Area: no prediction above the floor, deliberately.** Source-identical modules on this design redistribute by
  about ±1,000 LUT each (measured on tree A: `dom_switcher` +1,006, `dyn_unit` +1,001, `IRO` −918, `scoreboard`
  −913, `lsu` +317, `rev_node` +154, all with no source change). A 36-line change cannot be predicted above that,
  and tree A's refuted +154 is what claiming otherwise looks like. Expected TOP: within ±1,500 LUT of 180,838.
- LUTLP-1 0.
- Loop membership: **no prediction that it is unchanged**, which was tree A's miss. Instead: no member inside the
  MMU or either TLB, because the new signal reaches only the valid flops' clear term, which is a flop input and not
  a combinational path into the lookup.
- **The discriminator, an endpoint question rather than a stem one:** will a failing endpoint appear inside the
  MMU's TLB cone? Predicted NO, for the reason above. A failing endpoint there would mean the flush term reached
  the lookup path, which is the one way this change could cost timing.

**The three sufficiency questions, and the third one does not pass.**
1. *Is it enough?* As RTL, yes for S-29: the window is closed in both directions and both TLBs, validated in
   simulation.
2. *Can the instrument fail silently?* **Yes, on silicon.** There is no readback of the flush. If it did not fire on
   hardware, nothing in a board run would say so; the proof that it fires is the simulation pair.
3. *What will it READ on silicon if the best hypothesis is true?* S-23's fetch panics stop. But with N = 3 panics
   that is weak evidence, and the board lane's own reading of its one capture favours the DATA face, which this
   change does not touch. So a boot that stops panicking cannot be told apart from one that was never going to,
   and a boot that keeps panicking cannot be told apart from a flush that did not fire. **That reading is
   uninformative, so by the standing rule this change does not justify a bitstream cycle on its own.**

**Recommendation: BATCH it.** The RTL is ready and gated and should be carried forward, but its own silicon reading
is too weak to buy a cycle. The candidates to batch with it are the consuming resume (ruled, not yet written),
R-57's shared saved-exception register (audited SHARED-SAFE, and the area it saves is in the capture mux and the
read-side priority chain rather than the flops), and an R-38-class fix if the index-only-compare arm confirms that
candidate. The decision is the lead's; this entry exists so that the alternative was weighed rather than defaulted
into a build.

## The escape-kind arm: first item of the batch after tree B, designed 2026-10-19 from the kind table

**Why it is first.** Two things rest on "every escape returns to CSRUNV+4 whatever its kind", and both are
currently design-static plus one observed kind: S-29's conclusion that the data-side window is an ABI rule a
single fence at CSRUNV+4 closes, and the consuming resume's design. One arm closes both.

**The kind table, from `commit_stage.sv:264-268`, which determines the whole construction:**

| kind | cause | condition |
|---|---|---|
| 1 | preempt | the quantum or an interrupt |
| 3 | 11 | an ecall, and **only if the frame's options bit 0 is set** |
| 4 | 12, 13 or 15 | a page fault, and **only if options bit 1 is set** |
| 5 | insufficient system resources | node pressure |
| 2 | anything else | terminal |

**This corrects how the existing arms' readings were described.** `va-satpstale`'s frame sets options = 1, so bit 1
is clear, and that is why its page-fault and refused-load escapes read **kind 2, terminal**: the terminality comes
from the OPTIONS WORD, not from the nature of the fault. With options = 3 the same page fault is kind 4 and
resumable. The readings stand; the explanation in their headers should say "terminal because options bit 1 is
clear" rather than implying a fault is terminal in itself.

**The construction**, one source with three sub-arms, frame options = 3 throughout so the ecall and page-fault
escapes are both resumable kinds:
- **kind 1** by setting `CSUPQUANTUM` small. **CORRECTED: the quantum is a per-CYCLE budget, not an instruction
  count.** `csr_regfile.sv:2145`'s condition has **no retirement term** and the block at `:2089` is evaluated every
  cycle; what the comment restricts is WHEN it counts (while the domain runs, never during the switcher's walks),
  not what it counts. I wrote "retired instructions" here and said it twice to the semantics owner, inferring it
  from the condition without noticing the absent term.

**AND A FINDING THAT CAME OUT OF GETTING THAT WRONG.** With the budget at 400 or 1,000 cycles the context retires
**nothing at all**: the instruction after the CSRUNV in the trace is the caller's own marker at CSRUNV+4. At 4,000
it completes its whole loop. So more than a thousand cycles are charged to the context between the switcher's walk
ending and the context's first retirement. Two of my hypotheses are refuted: the caller's pre-entry instructions
are not charged (the counter loads at the entry, `:2097-2101` and `:2110-2114`), and it is not the cold page-table
walk after a TLB flush (the same 1,000-cycle quantum without the pre-entry fence also retires nothing). I have not
offered a third. **Why it matters:** a context armed below that cost retires nothing, so a scheduler re-entering it
burns the budget and makes no progress, which is the livelock the exclusion's own comment exists to prevent, and
the kit's armed quantum must exceed it.

**RESULT LINES AND SOURCE, so this stops being prose.** Every quantum value's readings are in
`docs/history/19-10-2026_15-30-00_quantum-zero-budget-s26-rtl-half.md`, and the arm source with all its builds is
`capstone-ariane` branch `virtual-profile-2-c` commit **`9b272ff2d`** (a submodule, which is why a search of this
repository alone does not find it). The startup cost is **between 1,500 and 2,000 cycles**: at 1,500 the context
retires nothing, at 2,000 its loop completes while the budget still expires before its own exit. A **zero** budget
is bit-identical to every sub-threshold value, which is S-26's RTL half.
- **kind 3** by the context's ecall, as today.
- **kind 4** by a load from an address the context's chain does not map, with options bit 1 set.
- kind 5 is left out: node pressure needs the allocator's reserve and is a different arm.
- **a fourth sub-arm, a cause-24 CAPABILITY refusal under options = 3, predicted kind 2.** The semantics owner
  asked for it so the capability case is a reading rather than either lane's inference, and the prediction is what
  distinguishes "the options word re-kinds only page faults" from "it re-kinds any resumable-looking fault". The
  table re-kinds to 4 only for causes 12, 13 and 15 (`commit_stage.sv:266`), and 24 is not among them, so it
  should fall through to 2 whatever the options say. A reading of kind 4 would mean the table is wrong about its
  own cause set, which would reopen Q-31.

**A correction that belongs with this design.** I first wrote that the options word makes a cause-24 refusal
resumable. It does not: the re-kinding is restricted to that three-cause set. The correction applies only to the
PAGE-FAULT reading of the S-29 arm, where kind 2 is a consequence of options bit 1 being clear. The arm's header
carried the overclaim briefly and now carries the cause-set reason instead.

**What each sub-arm must read**, so one arm serves both claims:
1. the escape's kind and cause, from the frame's event words, as today;
2. **the saved return pc**, which no existing arm reads: the frame's id-0 slot at +64, whose cursor is the parked
   pc. The claim under test is that it equals CSRUNV+4 for every kind;
3. the caller's first data access after the escape, both unfenced and with one `sfence.vma` at CSRUNV+4, which is
   S-29's ABI-rule claim tested per kind rather than for the ecall kind alone.

**Pre-registered expectation:** the saved pc equals CSRUNV+4 for kinds 1, 3 and 4; the unfenced reading is the
CONTEXT's value and the fenced reading the caller's own, for each of the three. A saved pc that differs for any
kind refutes the ABI rule as stated and makes the escape direction a hardware problem again rather than a
software one, which is exactly why this arm comes before the consuming resume.

## The data face's transient candidate: the three source answers the arm needs, 2026-10-19

The persistent form of the index-only candidate is **refuted**, by the board lane's localisation rather than by
argument: both refused addresses are in CPU 0's per-CPU unit, and `_raw_spin_lock_irqsave` reads `__mmiowb_state`
in that page on every acquire, so the page was read successfully throughout the boot. A persistent deny of the
slot or the page is excluded. The TRANSIENT form survives, and these are the three things it rests on.

**1. The broadcast is the walk's own traffic, so it can be late.** `ex_stage.sv:1615-1616`:
`revnode_invalidation_valid_o = mem_rev_wr_req_valid && !node_wr_req[31] && !vp_hdr_wr`, id `node_wr_req[29:0]`,
where `mem_rev_wr_req_valid` is the rev-node unit's `_mem_ch_write_req_valid` (`:1533`). So the broadcast tracks
the WALK's memory writes, not the retirement of the instruction that started it. R-27's drain
(`~(|rev_drain_q)` in `capstone_dyn_ready_o`) exists because such traffic outlives a flush, and a switch flushes.
A clear after the CSRUNV escape has retired is therefore structurally possible.

**2. The compare is index-only.** `pmp_data_if.sv:96-108`: the broadcast carries the full 30-bit id but the test is
`revnode_invalidation_id_i[15:0] == cpmp_tracked_revnode_id_d[i][15:0]`, deliberately, because a 30-bit compare
would stop firing once generations advance. So a context-namespace node at generation 0 and index N matches a
kernel entry tracking (g, N) for any g.

**3. WITHDRAWN AND CORRECTED. I claimed `ns_flush` has only a software driver and that is WRONG.** I had grepped
`ns_flush_strobe_d` in `csr_regfile.sv`, found one assignment, and called it the only driver without tracing the
signal that actually arrives at the tracker. The signal is `revnode_ns_flush_o`, and `ex_stage.sv:1388` computes it
from **three** terms:
`(vctx_rev_q != vctx_rev_qq) || (srevroot_rev_q != srevroot_rev_qq) || ns_flush_strobe_i`,
where the first two are edge detectors on `vctx_table_i` and `srevroot_i` (`:1379-1385`). **So a namespace change
fires the flush automatically**, re-arming every tracker as assumed-live (`pmp_data_if.sv:109-114`), and the
software strobe is only the third way in.

**What decides whether an ESCAPE fires it, which is now the live question.** `csr_regfile.sv:2865-2867`:
`vctx_table_o` is true when a supervised or virtual context is active, OR at S outside one with
`srevroot_q != 0`, OR at M with `csupctl.table`. Across an escape `sup_active_q` goes 1 to 0, so:
- if the kernel at S has **`srevroot_q != 0`** the flag stays true, there is **no edge, no flush**, and a cleared
  validity bit **persists** across the escape;
- if **`srevroot_q == 0`** at S the flag falls, the flush fires, and every tracker is re-armed at the escape.

So `srevroot_q` and `sup_active_q` at the fault are the discriminator, and the arm must CONTROL `srevroot` across
the escape rather than inherit whatever the kernel had. The adopt path at `pmp_data_if.sv:89-94` is still a second
re-arm route, on a rewrite with a **different** id, and still does not fire when the same capability is written
back; it is just not the only one.

So the transient shape is: a late broadcast clears the slot, the kernel's next access in that window is refused,
and the next swap putting a different region there re-adopts it. That matches "read successfully all boot, refused
once, in interrupt" with no software strobe involved.

**The matched pair, which is the semantics owner's design and better than reproducing the failure.** The two arms
must differ in **exactly the index**: the failing arm retires a context node whose 16-bit index EQUALS the index
the caller's CPMP slot tracks, the control retires one whose index DIFFERS, and everything else is identical, the
same walk length, the same quantum, the same caller access. The index-only mechanism then predicts a cause-5
refusal on the first and a clean read on the second, while ANY other mechanism, including a generic switch
hazard, predicts the same reading on both. That is what makes the arm discriminate rather than merely reproduce:
a refusal on both arms would confirm a window and refute this explanation of it.

**Three accesses after the refusal, not one** (the semantics owner's design, once persistence reopened): the
first with no intervening rewrite of the slot, where a persistent deny refuses again; the second after the slot is
rewritten with a different region and back, where the adopt re-arms it and the read is clean. Predicted for the
index-only mechanism on the equal-index arm: **refused, refused, clean**; and clean throughout on the
different-index control. Both arms must run with `srevroot` CONTROLLED across the escape, for the reason in point
3 above: with `srevroot != 0` at S the escape fires no namespace flush and a cleared bit persists, while with
`srevroot == 0` it re-arms at the escape and the second access would read clean for a reason that has nothing to
do with the index.

**If the pair is clean on both**, record the walk's last memory write cycle relative to the escape's retirement.
That distinguishes a window that exists and was missed from one that does not exist, and it is the reading that
tells the next person whether to try a longer walk or to stop.

**What the arm must pin, and what this note does NOT establish.** The timing claim: whether a context's retire
walk can still be writing when a kind-1 quantum escape returns the caller. Structural possibility is not
occurrence. And what the kernel's slot 7 is tracking at that moment, which is a monitor question. The arm comes
after the escape-kind arm and after the localisation fork the semantics owner set: if the refused pages are the
module's pinned arena pages or their struct pages, the face is a page-lifetime question and this candidate is not
the one to build.
