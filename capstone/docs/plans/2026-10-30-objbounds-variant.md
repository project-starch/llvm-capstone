# ObjBounds: bitstream 3 without the per-access lifetime check

*RTL lane, 2026-10-30. The performance ladder's ObjBounds rung. Runtime lane's request on the lead's word ("Если нужен
новый битстрим, то делай его"); scope ruled by the runtime lane the same day. The flash is the runtime lane's, on
the lead's word in that session (board 2, after B's two check boots).*

## Why

The microbenchmark ladder (103 programs, BEEBS/RV8/CoreMark, results on dev) has these rungs:

| Rung | What it has |
|---|---|
| Plain | native RV64 |
| Caps | an allocator with coarse bounds |
| ObjBounds | per-object bounds, no lifetime hardware |
| MapLife (T1) | per-object bounds, plus the per-access lifetime check and per-mapping lifetimes |
| ObjLife (T2) | MapLife, plus a lifetime per malloc object |

"The cost of temporal safety" is ObjLife / ObjBounds. It needs a bitstream on which a capability access skips the
lifetime check and nothing else changes. That bitstream is bitstream 3 (capstone-ariane e7af48457), because every
microbench cycle count was measured on it.

## What the check costs on e7af48457 (from source)

There are two different mechanisms, by which functional unit the access goes through. The first analysis sent to
the runtime lane described only the first, and was retracted the same day ("Corrections" below).

**Plain loads and stores (fu LOAD/STORE): a cached check.**

- **Hit: free.** The verdict is a same-cycle lookup in the `cap_violation_detection` chain, after NOT_CAP, type,
  permission and bounds (`load_store_unit.sv:1366-1403`). It looks up a registered 4-way x 64-set cache of node ids:
  the set is index[5:0] and the tag is all 30 bits. The cache is filled only by watching the rev-node unit's writes
  (a CDERIVE, SPLIT or MREV child is resident from birth) and its reads (LCC, a probe). A hit reads no record and
  stalls nothing.
- **Miss: a replay.** On a miss:
  1. The access takes an internal replay marker on the registered exception path.
  2. A probe goes to the rev-node unit, which reads the record through D-cache port 2: its own read port
     (`cva6.sv`, `dcache_req_to_cache[2] = rev_rd`). Its writes share port 3 with the store unit. The record read
     can hit L1D or miss to DRAM.
  3. Commit holds the marked head until the probe resolves, then flushes and re-fetches the same pc.
  4. The re-executed access hits.

  The two probe timeouts bound the wait, and the check fails closed.
- **The namespace flush empties the cache** at every context entry and escape. After each escape, every distinct node
  a context touches misses once.

**Capability loads and stores (LDC, STC): an uncached query on every access.** The decoder sends LDC and STC to the
DYN unit (`decoder.sv:1402/1410`, fu CAPSTONE_DYN), so they never reach the LSU's check. After its type, permission,
bounds and alignment checks, the DYN unit asks the rev-node unit for rs1's node validity on every LDC and every STC
(`capstone_dyn_unit.anvil:685/:753`, `get_node_query_validity`). The rev-node unit reads the record
(`capstone_rev_node.anvil:423-432`) and the instruction waits for the answer. No cache sits in front of it.

**Fetch: free.**

- **Fetch costs nothing.** The PC tracker is one id and one valid bit (`commit_stage.sv:389`). It adopts a new PC
  capability's id as live and is cleared by a broadcast. It never stalls and reads no record.

So a plain load or store pays only on a miss: a working set of at most ~256 live nodes pays one replay per node per
escape, and a larger one pays a replay per access. Every LDC and STC pays a record query, whatever the working set.

### Every lifetime consumer on e7af48457

Enumerated by the call sites of the check, not by the cause number. The rev-node unit's query site says which ops it
fronts (`capstone_rev_node.anvil:428`: LCC, SPLIT, REVOKE, TIGHTEN, CALL, RETURN, LDC, STC), and every
`ex_code::INVALID_CAPABILITY` in the DYN unit was mapped to its op:

| Consumer | Per access? | In the variant |
|---|---|---|
| (a) LSU verdict, plain loads/stores, AMOs | yes, cached | removed (macro) |
| (b) PC tracker's 25 at commit (fetch) | yes, free | removed (macro) |
| (e) DYN query on LDC/STC | yes, uncached | removed (DYN Anvil source; not under the macro) |
| (c) CPMP trackers (integer-mode accesses) | no cost; physical broadcasts only | unchanged |
| (d) DYN: DROP, REVOKE, MREV, SPLIT, CDERIVE, CREVOKE (both operands), DELIN, TIGHTEN, CALL, RETURN, CSSUPERVISE, and LCC's query | per op | unchanged (ISA semantics) |

The domain switcher (including CSRUNV entry) takes no lifetime verdict (no node reference in
`capstone_dom_switcher.anvil`), and the FLU unit only copies ids.

## The change (capstone-ariane; the variant is `objbounds-b`, its control `objbounds-off-b`)

**The variant to synthesise is objbounds-b 603528b8b. Its control is objbounds-off-b 1782d20bd** (the runtime lane's
ruling, option B).
- 603528b8b is 24e588199 plus two commits that change no RTL token:
  - 13c772ee4 corrects the comment beside the define in the config package. It had said every other behaviour is
    e7af48457's, which ignored the DYN-unit change. With comments stripped, the file has the same 374 tokens.
  - 603528b8b re-baselines the lint REF (UNUSEDSIGNAL 1431 -> 1435, attributed under Gates). It is verif/ only.
- Every simulation gate below ran on 24e588199.

`git diff 1782d20bd 603528b8b -- core` is exactly:
- the config package's `define CAPSTONE_NO_LIFETIME_CHECK` and its comment (5 lines);
- in `capstone_dyn_unit.anvil`, LDC's and STC's `let validity = call get_node_query_validity(rs1_v.metadata.revnode_id)`
  becomes `let validity = 1'd1` (two lines, each with a comment naming what bitstream 3 does there).

Anvil has no preprocessor, so site (e) is not under the macro. The control's identity with e7af48457 therefore rests
on two facts:
- its core/ preprocesses to e7af48457's tokens with the macro undefined;
- its DYN source is e7af48457's (core/ against e7af48457: only the `ifndef` wrappers of load_store_unit.sv and
  commit_stage.sv).

How it got here, all on the lane branches:


- **c7b020d7b** puts `ifndef CAPSTONE_NO_LIFETIME_CHECK` around two sites:
  - the LSU verdict (all six arms: hit-dead, same-cycle invalidation, ALLOW, probe DEAD, replay marker, tie-off
    deny);
  - the PC check's `!pc_revnode_valid_d -> 25`.

  With the macro defined, an access that passed NOT_CAP, type, permission and bounds is allowed. It gets no 25 and no
  replay marker, so no probe starts and no record is read (the probe starts only from `rvc_miss_now`). The cache keeps
  filling and is never read (in simulation; synthesis should trim it, Gates 5), and the R-43 refusal record
  (apertures 204-208) reads 0.
- **0af4307fc (option A, measured and not taken)** adds site (e): while the DYN unit's in-flight op is LDC or STC, ex_stage.sv answers the DYN unit's
  query itself, live, without reaching the rev-node unit.
  - The op is latched at dispatch; the unit serialises ops, so the latched op is the one that queries.
  - The request is accepted at once and the answer comes one cycle later. There is no record read and no wait for one,
    only the channel handshake (its residual is measured below).
  - Every other op's query passes through unchanged, and the generated DYN unit is untouched.
  - A flush drops an owed answer, the R-27 rule.
- **3c5c43996** defines the macro beside `CAPSTONE_ITFC_EN` in
  `capstone_cv64a6_imafdc_sv39_config_pkg.sv`. That is the mechanism the target already uses for that macro, so the
  synthesis flow is unchanged.
- **objbounds-off f32ecccb1** is the define-off twin: c7b020d7b with 0af4307fc's changes, and nothing else. It is the
  control tree for the A/B and the sweep. `git diff f32ecccb1 0af4307fc` is the config package's define, 3 lines.
- **objbounds-noquery 233e0e81d** is a SIM-ONLY measurement tree, never to be synthesised: 0af4307fc with the DYN
  unit's two LDC/STC query calls replaced by a constant. It measured option A's residual, and it is option B's
  behaviour: B differs from it only by the removal of A's ex_stage answer, which goes idle once LDC/STC stop
  querying.
- **Why B and not A.** A's answer is accepted in the request cycle and given the next, the minimum. The rest of the
  call's sequencing (set waiting >> send >> recv >> set validity) stays inside the DYN unit. On one resident node
  (sim, memory delay 12) it kept 1.98 of the LDC query's 3.96 cycles and 2.95 of the STC query's 4.67: about half of
  what the variant exists to remove. B's residual is the constant's own `>>`, 0-1 cycle per op. The runtime lane's
  reason: "a residual that keeps 50-63 % of the LDC/STC query on resident nodes would make ObjBounds read too slow
  and the cost of temporal safety read too low".
- **objbounds-b 24e588199** is 0af4307fc with site (e) moved into the DYN source and ex_stage.sv back to e7af48457's
  text; 13c772ee4 corrects its config-package comment and 603528b8b re-baselines lint. **objbounds-off-b 1782d20bd** is f32ecccb1 with ex_stage.sv
  back likewise.
- **Scope**, as ruled by the runtime lane:

  | Site | Under the macro? |
  |---|---|
  | (a) the LSU verdict | in |
  | (b) the PC tracker's 25 | in |
  | (c) the CPMP trackers (integer-mode accesses, physical broadcasts only, zero cycles) | out, unchanged |
  | (d) the DYN unit's 25s on CDERIVE, CREVOKE, CSRETIRE and LCC operands | out: the ISA semantics, so the same SDKs and kits run |

- **Off-equality at source.** With the macro undefined, both files preprocess (`verilator -E -P
  +define+CAPSTONE_ITFC_EN`) to e7af48457's token stream exactly: load_store_unit 3,044 tokens, commit_stage 2,412.
  Defined, 79 and 11 tokens drop out, the lifetime arms only. ex_stage.sv (site e) is likewise token-identical
  undefined (4,118 tokens) and adds 99 defined.
- **Batched with it: nothing.** r64-flags (R-64/65/66) is gate-complete, but it changes the FPU netlist, and this build
  exists only for cycle comparability with e7af48457.

## Gates

All on the pair: variant 24e588199 (vp8; 603528b8b by tokens) and control 1782d20bd (vp9). Each model was rebuilt
from a removed `work-ver/` after the Anvil prologue, and every PROVENANCE reads `rtl_dirty=no`.

1. **The control is bitstream 3 by its generated units.** The prologue's sha256 prefixes for the control's five
   generated Anvil units, ex_stage.sv and ariane_pkg.sv equal e7af48457's own clean runs (vp8 `.sup/r62ud`, vp9
   `.sup/sweepr62`). The variant differs in one of them:

   | Unit | e7af48457 = 1782d20bd | 24e588199 |
   |---|---|---|
   | anvil_dyn_unit | 7accda6a28c7bf08 | 131818fb5bfda03c |
   | csr_unit, dom_switcher, flu_unit, rev_node, ex_stage, ariane_pkg | same | same |

2. **Arms** (memory delay 12, R43_TRACE; vp8 `.sup/arms-b-on`, vp9 `.sup/arms-b-off`; the no-query column is
   objbounds-noquery 233e0e81d, `rtl-base-wt/.sup/objbnq`). Cycles:

   | Shape | Control 1782d20bd | Variant 24e588199 | No-query 233e0e81d |
   |---|---|---|---|
   | s1: 4,096 plain ld through one capability | 16,770 | 16,770 | not run |
   | s2: 64-node chase, 1,024 ldc | 9,251 | 5,215 | 5,215 |
   | s3: 1,024-node chase, 2,048 ldc | 174,952 | 145,274 | 145,274 |
   | s4: 64 nodes after a namespace flush, pass 1 / pass 2 | 656 / 708 | 464 / 464 | 464 / 464 |
   | s5: churn / passes | 663,679 / 285,921 | 663,618 / 235,396 | 663,618 / 235,396 |
   | s6: 1,024 ldc, one node | 9,264 | 5,205 | 5,205 |
   | s7: 1,024 stc, one node | 11,378 | 6,592 | 6,592 |

   - The variant equals the no-query tree on every cycle count. The control equals the earlier define-off twin
     f32ecccb1 on every value and every cycle count.
   - Probes: s5 2,049 on the control, 0 on the variant; s3 and s4 1 each on the control, 0 on the variant.
   - Every other CAPPRINT value is equal across the pair.
   - Per op, on one resident node: 3.96 cycles per LDC (s6) and 4.67 per STC (s7). These are the query's sim cost
     under memory delay 12, and a pattern, not a silicon value.
3. **The FIRE arms.**
   - `stale-access` reads 0x19 ×4, 0, 0x19 ×2, 0x77, 6 on the control. That is the vector `sublet-bare` read on
     silicon on the Sublet bitstream (`caplifive_sublet_d93a8d66b`, 2026-10-16). On e7af48457 it had not been read on
     silicon before this kit's bitstream-3 arm, which read it on board 2 (Silicon, below). On the variant it reads
     0 ×7, 0x3333, 0: every 25 is gone and the stale store lands.
   - `r43-evict-live`: every 0x19 on the control reads 0 on the variant. Both trees end at the test's own capmode
     RVTEST_PASS (rc 1), as on every earlier tree.
   - `objb-pcc` completes on the variant (0xbb5 0 0). On the control it loops in its trap handler (76,882 trap lines)
     until the cycle limit.
   - `objb-pcc-ctrl` reads 0xbb5 0 0 on both.
4. **Lint** (`verif/sim/rtl-lint-gate.sh` on 24e588199, vp8 `.sup/lint-b-on.txt`).
   - Every counter equals e7af48457's except one: LATCH 44, MULTIDRIVEN 3, ALWCOMBORDER 0, COMBDLY 0, UNOPTFLAT 39
     (the REF's 39 names exactly), BLKSEQ 2, UNDRIVEN 25, CASEOVERLAP 2, SELRANGE 100, ANVIL_UNOPTFLAT 0.
   - UNUSEDSIGNAL is 1435 against REF 1431, so the gate FAILed on the ceiling alone. The delta is attributed by file
     against the bitstream-3-era lint (a73a7c670, 1431):
     - load_store_unit.sv 9 -> 15 (+6): rvc_inv_now, rvc_lu_hit, rvc_lu_nodeval, rvc_ok_match, rvc_dead_match and
       rvc_dead_tmo_q. Each is read only by the verdict that the macro removes.
     - The generated DYN unit 311 -> 309 (−2). Its wire names renumber with any source change, so this part is
       attributed by count only.
   - 603528b8b sets the REF to 1435, verif/ only.
   - The paired reading is the control's lint, on its own tree with REF 1431 (vp9 `.sup/lint-b-off.txt`): PASS, every
     counter at baseline. Its 1,431 UNUSEDSIGNAL warnings equal the bitstream-3-era lint's as a multiset, by file and
     by full message.
5. **claim-auditor over the diff: SUPPORTED** for no new latch, loop, multi-driven or undriven signal, and for what
   the macro does. Its readings:
   - Every variable the removed arms assigned is defaulted at the top of `cap_violation_detection`:
     `cap_exception = '0`, `rvc_miss_now = 1'b0`, `rvc_den_arm_now = 4'b0000` (load_store_unit.sv:1321-1324).
     commit_stage's chain has no trailing `else` after the 25 arm, and its defaults are at :371-372.
   - So `rvc_miss_now` and `rvc_den_arm_now` are constant 0. The probe, the ALLOW/DEAD records and the refusal record
     stay at reset, and the LSU's probe and refusal ports are driven by constants.
   - The define reaches both files by `CAPSTONE_ITFC_EN`'s path. Flist.cva6:62-63 puts the config package before
     :125 and :141; the Makefile's fpga target writes add_sources.tcl from that Flist (`target ?=
     capstone_cv64a6_imafdc_sv39`); run.tcl has no verilog_define. Both sites sit inside `ifdef CAPSTONE_ITFC_EN`.
     The synthesis host's launcher was not read.
   - The Anvil edit removes both halves of `get_node_query_validity`'s send/recv pair (:173-180). Ten call sites
     remain, so the channel stays live. In the generated unit, `_rev_node_ep_query_res_0` has 10 consumers against
     the control's 12; only the query request, its valid and the response ack change shape, two sites each. Port
     lists are identical. An undriven-signal scan (positive control injected) reads the same set on both trees.
   - What it adds to the scope: CALL, RETURN and CSSUPERVISE still query the node they enter (scope d), so the rung
     pays one query per domain entry. The CPMP trackers still gate CPMP-region permission (scope c).
   - "The cache keeps filling and is never read" is true in simulation only. In synthesis the cache has no observer
     and should be trimmed, together with its four `(* ram_style = "distributed" *)` tag arrays
     (load_store_unit.sv:1168-1174, 4 × 64 × 30 bits). That is key 3.
6. **Kit image sims** (vp8 `.sup/kit-b-on`, vp9 `.sup/kit-b-off`): `objb-bare-2026-10-30/PREREG.md`.
   - Every non-cycle position is equal across the trees.
   - objb-stale and objb-pcc separate the two bitstreams.
   - The cost shapes read the same pattern as the arms.
7. **95-test sweep, both trees** (testlist_capstone.yaml, seed 20260922, memory delay 12; vp9 `.sup/sweep-b-off`,
   vp8 `.sup/sweep-b-on`; each tree SWEEP-RAN 95, 184 logs, `rtl_dirty=no`).
   - **The control equals bitstream 3's own sweep** (vp9 `.sup/sweepr62`, e7af48457). 92 tests are identical in
     exceptions, readings and retired count; 0 differ; 3 random-generator entries are missing on both sides, as
     always. The cycle counts are identical in all 92 tests that have an end line on both sides.
   - **The variant against the control:** 88 identical, 4 differ, 3 missing. 11 of 92 tests differ in cycles, and every
     one is faster on the variant: stc-counter-pair −93,886, cap-tag-cache −1,948, excode-base-audit −130, the three
     stc-ldc-sbpressure arms −75, s07-wbuf-tag-reorder(-ctl) −32, s07-wbuf-liveness −20, s06-mechanism-probe −10,
     cap-pointer −5.
   - **The four that differ, each read from a traced re-run** (SUP_TRACE, the sweep's seed, delay and time_out;
     vp9 `.sup/sw4-off`, vp8 `.sup/sw4-on`; the re-runs reproduce the sweep's end lines to the cycle):
     - **excode-base-audit.** ARM I's LDC through a revoked capability (0x800000c0) raises INVALID_CAPABILITY on the
       control and completes on the variant. The test exists to pin that 25, so the variant fails it at TESTNUM 2. It
       is a 25-deny now completing, at site (e).
     - **data-sharing.** The LDC after REVOKE (0x80000088) raises INVALID_CAPABILITY on the control. On the variant it
       completes, loading 0x80003010, the shared region capability's cursor, and prints it. Then the harness's capmode
       exit store traps, and both trees park at the cycle limit. It is a 25-deny now completing.
     - **s07-ldc-chain-forward.** Both trees take the same single trap, the LCC control's UNEXPECTED_OPERAND at
       0x80000050, and both park in the same plain-load loop at the cycle limit. The variant enters that loop 204
       cycles earlier (4,180 against 4,384) and so retires 511 more instructions by the limit. This is timing only.
     - **cap-tag-cache. NOT a 25-deny: the T4 eviction arm's positive control fails on the variant** (tohost 42).
       - The control reads hot LDC 9 cycles and swept LDC 88. The variant reads 5 and 2, and the arm requires
         swept ≥ hot + 8.
       - The trace shows why. On the variant the swept LDC (0x800001c4) writes back at cycle 334,980, the cycle the
         window's first `csrr mcycle` commits. Its miss ran in the shadow of the sweep loop's last missing `ld`, which
         wrote back at 334,975.
       - On the control the same LDC writes back at 335,066, 86 cycles after the window opens. Its query's record read
         and its data load are two serialised misses.
       - The loaded value is correct on both trees: 0x80003000, type check 41 passes. So the tag survives the refill.
         What fails is the arm's assumption that the swept LDC's latency lands inside its mcycle window.
       - The pre-registered criterion ("each difference is a 25-deny now completing") is NOT met for this test, by
         the letter. The difference is explained by trace as timing, with every value correct.
   - **Instrument note.** This lane's cycle column read the `.log` until today, and the end line is in the
     `.log.iss`. It printed 0 everywhere, and the retraction is dev b71a4db444f0. The readings above come from the
     fixed script. Its positive control is this variant's 11 differing tests.

## Synthesis pre-registration (objbounds-b 603528b8b; written and pushed before the gate request)

The control for every key is bitstream 3's own build of e7af48457, read by the same query from its own checkpoints.

**Size gate** (synthesis only, existing settings, 40 GB guard):
1. **Mapping.** Technology mapping peak at the marker is in bitstream 3's class: 4.8-5.6 GB (b3 5,004.9 MB). The
   whole-run peak is under the 34 GB watchdog. No single Vivado process goes above 9 GB.
2. **Loops.** "Found timing loop" headers 0, Synth 8-326 0, check_timing 0 combinational loops (b3: 0 / 0 / 0).
3. **The macro reached synthesis.** On the synthesised checkpoint, the cells named `*rvc_occ_q*`, `*rvc_nodeval_q*`,
   `*rvc_rr_q*`, `*rvc_ok_q*`, `*rvc_dead_q*`, `*rvc_pend_q*`, `*rvc_den_v_q*` and `*pc_revnode_valid_q*` number 0, and
   so do the tag arrays' cells, `*g_rvc_tag*`.
   - Positive control: the same query on bitstream 3's synthesised checkpoint reads > 0.
   - Negative control: `*cpmp_revnode_valid_q*` (the CPMP trackers, ruled out of scope) reads > 0 on both.
3b. **The DYN-unit change reached synthesis.** Site (e) is in the Anvil source, not under the macro, so key 3 cannot
   see it.
   - The `core/capstone_dyn_unit.anvil.sv` that the synthesis tree generates has sha256
     131818fb5bfda03c24490cf3f8d4cf6867c2cd7a93936a3067fe93f8632e9d1f. That is this lane's prologue reading for
     24e588199, and 603528b8b's Anvil source is the same.
   - Control: the same file in bitstream 3's synthesis tree reads `7accda6a28c7bf08…`
     (7accda6a28c7bf0863ee16166d8f28c1eb437101cdd38ca3f9e479bca941d2b2).
   - If synth's anvilc output differs from this lane's at both commits, the key is instead read as a diff: synth's
     603528b8b file against synth's e7af48457 file must differ only in LDC's and STC's query paths.
   - The DYN unit's own LUT/FF rows (variant against b3) are reported, not graded.
4. **Size.**
   - The core's FF count falls by at least 700. The named flops alone are about 820: occ 256, nodeval 256, rr 128,
     the ALLOW/DEAD/pending records about 95, the probe wait counter 20, the refusal record 35, the PC tracker 31.
   - The core's LUT count does not rise above b3's 144,584 by more than the ±2,000 top-level floor.
   - The LUT fall itself (the tag storage and its compares) is reported, not graded. The 7,680 tag bits are
     requested as distributed RAM (load_store_unit.sv:1168-1174), but bitstream 3's mapping of them is not on record
     here. Synth's b3 reading of the matched cells goes beside it.
   - ex+issue rises by no more than 2,500 over b3's, read on the same checkpoints. This key is one-sided: the cache
     sits inside ex_stage (lsu_i), so a large fall is the expected direction, not a failure.

**Full build** (on a gate pass):
5. **Placed area.** LUT ≤ 169,061 + 2,000; FF ≤ 63,647 − 700 + 300.
6. **Congestion.** The placer's final estimate has no window above level 5. The router's level is ≤ 5 if the router
   reports one (b3's routed `-congestion` report segfaulted twice; its placed reading is the control).
7. **Routing.** 0 routing errors, 0 overlaps.
8. **Loops and warnings.** check_timing 0 loops; TIMING-23 0 and LUTLP-1 0, each with its b3 control firing.
9. **The macro reached the routed netlist.** Key 3's query on the routed checkpoint reads 0, with b3's routed
   checkpoint > 0 as the control.
10. **WNS.** Reported, not graded (the lead's ruling). b3 is −10.047.
11. **The bitstream.** .bit sha256, size and path.

**Size gate readings (synth, 2026-10-30, synthesised dcp, ObjBounds against b3; graded here).**
- **Key 1: PASS.** The mapping peak at the marker is 4,951.7 MB (b3 5,004.9). The whole-run peak is 22.24 GB, and the
  largest single process 4.95 GB.
- **Key 2: PASS.** "Found timing loop" 0, Synth 8-326 0, check_timing 0. The matcher's control, vp3g, reads 7 and 73.
- **Key 3: PASS.** Cells, ObjBounds / b3:

  | Name | ObjBounds | b3 |
  |---|---|---|
  | rvc_occ_q | 0 | 655 |
  | rvc_nodeval_q | 0 | 1,124 |
  | rvc_rr_q | 0 | 414 |
  | rvc_ok_q | 0 | 19 |
  | rvc_dead_q | 0 | 10 |
  | rvc_pend_q | 0 | 2 |
  | rvc_den_v_q | 0 | 20 |
  | pc_revnode_valid_q | 0 | 14 |
  | g_rvc_tag | 0 | 817 |
  | rvc_ok_id_q | 0 | 63 |
  | LUTs named `*lsu*rvc*` | 0 | 1,980 |
  | NEG `cpmp_revnode_valid_q` | 179 | 238 |

  b3's routed dcp gives the same counts, apart from one carry cell in occ (654), so key 9's control is ready.
- **Key 3b: PASS.** The generated DYN unit in synth's tree is 131818fb…9d1f, and b3's is 7accda6a…d2b2: both equal
  this lane's.
- **Key 4, LUT and ex+issue: PASS.**
  - Core LUT 142,909 (−1,675 against 144,584). That includes LUTRAM −480 (870 → 390).
  - ex+issue LUT +1,962 (ex −6,318, issue +8,280, the re-homing A showed). ex+issue FF −605.
  - In b3, the removed cells were 1,679 LUTs, 600 distributed-RAM cells (120 RAM64M + 480 RAMD64E) and 645 named FF.
- **Key 4, core FF ≥ 700: NOT MET by the letter.** The core reads 40,782 FF, −641.
  - Where it moved:
    - lsu_i −805, of which its own cells −790;
    - commit −32, the PC tracker's 31;
    - ex_stage's own cells −54; issue −14; csr +10;
    - the FPU **+284**, in opgroup_block[0] / fmt_slice[1].
  - The FPU's RTL is unchanged. Synthesis retiming flipped:
    - b3: forward 2, backward 42, registers added 146, deleted 31;
    - ObjBounds: forward 100, backward 169, added 1,280, deleted 281.

    1,085 of the 1,179 retimed register names are under the FPU.
  - **The clause was mis-scoped.** It was written at the core total, where retiming in an untouched module moves
    hundreds of flops. It set its threshold 120 below an estimate (~820) that proved high: b3's named flops are 645,
    because the tags are distributed RAM.
  - What the clause existed to show is that the removal reached synthesis. That is shown by key 3, with its POS and
    NEG controls, and in flop terms by the stems: lsu_i −805 and commit −32.
  - This is a re-scoping made after the reading, so it is recorded as one and carries less weight than a
    pre-registered key. The same lesson was recorded earlier for A's ex_stage band: grade the stem, not the total.
- **GO for the full flow on that basis** (this lane's ruling). Before launch, key 5's FF clause is restated the same way:
  - **Key 5, as restated before the full build:**
    - Placed LUT ≤ 169,061 + 2,000, unchanged.
    - FF: placed lsu_i falls by ≥ 700 and commit_stage by ≥ 25, against b3's placed checkpoint by the same query.
    - The core and top FF totals are reported, not graded.

**Full build: SEALED 2026-10-30 (synth). Graded here: PASS on every graded key.**
- **The run.** Exit 0 in 1 h 33 m 11 s. The guard's peak was 21.85 GB and the watchdog's 21.91 GB; collection exit 0
  at a peak of 6.90 GB.
  - write_bitstream completed, and PROVENANCE reads 603528b8bce2.
  - Synthesis matched the gate: the mapping peak was 4,950.0 MB, Synth 8-326 0, timing-loop headers 0, and retiming
    100 / 169 / 1,280 / 281, identical to the gate's.
- **Key 5 (as restated before launch): PASS.**
  - Placed LUT 167,258 against a limit of 171,061.
  - lsu_i FF 7,341 against b3's placed 8,146: −805, which meets ≥ 700.
  - commit_stage FF 210 against 242: −32, which meets ≥ 25.
  - Reported only: core FF 40,782 (−641), top FF 63,006, core LUT 142,174 (b3 143,978), fpu_i FF 2,210, slices
    47,860 (93.94 %).
  - Control: b3's placed totals read 169,061 / 63,647, the numbers on record.
- **Key 6: PASS.** The placer's largest window is level 4 (16×16, east, global). The router's effective maximum is 3.
- **Key 7: PASS.** Failed, unrouted and partially routed nets 0 / 0 / 0. Overlaps went 74,820 → 0.
- **Key 8: PASS.** check_timing on the routed dcp reads 0 loops. LUTLP-1 is 0, with its control PDRC-138 firing at 275
  (b3 288). TIMING-23 is 0, with its control TIMING-16 firing at 1,001 (b3 1,001).
- **Key 9: PASS.** On the ROUTED dcp all nine names read 0, and NEG `cpmp_revnode_valid_q` reads 179. b3's routed dcp
  is above 0 on every name: occ 654, nodeval 1,124, rr 414, g_rvc_tag 817, and so on.
- **Key 10, reported and not graded.**

  | | ObjBounds | b3 |
  |---|---|---|
  | WNS | −9.433 | −10.047 |
  | TNS | −49,930.352 | −44,010.926 |
  | failing endpoints | 11,182 of 183,832 | 13,396 of 188,701 |
  | WHS | +0.054 | +0.029 |
  | THS | 0 | 0 |

  The synthesis lane flags every build with WNS < 0 as NOT FLASHABLE; that is its standing rule. WNS is not an
  acceptance criterion: that is the lead's ruling, recorded in the SPEC board PREREG's Addendum 14. Bitstream 3, at
  −10.047, is in service on both boards.
- **Key 11.** `ariane_xilinx.bit`, 11,443,722 B, sha256
  0f03b097bbd1693bdc3424a872eb2c88f086d5d5fc3d771b29d4945490ec18c1. It is in the synthesis host's
  `~/scratch/capstone-ob/corev_apu/fpga/work-fpga/`. The tarball `synth-603528b8b-exit0.tar.gz` (376,010,792 B,
  sha256 40039facf9c1f182…) holds the routed dcp, TIMING-FORENSICS and PROVENANCE, and its .bit hashes identically.
- **Grade: a flash candidate.** Before the flash, the board side checks its copy's sha256 against the line above, and
  then runs the kit's ObjBounds arm (PREREG addendum 1).

**Failure branches.**
- A mapping blowup (a single process > 9 GB, or the 34 GB watchdog) is graded by genome and Synth 8-326 targets, with
  no retry.
- A routing failure gives the placed area and keys 1-5 plus a congestion census, and no WNS.

## Silicon (board 2, both arms; kit `capstone/tests/rtl-smoke/objb-bare-2026-10-30`)

Every reading below was graded here by script (`objb-grade.py`, `objb-wgrade.py`) from the raw run directories, against
the kit's PREREG and its addenda: image hashes recomputed, the resident name read from each run.log, SR/SV by value.
The runtime lane ran the boots and graded both arms with its own script; the two grades agree. For the witness it
relayed the numbers only, and the verdict below is this lane's.

**The flash (runtime lane, 2026-10-31 00:43-00:46).**
- The local copy's sha256 is the seal's: 0f03b097…c18c1. The runtime lane checked it before the flash, and it was
  recomputed here.
- After the mandatory power cycle the resident name read back `caplifive_vp2c_603528b8b.bit`. The console does not
  expose a sha256.
- The simulated RTL is the flashed RTL: the kit's variant sims ran on 24e588199, and 24e588199 → 603528b8b changes
  only the config package's comment and `verif/sim/rtl-lint.REF.txt`.

**Bitstream-3 arm (e7af48457, 2026-10-30 19:50-20:01): AS EXPECTED, 10 of 10.**
- objb-pcc-ctrl read `0xbb5 0x0 0x0`.
- objb-stale read `0x19 0x19 0x19 0x19 0x0 0x19 0x19 0x77 0x6`, its first bare reading on e7af48457.
- Every cost shape's non-cycle positions equal the table.
- objb-pcc entered and did not return, which is its expected reading.

**ObjBounds arm (603528b8b, 2026-10-31 00:46-00:55): CONFIRMED, 10 of 10.**
- objb-stale read `0x0 0x0 0x0 0x0 0x0 0x0 0x0 0x3333 0x0`. objb-pcc returned `0xbb5 0x0 0x0`, equal to its control.
- Every cost shape's non-cycle positions equal the table.
- The cost pattern, board 2 against board 2, same images. Silicon cycles; the simulated ratio under memory delay 12 is
  for reference and is not graded:

  | shape | bitstream 3 | ObjBounds | ratio | predicted | sim ratio |
  |---|---|---|---|---|---|
  | s1: 4,096 ld, one capability | 16,535 | 16,535 | 1.000 | equal within 1 % | 1.000 |
  | s2: 64-node ldc chase | 9,262 | 5,163 | 0.557 | lower | 0.564 |
  | s3: 1,024-node ldc chase | 94,864 | 76,125 | 0.802 | lower | 0.793 |
  | s4: 64 nodes after a flush, pass 1 | 622 | 389 | 0.625 | lower | 0.707 |
  | s4: pass 2 | 634 | 391 | 0.617 | lower | 0.655 |
  | s5: churn over 1,024 objects | 203,210 | 203,722 | 1.003 | equal within 1 % | 1.000 |
  | s5: two passes | 190,224 | 136,745 | 0.719 | lower | 0.760 |
  | s6: 1,024 ldc, one node | 9,232 | 5,131 | 0.556 | lower | 0.556 |
  | s7: 1,024 stc, one node | 11,269 | 6,153 | 0.546 | lower | 0.557 |

  Every prediction holds. On one resident node the query costs 4.00 cycles per LDC (s6) and 5.00 per STC (s7) on
  silicon; the simulation read 4.01 and 4.92.

**What the identity reading separates, and the witness.**
- objb-stale rules out a probe that never reaches the check. The hash-identical image took 25 at all six stale
  accesses on bitstream 3, on the same board, and on ObjBounds stale store #7 retired and wrote 0x3333.
- It cannot rule out that the revocations themselves were ineffective on the ObjBounds bitstream. Every ObjBounds
  reading fits that hypothesis as well as a removed check.
- Its datum covers one access of six.
- The witness `objb-stale-w` (PREREG addendum 2, pre-registered at land/rtl-objb a6c75c1f0343 before the boot) runs
  TIGHTEN, whose node query the variant keeps, on each revoked node. It also reads path B's store back.
- **The witness on board 2 (2026-10-31 06:22-06:23): WITNESSED.** The image named in run.log hashes to ae0e1cb3…,
  which is addendum 2's hash. The resident name is `caplifive_vp2c_603528b8b.bit`, and SR is 15. The vector is
  `0x0 ×7 0x3333 0x0 0x0 0x19 0x19 0x19 0x2222 0x3`, the ObjBounds row exactly.
  - TIGHTEN on the live node (#10) reads 0, so the op runs to completion.
  - TIGHTEN on path A's child, path B's piece and path C's reloaded capability (#11-13) reads 25 on each. On the
    ObjBounds silicon all three revocations took effect, so the zeros at positions 0-6 come from the removed check.
  - Path B's stale store landed 0x2222 (#14), so two of the six stale accesses now have a value witness. The trap
    count (#15) is 3: the three TIGHTENs.
  - This is N=1, on board 2 only. The witness ran as a boot of its own after the ObjBounds arm, and board 2 then
    went back to bitstream 3 (the runtime lane's reflash).

## Corrections

- **Corrected 2026-10-30, the same evening, on the lane only.** c8be354d28ef's Gates section called stale-access's
  control vector "sublet-bare's bitstream-3 silicon vector". sublet-bare read it on `caplifive_sublet_d93a8d66b`, not
  on e7af48457 (`sublet-bare-2026-10-15/PREREG.md`, the board section). No bare stale-access reading on e7af48457 is
  on record. Found while writing the kit's PREREG, by checking the claim against that file.

- **Retracted 2026-10-30, the same day, before anything went to synthesis.** The first cost analysis sent to the runtime
  lane said that the check's cost is set by misses and that a hit is free. It listed four consumers, with every DYN-unit
  25 under "semantic ops' operands". The first variant (3c5c43996) followed that list.
  - **What it missed:** LDC and STC are DYN-unit ops, which query the node on every access. The variant as first built
    left that query in place.
  - **Found by the first A/B.** The pointer chases (shapes 2-4, built on ldc) read equal cycles on both trees and
    printed zero LSU verdict lines.
  - **The slip:** the consumers were enumerated from `grep d25` over core/*.sv plus the blocks already read. The DYN
    unit raises `ex_code::INVALID_CAPABILITY`, an Anvil enum, and its prints were seen and not broken down by op.
  - **The fix:** site (e), ruled IN by the runtime lane: first in ex_stage (0af4307fc, option A), then in the DYN
    source (24e588199, option B), after A's residual was measured.
  - **The method now:** a check's consumers are enumerated by the call sites of the function that implements it, and
    by the decoder's fu routing.
