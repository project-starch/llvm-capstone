# LDC/STC without the DYN unit's serialisation: design note (lever (b) + lever (a)), 2026-11-13

*RTL lane, on the SPEC lane's decision, inside the lead's goal "every row under 1.20". Base: capstone-ariane
`perf/ids55-best` 5f41c486e (= b98f5a517 + HPM event 23 + R-64/65/66). Written before any RTL. The name if built is
`caplifive_ids55-native-tagcache-mlp-fpieee-hpmreplay-dynbypass_<sha>`.*

## What costs the cycles today (measured, simulation)

capstone-ariane `perf/ids55-opcost` 6c551dc41, `c5a-cost.S -DCOST_DIST`, run `.sup/opdist` in sim14-wt. Eight
distinct destination registers, 1,024 accesses, D$ hits. Cycles per access:

| access | verdict-cache hit | verdict-cache miss |
|---|---|---|
| integer ld (riscv-base shape) | 1.52 | 1.55 |
| ld through a capability (marked page) | 1.64 | 14.83 |
| LDC | **4.22** | 16.46 |
| integer sd | 1.73 | 1.73 |
| sd through a capability (marked page) | 2.54 | — |
| STC | **6.29** | 19.20 |

A first cut of this table, with all loads writing one register, read LDC = ld = 4.2. That reading was write-after-write
serialisation; it is withdrawn.

**Mechanism** (the 4 Nov trace, land/rtl-dyntrim `2026-11-04-dyn-handshake-trim.md`, "After stage A"):
- The DYN unit takes one op at a time: `loop { send rtr >> recv req >> match ... >> cycle 1 }`
  (`capstone_dyn_unit.anvil:858-882`).
- `capstone_dyn_ready_o = !capstone_dyn_valid_i & capstone_dyn_ready & ~(|rev_drain_q)` (`ex_stage.sv:1264`), and issue
  sees `fus_busy.capstone_dyn`.
- LDC holds the unit for 4 cycles; STC holds it for 5 cycles plus 1 cycle of rs1 write-after-write.
- Plain loads through a capability pipeline at ~1.6.

**gcc train01 estimate.** The compiler lane's class diff gives capload +1.72 G and capstore +1.13 G.
- 1.72 G x 2.7 + 1.13 G x 4.6 is about 9.8 G cycles, about 390 s at 25 MHz, of a 2,999 s run.
- That is an upper bound: it assumes independent streams, and dependent code pipelines less.

## Today's path (read at 5f41c486e)

- **Dual dispatch.**
  - An LDC/STC is ALREADY dispatched to both the DYN unit and the LSU (`issue_read_operands.sv:1360-1369`). The LSU
    performs the access, and the DYN unit performs the checks and assembles the result.
  - All LSU results pass through two single-record Anvil syncers. A result whose trans_id is not the DYN op's goes out
    as `normal_res` (`capstone_dyn_unit.anvil:905-976`, :1074+; `ex_stage.sv:1392-1403`, :1536-1548).
  - So the bypass question is not "send it to the LSU". It is "stop the result and the exceptions from depending on
    the DYN unit".
- **Checks that exist only in Anvil:**
  - LDC: NOT_CAP 24; type in {LINEAR, NONLIN, SEALEDRET} 26; read permission 27; bounds `[start, end-16]` (SEALEDRET
    `[start+48, start+1008]`) 28; 16-byte alignment.
  - STC: the same, plus UNINIT (with `imm == 0`).
  - LDC's post-load rule: loading a linear-family capability through an rs1 without write permission gives 27
    (`capstone_unit.anvilh:721-747`).
  - The LSU's own capability check covers plain LOAD/STORE only. It is gated on capmode, M-mode and fu LOAD/STORE, and
    admits only LINEAR/NONLIN (`load_store_unit.sv:1344-1430`).
  - For LDC/STC the LSU runs only the lifetime verdict (`dynlt`, :1471-1500).
- **Results.**
  - LDC: rd = the loaded capability; rs1 unchanged (and not written at commit since stage A).
  - STC: rd = rs2 (`decoder.sv:1413`), cleared to cnull for a linear-family rs2; rs1 written back at commit, changed
    only for UNINIT (cursor + 16).
  - Capability metadata reaches the scoreboard only on write-back ports 0 (FLU) and 4 (CAP/X/ACC); every other port
    writes `cap_result = '0` (`scoreboard.sv:240-247`).
- **Holds that assume one op in flight:**
  - the item-B LDC exception hold `ldc_lsu_ex_q` and its STC twin;
  - the C.5a verdict record `dyn_ldst_q` / `dynlt_v_q`;
  - `capstone_dyn_ftval_q` and `capstone_dyn_req_pc_q` (`ex_stage.sv:1242-1295`, :1415-1532).
- **The load unit holds an LDC until its data returns.** It goes to `SEND_TAG_LDC` with no pop (`load_unit.sv:446-458`),
  because a linear-family LDC must clear its source granule (`LDC_CLEAR_WAIT`, single `ldc_saved_*` registers). On a
  D$ hit the trace shows this as 2 cycles (`IDLE -> SEND_TAG_LDC -> IDLE`).
- **STC claims rs1 at issue** (`issue_read_operands.sv:616-619`, `stall_waw_rs1` :1484). A younger capability op
  through the same base waits until the STC commits. Compiled spill code stores through one base (sp), so this
  serialises STC streams even with the DYN unit gone.

## The design

**(b) LDC/STC become LSU operations with their own check.**
1. **The checks move into the LSU.**
   - The Anvil LDC/STC checks are written in SystemVerilog beside `cap_violation_detection`, evaluated on `lsu_ctrl`
     (registered, the same place the plain check sits, NOT at issue) for `fu == CAPSTONE_DYN && op inside {LDC, STC}`.
   - They cover every Anvil arm: the types including SEALEDRET and UNINIT, 16-byte bounds, the SEALEDRET window,
     alignment, and UNINIT `imm == 0`.
   - The priority order is kept: today's LSU-exception > DYN code > 25 > replay (`cva6.sv:1550-1573`).
   - A failing op is suppressed in the LSU and reports its exception on its own port with `tval` as today (from
     `capstone_dyn_ftval_q`'s rule).
   - LDC's post-load 27 is checked where the load result is formed, on the loaded tag/type and rs1's write
     permission (known at request).
2. **The lifetime verdict joins the access's exception**, as it already does for plain accesses (`cap_exception`),
   instead of the side record `dynlt_*`. The R-43 replay marker travels the same way.
3. **The results ride the LSU's own write-back ports, with metadata** (revised 2026-11-13, before RTL).
   - The LDC result (the loaded capability, made by the same `decompress_cap_tagged` -> `cap_lane` pair as today, so
     it is bit-identical) goes out on LOAD_WB (port 2). The STC result (rd = rs2 or cnull, the rs1 lane) goes out on
     STORE_WB (port 1). `cap_wbdata` grows from two entries (ports 0 and 4) to four. Plain loads and stores keep
     `cap_result = '0`.
   - **Why not port 4 (the first version of this note):** the DYN unit's `ep.res` is statically timed
     (`capstone_unit.anvilh:711`, `res : (result_pack@#1)`, no ack), so it cannot be held back. Sharing port 4 with
     LSU results would need a queue in front of CAP_WB as deep as the in-flight window: about 8 entries x ~600 bits,
     plus a cycle of latency on every DYN result.
   - **The cost of this choice, and the risk the SPEC lane asked to hear about.** Metadata forwarding already walks
     every write-back port's `cap_data` (`issue_read_operands.sv:749-757`), and ports 1/2 are constant zero today
     (`scoreboard.sv:320-321`), so synthesis prunes them. Making them live adds live inputs to the rs1/rs2/rs3
     metadata forwarding muxes, which feed `fu_data_q`, the worst timing family (`cpmp_q -> fu_data_q`). It adds no
     new loop: the muxes already exist. The synthesis-only gate reads the per-module delta and the worst-path
     family. The S-07 detector (`scoreboard.sv:327-344`), which flags an LDC retiring on port 2, is updated to the new
     rule.
4. **Issue.** `fu_busy` for LDC/STC drops the `fus_busy.capstone_dyn` term and keeps `fus_busy.load` / `fus_busy.store`
   (`issue_read_operands.sv:575-576`). **This edits a standing loop cone** (`issue_pointer`, `fu_busy`): the
   synthesis-only gate decides, and the SPEC lane is told before synthesis (its condition).
5. **The single-op holds** (`ldc_lsu_ex_q`, the STC twin, `dynlt_v_q`, `ftval`, `req_pc`) are no longer used for
   LDC/STC. Exceptions are per access, carried by the LSU.

**(a) STC stops claiming rs1.**
- STC changes rs1 only when rs1 is UNINIT (cursor + 16). The issue stage cannot know the type without putting the
  forwarded operand into the claim (the worst timing family), and the dyntrim note rejected that variant.
- The design instead: STC never claims rs1. At commit, an STC whose rs1 was UNINIT writes rs1 and raises
  `flush_commit` (the same mechanism as REVOKE/CREVOKE, `commit_stage.sv:889-894`), so younger instructions re-read
  the advanced cursor.
- The commit-time rs1-lane write (`cap_we_o`, `commit_stage.sv:565-566`) becomes conditional on rs1 having been
  UNINIT, the way LDC is already excluded. In-order commit is not relied on to make an unconditional stale write
  harmless.
- Cost: one flush per UNINIT STC.
- **The compiler lane's count (2026-11-13, native route, nat-fair-cache SDK):** no store of any kind through an UNINIT
  rs1 in 500.perlbench_r test (>= 37.7 M counted capability-mode stores: 22.3 M STC, 15.5 M slow-path scalar) or 502.gcc_r
  train01 (>= 12.6 M: 5.46 M STC, 7.1 M scalar).
  - Instrument: a counting copy of the twin, native01 qemu55 a0501ebc39, at capstone_check_access.
  - The stack is NONLIN, by a native CSLCC type probe of sp, a stack local, a malloc'd object and a fresh heap grant.
  - The totals are lower bounds (progress lines; the exit line is lost).
  - **Positive control.** The same counter code, ported to stock capstone-qemu deb7d75756 (its checks live in
    `_helper_access_with_cap`), counts 256/256 on the module-route probe `uninit_init_then_use_ok`: the probe's
    4,096-byte fill is 256 STCs through UNINIT x13. The negative control `linear_no_drop_ok` reads 0.
  - That twin binary itself cannot run the probe: it halts with cause 24 before the fill. So the zero there rests on
    the same counter code plus the routing argument: the inline fast path admits only types 0/1, and STC always
    takes the helper.
  - A miscount could only make (a) costlier, never incorrect.

**What stays.**
- The load unit's LDC hold, 2 cycles on a hit, needed for the linear clear. A later lever could pop early when rs1
  lacks write permission (no clear is possible), but that is not in this change.
- The STC's store-buffer entry.
- The tag path.
- The R-43 probe and replay.
- The linear-LDC `commit_lsu` on port 0.

## Exception order and tval (settled before RTL)

- **Order.** Today an LDC/STC's LSU exception (page fault, access fault) outranks the DYN codes, because ex_stage
  merges them in that order (`cva6.sv:1550-1573`). A plain access through a capability is the opposite: its check is
  folded in before translation (`load_store_unit.sv:1145-1146`).
- **The twin.** capstone-qemu deb7d75756, `_helper_access_with_cap` (`target/riscv/op_helper.c:1396+`), raises
  24/26/27/28/25 and misalignment inside the helper, BEFORE the guest load or store (`tcg_gen_qemu_ld`/`st`, where a
  page fault would come from). So the twin puts capability faults first for LDC/STC too.
- **Decision.** The bypass folds the LDC/STC checks into `cap_violation_detection`, which gives the twin's order
  (capability cause first).
- **Not a measured divergence.** The SPEC lane's independent read (rtl-oracle, capstone-qemu deb7d75756) agrees on the
  order (`trans_capstone.c.inc:169 -> :181` LDC, `:273 -> :283` STC; plain `trans_rvi.c.inc:278-279`, `:390-391`).
  It adds that in that tree LDC/STC exist only in M-mode with bare translation (`cpu_helper.c:850-855`), so no page
  fault can arise there. The native route's twin is a different build. So the corner "a capability-faulting LDC/STC
  to an unmapped page" is unexercised on both sides; it is not a recorded silicon/twin divergence.
- **Codes, as that read records them.** This tree's twin also differs from silicon in its codes:
  - out-of-bounds raises 5/7, and 28 has no raise site (`op_helper.c:1539`);
  - the access path has no permission check (`:1508`);
  - misalignment outranks bounds;
  - tval is cursor+imm for 4-7 and 0 for 24/25/26/29 (`cpu_helper.c:1672-1695`).
  These are twin divergences to record, not to copy. The bypass keeps silicon's codes, order of checks and tval
  rule, so the equivalence gate (silicon before vs after) holds.
- **tval.**
  - Silicon reads tval = rs1's cursor for the DYN capability codes: ldcstc-excodes L5, LDC with imm -16, reads obj.
  - Misaligned LDC/STC read tval 0.
  - The bypass keeps that rule, so the equivalence gate holds; the plain-access rule (tval = the effective address)
    is not applied to LDC/STC.
  - Whether the twin's tval equals silicon's here is not read; that goes to the SPEC lane as a twin question, not as
    an RTL one.

## The alternative considered: keep the DYN unit and remove only its wait

**The variant.**
- The DYN unit sends its result without waiting for the LSU: the exception code, rs1, and STC's rd, with no
  `recv cap_load_ri.res` in the LDC/STC threads.
- ex_stage holds that result per trans_id, in an NR_SB_ENTRIES-deep table replacing the single `ldc_lsu_ex_q` /
  `stc_lsu_ex_q` / `dynlt_v_q`, until the LSU's result for the same trans_id arrives, then writes back on port 4.
- The checks stay in Anvil, unchanged. Exception priority stays in ex_stage, port 4 is as it is, and `fu_busy` is
  untouched, so no loop cone is edited.

**What it buys.**
- The DYN unit's per-op occupancy falls to its own turnaround: rtr, the registered dispatch and the `cycle 1` after
  each op, about 3 cycles.
- Estimate: LDC 4.2 -> ~3 against ~2 for the full design. That is about one cycle per LDC (~1.7 G cycles on gcc)
  less than the full bypass.
- The `cycle 1` cannot simply be dropped: Anvil requires a positive loop delay (see the measurements).

**Decision.** The full design goes ahead (the SPEC lane's decision, the lead's "all efforts"). The variant is the
fallback if the full design's equivalence gate or its loop-cone edit fails, and it reuses the per-trans_id table. The
variant was weighed as follows:
- it is the smaller equivalence surface (no Anvil check is rewritten);
- it stops at ~3 per LDC because the DYN unit still takes one op per turnaround;
- it leaves STC's rs1 claim, so it needs (a) either way.

## Measurements taken before RTL

**The one-base stream.** capstone-ariane `perf/ids55-opcost` 9dd319fa8, `-DCOST_DIST -DCOST_BASE1`: eight accesses
through the heap capability at offsets 0..112, distinct registers. Run `.sup/opbase1` in sim14-wt. Cycles per access,
D$ hits:

| access | one base | distinct bases (`.sup/opdist`) |
|---|---|---|
| ld through a capability | 1.64 | 1.64 |
| LDC | 4.22 | 4.22 |
| sd through a capability | 2.54 | 2.54 |
| STC | **7.21** | 6.29 |

- STC's rs1 claim costs +0.92 per STC in the spill shape today. Loads do not claim rs1 and do not move.
- With the DYN wait gone, the claim becomes the larger share. A younger STC on the same base waits for the older
  STC's commit, so (a) matters more after (b) than before it. That is why (b) and (a) are one change.

**The `cycle 1` probe did not run.** `perf/ids55-dyncyc` f07d2e231 removed the `cycle 1` after an op in the DYN main
loop. anvilc rejects it with "Event graph error: Recurse delay must be greater than 0!" (`capstone_dyn_unit.anvil:859`):
some arms of the match take no cycles. Dropping the turnaround therefore needs the loop restructured, with the delay
moved into the zero-delay arms. The alternative's "~3 per LDC" stays an estimate. The probe branch stays local; it is
not for synthesis.

## Prediction (simulation, `c5a-cost -DCOST_DIST`, before any RTL)

| arm | today | predicted with (b)+(a) | basis |
|---|---|---|---|
| LDC hit, distinct rd | 4.22 | 2.0-2.5 | the load unit's 2-cycle LDC hold; plain ld through a capability is 1.64 |
| STC hit, distinct rs2 | 6.29 | 2.5-3.0 | sd through a capability on a marked page is 2.54 |
| LDC / STC miss | 16.46 / 19.20 | ~14.8 / ~15.5 | the replay path stays; a miss then costs what the plain capability access's miss costs (ld 14.83), and the STC adds the store side (sd through a capability, 2.54 against ld's 1.64, +0.9) |
| STC stream through ONE base (spill shape) | 7.21 | ~2.5-3.0 with (a); ~4+ with (b) alone (the issue waits for the older STC's commit) | sd through one base is 2.54 |

- **gcc train01 (upper bound):** 1.72 G x ~2.0 + 1.13 G x ~3.6 is about 7.5 G cycles, about 300 s of 2,999 s.
- **mcf:** in proportion to its LDC/STC count, which comes from the SPEC lane's class diff. Its D$-footprint share
  is untouched.
- **Refutation keys (written now):**
  - LDC hit above 3.0 or STC hit above 4.0 means the design missed its mechanism; it is read from the trace before any
    synthesis request.
  - Any value difference in the 95-test sweep, or in an arm covering every Anvil exception code, stops the change.

## Gates

1. **Directed arms for every Anvil LDC/STC exception arm**, run on base and change with values identical: NOT_CAP,
   each type, permission, bounds edge ±1, the SEALEDRET window, UNINIT `imm != 0`, misalignment, the post-load 27, and
   each `tval`. Several exist already (`c5a-arms.S`, the ldc/stc tests); the missing ones are written first.
2. **The cost arms above, plus the one-base STC arm.**
3. **`rtl-lint-gate.sh`**, paired with the base.
4. **claim-auditor, aimed at synthesis risk:**
   - the `fu_busy` edit in the `issue_pointer` loop cone;
   - the live metadata on write-back ports 1 and 2 (forwarding into fu_data_q);
   - the new checks feeding `cap_exception` (the dominant slack family);
   - equivalence with Anvil arm by arm.
5. **The 95-test sweep with RVFI compare**, against perf/ids55-best's.
6. **Synthesis-only gate**: per-module LUT delta (lsu, ex_stage, issue, dyn_unit; floor first, +/-2,500), loops 0,
   mapping in class. The full build only on a pass.

**Size.** Several days: the checks and their arms (1-2), the result path on ports 1 and 2 (1), (a) with the flush (1), and the
gates. Built in that order, with each piece's arms green before the next.

## Built and measured (2026-11-13, simulation), and lever (a) withdrawn

capstone-ariane `perf/ids55-dynbypass`.

**Pieces.**
- Piece 1, 1c8691a11: the Anvil LDC/STC checks in the LSU.
- Piece 2, 3b8bfe890: LDC/STC off the DYN unit, with their results on LOAD_WB/STORE_WB.
- (a), cc7a73cbb: withdrawn at 44e7b933c.
- `+define+CAPSTONE_NO_DYN_BYPASS` turns the bypass off on the same tree. Off reproduces the base's cost vector
  exactly.

**Equivalence.** 19 LDC/STC tests plus ldcstc-excodes (every Anvil check arm) read identically off and on, retired
counts included. No invariant fires.

**Cycles per access, off -> on** (`c5a-cost -DCOST_DIST`, D$ hits):

| arm | off | on |
|---|---|---|
| LDC hit | 4.22 | 2.75 |
| STC hit | 6.29 | 2.73 |
| LDC miss | 16.46 | 15.82 |
| STC miss | 19.20 | 15.54 |
| STC through one base (spill shape) | 7.21 | 5.28 |
| plain ld | 1.64 | 1.64 |

LDC is 0.25 above its prediction band (2.0-2.5) and inside the stop key (3.0).

**Lever (a) as built (cc7a73cbb) was defective.**
- With it on, the one-base STC read 2.73, the predicted gain.
- But r30-fill-init hangs at its `fence iorw,iorw`, and ldcstc-excodes S5c loses the second of two back-to-back
  UNINIT STCs.
- Likely cause: the UNINIT STC commits its store (`commit_lsu_o`) in the same cycle that (a) raises `flush_commit`.
  No store-carrying op had ever flushed at commit before. Commit port 1 can also retire a younger reader in that
  cycle.

**Outline for (a), next bitstream.** Commit the UNINIT STC normally with commit port 1 held. Then replay the next
valid head through the R-43 path: flush, then refetch the same pc (`replay_commit_o`; frontend `refetch_same_pc`),
without committing it. The work is forcing a clean head not to commit: every commit output (gpr/fpr/cap writes,
`commit_lsu`, CSR, fence, AMO) must be held that cycle. Arms: r30-fill-init, S5c, the excodes vector, the one-base cost
arm. The flush is cold in perl/gcc: the compiler lane counts 0 UNINIT-rs1 stores, with the counter validated 256/256
on a module-route probe.

## claim-auditor on 44e7b933c, and what was done (2026-11-13)

The audit found four value divergences that the off/on equivalence tests could not see.

- **A, fixed (e66746899).** Under translation, the folded LDC/STC capability fault made the MMU skip the walk, and
  PMP judged the VA as if it were a PA. Without translation the fold stays. Under translation the fault is registered
  on the MMU's request strobe and raised after a fault-free translation, behind any page or access fault: the DYN
  path's order.
- **B, a deliberate change from the DYN path, kept (the SPEC lane's ruling).** A dead node, an rs1 without write
  permission and a linear-family granule now read 25 where the DYN path read the post-load 27. The verdict is checked
  before the access, as the twin does.
- **C, fixed (294170bc3).** A plain load's raw user lane reached the forwarding muxes on port 2. Forwarding ignores
  cap_data's valid bit. The lanes are now '0 unless an LDC/STC lane is valid.
- **D, fixed (294170bc3).** A faulting LDC/STC's tagged lane was forwardable before its flush. The lane's valid now
  requires no exception.

Synthesis risk named by the audit: no new combinational loop, but a new deep path into `fu_data_q` (load pipe register
-> decompress -> syncer -> compress -> port-2 forwarding mux), plus the dispatch-time decompress/compress into the
trans_id table. The synthesis-only gate reads it.

## Pre-registration of the speculative builds (2026-11-13, before any build)

**The lead's instruction (relayed by the SPEC lane):** every optimisation is built in parallel, with synthesis
starting alongside its remaining checks. A check that fails voids that build: it is discarded and recorded.

**Builds:**
- **DYN bypass v2:** capstone-ariane perf/ids55-dynbypass-v2 **eb101c351** (pushed). It is 5f41c486e + piece 1 +
  piece 2 + the C/D lane gating + the translation merge (capability fault outranks the MMU's result) + the trans_id
  table with no flush clear. Its last commit is sim-only. Name: `caplifive_ids55-native-tagcache-mlp-fpieee-hpmreplay-dynbypass_eb101c351`.
- **Combined:** perf/ids55-combo **ea174bfdd** = eb101c351 + the E2 config (DTLB 64, L1D 64 KiB). Name:
  `...-hpmreplay-dynbypass-dtlb64-l1d64k_ea174bfdd`. It is speculative: it flashes only after each single change has
  shown positive on silicon.

**Checks still running when the builds start:** v2's bypass-on arms (every translated arm, excodes, the 19 tests,
the cost arms), the 95-test sweep against perf/ids55-best's, and lint. v2 lint was already read on 0a1c0cf02, whose
synthesisable RTL is identical: PASS, UNUSEDSIGNAL 1479 (-1), the rest at baseline. The combined variant's lint is run
too.

**Synthesis-only gate keys** (against the ids55-best gate checkpoint):
1. Mapping in class.
2. Loops 0 (check_timing, "Found timing loop", 8-326, LUTLP-1).
3. Area report-only at the +/-2,500 floor, with the change's own rows.
   - Bypass: load_store_unit, load_unit, ex_stage, issue_read_operands, scoreboard, the DYN unit.
   - Combined: also the DTLB and wt_dcache.
4. The worst-path family against the base. Watched: a new family from `i_pipe_reg_load` through the bypass's
   decompress/compress into `fu_data_q` (the claim-auditor's named risk), and anything ending in the DTLB or the D$
   way select.

**Full build:**
- placed LUT reported against 181,565;
- placer <= 5 and router <= 5;
- 0 routing errors;
- 0 loops;
- WNS against the base's -15.077 (null band; a value);
- sha256.

A routing failure or a WNS outside the band is a NEGATIVE result: its artefacts are collected and labelled, with no
further runs on that variant, and nothing is tuned to fit.
