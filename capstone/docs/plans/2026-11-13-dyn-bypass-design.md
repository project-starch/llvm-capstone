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

## Pre-registration: DYN bypass v2 + DTLB 64, without the L1D (2026-11-13, before any build)

**Why.** The SPEC lane asked for it: this is the realistic final bitstream if both single-change A/Bs come out
positive, and it should not wait behind the L1D 64 KiB build's routing verdict (+7 k LUT, TOP ~191 k).

**The build.**
- capstone-ariane perf/ids55-dynbypass-dtlb64 **bec4beb51** (pushed) = eb101c351 + 097401036's config change
  (DataTlbEntries 64).
- Name: `caplifive_ids55-native-tagcache-mlp-fpieee-hpmreplay-dynbypass-dtlb64_bec4beb51`.
- Host: apollo, through the board lane (leffe is at its cap).
- Speculative: it flashes only after both single changes have shown positive on silicon.

**The checks it rests on.**
- The bypass's checks all passed on eb101c351: arms 33/33 identical off vs on, sweep RVFI 92/92, lint PASS.
- DTLB 64's synthesis-only gate passed: core +3,005 LUT / +4,526 FF, the DTLB 881 -> 4,894 LUT, and no DTLB/MMU path
  in the worst 200.
- Its own lint is sent with the hand-off.

**Predicted area.** Base placed 181,565. The DTLB adds about +3.0 k LUT, and the bypass's own delta comes from its gate
(running on leffe), so roughly 185 k placed or more. This is a report, not a key.

**Keys.** As above: the gate keys, plus the full build's congestion <= 5, 0 routing errors, 0 loops, WNS against
-15.077 (a value), and sha256. A routing failure or a WNS outside the band is a NEGATIVE result: collected and
labelled, no reruns.

**Prepared, not launched: L1D 48 KiB / 12 ways** (perf/ids55-e2-l1d48k b9ab75c41).
- It is NOT config-only. The D$ victim LFSR is $clog2(ways) wide and would name ways 12..15, which do not exist, so
  those values fold onto 8..11 (wt_dcache_missunit.sv). A power-of-two way count never takes the fold.
- Its lint and a simulation smoke test against the base are running. It launches only if L1D 64 KiB fails to route,
  on the SPEC lane's word.

## Bypass v3 (raw lanes) and its speculative builds (2026-11-13, before any build)

**The finding.** bypass v2 + DTLB 64 (bec4beb51, sealed on apollo, sha256 5190f43a...) placed at 193,954 LUT: +12.4 k on
the base, of which the DTLB is ~3 k, at 99.6 % slices and router congestion 6. It routed with 0 errors and WNS -14.885
(inside the null band).
- The bypass's bulk is decompress -> compress round trips: one compress on the LDC result path; two decompress and two
  compress at STC dispatch.

**v3.** capstone-ariane perf/ids55-dynbypass-v3 **3ef8dd947**, `CAPSTONE_DYN_BYPASS_RAWLANE`:
- the LDC lane is the memory word's own {slot, tag, metadata};
- the STC table stores the operands' raw lanes, and reads only the tag bit and the type field.
- Correct only if cap_lane(decompress(L)) == L on every lane a register or memory holds. Two checks run in parallel
  with the builds, and a failure of either voids them:
  - (1) an identity test of the package's own functions (perf/lane-proof/);
  - (2) the same tree with `+define+CAPSTONE_NO_DYN_RAWLANE`, which restores the round trip, compared arm for arm,
    then the 95-test sweep.

**Builds, in parallel with the checks (the lead's standing order):**
- v3 + DTLB 64: perf/ids55-dynbypass3-dtlb64 **39dfed222**, apollo (board lane). Name
  `caplifive_ids55-native-tagcache-mlp-fpieee-hpmreplay-dynbypass3-dtlb64_39dfed222`.
- v3 alone (3ef8dd947): leffe, when a slot frees.
- Predicted: placed well below bec4beb51's 193,954. The round trips were the bypass's bulk, so roughly 185-188 k.
  A report, not a key.
- Keys as for v2: mapping in class; 0 loops; 0 routing errors; WNS against -15.077 (null band); congestion reported.
  A routing failure or a WNS outside the band is a NEGATIVE result.

## Pre-registration: the realistic final, v3 + DTLB 64 + the SHRINKTO/CAPBOUND/CAPENTER fix (6d0e754d5), before any build

**What changed since the v3 section above.** Neither v3 build launched:
- v3 + DTLB 64 (39dfed222) failed the lint gate at UNUSEDSIGNAL 1486. The fix, 38ae2d775, hoists the round-trip
  temporaries under `ifndef` and reads the type bits directly. v3 + DTLB 64 on it is c1fbeb632, lint PASS 1480.
- v3 alone (38ae2d775) has a VOID lint: two Anvil regenerations ran at once and left an empty
  `capstone_dyn_unit.anvil.sv`. It is re-run before that tree goes anywhere.

**The tree.** capstone-ariane perf/ids55-final **6d0e754d5** (pushed) is c1fbeb632 + ee82e1c6a + 6d0e754d5:
- SHRINKTO refuses an immediate that wraps below the cursor;
- CAPBOUND refuses inverted bounds and lengths >= 2^64 - 2^55;
- CAPENTER's operand form refuses operands past MEM_END;
- all three with ILLEGAL_OPERAND_VALUE (29).

That FLU fix closes the E = 52 lanes, the one class where the raw-lane identity proof does not hold. Name:
`caplifive_ids55-native-tagcache-mlp-fpieee-hpmreplay-dynbypass3-dtlb64-shrinkfix_6d0e754d5`.

**Gates passed** (sim14-wt, rtl 6d0e754d5, dirty=no):
- Lint: PASS, every counter at baseline (LATCH 44, MULTIDRIVEN 3, UNOPTFLAT 39, UNDRIVEN 25, UNUSEDSIGNAL 1480,
  SELRANGE 100, the rest 0 or at baseline).
- Arms, compared by value:
  - shrinkto-neg = the fixed reading (ffb82d5bb, 25 values: N, B and E read 29; P and O unchanged).
  - ldcstc-excodes, vb-trans-vp2, vb-trans-capa-vp2, va-runv, c5a-cost-ldc/stc-hit8-dist = v3's readings
    (sim10 .sup/rlon on 3ef8dd947), 6 of 6 SAME.
- v3's raw-lane A/B on 3ef8dd947: 39 of 40 arms identical with `+define+CAPSTONE_NO_DYN_RAWLANE` (confirmed in the
  build line) against raw lanes on, cost arms included. stc-neighbour-load parks at wfi on both sides, as on every
  sweep since vp3.
- Running or queued at hand-off; a failure of either voids the build: the 95-test sweep on 6d0e754d5 itself (it covers
  v3, DTLB 64 and the fix together), and v3's sweep on 3ef8dd947.

**Host and order.** apollo (board lane), speculative, under the lead's parallel order. It is a hedge: the SPEC lane
now prefers a final with the winning I$ change and WITHOUT DTLB 64, unless mcf's event-4 count is large. This build
also gives the first placed reading of v3.

**Predicted:**
- Placed about 185-189 k LUT: bec4beb51 placed at 193,954, and v3 removes the round trips that were most of the
  bypass's +9 k.
- The FLU fix adds a few hundred LUT (three 64-bit compares).
- A report, not a key.

**Keys** (base 5f41c486e: placed 181,565, WNS -15.077):
1. Mapping in class (5.0-5.5 GB at the marker). A blow-up fails; its partition is read.
2. Loops: check_timing 0, "Found timing loop" 0, Synth 8-326 0, LUTLP-1 0.
3. 0 routing errors.
4. Congestion: placer max <= 5, router effective <= 5. bec4beb51 read 6 and routed; a 6 that routes is reported as
   not met by letter, not as a failure.
5. WNS inside the null band of -15.077 (+/- ~3.7 ns), reported as a value.
6. sha256 recorded.

**Failure branch.** A routing failure is a NEGATIVE result: placed area, congestion and the census are read; no WNS
is recorded; there is no rerun with identical inputs.

## Checks completed, DYN bypass v2 sealed, and the v3-alone build (2026-11-13)

**DYN bypass v2 (eb101c351), sealed on leffe** by the synth lane; graded against the v2 keys above.

| key | reading | grade |
|---|---|---|
| placed Slice LUT (report) | 189,064 (+7,499 on 5f41c486e) | reported |
| congestion: placer max | 4 | PASS |
| congestion: router effective | N5 S5 E3 W5 | PASS |
| routing | 0 errors (238,908 nets routed); one Route 35-447 warning in Phase 5, the same phase as b98f5a517's base reading | PASS |
| loops | 0; LUTLP-1 0 | PASS |
| WNS | -15.382 (TNS -390,257.75; 56,255 of 211,715 failing), inside the null band of -15.077 | PASS |
| sha256 | aeb8639cdd04389d42358c30aa4d4db2468459bf4658942ce7fd3f6c581700d0 (tarball synth-eb101c3519bc-exit0.tar.gz, sha256 7c520e49…; its .bit matches) | |

The combined build (bypass + DTLB 64 + L1D 64 KiB, ea174bfdd) is NEGATIVE, as recorded above: placement failed.

**Bypass v3: every check that could void its builds has passed.**
- The raw-lane A/B on 3ef8dd947: 39 of 40 arms identical, as recorded in the final's section.
- The 95-test sweep on 3ef8dd947 (sim13 .sup/swrl), against the base 5f41c486e (.sup/swbest):
  - `sweep-compare.py`: 92 identical (exceptions, readings, retired), 0 differ; 3 expected-missing on both sides.
  - `sweep-compare-rvfi.py`: 92 identical retirement traces, 0 differ, 26 compared over the prefix only (parked).
- The final 6d0e754d5's own sweep (sim14 .sup/swfinal), against the same base: the same result, 92 / 92 by value and
  by trace. So the condition "a sweep failure voids the build" is cleared for the apollo build.
- Lint on v3 alone, 38ae2d775, re-run after the void reading (vp8 .sup/lint-v3alone-lint2): PASS, every counter at
  baseline (UNUSEDSIGNAL 1480).

**Pre-registration: v3 alone (38ae2d775), leffe, before the build.** The tree is 3ef8dd947 plus the lint fix only:
the round-trip temporaries are hoisted under `ifndef` and the type bits are read directly, with no logic change.
- Name: `caplifive_ids55-native-tagcache-mlp-fpieee-hpmreplay-dynbypass3_38ae2d775`.
- It is the single-change bitstream for the SPEC lane's same-board A/B of the bypass against the base.
- Predicted placement: 5f41c486e's 181,565 plus v2's +7,499, minus the round trips. This is a report, not a key.
  The final's post-synth rows against bec4beb51 (board lane) read:
  - dyn_unit -6,114 and rev_node +7,551, with the design TOP at -6,887;
  - capstone_rev_node.anvil has no changed lines between the two trees;
  - the FLU's own row is 4-6 LUT in every build, so the FLU fix's comparators are homed in other rows too.
  The module table cannot split rev_node's rise between moved bypass cones and the fix, so only the design total is a
  reading. Roughly 183-188 k placed.
- Keys, as for v2:
  1. mapping in class;
  2. 0 loops;
  3. 0 routing errors;
  4. congestion <= 5, with a routed 6 reported as not met by letter;
  5. WNS inside the null band of -15.077;
  6. sha256.
- A routing failure is a NEGATIVE result: no WNS is recorded, and the build is not rerun.

**The final-IC tree, prepared: v3 + the SHRINKTO fix + 64 B I$ lines, without DTLB 64.**
- capstone-ariane perf/ids55-final-ic64b **105c464bc** (pushed). It is 38ae2d775 with the probe tests, the fix
  commits and 5384cf84c's config line cherry-picked on top.
- Against 6d0e754d5 it differs in exactly two config lines: DataTlbEntries 64 -> 16, IcacheLineWidth 128 -> 512.
- Lint (vp8 .sup/lint-fic64-lint): PASS at baseline, UNUSEDSIGNAL 1480.
- Arms (vp8 .sup/fic64arms, rtl 105c464bc dirty=no): every value is identical to its reference. The references are
  6d0e754d5's finalarms, v3's rlon and the 64 B I$ variant's ic64b. Only cycle fields move:
  - c5a-cost-ldc-hit8-dist 0xaff -> 0xa8a;
  - c5a-cost-stc-hit8-dist 0xaee -> 0x6cd, -38 %, with identical instruction counts. v3 and the final read the same
    as each other, so this move belongs to the 64 B lines. It is not attributed further.
  - c5a-cost-ldc-miss8-hpm: +1,186 cycles and replays 1,021 -> 1,024 against ic64b, which has no bypass. That is the
    bypass's miss path, not the I$.
- Its build waits for the 64 B I$ gate on leffe. Its own pre-registration is written before that launch.

## Pre-registration: the two final-IC candidates (before either build, 2026-11-13)

**Why two.** The SPEC lane and this lane agreed to replace DTLB 64 with an I$ change. The silicon gcc reading caps
DTLB 64 at <= 0.8 %, while L1I misses are 7.84 per 1 k cycles. The I$ model cannot pick the geometry. It reads 0.76x
of silicon on gcc and 0.22x on perl. On gcc, 8 ways beat 64 B lines; on perl, 64 B lines read WORSE than today and
8 ways remove nearly every miss. So both candidates are built speculatively under the lead's parallel order, and the
board's hpm3 and run time on each decide. Neither waits for its I$ synthesis-only gate (leffe, in synthesis). A gate
failure or a routing failure is a recorded NEGATIVE.

| candidate | commit (pushed) | against the final 6d0e754d5 | name |
|---|---|---|---|
| final-IC 64 B | perf/ids55-final-ic64b **105c464bc** | DataTlbEntries 64 -> 16; IcacheLineWidth 128 -> 512 | `caplifive_ids55-native-tagcache-mlp-fpieee-hpmreplay-dynbypass3-shrinkfix-ic64b_105c464bc` |
| final-IC 8-way | perf/ids55-final-ic32k **9972ae79d** | DataTlbEntries 64 -> 16; IcacheSetAssoc 4 -> 8, IcacheByteSize 16384 -> 32768 | `caplifive_ids55-native-tagcache-mlp-fpieee-hpmreplay-dynbypass3-shrinkfix-ic32k_9972ae79d` |

Both trees are 38ae2d775 (v3) + the probe tests + ee82e1c6a/6d0e754d5 (the fix, cherry-picked) + one I$ config
commit.

**Gates:**
- 105c464bc: lint PASS at baseline, and every arm value-identical (the section above).
- 9972ae79d: its lint and arms are running in vp8 (the same list). A failure voids its build.
- v3's sweep and the final's sweep are clean (92 / 92 by value and trace). The fix and the I$ geometry are config or
  FLU-only changes on top of those.

**Predicted, as reports:**
- Placed area: the v3 build's placed area plus the I$ delta measured by its gate. DTLB 64's +3.1 k is removed.
- I$ deltas, as guesses:
  - 8-way: +8-10 BRAM36, +0.3-1 k LUT;
  - 64 B: +8 to +24 BRAM36, +1-3 k LUT.

**Keys,** the same as for the final (base 5f41c486e: placed 181,565, WNS -15.077):
1. mapping in class;
2. 0 loops (check_timing, found-loop, 8-326, LUTLP-1);
3. 0 routing errors;
4. congestion placer and router <= 5, with a routed 6 reported as not met by letter;
5. WNS inside the null band of -15.077, as a value;
6. sha256.

A routing failure is NEGATIVE: placed area, congestion and the census are read, no WNS is recorded, and the build is
not rerun.

**Silicon (the SPEC lane's):** board 2, problem rows first, each candidate with its own same-board base run
(5f41c486e, bkit61h). hpm3 (L1I misses) and run time are the readings.

## The realistic final (6d0e754d5): SEALED on apollo, every key met

`caplifive_ids55-native-tagcache-mlp-fpieee-hpmreplay-dynbypass3-dtlb64-shrinkfix_6d0e754d5.bit` was built by the board
lane on apollo and sealed at 2026-11-13T12:28:58Z.
- run.exit 0; run peak 30.91 GB; collector peak 12.24 GB.
- PROVENANCE HEAD 6d0e754d5e00..., and the tarball's .bit is identical to the built .bit.
- Stored 0444 at <mnt-hdd>/capstone/bitstreams/ and /tmp/capstone/_bitstreams/; the sha reads back in both.

Graded against this note's pre-registration (b5e9bfef7045):

| key | reading | grade |
|---|---|---|
| 1 mapping | 5,170 MB at "Finished Technology Mapping" (30:34 elapsed, against 17:43 for bec4beb51) | PASS |
| 2 loops | check_timing 0 combinational loops; latch loops 0; found-loop 0; 8-326 0; LUTLP-1 0 on the synth and opt checkpoints | PASS |
| 3 routing | 0 errors, 240,195 / 240,195 nets routed; Route 35-447 absent (bec4beb51 and the tagcache build had it) | PASS |
| 4 congestion | placer max 4; router effective N5 S5 E4 W4, max 5 | PASS (bec4beb51 read 6) |
| 5 WNS | -12.363 (TNS -150,200.9, 35,978 failing; WHS +0.051), inside the null band of -15.077 | PASS; +2.7 ns is not an improvement claim |
| 6 sha256 | 72bf215d1b4bdc743b1996a5e5cb0c1c17c12003fa6c3bd9ed679347178c76b9 (11,443,722 B) | recorded |

- **Placed:** Slice LUT 187,333 (91.92 %), +5,768 on the base, inside the predicted 185-189 k. Slices 50,654 of
  50,950 (99.42 %); registers 75,467; BRAM 63.
- **Post-synth rows against bec4beb51** (board lane, the same flow):
  - TOP 189,704 vs 196,591 (-6,887);
  - dyn_unit -6,114; lsu -2,475; issue_read_operands -3,052; csr_buffer -1,844; i_dtlb -539;
  - rev_node +7,551, although its source is unchanged.

  The FLU's own row is 4-6 LUT in every build, so the module table cannot apportion rev_node's rise. Only the design
  total is a reading.
- **The sweep condition is cleared:** 92 / 92 against the base, by value and by trace.
- It is a flash candidate. Silicon is the SPEC lane's: problem rows first, with a same-board base run.

## The three I$ gates (leffe), and pre-registration of the fourth candidate: the final + the 8-way I$ (3c6b9906c)

**I$ synthesis-only gates** (synth lane, each against the ids55-best gate: core 157,186 LUT, TOP 184,082; I$ 795 LUT /
12 RAMB36). All three PASS every key: mapping in class; found-loop 0, 8-326 0, LUTLP-1 0 and check_timing 0, with the
S-10b control reading 1 by the same script.

| variant | core LUT | I$ (i_cva6_icache) | adapter FF | worst path |
|---|---|---|---|---|
| 32 KiB 8-way (8fe6375f9) | +253 | LUT +264, RAMB36 12 -> 24 | +1 | +1.880 ns, base family; 0 I$ endpoints in the top 200 |
| 32 B lines (582f23f83) | +257 | LUT +185, RAMB36 12 -> 20 | +131 | +1.884, base family |
| 64 B lines (5384cf84c) | -101 | LUT +449, RAMB36 12 -> 36 | +389 | +2.200, base family |

- These are synthesis estimates, before placement.
- Against the guesses in the pre-registration:
  - the 8-way's BRAM read +12, against the guessed +8-10;
  - every LUT delta came in at or under the guess.
- So neither final-IC build is void.

**Why a fourth candidate.** The SPEC lane read mcf on the base: DTLB walks 471.0 M (2.55 per 1 k cycles, 184.43 G run
cycles). At 15-30 cycles per walk that is 4-8 % of mcf's time, if 64 entries remove most of the walks. So DTLB 64 is
kept, and the 8-way I$ is added on top of it.

**Launch rule, written before the gate read:** launch if lint is clean and the 8-way gate's I$ delta is <= ~1.5 k
LUT. It read +264 LUT, so it launches.

**The tree.** capstone-ariane perf/ids55-final-dtlb64-ic32k **3c6b9906c** (pushed) = the sealed final 6d0e754d5 +
8fe6375f9's two config lines (IcacheSetAssoc 4 -> 8, IcacheByteSize 16384 -> 32768), nothing else. Name:
`caplifive_ids55-native-tagcache-mlp-fpieee-hpmreplay-dynbypass3-dtlb64-shrinkfix-ic32k_3c6b9906c`.

**Gates:**
- Lint (vp8 .sup/lint-fd8-lint): PASS, every counter at baseline (UNUSEDSIGNAL 1480).
- Arms (vp8 .sup/fd8arms, rtl 3c6b9906c dirty=no): 11 of 11 identical to the references (the final's arms and the
  8-way final-IC's), cycle fields included.

**Predicted, as a report:** placed = the final's 187,333 + about 0.3 k LUT, BRAM 63 -> 75.

**The risk is SLICES, not LUT.** The final placed at 99.42 % of slices, and the full E2 combination failed placement
on slices.

**Keys:** the final's six. A failure to place or to route is a NEGATIVE result: the placed or utilisation reading and
the census are read, no WNS is recorded, and there is no rerun.

## final-IC 8-way (9972ae79d): SEALED on apollo, every key met

`caplifive_ids55-native-tagcache-mlp-fpieee-hpmreplay-dynbypass3-shrinkfix-ic32k_9972ae79d.bit` was sealed by the board
lane at 2026-11-13T14:05:35Z.
- sha256 d00bb291dc5e507d79dfd9eb20e29410b84744517aee62c319a03e136bce51a2 (11,443,722 B).
- Stored 0444 at <mnt-hdd>/capstone/bitstreams/; the tarball's .bit is identical, and PROVENANCE HEAD is 9972ae79d4f2...
- run.exit 0; run peak 22.36 GB.

| key | reading | grade |
|---|---|---|
| 1 mapping | 4,993 MB | PASS |
| 2 loops | check_timing 0; latch loops 0; found-loop 0; 8-326 0; LUTLP-1 0 on the synth and opt checkpoints | PASS |
| 3 routing | 0 errors, 235,265 / 235,265 nets; Route 35-447 PRESENT (absent on the final) | PASS |
| 4 congestion | placer max 5 (E, 32x32); router effective N4 S4 E5 W4 | PASS, at the limit |
| 5 WNS | -13.038 (TNS -226,474.2, 43,602 failing; WHS +0.061), inside the null band | PASS |
| 6 sha256 | above | recorded |

**Area against the prediction (report):**
- Placed Slice LUT 185,791 (91.16 %): -1,542 against the final. With DTLB 64's ~3.1 k removed, the placed cost of the
  8-way I$ is about +1.5 k LUT, which is MORE than the gate's synthesis estimate (+264) and the guess (+0.3-1 k).
- BRAM 75 against the final's 63 (+12).
- Slices 50,275 (98.68 %).
- Registers 70,628, which is -4,839, consistent with DTLB 64's FFs removed.

**Bearing on the fourth candidate (3c6b9906c, the final + the 8-way I$).** If the 8-way I$ costs about +1.5 k placed
LUT on top of the final as well, that build lands near 188.8 k LUT. The final already used 50,654 of 50,950 slices
(99.42 %), so the fourth candidate's placement is at real risk. This is a prediction, recorded before its seal.

## final-IC 64 B (105c464bc): SEALED on leffe, every key met

`caplifive_ids55-native-tagcache-mlp-fpieee-hpmreplay-dynbypass3-shrinkfix-ic64b_105c464bc` was sealed by the synth lane,
with a bitstream at 16:15Z.
- run.exit 0. sha256 89b7d825d54a18d0d079bb3c265094d84cd2575b4e986e111d59f078d4217da3 (11,443,722 B).
- Tarball synth-105c464bc687-exit0.tar.gz (sha256 1f15fa87...); its .bit matches, and PROVENANCE names 105c464bc687.

| key | reading | grade |
|---|---|---|
| 1 mapping | 5,261.0 MB | PASS |
| 2 loops | found-loop 0, 8-326 0; routed check_timing 0; LUTLP-1 0 | PASS |
| 3 routing | 0 errors, 236,711 nets; Route 35-447 0 | PASS |
| 4 congestion | placer max 5 (E 32x32); router effective 5 in every direction | PASS, at the limit |
| 5 WNS | -12.172 (TNS -251,246.5, 47,413 of 214,976 failing), inside the null band | PASS |
| 6 sha256 | above | recorded |

- **Placed Slice LUT 186,015.**
- **The 3h30m wall-clock alarm fired in route_design (about 15:56Z) and was overridden.** The ruling, agreed with the
  synth lane before the alarm, replaced the wall-clock rule with a 20-minute router-stall rule. Convergence was
  monotone: overlaps 142,426 -> ... -> 3 -> 0. The extra time came from synthesis under host load 90-100.

## The next batch: R-72's fix + lever (a) safe form, on the fourth-candidate tree (pre-registered before the build, 2026-11-14)

**The tree.** capstone-ariane perf/b-next-dtlb64-ic32k **d10608fc9**:
- 3c6b9906c, the sealed final 6d0e754d5 + the 8-way I$;
- + R-72's fix (commit_stage.sv: an instruction must lie wholly inside the PC capability);
- + lever (a) safe form, 647ef2f16 + 6784c69e8 (issue_read_operands.sv: an STC claims rs1 only when its rs1 lane was a
  tagged UNINIT at issue; the STC's own rs1 WAW stall is kept).

Against perf/b-next 6784c69e8 (6d0e754d5's core plus the same two changes) it differs only in the two I$ config lines.

Name: `caplifive_ids55-native-tagcache-mlp-fpieee-hpmreplay-dynbypass3-dtlb64-shrinkfix-ic32k-r72-stcnoclaim_d10608fc9`.

**Lever (a), how it got here.**
- The first safe form, 647ef2f16, also dropped the STC's rs1 WAW stall. That let an UNINIT STC issue beside an older
  in-flight writer of rs1, so excodes S5b's reader was forwarded the older writer's value: 0x80003000 for
  0x80003010.
- The STC_CLAIM_TRACE run (perf/b-next-dbg e286fc0bc) showed the claim flop itself correct. The fix, 6784c69e8, keeps
  the WAW stall.
- The same-tree A/B on 6784c69e8 (vp8 .sup/lvaoff2, lvaon2; the off-define confirmed in the build line) has 36 of 37
  arms identical by value. The one difference is the intended gain, c5a-cost-stc-base1 0x151a -> 0xaee cycles
  (5.28 -> 2.73 per STC), with the same instruction count.

**R-72's fix:** pcc-end-rvc flips as predicted (the straddler now faults at itself; both controls unchanged), 8
regression arms are identical, and lint is at baseline (vp9 .sup/r72arms, lint r72-lint).

**Running beside the build; a failure voids it:**
- lint on d10608fc9, 75e4fb3c2 and 6784c69e8 (the build waits for d10608fc9's);
- the 95-test sweep on 6784c69e8 against the base;
- a claim-auditor over both diffs.

**Keys** (base: the fourth candidate 3c6b9906c, routed WNS -11.930; the final 6d0e754d5 placed 187,333 at 99.42 %
slices):
1. Mapping in class. Note that 3c6b9906c's synthesis peaked at 33.45 GB, so the memory watchdog is set ABOVE that
   (36 GB) and under the 40 GB guard ceiling. A peak in the 33 GB class is that tree's known envelope, not a blow-up.
2. 0 loops.
3. 0 routing errors.
4. Congestion <= 5, with a routed 6 reported by letter.
5. WNS inside the null band of -15.077, reported as a value, and compared with 3c6b9906c's -11.930.
6. sha256.

**Placed area** is predicted at 3c6b9906c's + a few hundred LUT (the R-72 compare and 16 flops with their clobber
term). It is a report. The risk is slices, as on 3c6b9906c.

**Failure branch:** a placement or routing failure is NEGATIVE. Read the placed utilisation and the census, record no
WNS and do not rerun. The fallback tree is 75e4fb3c2 (the 8-way tree without DTLB 64).

## Seals: v3 alone (38ae2d775) and the fourth candidate, final + 8-way I$ (3c6b9906c), both on leffe, every key met

| key | v3 alone 38ae2d775 | final + 8-way I$ 3c6b9906c |
|---|---|---|
| run | 4h19m; 3h30m alarm overridden in opt_design (router-stall rule) | 2h56m |
| 1 mapping | 5,168.4 MB | 5,426.0 MB; synth peak 33.45 GB (watchdog raised 34 -> 40, the guard ceiling) |
| 2 loops | 0 (synth, routed check_timing, LUTLP-1) | 0 |
| 3 routing | 0 errors, 232,542 nets; Route 35-447 0 | 0 errors, 241,123 nets; Route 35-447 0 |
| 4 congestion | placer 5, router effective max 5 | placer 4, router effective max 5 |
| 5 WNS | -16.475 (1.40 ns below -15.077, inside the band) | -11.930 |
| placed Slice LUT | 183,890 (predicted 183-188 k) | 187,646 (slices 99.48 %, BRAM 75) |
| sha256 | a6e2e92c8293a7730a83942504dad4d9e4a379f0dc5f96fdafa15f920e4d36b6 | 141f840eb8a59149762605e47aa6fb22a7b9c2525bc51f7df7c8ace85114e7d3 |

- v3 alone is the single-change bypass bitstream for the same-board A/B.
- 3c6b9906c placed, against the slice-risk prediction recorded before its seal. It is the base of the next batch,
  d10608fc9.

**Correction to the batch pre-registration above, before the build:** the memory watchdog is 48 GB, not 36 GB.
3c6b9906c, the same tree without R-72's fix and lever (a), peaked at 33.45 GB, so a 36 GB ceiling could have killed a
healthy run. 48 GB is about 1.4x that envelope, inside the rule (about twice the reference envelope, below what the
host can spare: apollo 125 GB total, 108 GB available), and far below the 100 GB pathological class. Raised at the
SPEC lane's request.

## Lever (a) REFUTED by the claim-auditor, batch HELD; R-72's synthesis-only gate alone (pre-registered before the gate, 2026-11-14)

**Lever (a) at 6784c69e8 is refuted (claim-auditor; the mechanism re-read here).**
- A non-UNINIT STC no longer claims rs1, but it is still an rs1 FORWARDING SOURCE: `check_fwd_rs1` lists STC
  (`ariane_pkg.sv:1098`), and scoreboard-entry forwarding matches on `sbe.rs1` independent of the clobber vector
  (`issue_read_operands.sv:792`; the WB-port path at `:713` is the same).
- So a younger writer of rs1 can issue while the STC is in flight, and a reader after it can be forwarded the STC's
  OLD rs1. That is the mirror image of S5b.
- The off-build is immune: the claim WAW-stalls the writer.
- The 37-arm A/B could not see it: no arm holds an STC uncommitted while a multi-cycle writer of its base issues.
- **The batch build of d10608fc9 was NOT launched.**
- **The directed arm,** stc-fwd-hazard.S (perf/b-next 7c49921eb), is running with the lever on and off: a missing
  load at the head, a non-UNINIT STC through s3, a div (or a mul) writing s3, then a read of s3. The predictions are in
  its header. The lever-on run must read the stale cursor first, or the shape does not reach the hazard.
- **The fix to follow:** an STC is an rs1 forwarding source only when its stc_uninit_q bit is set, in the entry and
  WB-port paths, for rs1/rs2/rs3. It then gets this arm, the A/B, the auditor and its own synthesis-only gate.

**R-72's synthesis-only gate, alone.** capstone-ariane perf/r72-on-fdk **84a23b65b** = 3c6b9906c (fdk, sealed) +
only R-72's pc_cap_check hunk (+5/-1). The claim-auditor found the logic SUPPORTED, but asked for a gate: the new
64-bit subtract and compare sit after the decompress of pc_metadata in `pc_cap_ex_valid`'s cone, which reaches
exception_o, and through it the CSR file's update_access_exception, on the standing UNOPTFLAT list.
- Its added arm, pcc-end-rvc PE_C (a 2-byte c.addi ending exactly at END must commit), is running.

Keys, against fdk's sealed 3c6b9906c:
1. Mapping in class: about 5.4 GB at the marker. fdk's whole-synthesis peak was 33.45 GB, so this run needs the
   40 GB guard and a watchdog above 34 GB.
2. Loops: found-loop 0, 8-326 0, LUTLP-1 0, check_timing 0.
3. Area, report only: the floor first (module, core and TOP 0 +/- 2,500). Read commit_stage's own row and the
   csr_regfile row, with the LUT delta against fdk.
4. The worst-path family and its slack: does any path through pc_cap_check's new compare enter the top 200?

The batch build waits for this gate AND for the fixed lever (a)'s own gate.

## Lever (a) fixed and passing; the batch moves to a47b588a1 (pre-registered before its build, 2026-11-14)

**The forwarding fix** (perf/b-next 11bb7427b): an STC is an rs1 forwarding source only when it claims rs1.
`fwd_rs1_src[e]` gates the six forwarding-request lines on the same per-entry flop. With the lever off, it equals
`check_fwd_rs1` exactly.

**The same-tree A/B on 11bb7427b** (sim13 .sup/fwdon, fwdoff; the off-define confirmed in the build line):

| arm | lever on | lever off |
|---|---|---|
| stc-fwd-hazard (div) | 0x8e (0x80003000 before the fix, on 6784c69e8) | 0x8e |
| c5a-cost-stc-base1 | 0xaee cycles (2.73 per STC) | 0x151a (5.28 per STC); instructions 0x503 both |

- Every other arm is identical: the mul hazard, the four pcc-end-rvc arms including PE_C, the RVC arms, excodes,
  the va-runv family and the cost arms.
- The hazard arm's positive control fired first: on 6784c69e8 the lever on read 0x80003000, and off read 0x8e.

**The batch tree** is capstone-ariane perf/b-next-dtlb64-ic32k **a47b588a1**: fdk 3c6b9906c + R-72's fix + lever (a)
(647ef2f16, 6784c69e8, 11bb7427b). It replaces d10608fc9, which carried the refuted (a). Against 11bb7427b it differs
only in the two I$ config lines. Name:
`caplifive_ids55-native-tagcache-mlp-fpieee-hpmreplay-dynbypass3-dtlb64-shrinkfix-ic32k-r72-stcnoclaim_a47b588a1`.
The fallback is perf/b-next-ic32k 31d416595.

**The build.** The SPEC lane relayed the lead's order to run synthesis in parallel with the checks: a full build on
apollo straight after a clean lint, with no separate gate.
- synth_design is its gate: `report_utilization -hierarchical` right after synthesis, for commit_stage, csr_regfile,
  issue_read_operands and the frontend.
- The 48 GB watchdog kills the run if mapping balloons.
- In parallel, any failure of which VOIDS the build before it reaches a board:
  - leffe's R-72-only synthesis gate, 84a23b65b against fdk;
  - the 98-test b-next sweep on 11bb7427b;
  - the claim-auditor's re-audit of the fixed (a).
- Nothing is flashed until every one of them is clean.

The keys are the batch's six above, against 3c6b9906c (routed WNS -11.930), with the 48 GB watchdog.
