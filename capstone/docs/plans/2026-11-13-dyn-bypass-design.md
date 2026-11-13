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
3. **The results ride write-back port 4, which already carries metadata.**
   - The LSU's LDC result (the loaded capability, made by the same `decompress_cap_tagged` -> `cap_lane` pair as today,
     so it is bit-identical) and the STC result (rd = rs2 or cnull, rs1 lane) are muxed onto CAP_WB.
   - The DYN unit no longer receives LDC/STC. Its other ops (CDERIVE, CREVOKE, ...) keep port 4 through a priority
     arbiter: LSU results first, DYN waits on its `ep.res` handshake.
   - Two LSU capability results in one cycle (an LDC and an STC) need a one-entry skid for the STC result, because
     the LSU cannot be back-pressured (its acks are unconnected today).
   - Keeping port 4 means the forwarding network into `fu_data_q` is untouched. That network is the worst timing
     family (`cpmp_q -> fu_data_q`) and the `issue_pointer` loop cone. Adding metadata to ports 1/2 is the
     alternative; it is rejected for that reason.
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
- Cost: one flush per UNINIT STC. **Open question for the SPEC/compiler lane:** how often do compiled programs store
  through an UNINIT capability (a census of the perf images)? If it is hot, the flush costs more than the claim.

**What stays.**
- The load unit's LDC hold, 2 cycles on a hit, needed for the linear clear. A later lever could pop early when rs1
  lacks write permission (no clear is possible), but that is not in this change.
- The STC's store-buffer entry.
- The tag path.
- The R-43 probe and replay.
- The linear-LDC `commit_lsu` on port 0.

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
| STC hit, distinct rs2 | 6.29 | 2.5-3.0 | sd through a capability on a marked page is 2.54, plus the port-4 skid |
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
   - the port-4 arbiter;
   - the new checks feeding `cap_exception` (the dominant slack family);
   - equivalence with Anvil arm by arm.
5. **The 95-test sweep with RVFI compare**, against perf/ids55-best's.
6. **Synthesis-only gate**: per-module LUT delta (lsu, ex_stage, issue, dyn_unit; floor first, +/-2,500), loops 0,
   mapping in class. The full build only on a pass.

**Size.** Several days: the checks and their arms (1-2), the result mux and arbiter (1), (a) with the flush (1), and the
gates. Built in that order, with each piece's arms green before the next.
