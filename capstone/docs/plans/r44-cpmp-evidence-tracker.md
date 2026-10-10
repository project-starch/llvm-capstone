# R-44 — the CPMP revnode tracker on positive evidence: design proposal for the cycle after 0568f93a9

*RTL lane, 2026-10-10. A proposal for the lead's decision, written while the R-51/R-52 bitstream is in synthesis.
Nothing here is implemented. Sources quoted at `0568f93a9` unless stated.*

## Why now

R-44 is R-35's mechanism per CPMP entry: `core/pmp/src/pmp_data_if.sv:82-93` adopts any revnode id a CPMP entry
presents as live ("assume the revnode is live until proven otherwise") and clears it only on a matching invalidation
broadcast. The LSU check that R-35 fixed is gated `ld_st_priv_lvl_i == PRIV_LVL_M`; the CPMP check is gated `!= PRIV_LVL_M`
(`pmp_data_if.sv`, `cpmp_check` and `cpmp_if_check`), so the CPMP is **all of S/U-mode capability enforcement**. Until this
month S/U mode ran nothing of ours. Since 2026-10-05 native Linux processes run on silicon (the process ABI, B0..B3), and
R-52 was found in one, so the stale-id re-adoption that `r12-recl-cpmp.S` probe P4 measures ("CPMP1 <- e, ld OK: a different
entry re-adopts the stale id") is now a live authority escape, not a latent one. Why it was deferred: copying R-35's
fail-closed tracker predicts a permanent deny on the genesis entries `cpmp(0..2)`, i.e. an unbootable board.

## What the RTL already has that the fix can reuse

- **Evidence taps.** The LSU receives every rev-node answer as `revnode_fill_valid_i / revnode_fill_id_i (30-bit) /
  revnode_fill_nodeval_i` and the read-fill variant `revnode_rdfill_*` (`load_store_unit.sv:202-204`, the R-35/R-43
  cache fills from them, :1286-1296). A fill with `nodeval = 1` for an id is exactly the positive evidence R-35 demanded.
- **A probe path.** The LSU's R-43 probe block (`load_store_unit.sv:1388-1420`) sends `rvc_probe_valid_o / rvc_probe_id_o`
  (a bare index) to the rev-node unit and resolves on `revnode_rdfill_*`; it is single-requester, registered, and its
  outputs never feed the load/store request combinationally. **It is also the cone the standing lesson is about**: OR-ing
  a second requester onto the rev-node channel broke the first on 2026-08-21 with no lint signature. Any second
  requester must be muxed in only while `rvc_pend_q = 0` and the pipeline is empty, and goes to synthesis before any board time.
- **Genesis ids are architectural.** `capstone_rev_node.anvil:336` `INIT_STAGE` sets `head := 16'd3`: indices 0, 1, 2 are
  never allocated by hardware. The monitor's three genesis capabilities carry them (`cpmp_q` itself resets to 0,
  `csr_regfile.sv:3154`; the ids arrive by CCSRRW). So "index < 3" is a seed the RTL can justify from its own allocator,
  not from the monitor.
- **Two install paths.** CCSRRW to CPMPn (`csr_regfile.sv:2525-2588`, serialised by the R-26 flush at commit) and the
  switcher's full exchange, which writes ids 9..24 (`csr_regfile.sv:1993`, `cpmp_d[reg_id - 9]`) in a burst of up to 16.

## The design

**A. Fail-closed on install, with the genesis seed.** In `cpmp_revnode_tracking`: when an entry presents a new id,
`cpmp_tracked_revnode_id_d[i] := id` and `cpmp_revnode_valid_d[i] := (id[15:0] < 3)`. Nothing else changes in the clear
path (the R-38 stage-0 post-adopt compare stays).

**B. Evidence sets the bit.** Add two inputs to `pmp_data_if`: `revnode_fill_valid_i / revnode_fill_id_i /
revnode_fill_nodeval_i` (the same wires the LSU has; routed from `cva6.sv`). Per entry:
`if (fill_valid && fill_nodeval && fill_id == cpmp_tracked_revnode_id_d[i]) cpmp_revnode_valid_d[i] := 1` (30-bit compare,
so a reissued generation is not vouched for by the old one), and `if (fill_valid && !fill_nodeval && fill_id[15:0] ==
tracked[15:0]) := 0` (the same clear the LSU applies, :1286). Clear beats set in the same cycle.

**C. A probe at install time, so evidence exists before the first S/U access.** On a committed CPMP write whose id is
not seeded, request one probe for that id. Two options, the first preferred:
1. **Through the LSU's existing probe block**, as a second request source accepted only when `!rvc_pend_q` and
   `flush_i`-quiescent (a CCSRRW commit is followed by the R-26 flush, so the LSU holds no capability access then; the
   switcher's writes happen while the core is gated on `dom_switch_active_q`). A small per-entry "probe owed" bit in
   `pmp_data_if` (16 flops) drives a request; the LSU's block services one owed probe at a time in index order; the
   resolution arrives on the fill taps and sets the bit by **B** with no extra wiring. The exchange burst therefore
   costs up to 16 sequential probes after a full switch: a few hundred cycles, once per switch, before the first S/U
   instruction — measured in step 3 below, and bounded.
2. A separate probe port on the rev-node unit for the CPMP. Rejected unless (1) fails synthesis: it is exactly the
   "second requester on the rev-node channel" shape.

**D. What a miss does meanwhile.** An S/U access through an entry whose bit is still 0 is **denied** (access fault), as
R-35 does in M mode; with **C** this window is the probe latency after an install and should never be seen by a
correctly written monitor. It is not replayed (unlike R-43) because the CPMP check sits in the MMU/PTW path and the
ifetch path, where a REPLAY cause does not exist; the deny is the fail-closed behaviour the R-35 line already chose.

**E. The PC sibling.** `commit_stage.sv`'s `pc_revnode_tracking` adopts the same way for the PC capability ("On domain
change ... assume valid"). It needs a stale CODE capability via CALL/RETURN, never constructed; same treatment (**A+B**)
is cheap and is in scope, with its own arm.

Not changed: `cpmp_check`'s priority scan, the ifetch last-match cache, the LSU's own cache, the rev-node unit.

## Verification, written to fail first

0. **The residual must deny.** `r12-recl-cpmp.S` P4 (CPMP1 <- stale e) reads FAULT; P1 (fresh install + probe) reads OK
   only after the probe resolved; P6 (reissued (1,3)) OK. Mutant: **A** without **B** denies P1 (the fail-closed shape
   that killed the boot in prediction) — the positive control that the evidence path is what allows.
1. **Genesis boots.** An arm installing ids 0, 1, 2 by CCSRRW then an S-mode load through each: OK with no probe sent
   (R43_TRACE shows none). Mutant: seed removed → deny.
2. **Install-time probe.** An arm installing a live non-genesis id: `R43 probe-sent id=` once, the S-mode load OK after
   it; the same with the id REVOKED first: probe resolves dead, the load FAULTs. Mutant: **C** removed → the load denies
   until some unrelated M-mode access to the same id fills the tap.
3. **The exchange burst.** A full switch installing 16 CPMP rows: 16 probes in order, `dom_switch_busy` plus the probe
   tail measured in cycles (the minimum quantum argument for supervised CALL grows by this number).
4. **Neutrality.** The 95-test sweep, `testlist_sup.yaml`, `testlist_r43.yaml` identical; lint at baseline (new
   UNUSEDSIGNAL re-baselined by name if the ports add one).
5. **Synthesis before anything else**, because of the probe-block mux: loop membership by stem unchanged, LUTLP-1 0,
   ~16 + 16x30 + 16 flops added (owed bits, tracked ids already exist, probe sequencer), WNS inside the null band; the
   refutation key is any failing endpoint starting in `pmp_data_if` or the LSU's `rvc_*` cells.
6. **Board acceptance** (the board lane): the first S-mode boot on the new bitstream IS the acceptance for the seed;
   then the R-43 list, then a native process (B3) — all three must be unchanged.

## Cost and risk, stated before the decision

- Area: small (tens of LUTs per entry for the compare, 16 owed bits, a 4-bit sequencer). Timing: the CPMP valid bit is
  already in the check cone; the new terms set it from registered fill taps, not from the access.
- The one real risk is the probe-block mux (C.1). It goes to synthesis before the board, and if synthesis shows a new
  loop or a failing path through `rvc_*`, option C.2 is the fallback and costs a second cycle.
- The monitor side needs nothing new if C holds; if the lead prefers to drop C, the monitor must touch each region
  through its capability in M mode before installing it (an `ldc` suffices), which is a contract worth avoiding.

## Decision asked of the lead

Whether R-44 is the content of the cycle after 0568f93a9. If yes, the RTL lane implements A–E on a branch off
`0568f93a9`, runs steps 0–4, and hands the hash to synthesis with the lint numbers and the predictions above.
