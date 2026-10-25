# Bitstream 3: scope, read-only (the RTL lane, 2026-10-26, for the semantics owner to compose and the lead to decide)

*Written on the semantics owner's request after bitstream 2 read on silicon at N=2 (`2026-10-23-next-bitstream-scope.md`,
the last sections). No RTL was edited for this note. Every row cites the source it was read from; costs marked
"estimate" come from counting state in the RTL, not from synthesis, and a size-gate reading replaces them before any
build (lint and reasoning cannot size a change: `lint-is-blind-to-resource-cost`, and the 2026-10-24 tracer removal blew
up technology mapping although it only removed logic).*

The base for any bitstream 3 is bitstream 2, capstone-ariane `dcc6daf6e` (`caplifive_vp2c_dcc6daf6e.bit`, WNS −9.790,
0 combinational loops, 169,076 placed LUT). Its 8.7 ns of slack and ~9k LUT of headroom against the batch are the room
the rows below spend.

## Summary

| row | what | RTL? | software (module/monitor) | estimate | recommendation |
|---|---|---|---|---|---|
| (a1) S-21 | the S caller's snapshot restored from a domain-reachable frame | optional | **the ruled fix (a) closes it**: retire the frame grant after create | B: a 64-bit CSR + two request fields | software first; B only with N slots |
| (a2) S-22 | the paused context's privileged words and stale slot copies in the frame | needed for the stale copies once a domain can reach a frame | fix (a) closes it while no domain can reach a frame | consuming resume: 31 writes per resume in the switcher walk | with N slots, not before |
| (b) uaf cause 13 | silicon-only 13 where 25 is expected, ~2.5 % | an observation-only record, if the free steps do not decide it | two free steps first | ~100 flops + aperture taps | free steps now; the record only if they do not decide |
| (c1) Q-26 N slots | one saved virtual context in the CSR file | yes | the module serializes today | ~72 flops per slot + N 64-bit frame compares on the commit path | the lead's call; it makes (a1)/(a2) live |
| (c2) R-60 narrowing | — | done in bitstream 2 | — | — | closed |
| (c3) Q-31 | ordering divergence | no (ruled: the RTL order is the reference) | a QEMU pin | — | out |
| (c4) stale status lines | R-58, R-59, Q-24 read OPEN though their RTL fixes are on silicon since the batch | no | — | — | corrected in this commit; silicon acceptance arms owed (below) |

## (a) S-21 / S-22: the frame is domain-reachable

The defect chain and the ruled fixes are in `docs/ref/ISSUES.md` (S-21, S-22) and the options were set out on
2026-10-18 (`2026-10-16-virtual-profile-on-silicon.md`, "S-21/S-22: where the caller's snapshot should live"). What
changed since, and two corrections to that note:

- **Exposure is still nil on the current bitstreams.** A child frame runs only in a second context slot, the RTL has one
  (Q-26), and from the Q-26 queue onward the module refuses CV_THREAD_CREATE while a frame is live (S-21's exposure
  paragraph). So S-21/S-22 become live exactly when (c1) lands, and not before.
- **Correction 1, Option A's bound.** The 2026-10-18 note proposed narrowing the domain's frame grant to `[frame, frame +
  1024)`. That keeps the snapshot at `+1024..` out of reach but NOT S-22's first mechanism: the context's mstatus at
  `+920` and the hidden satp/srevroot/options words at `+928..+944` lie below 1024. A bound that excludes them is
  `[frame, frame + 656)`: everything the ABI exposes (the slots at `+64 + 16i`, the event words, the a0..a7 views at
  `+576..+632`, the collection pointers at `+640/+648`) is below `+656` (the reference model's own limit,
  `capstone_supervisor.c:844`, per that note), and 656 is a multiple of the 16-byte granule, so it is exactly
  representable. It still leaves the slots, and with them S-22's stale LINEAR copies, in reach.
- **Correction 2, the "software clear" of the slots is not available.** The note offered "the trusted side zeroes the
  slots after each escape it handles" as the software half of the consuming resume. It cannot work: after an escape the
  slots ARE the paused context's state and the resume needs them; the stale copies the issue is about exist AFTER the
  resume, while the context runs. Only the switcher can consume the slots as it loads them.

**What can be done in software (the module side, the semantics owner's and the compiler lane's):**
1. **Fix (a) as ruled** (S-21's "Fixes, as ruled"): retire the frame arena's ancestor once the pinned page is in hand,
   keep the page kernel-only, and refuse CV_THREAD_CREATE if the retire fails. This closes S-21 AND both mechanisms of
   S-22 while it holds, because no domain can then reach any frame word. The two-sided run.py check is in the entry.
2. If the ABI ever needs the domain to keep a live handle on its child's frame, the narrowed grant `[frame, frame + 656)`
   closes S-21 and S-22's first mechanism, and not the stale copies.

**What needs RTL (only worth it with N slots, or if fix (a) is judged too fragile to carry a security boundary):**
- **B′, a kernel-only save base** (the 2026-10-18 Option B, extended): one 64-bit CSR written by the trusted side before
  entry, used as `save_base`/`restore_base` (`commit_stage.sv:630`, `:869` at dcc6daf6e) instead of `frame + 1024`, with the paused
  context's privileged words (+920..+944) relocated into the same kernel-only area. Estimate: 64 flops, a mux on two
  request fields, and the switcher's `+920`/`+928..+944` offsets re-based; a CSR address in `csr_regfile` (the module that
  hosted bitstream 2's failed congestion window, so it gets a size gate of its own). The model needs the same CSR.
- **A consuming resume**: the resume walk zeroes the 31 slots and their tags after loading them, as ENTRY's init strobe
  does. Estimate: 31 extra memory writes on every resume, measured against va-runv2's resumes before it is chosen; an
  Anvil edit to `capstone_dom_switcher.anvil`, the unit whose timing loop is now gone, so its lint and loop readings are
  part of the gate.
- **C, the snapshot in flops**: rejected on 2026-10-18 (about 8,600 flops per slot) and still rejected; with N slots it
  multiplies.

## (b) The silicon-only cause 13 on the uaf control

**What is known** (the board lane's records, `vp2-kit-2026-10-17/tier3-smoke/README.md`):
- The Juliet uaf control (`controls.dom uaf-malloc`) reads cause 13, a load page fault, at code+0x10c70
  (control_read_probe+0x38) in about 2.2 % of runs on both trees (A 16/720, B 26/1120) and 3 % on the batch (6/200,
  lane c473e792cc5f), where 25 is expected; every 13 comes with two "no lazy arena covers this address" refusals at its
  own address, and no revoked read ever completes.
- The retire took effect in every cause-13 run (retire_verify: post_live 0 over 4,400 retires), and **with the retired
  region kept MAPPED there is no 13 in 200**. So the 13 needs the munmap: the page-table walk must be reached, and the
  capability verdict must not have been delivered first.
- Simulation finds no route (rvk-unmap-cause, `e340ec8be`): with the page unmapped, resident or not, after a context
  exit and re-entry, at memory delay 12 and 30, the registered capability exception is delivered in the cycle after the
  request and the load reads 25.

**The hypotheses, each with what would separate it:**
1. **H1, the check never ran.** `cap_violation_detection` is gated on `capmode_i && ld_st_priv_lvl_i == PRIV_LVL_M`
   (`load_store_unit.sv:1326-1329` at dcc6daf6e). A virtual context runs at M, but a context mstatus with MPRV set makes the effective
   privilege MPP, which turns the whole check off while translation stays on; the 2026-10-23 scope note flagged this
   path (frame +920 / the init strobe). A load to the unmapped page then faults 13 with no capability verdict at all.
2. **H2, the check ran and ALLOWED** (a live record for a dead node: a missed invalidation, R-35's class).
3. **H3, the check ran, missed, and the R-43 replay lost the verdict** (the re-executed access faulted 13).
4. **H4, the check ran and DENIED, and the deny was lost.** The capability exception reaches the MMU only as
   `misaligned_ex_n.valid = misaligned_ex_i.valid & lsu_req_i` (`cva6_mmu.sv:532`): it is sampled in a cycle in which
   the load unit is requesting translation. If on silicon the request and the registered verdict are not aligned (a long
   D-cache or PTW latency the simulation's fixed delay does not produce), the verdict is dropped and the walk's 13 is
   what is reported.

**Free steps first (no bitstream):**
- **Step 0a (module, the runtime lane):** at the cause-13 escape, print the context's saved mstatus (frame `+920`): MPRV
  and MPP. MPRV = 0 refutes H1 outright; MPRV = 1 with MPP ≠ M confirms it, and the fix is then in whatever sets it.
- **Step 0b (board):** the uaf control ×200 on bitstream 2. Bitstream 2 moved the load unit's clear timing and removed
  33k flops; a changed rate is a reading about timing (H3/H4), an unchanged one is consistent with H1/H2.
- **Step 0c (board, free):** read the existing sticky refusal record after a cause-13 run (apertures 5'b01100..5'b10000,
  `cva6.sv:1349-1353`: the last cause-25 arm and its node id). If its id is the freed region's node and its arm is
  hit-dead, the check DID deny that node at some point in the run; it cannot say which access, so it supports but does
  not decide H4.

**The instrument, if the free steps do not decide it:** a per-access verdict record, observation only, read through the
debug apertures after the trap.
- **Latched for every LSU access** while `lsu_ctrl.valid`: the gate bits (`capmode_i`, `ld_st_priv_lvl_i`), the verdict
  arm actually taken (one-hot: allow-hit, allow-record, deny-hit-dead, deny-same-cycle, deny-probe-dead, miss-replay,
  no-evidence, another cap cause), the node id (30 bits), and `lsu_ctrl.vaddr[31:0]`.
- **Frozen** at the first exception the LSU delivers with cause 13 (or 15, R-61's relabel, kept for completeness), with
  one more bit: whether `misaligned_ex_q.valid` was set in the MMU when the 13 left it.
- **Read** after the escape; the vaddr is matched against the trap's mtval to tie the record to the faulting access.
- **Predicted readings:** H1 gate bit 0; H2 allow arm with the freed node's id; H3 miss-replay arm; H4 a deny arm with
  the MMU's misaligned bit clear. Each is a different reading, so the instrument is informative under every hypothesis
  on the table; the rule "what will it read if the best hypothesis is TRUE" is met.
- **Estimate:** about 100 flops and a handful of aperture windows in `cva6.sv`'s debug mux. Aperture taps have cost FSM
  encoding and LED slack before (`fpga-aperture-taps-cost-fsm-encoding-and-led-slack`), so it gets the size gate; it
  touches no loop cone (there are none on bitstream 2).
- **Fits bitstream 3:** yes, as a strict observation-only addition; it should be cut only after step 0a, because H1 would
  make it unnecessary.

## (c) The rest of the list

- **R-60's six-E-bit narrowing:** done and on silicon in bitstream 2 (`2026-10-23-next-bitstream-scope.md`).
- **Q-31:** ruled 2026-10-18 by the semantics owner that the RTL's order is the reference; the fix is a QEMU pin, not RTL.
- **Q-26, N saved contexts:** the per-slot state in `csr_regfile.sv:391-395` is `runv_used`, `runv_paused`,
  `runv_terminal`, `runv_kind[2:0]`, `runv_frame[63:0]` and `runv_options[1:0]`, about 72 flops per slot; the paused
  context's own registers live in its frame, so a slot is cheap. The cost that matters is the lookup: commit refuses
  action 0 when the slot is used and the request is not a resume of the same frame (`commit_stage.sv:271-275` at dcc6daf6e), which
  for N slots becomes N 64-bit frame compares plus a free-slot pick on the commit path, and every `sup_runv_*_o` output
  becomes a slot-select mux. Estimate for N = 4: under 1,000 LUT and about 300 flops; the commit-path compare is the
  timing risk, which bitstream 2's slack can probably absorb but only the size gate and the routed WNS can say.
  **It makes S-21/S-22 live**, so it should not land before the module's fix (a) is on the board, or without B′ and the
  consuming resume if the owner does not want a security boundary resting on fix (a). The QEMU side already has
  `x-capstone-supervisor-slots`.
- **Stale status lines corrected in this commit** (the RTL lane's own rows): R-58 (the caller's srevroot, batch item 2,
  `524326f34`), R-59 (the precheck, batch item 3, `750ed72e4`) and Q-24 (the medeleg mask admits causes 24-30,
  `3244170b0`) are FIXED IN RTL and present on every bitstream since the batch `c721c1a2a`, bitstream 2 included. None
  has a silicon reading yet: the batch's acceptance was sim-only for items 2-4.
- **Owed silicon acceptance, no bitstream needed (bare kits on bitstream 2, the board lane's to run if wanted):**
  `va-srevroot` (R-58: the caller reads its OWN srevroot back after a context), `va-runv-r59` / `va-runv-r59-low`
  (R-59: a malformed table root is refused at entry), and a Q-24 arm (firmware sets a capability cause in medeleg; an
  S-mode capability fault lands in S). The kits are built and simulated from their images here on request.

## Not in this note

R-44 (the CPMP tracker's adopt, deferred on a boot-kill risk), R-46 (accepted by the lead), R-28 (named by audit, no
reachable route), Q-27 (action 3 is illegal by design) and S-30 (a monitor carve, not RTL) are left as they stand; each
is a row only if the owner wants it, and none is costed here.
