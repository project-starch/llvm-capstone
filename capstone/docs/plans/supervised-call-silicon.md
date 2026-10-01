# Supervised CALL on silicon — RTL design: the full-exchange escape

*RTL lane, 2026-10-01. Design approved by the project lead on 2026-10-01; implementation on capstone-ariane branch `sup-call`. Before-audit done 2026-10-01: it refuted the first design, revision 1.1 below was accepted by the lead the same day; nothing synthesised. The scoping estimate that preceded it is summarised in the Context section.*
design, after-audit of the diff; apollo rules (CPUs 0-7,32-39, `--cpuset-cpus`, `taskset`+`nice`+`ionice`, `-j12`);
work on a lane/task branch of capstone-ariane, squash-land; synthesis only on the lead's word; never name people.*

## Context

Every application port of the last two weeks (mruby, Perl, CPython, PostgreSQL, tshark, FFmpeg; the signals/sockets/
threads contracts) runs on the delegated runtime (`capstone-exec`, application ABI v2), which since 2026-09-30 is the
only application runtime. It depends on the "supervised CALL" platform extension that exists only in capstone-qemu
(`origin/c128-qemu-merge`): the monitor arms a 5 ms quantum (`cssupervise`), CALLs the application, and every fault,
interrupt or quantum expiry inside it returns to the monitor as a *typed event* through the same CALL, with the
monitor's registers protected from the domain; resume is the same CALL again. The runtime's requirements are hard and
a cooperative variant does not meet them (`docs/plans/domain-process-runtime.md:206-225`, `runtime/applications.md:14-17`):
trusted cause/pc/address for SIGSEGV, survival of a destroyed stack or cleared vector, cancellation of a loop that never
yields, no re-entry after a fault.

On the resident RTL (`8f6a0af98`) none of that exists: a trap inside a domain runs the handler under the DOMAIN's PC
capability and cscratch (the trap-vector defect, registry M-1), the two opcodes do not decode, a domain CALLed from
M-mode can `csrw mie/mstatus/mtvec`, and the hardware domain switch exchanges only 8 registers (ids 0..7). The scoping
estimate sent to the board lane on 2026-10-01 concluded: no firmware-only path; the blocker is one RTL primitive. This plan designs that primitive and everything around it.

**Decisions already taken by the lead:** node-GC data via a read-only CSR (the `cssupervisor_gc` opcode is NOT
decoded; "collect" is a no-op on silicon because the revoke walk already frees); a CALL by a supervised domain is
ILLEGAL in v1 (no context stack). Naming: "trap-vector defect" = registry M-1; "node reclaimer" = R-12 / study M1.

## REVISION 1.2 (2026-10-01, the implementation on capstone-ariane branch `sup-call`)

Implemented as designed in 1.1 with the refinements below, each forced by the RTL or by a test, and verified
through step 7 of the ladder in simulation (retirement-trace readings, `capprint-readings.py`; every number
below is a measured reading). Not yet: the mutant controls (running), the step-8 neutrality sweep (running),
the after-audit, synthesis, silicon.

- **SAVE runs BEFORE the 0..7 exchange, RESTORE after it, and a resume's RESTORE starts at id 8.** The walks
  and the exchange both cover ids 3..7 (mstatus, mideleg, medeleg, mip, mie at +48..+87, inside the SEALEDRET
  window). With SAVE after the exchange the armed CALL would save the domain's image as "the monitor's" and the
  escape would save the monitor's values as "the domain's". So: armed CALL = SAVE the monitor's 3..66 into the
  private area, THEN the ordinary exchange (arguments in a0..a7 as before), then (resume only) RESTORE the
  domain's 8..66 from its seal: its 3..7 come from the exchange, out of the seal's slots 3..7 that the escape's
  SAVE wrote. Escape and supervised RETURN = SAVE the domain's 3..66 into its seal region, exchange 0..7,
  RESTORE the monitor's 3..66 from the private area, overriding whatever the exchange brought in from the
  writable window. The request carries `save_en/save_base/restore_en/restore_base/restore_lo`. Found by
  self-review; the tests of 1.1 could not see it (both sides had MIE = 0), so the tests now read the monitor's
  `mie` back (0x888) and the MTIP test resumes the domain and demands a second timer escape.
- **The CSR map.** `csupquantum` 0x7C3 (M RW), `csupctl` 0x7C4 (bit 0 = resume, consumed by the armed CALL),
  read-only `csupstatus` 0xFC0 = {kind[1:0], valid} (read-to-clear), `csupcause` 0xFC1, `csupepc` 0xFC2,
  `csuptval` 0xFC3, `csnodefree` 0xFC4 = 65535 - head + free_len (the rev-node exports `free_len`). All have
  addr[9:8] = 2'b11, so the supervision gate covers them.
- **`cssupervise rd, rs1(seal), rs2(save area)`**: statuses 0 armed / 1 dead node / 2 refused (rd in {x0, rs1,
  rs2}; rs1 not a synchronous SEALED or not 16-B aligned; rs2 not LINEAR RW, smaller than 1024 B or not 16-B
  aligned) / 3 an unread event (assigned by commit, which also drops the arm). rs1 = x0 is the forget form.
  **The seal's SIZE cannot be checked** (corrected the same day): a SEALED capability carries no end bound in a
  register (`decompress_cap_metadata` gives 0 for SEALED/SEALEDRET, the field holds reg_id/async); a 512-byte
  seal arms with status 0. The 1 KiB minimum the walks rely on is SEAL's alone (S-11, fixed below). **Status 1 is
  reachable and measured** (2026-10-01 evening, after the `sup-arm.S` repair recorded below): a seal parked in
  memory with STC, its node REVOKEd through the MREV handle, reloaded with LDC, reads SEALED (LCC 4) and
  `cssupervise` on it returns 1; without the revocation the seal survives the STC/LDC round trip (LCC 4) and arms.
- **The escape at commit**: `escape_d` (combinational, decision cycle T) strips `exception_o` and loads flops
  (pc, pc metadata, cause, tval, kind); `escape_q` issues the RETURN-shaped request at T+1 from the flops alone
  and is held until the switcher acknowledges. T and T+1 force every commit side effect off. The strip also fires
  while the switcher is busy (M7 below). A RETURN through a seal that is not the supervised one, or to a caller
  other than the supervising rd, is `foreign_return`: an escape with kind 2, cause 26.
- **Guards**: decoder -- SRET, MRET, WFI, CALL, CAPENTER, CSSUPERVISE, CAPCREATE/CAPTYPE/CAPNODE/CAPPERM/CAPBOUND
  (CAPPRINT stays legal) are illegal under supervision; CSR file -- every plain CSR with addr[9:8] != 0 and
  0x800/0x801/0x802/0x804/0x810/0x811, and CCSRRW to CIH and CPMP0..15 (CCSRRW is its own op code: the first
  form of the gate, keyed on the four plain CSR ops, let `CCSRRW cpmp0` through -- caught by step 6); `wfi_d`
  forced low. The domain cannot read mstatus/mie either; fcsr and the user counters stay legal.
- **The quantum**: a 32-bit down-counter in the CSR file, loaded from csupquantum at the armed CALL, counting only
  while `sup_active && !dom_switch_active`, holding at zero; `irq_ctrl.sup_quantum` is injected by the decoder
  after the ordinary interrupt block, cause 0x8000_0000_0000_0010.
- **Two pre-existing defects found by the ladder, both fixed on the branch:**
  (a) **CALL parks the wrong return pc after a jump or taken branch** (registry entry pending the audit):
  `ex_stage.sv` packed `pc_i + 4` at the dyn unit's response, and `pc_i` is the issue stage's pc, which a branch
  unit-resolved jump after the CALL has already advanced. Measured on the unmodified RTL (`call-retpc.S`, the
  parked pc read from the seal's slot 0): `c.j`, a 4-byte `j` and a taken `beq` after CALL park CALL + 8 (the
  caller resumed past them, readings 2/4/6 for 0), `addi`, `c.addi` and `csrr` park CALL + 4. First seen as the
  supervised monitor resuming at CALL + 8, mid-instruction. The corpus pads every CALL with nops. Fix: latch the
  pc and its metadata when the dyn unit accepts the request (the `capstone_dyn_ftval_q` precedent); all six shapes
  then park CALL + 4. (b) **A busy-exception hazard** (step 0, M7 below).
- **M7, the busy-exception detector over the 92-test corpus on the unmodified RTL (sweep0, 97 runs):** 5 hits,
  the 4 sup-strip arms by design and `revocation` (2): an ILLEGAL_INSTRUCTION on a speculatively fetched `0x0000`
  word past the callee's RETURN, delivered to the CSR file in the middle of a switch and visible in the
  retirement trace as two taken traps; `call-retpc.S` on the unmodified RTL likewise ends with a stray mcause 28
  from a trap taken during a switch. The strip removes both; the step-8 run must show exactly those differences.
- **Lint**: 0 errors; LATCH 52, MULTIDRIVEN 3, UNOPTFLAT 40, BLKSEQ 2, UNDRIVEN 25 at baseline; UNUSEDSIGNAL
  rises only by renumbered Anvil wires and bit-range shifts of the widened scoreboard entry (compared by name).
- **A lane-only retraction, for the record.** For one afternoon the supervision state machine sat between the
  CSR file's `if (csr_we) ... end` and its `else if` for the switcher's register writes, so the else-if attached
  to an always-true `if` and NO switcher write of a CSR executed: lint clean, every GPR-based reading green, and
  "mie 0x888 restored" trivially true. The hostile domain's mtvec reading and the MTIP arm caught it; every
  CSR-side reading below is from the repaired tree (`all3`).
- **Positive controls (one mutant per build, `verif/tests/custom/capstone/sup-variant.py`):** `no-strip` (the
  escape no longer strips `exception_o`): the five step-3 faults are TAKEN as traps -- 5 taken-exception lines
  in the retirement trace against 1 (the reference) -- while the monitor's readings stay green because mcause is
  restored from the private area, which is why the test now also reads the shared `mepc`; `resume-plus4`:
  the stored count is 300, not 400 (each resume skipped two compressed addis); `count-in-switch` (the quantum
  counts during the walks): quantum 16 never completes, 283 escapes until the simulation's ceiling against
  101 and completion; `no-set-ra`: the resume CALL faults on a null seal, cause 24; `no-mret-guard`: the
  domain's mret executes, drops to U-mode with MPP = 0 and the run ends in a trap at mepc = 0 (cause 1) instead
  of a kind-2 escape -- UNRESOLVED why that fault trapped rather than escaped (the guard makes it unreachable;
  worth a look before silicon); `status-swap` (2 <-> 1 in the dyn unit): the five refusals read 1; `no-csr-gate`:
  the domain's `csrw mepc` lands (the monitor reads 0x1234 back) and `csrw mie` runs, both ending in a quantum
  escape instead of a fault; `no-mint-guard`: CAPCREATE runs and the quantum ends the domain; `no-foreign-check`:
  the RETURN through the other domain's SEALEDRET is honoured and the core ends up in that domain's old spin loop
  with supervision cleared, never returning to the monitor (1.98 M retirements to the ceiling). One mutant-only
  observation stays UNRESOLVED: with the plain-CSR gate removed, a domain that READ csupstatus was never
  preempted (the quantum did not fire, 1.9 M cycles); the real gate makes that read a fault, and the legal
  `csrr cycle` followed by a spin IS preempted (step-6 arm 11, csupstatus 3), so the quantum survives a legal CSR
  read. (`csrr fcsr` with FS = Off in the image is itself illegal and escapes as cause 2, arm 12.)
- **The after-audit of the diff (2026-10-01) REFUTED readiness and found four defects, all fixed the same day,
  each with a discriminating arm added before the ladder was re-run:** (3C) a RESUME re-wrote the domain's x1 with
  a fresh SEALEDRET after the RESTORE walk, because the switcher's set_ra write runs last and the armed-CALL request
  kept CALL's set_ra = 1 -- any program preempted with a live return address would have crashed on its next `ret`
  (fix: set_ra = 0 on a resume; the quantum test now carries a plain sentinel in x1 across every resume);
  (4A) a CCSRRW to CIH or CPMP that the gate refused raised the exception but still performed its write, because
  csr_op_logic clears only csr_we on a violation and commit asserts ccsr_we regardless (fix: the capability-CSR
  block and the R-26 flush are gated on !privilege_violation; the guards test writes a tagged capability into
  CPMP0 from the domain and the monitor reads it back empty); (4B) CEPC/mepc were neither blocked nor walked, so
  the domain could plant the monitor's mepc and read its tagged cepc through the legal CCSRRW CEPC (fix: id 25,
  the reserved slot, now carries {cepc_tag, cepc, mepc} in both walks; the guards test plants both and the
  monitor reads its own back); (4D) the supervision CSR gate had no debug-mode exception, so a JTAG halt during a
  supervised domain would loop in the debug ROM on its first `csrw dscratch0` (fix: nothing of the gate, the
  decoder guards or the counter applies in debug mode, H5). Also from the audit: the escape flops are now loaded
  every cycle while no escape is pending (the decision no longer enables ~260 flops), the foreign-RETURN compare
  is no longer in the trap strip's cone (it only enters the decision), and the pre-existing PC-capability
  override also drops a faulting cssupervise's arm. Confirmed refutations worth keeping: the head cannot change
  between T and T+1 (commit_drop is constant 0 on this config, one commit port), the walk order is right slot by
  slot, a marked instruction cannot escape again in the monitor, the counter cannot run during a walk, the
  free-list counter cannot underflow. Residuals recorded: a legal CSR or AMO head that escapes on a PC-capability
  fault performs its side effect at T (only matters if a kind-2 event were resumed, which the contract forbids);
  quantum 0 or any quantum shorter than the post-switch refetch latency livelocks by construction (the contract's
  >= 50k minimum is the only floor); while armed, a CALL of another seal fails CLOSED and the arm survives every
  exception on the CALL head (the delegated decisions below; this residual used to read "silently disarms, so
  the monitor must mask interrupts between cssupervise and CALL"); and the first entry installs neither
  the seal's CPMP slots nor the GPRs -- the domain runs with the monitor's CPMP0..15 and x2..x31, as today.
- **RETRACTED (2026-10-01 evening, by the after-audit of the delegated decisions): "a SEALED capability stored
  with STC and reloaded with LDC reads back untagged (LCC 7) in simulation", and with it "status 1 is
  unreachable".** Both came from `sup-arm.S`, where the rd == rs2 block had overwritten the seal register with
  an integer (`lla a3, savearea2`) before the STC, so the round trip, the dead-node arm and the rd == rs2
  refusal itself were readings of an integer; the auditor found it in the trace's register write at the STC
  (x13 = the address of savearea2, not the seal). Repaired (those bounds now go through t1/t2), the arms read:
  rd == rs2 refused with 2; the seal after an STC/LDC round trip SEALED (LCC 4); after a REVOKE of its MREV
  handle still SEALED on reload (4); `cssupervise` on it 1 -- the dead node IS detected; its type still SEALED
  (4); mcause 0 (run fix3, 2026-10-01 evening, 1384 retirements). Nothing about sealed capabilities in memory is
  open from this plan, and the "next question for whoever picks this up" is withdrawn. The line "a revocation
  broadcast nulls the seal in its register (LCC reads 7)" was measured on the same integer and is withdrawn
  too: unmeasured, and not needed by the design.
- **Landed on the lane branch:** `capstone-ariane` `sup-call` = d38887426 (prerequisites) + 1dbf379b1 (S-11 in
  simulation, the Anvil lint) + 727ea6e93 (R-47) + 03b70667e (the implementation, tests, runner and mutant
  patcher). **Step 8 on that exact tree:** the 92-test corpus against the unmodified RTL -- 91 tests identical
  in taken exceptions, CAPPRINT count and retired count; `revocation` loses its three phantom mid-switch traps
  with readings and retired count unchanged; the mid-switch detector fires 0 of 92 times (5 of 97 before).
  Not landed on the submodule's shared branch and the parent's pointer not bumped: synthesis is the lead's call.
  **Later the same evening:** sweep3 on the decided tree (fail-closed + the S-11 size check, before the audit's
  changes) -- 91 of 92 identical, `revocation` -3 traps, the detector 0 of 92 (3 on the unmodified RTL by the same
  comparator); sweep4 on the FINAL tree (the audit's changes and the sealed cursor): 91 of 92 identical to the unmodified RTL in taken exceptions, readings and retired count, the one difference again revocation's three phantom mid-switch traps gone (detector 3 -> 0); against sweep3, the decided tree, 92 of 92 identical -- the audit's changes and the sealed cursor are invisible to the corpus. Every other ladder
  test's readings and retired count are identical between the all5 and all6 runs (the three that differ are the
  three whose tests changed: `sup-armclose`, `sup-arm`, `sup-sealsize-1023`). Lane-branch commits: b9acda022 (fail-closed, audited: one predicate, the arm persists), bd4c0d486 (the sup-arm.S retraction), c0ad507b2 (S-11 size and alignment), c0c546542 (the sealed cursor, the lint re-baseline and the gate guard; the veto point), 36a641e0b (the bracket test); tip 36a641e0b.
- **Two decisions the lead delegated (2026-10-01 evening, "audit and decide yourself"), both in this bitstream,
  both AUDITED after implementation (claim-auditor, adversarial: both supported, with changes that are now made):**
  (1) **M1 fails CLOSED.** A CALL of any seal other than the armed one, while armed, no longer switches and no
  longer drops the arm silently (which left the intended CALL running unsupervised, with no signal): it is held
  at commit and traps as an illegal instruction one cycle later from a flop, mepc at the CALL, the arm persisting
  -- QEMU's `capstone_supervisor_call` (raise ILLEGAL_INST before clearing the arm). The audit required one change
  and recommended one, both made: ONE predicate (`sup_mismatch_hit`) now feeds both the block in the commit
  branch and the trap flop -- the first version excluded debug mode from the trap only, which would have held a
  program-buffer CALL at the head forever (in debug mode the trap now goes to the debug ROM's exception address,
  `csr_regfile`'s `trap_vector_base`, with no mepc/mcause update); and the disarm on a CALL head that takes an
  exception is REMOVED -- an interrupt or debug request marked on the armed CALL dropped the arm, so the mret/dret
  re-execution ran UNSUPERVISED, silently; QEMU keeps the arm on every exception, and so does the RTL now: only an
  acknowledged armed CALL or the forget form clears it. Test `sup-armclose.S` (measured, runs all6/fix3): the CALL
  of B traps with cause 2 at its own pc; the handler mrets WITHOUT forgetting and the re-executed CALL traps AGAIN
  (the arm survived the trap and the mret); after the forget the same CALL runs B ordinarily (marker 0xBB, RETURN
  to CALL + 4); re-armed, a planted CALL of a LINEAR capability faults (cause 26) and the handler steps over it
  without forgetting; CALL A then ESCAPES on its ecall (csupstatus 5, csupcause 11) -- the arm survived the planted
  fault; the marker intact, mcause 0 (199 retirements). Mutants: `no-armclose` (fail open again: measured, the
  cause-2 readings are absent and 0xBB follows the arm status directly -- B ran unsupervised from the first CALL
  and the arm was gone); `no-armclose-trap` (the switch blocked but no trap: measured: the CALL wedges at the head, 114 retirements, one reading, the simulation's time-out); `disarm-on-fault` (the
  earlier disarm restored: measured: after the planted fault the arm is gone and CALL A runs unsupervised; its ecall then traps INSIDE the domain, whose trap vector is its seal's slot 1, and the core storms to the simulation's time-out (399306 exceptions, no escape reading, the test never completes) -- the M-1 shape itself; the pre-registered 'the handler prints 11' named the wrong observable, since the monitor's handler is unreachable from an unsupervised domain, but the discriminator (no 5/11, no completion) holds). The audit's other findings: the "enters no UNOPTFLAT cone" wording was
  wrong -- the commit block is on the standing loop already (`commit_ack -> dirty_fp_state -> csr -> cap_check`);
  the compare is the one `foreign_return` uses and every new source is a flop, the 40 UNOPTFLAT names are
  unchanged, and synthesis is the gate as for every edit there. No legitimate flow CALLs another seal while armed
  (the QEMU monitor's `supervised_invoke` and capstone-c's lowering emit one CALL of the armed seal). QEMU's owner
  (PCC) and rd checks are not reproduced: seals are linear, so only the holder of A can consume the arm, and a
  mismatched rd on resume makes the domain's RETURN a `foreign_return` (fails closed).
  (2) **S-11 is fixed, at 1024 bytes and 16-byte alignment, with R-32's off-by-one corrected AND the sealed
  cursor set to the region's start.** The audit supported the check as safe to ship and REFUTED its stated
  rationale: a SEALED capability carries no bounds in a register, its effective base is its CURSOR
  (`decompress_cap_metadata`), so a check on start/end bounds nothing once the cursor is advanced before SEAL --
  a 1024-byte region sealed with the cursor at +960 would put the exchange, the SEALEDRET window and the walks
  past the region. The fix, batched into this bitstream as the LAST RTL commit on the branch so the lead can veto
  it alone: `func SEAL` sets the result's cursor := `rs1.metadata.start` -- the spec says the cursor does not
  apply to a sealed capability, and QEMU's CALL uses `bounds.base`, so silicon and QEMU now agree on where a
  seal's exchange lands. Never check the cursor instead: the resident monitor seals its interrupt-handler region
  with the cursor at +88 (the compiled `cap_env_init`), harmless only because silicon never uses CIH (`cap_cih :
  '0` in ex_stage); with the fix that seal's base is its start, still unused. Sealer census (source level): the
  monitor's domain seals 1536 B, page-aligned, cursor = start (its zeroing loop leaves it there); the
  interrupt-handler seal exactly 1024 B at 16-byte alignment; the runtime's contexts 1024 B (QEMU only); every
  directed test >= 2048 B via CAPBOUND (cursor = start) except the deliberate S-11 arms. Measured
  (`sup-sealsize.S`, runs all6/fix3): 64 and 1023 bytes trap with cause 29; 1024 and 2048 seal with every canary
  word intact; a 1024-byte region starting 8 bytes into the aligned block traps with 29 (the alignment clause
  alone -- an STC through a misaligned cursor traps on its own with 6, so nothing is planted first); the same at
  +16 seals as the control; the cursor arm (1024 bytes, images planted at +0 and at +960, the cursor moved to
  +960 before SEAL) reads as the control with the fix. Mutants: `no-seal-check` (measured: 64 and 1023 bytes seal (LCC 4), the 64-byte region shows the spill pattern again (medeleg/mip/mie images at canary[0..2]), and the misaligned 1024-byte region seals and its CALL storms to the time-out, the 29s gone); `no-seal-cursor`
  (measured: the cursor arm shows the 64-byte arm's spill pattern (#3 0, #4 the mip image 0x80, #5 0, #6 canary, #7 the canary pattern) -- the exchange happened at +960 and wrote past the region -- while the 1024-byte control is unchanged). `sup-arm.S`'s 512-byte arm, moved last, ends that test with the trap. QEMU's minimum is 528
  with a different cause (INSUF_CAP_PERMS); the divergence is in the safe direction and the runtime lane should
  raise QEMU's `CAP_SEALED_SIZE_MIN` to 1024. `seal-minsize-boundary.S` on the `board/r35-directed-repro` and
  `s12-ldc-rolling-filter` branches expects a 1022-byte region to seal and flips if merged. The Anvil relational
  lint reports 0 findings. Lint gate on the final tree: every hazard counter at baseline (LATCH 52, MULTIDRIVEN 3,
  UNOPTFLAT 40, BLKSEQ 2, UNDRIVEN 25, ANVIL_UNOPTFLAT 0); UNUSEDSIGNAL 830 -> 832, attributed line by line to the
  generated FLU alone (the SEAL edit's two discarded intermediate wires, the rest renumbered), re-baselined.
  **Contract items from the audit (for the FPGA monitor):** the silicon forget form is rs1 = x0 -- QEMU's
  `cssupervise rd, d, x0` is a status-2 refusal on silicon and leaves the arm; the monitor's trap path must forget
  (`cssupervise x0`) when it takes cause 2 at a CALL, or every later CALL of another seal traps; a re-arm while
  armed overwrites (QEMU refuses); the arm survives every exception on the CALL head.
  **Residuals (recorded, not changed here):** CSSUPERVISE has no privilege term in the decoder, like CALL and
  CAPENTER on this RTL, where QEMU requires C-mode; the SEALEDRET window [cursor+48, cursor+1008) moves with its
  cursor and CINCOFFSET does not refuse a SEALEDRET (source read only, no run: registry R-48).
- **Step 9 pre-registration, FINAL (2026-10-01, written before any synthesis number exists; supersedes the
  "~476 FFs" of revision 1.1):** own-cell flop delta = csr_regfile supervision state 524 bits (armed 1, active 1,
  base 64, save base 64, seal 129, rd 5, csupquantum 32, counter 32, resume 1, event valid 1 + kind 2 + cause 64
  + epc 64 + tval 64) + commit escape flops 259 (escape_q 1, pc 64, pc metadata 64, cause 64, tval 64, kind 2) +
  ex_stage's CALL pc latch 128 + the rev-node's free_len 16 + the switcher's phase 2 and its widened request
  register 137 + the fail-closed flop `sup_mismatch_q` 1 = **1067**. The scoreboard entry's `dom_switch_req_t` grew by 137 bits and `cap_wbdata_t` by 1
  (sup_arm_en), i.e. 138 bits per entry: the request's new fields are constant 0 at every pack site (commit is
  their only writer, on its own output), so opt_design should prune those flops -- a delta near +1067 confirms
  it, a delta near +2171 means they were NOT pruned and the walk fields must move off the entry type. ORDER-like
  test (0/500 cells before the load unit, the D-cache arbiter and i_frontend): the only NEW combinational inputs
  are `csr_exception_i.valid` and the 64-bit `foreign_return` compare into `dom_switch_valid_o`/`commit_lsu_o`
  (the after-audit's 1b); `exception_o.valid` gains flop-sourced terms only, the foreign-RETURN compare having
  been kept out of its cone; the escape request itself is issued from flops. Loop membership: the 40 UNOPTFLAT
  names unchanged (verified by name against the prerequisite tree). LUTLP-1 = 0. WNS within the null band of
  -9.595 +/- 3.74. `timing-forensics.tcl` startpoint census unchanged under commit_stage/csr_regfile's dom-switch
  names. If the reading would be uninformative the build does not go: here it is not -- the design is inert
  unless cssupervise runs, so the bitstream's first job is the R-43 acceptance list unchanged, then the first
  capstone-exec program.
- **Final ladder on the audited tree (`all4`, every test, 2026-10-01 evening):** identical to the readings below
  where they overlap, plus the audit's arms: the x1 sentinel 0x1234 survives 53 resumes (quantum 64) and 103
  (quantum 16) with the count exactly 400 both times; CCSRRW cpmp0 with a tagged capability escapes (5) and the
  monitor reads CPMP0 back empty (LCC 7); CCSRRW cepc with a capability is legal, the domain is preempted (3),
  and the monitor reads its own mepc (0) and an empty cepc back; a legal `csrr cycle` before a spin is preempted
  (3); every earlier arm unchanged; the 8-register control unchanged; all seven CALL shapes park CALL + 4.
- **Verification, steps 0-7 (repaired tree):** step 2 -- statuses 0/0/2/2/2/2/2 as listed, the 512-B seal 0
  (above); step 3 -- ecall 11, illegal 2, `csrw mepc` 2, misaligned load 4 with tval 0x3001 equal to the
  reference trap's mtval, mret 2: csupstatus 5, epc = the faulting pc, seal back SEALED, x7/x29/x30 and mie
  0x888 restored, mcause 0, csupstatus 4 after the read; step 4 -- quantum 64: 52 resumes, the stored count
  exactly 400; quantum 16: 101 resumes, exactly 400; mie 0x888, no trap; step 5 -- a monitor-side reference
  trap (mcause 0x8000000000000007 after 243 spin iterations), then the domain escapes on MTIP with that cause,
  mip.MTIP still pending in the monitor, and a second MTIP escape after a resume (the domain's mie survived);
  with mie = 0 in the image the quantum fires instead; step 6 -- all ten guards: 5/2/epc 0/SEALED/mcause 0, mepc
  0 after the blocked `csrw mepc`, and the legal control returns with csupstatus 1; step 7 -- the hostile domain
  gives back ra, sp, gp, tp, x8, x22, x30, x31 and mtvec exactly, kind 1; the foreign RETURN escapes with
  kind 2 cause 26 at its own pc, registers intact.

## REVISION 1.1 (2026-10-01, after the before-audit) — the monitor's continuation lives in a PRIVATE save area, not in the seal

The audit refuted the design below on its central premise, and both critical findings are verified in source:

- **C1.** The CALL hands the domain a SEALEDRET in x1, and `LDC`/`STC` through a SEALEDRET are allowed inside
  `[seal+48, seal+1008)` (`capstone_dyn_unit.anvil:364-365, 373-374, 429-453`; no permission test for that type; the
  LSU's type gate covers only LOAD/STORE, `load_store_unit.sv:1257,1280`). A full exchange would park the monitor's
  mstatus..mie, CPMPs, x1..x31 and mcause..satp at +48..+943 — inside that window. QEMU is immune because its caller
  snapshot is hidden state (`capstone_supervisor.c:345, 388`), and the window is contractual
  (`runtime/tests/application/context-probe-asm.S:278-285`). Today's 8-register CALL already exposes +48..+87.
- **C3 (pre-existing, made live by full mode).** The switcher's GPR write arm forces `cap_we_pack[i] = 1'b1`
  (`issue_read_operands.sv:1653-1661`) while the rs1 write lane carries the frozen head's `cap_result` without ack
  gating (`commit_stage.sv:297-300`), so each switch write also rewrites a stale register.
- **C2.** The strip must act at T (the trap is otherwise taken in the decision cycle, `csr_regfile.sv:2019`).
- **H1/H2/M3.** Resume must not rewrite x1; FP state is not in any exchange; a full exchange on the FIRST entry breaks
  argument passing in a0..a7 (QEMU's first entry is the ordinary CALL, `capstone_supervisor.c:351`).

**The revised primitive (matches QEMU's structure exactly):**

```
   cssupervise rd, rs1(seal), rs2(SAVE: a LINEAR RW capability to a 1 KiB monitor-private region)   quantum in CSR csupquantum
   armed CALL (first entry):   the ORDINARY 8-register exchange with the seal (args pass in a0..a7, contract unchanged)
                               + SAVE the monitor's ids 3..66 (mstatus..mie, offsetmmu, CPMP0..15, x1..x31 with metadata,
                                 mcause..satp) into SAVE                       -- the protected continuation
   escape / supervised RETURN: exchange ids 0..7 with the seal (the domain's pc+PCC, ctvec, cscratch, 8 regs parked, as
                               a RETURN does today), SAVE the domain's ids 3..66 into the seal's own region [+48, +944)
                               (its own state, reachable only by itself), then RESTORE the monitor's ids 3..66 from SAVE
                               -- so the seal's +48..+87 contents, which the domain can rewrite, never reach the monitor
   resume (armed CALL, resume flag): exchange 0..7 with the seal, SAVE the monitor's 3..66 into SAVE, RESTORE the domain's
                               3..66 from the seal's region; set_ra = 0 (x1 restored from the image, not rewritten)
```

The switcher gains two walk kinds beside EXCHANGE: SAVE(base, ids 3..66) = registers → memory, RESTORE(base, ids 3..66)
= memory → registers, run in sequence after the 0..7 exchange; `dom_switch_req_t` gains `save_en/save_base`,
`restore_en/restore_base`, and the id range 3..66 reuses the full-mode slot sizes (ids 3..8 at 8 B, 9..56 at 16 B,
57..66 at 8 B; 896 B at +48..+944, so a 1 KiB save area with the same layout as the seal's). `sup_save_base` is hardware-held from cssupervise's rs2; the domain has no
capability to it. The SEALEDRET window now contains only the domain's own images (harmless: they are rewritten at the
next escape before any resume). C3 is fixed by `cap_we_pack[i] = cap_we_i[i]` in the dom-switch arm. The quantum is a
CSR (`csupquantum`, M-level RW, blocked under supervision) so cssupervise's rs2 can carry the save capability; the
forget form is rs1 = x0.

Audit items folded in: C2 — strip at T from `sup_active_q && head.ex.valid && !debug_mode && cause ∉ {REPLAY,
DEBUG_REQUEST}` (plus the bad-RETURN and CSR/pc-cap cases), with the head's pc/metadata/cause/tval latched into flops at T
and the dom-switch issue at T+1 from those flops (H3), every commit output forced off under `escape_q`; H1 — the resume
flag zeroes set_ra; H4 — a RETURN that is not through the supervised seal (base ≠ sup_base or set_ra ≠ sup_rd) is itself
the escape and its own switch is suppressed; H5 — escape, injection and the counter are gated on `!debug_mode_q`;
M1 — the armed CALL must name the cssupervise'd seal (base compare) or it is an ordinary CALL and the arm is dropped;
M2 — id 25's real defect is `cpmp_q[16]`: the arm returns 0 and writes nothing; M4 — CCSRRW to CTVEC/CEPC/CSCRATCH stays
legal (QEMU allows them, the runtime's probe uses them), CIH/CPMP are blocked; M5 — the plain-CSR gate blocks
`addr[9:8] != 0` AND the explicit list CID/CIC/CIS (0x801/0x802/0x804), 0x800, 0x810, 0x811 (fcsr stays legal); M6 —
`csupstatus` is read-to-clear and cssupervise refuses (status 3) while an event is unread, so an interrupt handler
cannot overwrite an unread event; M7 — the busy-strip is first run as a DETECTOR over the 92-test sweep on the unmodified
RTL, and any test that trips it is examined before the strip ships. **H2 (FP):** v1 contract — the monitor saves and
restores f0..f31 and fcsr in software whenever the domain it resumes is not the one that ran last; v2 — FPR ids in the
hardware walks. Pre-registration corrections (G): the id-25 mutant's observable is an out-of-range CPMP read, not a
deadlock; the FF count is ~420 + the save/restore request fields; the WNS band is −9.595 ± 3.74; the ORDER-like test
exempts the strip's legitimate path into `exception_o`/`i_frontend` and binds the escape-issue path only.

Everything below this line is the original approved text, kept for the record; where it conflicts with the revision,
the revision wins.

## The primitive: a trap-initiated FULL exchange, on the existing switcher

The switcher already implements a full context exchange — `is_full=1` walks ids 0..66: pc capability, ctvec/mtvec,
cscratch/mscratch, mstatus, mideleg, medeleg, mip, mie, offsetmmu, CPMP0..15, x1..x31 with metadata, mcause..satp
(`core/anvil_build/capstone_dom_switcher.anvil:113-126`; ids: frontend 0, csr_regfile 1..25 & 57..66, GPRs 26..56 via
`issue_read_operands.sv:1568-1569`). Nothing sets `is_full=1` today; CALL/RETURN use ids 0..7 (`capstone_dyn_unit.anvil:311,346`).
The seal is 96 × 16 B = 1536 B; full mode needs 944 B. SEAL is MEANT to require ≥ 1024 B (`capstone_flu_unit.anvil:207-215`) but that check is inert on every bitstream so far (S-11 / R-32, the Anvil relational-precedence fold; QEMU raises since 2026-09-26 with a 528-byte minimum), so `cssupervise` checks the save area's size and alignment and the seal's alignment itself; the seal's SIZE cannot be checked from a register (a SEALED capability has no end bound there, `decompress_cap_metadata`), so the walks rely on SEAL's minimum, i.e. on S-11 being fixed or on the monitor sealing >= 1 KiB as it does today. [Corrected 2026-10-01; the full-mode slot map was measured the same day after the switcher's own instance of the fold was fixed.]

```
   monitor                                   supervised domain (runs in M-mode, capmode on)
   cssupervise rd, seal, quantum  --(status)-->
   CALL rd, seal   [armed -> is_full=1]       seal slots 0..66 := the MONITOR's pc+PCC, CSRs, CPMPs, x1..x31 (+metadata)
          |                                   registers       := the seal's image (the domain's context)
          |                                   quantum counter starts (core-internal, cannot be masked or retargeted)
          |          ... domain runs ...
          |          fault / injected interrupt / quantum expiry reaches COMMIT as ex.valid on the head
          |          -> ESCAPE: no trap (no mcause/mepc/mtvec), head not retired,
          |             synthesized dom_switch_req {is_full, base=seal, pc=head.pc (+its PCC), set_ra=rd, ra=seal resealed}
          |             seal slots := the DOMAIN's full context;  registers := the monitor's parked context
   <------+  monitor continues at CALL+4 with ITS registers, x[rd] = the SEALED seal,
             reads csupstatus/csupcause/csupepc/csuptval (hardware-written, read-only)
   resume: cssupervise + the same CALL -> the exchange restores the domain at head.pc; it re-executes once.
   RETURN by the domain while supervised: the existing RETURN, forced is_full=1, kind 0.
```

Nothing changes in the frontend (the switcher's own redirect installs pc + metadata, `frontend.sv:462-472`) or in the
switcher's handshake. The trap path's CSR side effects never occur on an escape (the R-43 replay strip is the template,
`commit_stage.sv:684-687`).

## Design

**A. Supervisor state (flops in `csr_regfile.sv`, written only by hardware events and by `cssupervise` at commit):**
`sup_armed`, `sup_active`, `sup_base[63:0]` (seal start, recorded at the armed CALL from the request), `sup_seal[128:0]`
(the SEALED capability to hand back in rd on escape), `sup_rd[4:0]`, `sup_quantum[31:0]`, `sup_count[31:0]`, `escape_q`,
event `{valid, kind[1:0], cause[7:0], epc[63:0], tval[63:0]}`, `free_len[15:0]` (from the rev-node). Read-only CSRs at
0xFC0..0xFC4: `csupstatus = {valid, kind}`, `csupcause`, `csupepc`, `csuptval`, `csnodefree = 65535 − head + free_len`
(custom M read-only, `addr[9:8]=11`, so the privilege gate below covers them; reads while `sup_active` → illegal → escape).

**B. `cssupervise rd, rs1(seal), rs2(quantum)`** — opcode 0x5B, funct3 001, funct7 0x22, `fu = CAPSTONE_DYN` (it needs
the node-validity query, `capstone_dyn_unit.anvil:137-144`). rs2 == x0 → the "forget" form: disarm, status 0. rs1 not
SEALED or async → status 2 (REFUSED); node dead → status 1 (DEAD); else status 0 and the result carries `{quantum}` plus an
arm flag that commit turns into `sup_armed := 1, sup_quantum := rs2` (commit-time action modelled on CAPENTER,
`commit_stage.sv:192-209`). rd := status (integer); rs1 returned unchanged (the two-writeback rule, `:297-300,345`).
rd == x0 → status 2 (the switcher skips `set_ra == 0`, `capstone_dom_switcher.anvil:64`, so an escape could not hand the
seal back).

**C. The armed CALL.** At commit of a CALL head (`:372-378`): `is_full |= sup_armed`; on the switcher's ack with
`sup_armed`: `sup_active := 1`, `sup_base := req.base_addr`, `sup_seal := req.ra_data re-typed SEALED`, `sup_rd := head.rd`,
`sup_count := sup_quantum`. `sup_armed` is cleared on ANY CALL head, faulting or not (an armed CALL that faults must not
leave the next CALL full). The monitor's x1..x31 with metadata, CPMPs, mcause..satp are parked in the seal: the protected
continuation.

**D. The escape.** Condition (combinational, then REGISTERED into `escape_q` — the decision cone includes
`pc_cap_ex_valid`, which carries the revocation broadcast, `commit_stage.sv:240,259-265`, and must not reach the dom-switch
interface combinationally): `sup_active && head.valid && !drop && !halt && !busy && ((head.ex.valid && cause ∉ {REPLAY,
DEBUG_REQUEST}) || (!head.ex.valid && (csr_exception_i.valid || pc_cap_ex_valid)))`. At T+1, from `escape_q` and head
flops, after the R-43 block (`:547`): `dom_switch_valid_o = 1`, `dom_switch_req_o = {is_full=1, base_addr=sup_base,
pc_next=head.pc, pc_next_metadata=head.pc_metadata, pc_next_tag=(pc_metadata[30:28]!=0) as ex_stage derives it
(`ex_stage.sv:1389-1391`), set_ra=sup_rd, ra_data=sup_seal, set_pc=1}`; event := {kind: 1 if the cause has the interrupt
bit (interrupt or quantum) else 2; cause; epc=head.pc; tval=head's tval (0 for interrupts)}; `sup_active := 0`. No
`commit_ack` (the head is not retired; resume re-executes it exactly once, like an interrupt return). `exception_o.valid`
is stripped whenever `escape_q || dom_switch_busy_i` (beside `:684-687`) — the second term also closes a latent hazard: a
younger interrupt-marked head one cycle after a CALL's handshake could trap mid-switch and corrupt the saved mstatus.

**E. The quantum.** `sup_count` decrements each cycle while `sup_active && !dom_switch_busy` (not during the exchange,
or a short quantum livelocks on resume). `irq_ctrl_o.sup_quantum = sup_active && sup_count == 0` — a new field in
`irq_ctrl_t` (`cva6.sv:223-229`) — injected in the decoder AFTER the global-enable block (`decoder.sv:1963-1995`, inside
`if (~ex_i.valid)`), bypassing `mie`/`mip`/`mstatus.mie` by construction, cause `CAPSTONE_SUP_QUANTUM_CAUSE =
64'h8000_0000_0000_0010` (interrupt bit + 16, platform range; unambiguous on a full-word compare; a leak would land in a
legal platform-interrupt mcause, never alias 25). The level stays asserted until the escape clears `sup_active`, so a
flushed marked instruction is re-marked.

**F. RETURN while supervised.** Honoured only if `req.base_addr == sup_base && req.set_ra == sup_rd` (a domain could
RETURN through any SEALEDRET it holds and exchange into a foreign region, stranding the monitor's continuation in the real
seal); then `is_full := 1`, `sup_active := 0`, event kind 0. Otherwise it is an escape with kind 2, cause 26.

**G. Guards while `sup_active`** (what QEMU enforces with "never leaves C-mode" and the CSR gate): illegal — `wfi`
(with `halt_i` set the commit block never runs and the quantum can never reach it), `mret`, `sret`, CALL, CAPENTER (capmode is
sticky; re-entering is a mint), `cssupervise`, every OpcodeCustom3 funct3=000 mint op (CAPCREATE/CAPTYPE/CAPNODE/CAPPERM/
CAPBOUND decode unconditionally today, `decoder.sv:1111-1147` — a supervised domain could mint a capability over its own
seal), every CCSRRW (CEPC/CIH/CPMP are not in the exchange or are live enforcement state), and every plain CSR with
`addr[9:8] != 0` (mepc, pmpcfg/pmpaddr, mcounteren, menvcfg, dcsr … are shared across a full switch). The decoder gets a
`sup_active_i` input (precedent `tsr_i`/`tw_i`); the CSR gate lives in `privilege_check` (`csr_regfile.sv:2707-2716`).
Belt and braces: `wfi_d = 0` under `sup_active`.

**H. Node census.** `free_len` register in the rev-node (+1 at the walk's push `capstone_rev_node.anvil:59-65`, −1 at the
pop `:113-134`, reset in INIT_STAGE), exported on the debug channel (`capstone_unit.anvilh:556-561`, binding pattern
`ex_stage.sv:1196`), routed ex_stage → cva6 → a new `csr_regfile` input (precedent `csr_hs_ld_st_inst`), read arm only.

**I. Full-mode defects to fix first** (full mode has never run): (a) the GPR read for the switch returns `rdata[0]`
without metadata (`issue_read_operands.sv:1573`) because the capability read port 0 lacks the dom-switch address mux the
scalar port has (`:1580-1586` vs `:1643`) — add the mux and return `{tag, metadata, cursor}`, mirroring the write at
`:1653-1661`; (b) id 25 indexes a non-existent `cpmp[16]` (`csr_regfile.sv:426-428, 1930-1933`) and the switcher blocks on
the response — add an explicit `7'd25` read arm returning 0, no write arm, keep the ack. Documented, no code: mstatus restore
then fix-up (`:1914, 1949-1955`), priv not exchanged (both sides M), satp restored without a TLB flush (harmless with
mprv=0 and mstatus writes blocked), mip hardware bits re-driven after restore (`:2004-2008`).

**J. Unchanged:** `frontend.sv`, `controller.sv`, the switcher's handshake and `process()`, the LSU, the revocation cache.

## Files

- `core/include/ariane_pkg.sv` — `CSSUPERVISE` after CAPENTER in `fu_op` (`:566`), in `check_cap_op` (`:939-949`) and
  `check_op_cap_dyn`; `CAPSTONE_SUP_QUANTUM_CAUSE` beside the replay cause (`:1234`).
- `core/anvil_build/capstone_unit.anvilh` — `CSSUPERVISE` after CAPENTER (`:277`); `free_len` in `rev_node_debug_ch`.
- `core/anvil_build/capstone_dyn_unit.anvil` — `func CSSUPERVISE` (statuses as in B), dispatch arm (`:557-570`).
- `core/anvil_build/capstone_rev_node.anvil` — `free_len` (+push, −pop, reset, debug send).
- `core/decoder.sv` — funct7 `010_0010` arm beside `:1268-1287`; `sup_active_i`; the illegal set of G; the quantum arm.
- `core/cva6.sv` — `sup_quantum` in `irq_ctrl_t`; route `sup_active` to id_stage; `rev_node_free_len` ex → csr.
- `core/ex_stage.sv` — the `free_len` debug port (`:117, 1152, 1196`). The pack (`:1376-1398`) is unchanged.
- `core/csr_regfile.sv` — the sup flops and ports (`sup_arm_i`, `sup_call_i`, `sup_event_i`, `sup_clear_i`; `sup_*_o`), the
  `dom_switch_active` input for the counter gate, `irq_ctrl_o.sup_quantum`, read arms 0xFC0-0xFC4, the privilege gate, the
  `7'd25` arm, `wfi_d` gate.
- `core/issue_read_operands.sv` — the cap read-port mux (`:1643`) and the tagged response (`:1573`).
- `core/commit_stage.sv` — the strip (`:684-687`); the CSSUPERVISE commit action; the armed-CALL override and `sup_armed`
  clear (`:372-378`); the RETURN check (F); `escape_q`; the escape issue site after `:547`, never inside `:502-509` (which
  zeroes `dom_switch_valid_o` on a pc-cap fault) and never driving `commit_ack_o`.
- Tests: `verif/tests/custom/capstone/sup-*.S`, `verif/tests/testlist_sup.yaml`, a variant patcher like `.r43-variant.py`,
  a batch runner like `.run-r43k.sh`; `asm_insn.h` gains `CSSUPERVISE(rd,rs1,rs2) = .insn r 0x5B,0x1,0x22,...`.

## Verification — written to fail, smallest first; every step with its mutant (positive control)

Values from the CAPPRINT registers in the retirement trace and `SUP_TRACE` `$display` lines; never `cva6.py`'s exit code.
`S12_MEM_DELAY=12`, `--sv_seed 1`, `--iss_timeout 7200`, container pinned.

0. **The strip alone**: a CALL with a timer interrupt timed one cycle after the handshake — no mcause/mepc change across
   the switch (an `always_ff` detector on `exception_o.valid && dom_switch_busy_i`). Mutant: strip removed → it fires.
1. **Full-mode round trip** under a sim-only `+define+SUP_FORCE_FULL` (forces `is_full` at commit), with I(a)/(b): the
   caller fills x5..x31 with tagged capabilities and integers, CPMP0..15, mscratch; the callee sees the seal image and
   RETURNs; every register checked (LCC = 1 on restored capabilities, CAPPRINT), CPMPs via CCSRRW. Mutants: revert the
   `:1643` mux (tags lost); revert the id-25 arm (deadlock, the harness must time out, not pass).
2. **`cssupervise` statuses**: live SEALED → 0; LINEAR → 2; REVOKE'd → 1; the x0 forget form; rd = x0 → 2. Mutant: 1/2 swapped.
3. **Synchronous-fault escape**: out-of-bounds load (28), illegal instruction (2), ecall (11), a CCSRRW and a privileged
   CSR (now illegal under G): the monitor resumes at CALL+4 with every GPR intact, `x[rd]` SEALED, `csupstatus` kind 2,
   cause/epc/tval as a trap would have given. Mutant: strip removed → a real trap (the step-0 detector).
4. **Quantum escape + exactly-once resume**: a domain increments a memory counter per iteration with a quantum shorter
   than the loop; after N resumes the counter equals the iteration count exactly. Mutants: `pc_next = head.pc + 4` (off by
   one); `set_ra` dropped (`x[rd]` not SEALED); the counter decrementing during `busy` with a tiny quantum (livelock).
5. **MTIP escape**: domain with `mie.MTIE` and `mstatus.mie` set from its seal image and the CLINT armed → kind 1, cause
   M_TIMER; with `mie = 0` in the image the quantum fires instead.
6. **Guards**, one arm each: `mret`, `sret`, `wfi`, `csrw mtvec/mstatus/mie/mscratch`, `ccsrrw cpmp0/cepc/ctvec`,
   `cssupervise`, CALL, CAPENTER, CAPCREATE, `csrr csupstatus` from the domain → kind 2 cause 2, monitor intact. Mutant: one
   guard dropped → the write lands (visible after return).
7. **Hostile domain**: clobbers every GPR including x1 and sp, destroys its stack, sets `mie = 0`, `mtvec = garbage`, never
   yields; and a RETURN through a foreign SEALEDRET (F) → kind 2. The monitor returns intact, register by register.
   Mutant: F's check removed → the continuation is stranded (hang or a return into the wrong region).
8. **Neutrality**: the 92-test sweep, `testlist_r43.yaml`, `testlist_r35rot.yaml` — 0 trap-count differences; the lint gate
   at baseline (UNUSEDSIGNAL re-baselined by name if the debug send adds one).
9. **Synthesis pre-registration** (sent to the synth lane before any number exists): loop MEMBERSHIP the same as R-42;
   LUTLP-1 = 0; ORDER-like counts 0/500 for `sup_*`/`escape` cells before the load unit, the D-cache arbiter and
   `i_frontend`; WNS within the null of −9.595; own-cell FF delta = the declared bits (~476: sup state 1+1+64+129+5+32+32+1,
   event 3+8+64+64, `free_len` 16); `timing-forensics.tcl` startpoint census unchanged under commit_stage/csr_regfile's
   dom-switch names. Plan on 2–3 synthesis iterations (this project's record: R-35 three, R-43 two).
10. **Board acceptance** (the board lane, after the lead's flash): the existing acceptance list of R-43 must still pass
    (the change is inert unless `cssupervise` runs); then the first `capstone-exec` program on silicon with the FPGA monitor.

## Monitor contract implied (the runtime lane's work, named here so it is not forgotten)

`supervised_invoke(d, quantum)` = `cssupervise rd, d, quantum` (rd ≠ x0; status 0/1/2/3 in rd) then the ordinary `__domcall` of the SAME seal (a CALL of any other seal while armed traps as an illegal instruction; `cssupervise x0` forgets the arm; mask interrupts between the two);
on return `x[rd]` is the SEALED seal; `events[0..3]` is replaced by `csrr csupstatus/csupcause/csupepc/csuptval` (kind 0
returned, 1 preempted — cause M_TIMER or the quantum cause — 2 fault); resume = `cssupervise` + the same CALL (the quantum
re-arms per CALL); kind 2 → never CALL that seal again; `cssupervisor_gc` mode 0 → no-op, the census = `csrr csnodefree`.
The arm survives every exception on the CALL head and a CALL of any other seal while armed traps (cause 2, mepc at the CALL): the trap path forgets with `cssupervise x0` -- the silicon forget form; QEMU's `cssupervise rd, d, x0` is a status-2 refusal here and leaves the arm. A re-arm while armed overwrites. SEAL now requires >= 1024 bytes at 16-byte alignment and sets the sealed cursor to the region's start.
The monitor must populate seal slots 3 (mstatus) and 7 (mie) as it does today. [Superseded by revision 1.1/1.2: the first entry is the ordinary 8-register exchange, so slots 9..24 (CPMP) and the GPR slots are NOT installed on entry -- the domain runs with the monitor's CPMP0..15 and x2..x31 until it is first preempted; slot 25 carries cepc/mepc.] Minimum quantum ≥ ~50k cycles (it counts only
outside the exchange, but must exceed the longest uninterruptible stall, e.g. a REVOKE walk). A `TARGET=fpga` define
(e.g. `CAPSTONE_SUPERVISOR_CSR_EVENTS`) selects this form.

## UNRESOLVED, to be settled by the before-audit or by step 1–4 simulation

- Whether the QEMU contract makes an invalid seal an illegal-instruction trap rather than a DEAD status (the two exploration
  reads disagree); the RTL returns statuses, and the FPGA monitor build adapts either way.
- The exact event `tval` for each fault class (vaddr for misaligned, `lsu_ea_full` for capability causes, instruction bits
  for CSR/illegal, pc for pc-cap) — read from the head's `ex.tval`; confirm per class in step 3.
- The exchange's cycle cost in full mode (67 slots × 4 transactions + misses): measure `dom_switch_busy` in step 1; it sets
  the minimum quantum above.
- The re-entry wedge recorded under the trap-vector defect (2026-09-16, a fault after `domreturn` + re-entry): the escape
  path does not use the trap vector at all, so it should not apply, but step 4's repeated resumes are the test.
- A tracer entry for the escape (the hardware tracer reads the head's `ex` directly, as it did for the R-43 marker): decide
  whether to mask it or document it, after step 3.

## Process

- Before-audit of this design (claim-auditor, adversarial; name F, G and the escape cone as the weakest links) before any
  RTL is written; after-audit of the diff before synthesis.
- Branch `sup-call` off `r43-query-on-miss` (5aa316e0d) in a fresh worktree (tools symlink target mounted, `riscv-tests`
  copied, nested submodules initialised — the 2026-09-29 lessons); every step its own commit on the lane branch; the folder
  `tests/fpga-repros/`-style report lives under `docs/plans/` until there is silicon evidence.
- Synthesis and the reflash are the lead's call; the FPGA monitor build is the runtime lane's.
