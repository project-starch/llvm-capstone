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
The seal is 96 × 16 B = 1536 B; full mode needs 944 B. SEAL is MEANT to require ≥ 1024 B (`capstone_flu_unit.anvil:207-215`) but that check is inert on every bitstream so far (S-11 / R-32, the Anvil relational-precedence fold; QEMU raises since 2026-09-26 with a 528-byte minimum), so `cssupervise` checks the seal's and the save area's size and alignment itself. [Corrected 2026-10-01; the full-mode slot map was measured the same day after the switcher's own instance of the fold was fixed.]

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

`supervised_invoke(d, quantum)` = `cssupervise rd, d, quantum` (rd ≠ x0; status 0/1/2 in rd) then the ordinary `__domcall`;
on return `x[rd]` is the SEALED seal; `events[0..3]` is replaced by `csrr csupstatus/csupcause/csupepc/csuptval` (kind 0
returned, 1 preempted — cause M_TIMER or the quantum cause — 2 fault); resume = `cssupervise` + the same CALL (the quantum
re-arms per CALL); kind 2 → never CALL that seal again; `cssupervisor_gc` mode 0 → no-op, the census = `csrr csnodefree`.
The monitor must populate seal slots 3 (mstatus), 7 (mie) and 9..24 (CPMP) for the domain's confinement — the full exchange
installs them where the 8-register CALL never did; slot 25 stays zero. Minimum quantum ≥ ~50k cycles (it counts only
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
