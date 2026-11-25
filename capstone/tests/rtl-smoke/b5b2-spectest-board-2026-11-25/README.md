# b5b2-spectest-2026-11-25 on board 3: the board lane's run record

This folder records boot 5 of the runtime lane's b5b2-spectest kit: the first SPEC CPU2017 test inputs on B5b-2 Linux.
The pre-registration is `capstone/tests/rtl-smoke/b5b2-spectest-2026-11-25/PREREG.md` on branch runtime/b5b2-linux, at
510a064d9774. Amendments 1 and 2 (5aaa7bbb2c6e, 9369457895ad) cover the chain that followed.

## Boot 5

The run:
- Board 3, hub booking 1504, 2026-11-25 04:53-07:16 UTC.
- The resident `caplifive_…-r75d-b5b2n_904cd9c6b.bit` was read back by name before and after. No flash.
- The image was `fw_payload.bin` `222cfdc50fa3…`. Its stages' plan.json and run.sh were checked against compiler's
  `board-stages/B5T`.
- The procedure was b5b2-linux's: upload, power-cycle, `reset halt`, `load_image` at 0x80000000 (166 s), `continue`.
  The bound was 15,300 s after the load.

### What it read

The result lines are `boot5-result-lines.txt`, and the grade is `boot5-grade.txt`:
- **C AS PREDICTED:** `okTRAPTEST: control smoke.dom rc=7`.
- **The first stage, `B5T-505.mcf_r` (first by glob order), began.** The kernel logged
  `sublet: mcf_r_base.nati[57] enters 0x3ff7fe0000` at 33.26 s.
- **Then nothing but `/init`'s 120 s heartbeat,** the last at uptime 2,799.49 s (about 05:44Z), before the stage's
  3,600 s watchdog could fire.
- **After that the console was silent for over 80 minutes.** The heartbeat stopped too, so the whole system stopped,
  not one stage.
- **Grade:** mcf is NO READING (no `@@QSTEP-END`). lbm, nab and xz never ran in this boot.

### Stopping it without disturbing the wedge (the runtime lane's ruling)

1. The run's wrapper was paused (SIGSTOP), and its runner was SIGKILLed, so the runner's `finally` never powered the
   board off. No power, reset, halt, gdb or load touched the board until step 4.
2. The wedge snapshot was read through the debug apertures, from a new console session (`run_sup_bare-snap.py`,
   sha256 prefix d908fce8cf17df9a). That session had to wait for the dead runner's hub lease to lapse; the hub's
   clock runs about 92 s behind apollo's.
   - Each aperture was read twice, with 0 UNSTABLE.
   - The switch walk went bit by bit in an order that never holds 191. At 904cd9c6b, 191 is
     `dom_switch_log_clear` (cva6.sv:1109), the only switch-keyed side effect: no tracer is built, so sw[1] arms
     nothing.
   - The walk parked at 0 afterwards.
3. The console replay the hub sends on connect (113 KB) held all of boot 5's output, 8,986 chars from its OpenSBI
   banner. That replay is the capture graded above.
4. The wrapper was resumed. It read the resident back, graded the capture and released the booking. The board was
   left powered, and the next boot's power-cycle reset it.

### The snapshot (`boot5-snapshot.txt`), labelled from 904cd9c6b's own mux

The labels come from cva6.sv's debug mux, load_store_unit.sv:412-413 and store_unit.sv:480. fu_op is counted from
ariane_pkg.sv: 238 entries, with the `CAPSTONE_SUBLET_FMT` block after index 235.

| aperture | raw | reading |
|---|---|---|
| 255 trap summary | 0x89 | seen 1, mcause 9 (ecall from S) |
| trap-log mepc / tval | 0xffffffff80006040 / 0x73 | the last nontrivial M trap, an S-mode ecall in the kernel |
| 224 | 0x0c | ex_commit 0, load/store syncers 0, **lsu_ready 0**, dyn_ready 1, flu_ready 1, flush 0, privM 0 |
| 225 | 0x80 | bit 7 trace_buf_empty is a constant 1 (no tracer); bit 3 dom_switch_busy a constant 0; the driven bits read no dyn waits, no stall_issue, no mem_write/mem_wait |
| 226 / 227 / 228 | 0 / 0 / 0x80 | the switcher's side is tied to constants under the Sublet format (cva6.sv:2398-2411); the driven bits (226: the LSU's data ack and response valid, the register acks; 227: commit's switch valid, the three register-response valids) read 0. 228 is the constant 0x80 |
| 192 | 0x01 | commit slot 0 valid |
| commit pc (230-237) | 0x3ff7fe3506 | inside mcf's domain (enters 0x3ff7fe0000); slot 0, stale after a flush |
| 193 | 0x04 | store_buf_commit_cnt 4 |
| 194 | 0 | store_state 0 (store_unit.sv:123, `state_q`) |
| 195 / 229 | 0 / 0 | **no reading**: `load_state` is undriven at 904cd9c6b, so it reads 0 on every boot (correction below) |
| 219 lsu_head_op | 0xd7 | fu_op 215 = **LDC** |
| 220 lsu_head_debug | 0x8c | lsu_ctrl.valid 1, lsu_ready 0, fu 12 = CAPSTONE_DYN |
| 221 store_debug | 0xf8 | commit queue valid 1111, **data_req 1, data_gnt 0**, no_st_pending 0 |
| 222 | 0 | adapter_tag_debug, constant '0 in this build |
| rev_node debug / head / serving | 0 / 0 / 0 | **no reading**: constant 0 in the Sublet build (capstone_sublet_life.sv:226-229) |

**Correction (2026-11-25, after the first push of this record, 57c5f32e3880).** The row above for sw=195 and sw=229
first read "store and load states idle". `load_state` has no driver: `load_unit.sv:109` declares `state_o` and never
assigns it (904cd9c6b; the same holds at 715bdd1fe). So both apertures read 0 on every boot, and say nothing about the
load unit. The rtl lane found it, after reading sw=195 = 0 as "the load unit was idle with the LDC presented, so the
LDC never issued, so not R-29"; that reading is withdrawn, and R-29's involvement is open. sw=192, 193 and 194 are
driven (`commit_instr_id_commit[0].valid`; store_buffer.sv:110 `commit_status_cnt_q`; store_unit.sv:123 `state_q`).
What this check missed: it took each aperture's label from the mux, and never looked for the driver behind it.

**Every aperture's driver at 904cd9c6b (traced after the correction, at the runtime lane's request).** Each net was
followed from the mux down to its `assign` or procedural write. The Anvil units' outputs were followed in the generated
`.anvil.sv` of 904cd9c6b's synthesis tree: `capstone_dyn_unit.anvil.sv:19227-19230`, its syncers' req_set at :19823,
and `capstone_flu_unit.anvil.sv:5361`. The tracer was first shown to find `load_state` undriven and `store_state`
driven.
- **Driven:** 255; 224 (all eight bits); 192, 193, 194; 219, 220, 221; the commit pc (scoreboard.sv:149); the trap-log
  mepc and tval (cva6.sv's procedural log); 225 bits 6-4, 2-0; 226 bits 6, 5, 2, 1; 227 bits 7, 5, 4, 3.
- **Constant in this build:** 225 bit 7 (trace_buf_empty = 1, no tracer) and bit 3 (dom_switch_busy = 0); 226 bits 7,
  4, 3, 0; 227 bits 6, 2-0; 228 (0x80); 222 (0); the rev_node debug word, head and serving index (0,
  capstone_sublet_life.sv:226-229).
- **Undriven:** 195 and 229 (load_state).

The board lane's snapshot tool now prints each value with this status (run_sup_bare-linux4.py, cdad66bfa4eb71b6), so
an undriven or constant value is never reported as a reading.

Read as raw readings, attributed to nothing:
- a valid LDC sits at the LSU head with the LSU not ready;
- four committed stores are queued, and the store unit is requesting the D$ without a grant;
- commit waits on slot 0, inside mcf's domain.

The runtime lane owns the attribution, so what this means is not decided here.

### What changed for the chain because of it (Amendment 2)

- mcf now runs last in every image that carries it.
- The board lane's runner (`run_sup_bare-linux4.py`, 6f17eb276c71c7e6) stops a boot after 1,200 s with no new console
  byte. Before writing the capture or powering off, it reads this same snapshot on its own live session.
- A stalled stage is STALL / NO READING.
