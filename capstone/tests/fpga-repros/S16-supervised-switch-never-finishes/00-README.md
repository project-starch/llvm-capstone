# S-16 — a domain switch that starts while the store buffer's commit queue is FULL never finishes on silicon (and loses a committed store)

**Sibling, a different signature: [S-17](../S17-ldc-after-supervised-switch-lsu-stuck/).** There, after a FINISHED switch,
an LDC behind `ccsrrw sp <- cscratch` leaves the LSU not ready. Apertures 224 / 225 read `0x0d` / `0x80` there; here
they read **`0x1f` / `0x88`**. Use both bytes: 225 = `0x80` alone is the resting value of many unrelated wedges.

## The signature
Bitstream `caplifive_supcall_36a641e0b.bit`, the supervised-CALL bitstream. Its CSR 0xFC4 (csnodefree) exists: it read
0xFFCD in the Linux identification boot and reads 0xFFFC in these bare images. The LED-path apertures after the hang
(labels from `cva6.sv` at capstone-ariane 36a641e0b, MSB first; raw lines in `results/result-lines.txt`):

| aperture | value | meaning |
|---|---|---|
| 224 {ex_commit.valid, load_syncer, store_syncer, lsu_ready, dyn_ready, flu_ready, flush, privM} | **0x1f** | flush asserted; no exception at commit; no syncer request |
| 225 {trace_buf_empty, wait_store_syncer, wait_load_syncer, wait_rev_res, dom_switch_busy, stall_issue, mem_write, mem_wait} | **0x88** | **dom_switch_busy = 1**. The rev-node unit's memory flags are 0 (they do not show the switcher's walk) |
| 226 {data_valid, data_ack, data_resp_valid, data_resp_ack, reg_valid, reg_ack, reg_resp_valid, reg_resp_ack} | **0x80** | the switcher's DATA request is valid and **never acknowledged** |
| 227 (..., data_req.write_en, reg_req.is_set, ...) | **0x04**, or **0x06** | that request is a **WRITE**; with `is_set` too (0x06) it is an EXCHANGE write, without it (0x04) a SAVE write |
| 228 / 239 / 240 | idx **7**, or **4** | the walk index: 7 for a switch that SAVEs first, 4 for one that starts with the exchange (see "Which switches" below) |
| 193 store_buf_commit_cnt | **4** | the commit queue's count reads full |
| 194 store_state | **3** | `WAIT_STORE_READY` (store_unit.sv:155-160) |
| 195 / 229 load_state | 0 | the load unit is idle |
| 192 commit_instr[0].valid | 0 | |

**The commit-pc aperture (230..237) during a stuck switch is STALE scoreboard slot 0**, not a live instruction:
`scoreboard.sv:149` reads the slot whether valid or not, and the switch's held flush resets the pointer to 0.
- **Which instruction it shows.**
  - The issue pointer also restarts at 0 after every flush and WRAPS over the scoreboard's 8 slots
    (`scoreboard.sv:302-303`; NrScoreboardEntries = 8 in `capstone_cv64a6_imafdc_sv39_config_pkg.sv:60`).
  - So slot 0 holds the first instruction issued after the previous flush ONLY until eight more have issued.
  - After that it holds the latest instruction whose issue index since that flush is a multiple of 8, i.e. one of the
    last eight issued.
- **The short case is exact.** The monitor's `li sp, 0; CALL` comes two issues after the CCSRRW's flush. Positive
  control: with a `nop` inserted after `ccsrrw x0, cscratch, sp`, the reading moved from `li sp, 0` to that nop
  (`s16nop-nt`).
- **For a domain that has run longer, the reading locates the stuck switch to within eight issued instructions.** It
  does not give the last resume point.
- The CALL itself had retired; it retires when the switcher accepts it (commit_stage.sv:532).

## Reproductions
**Bare images** (`images/`). Rebuild both with `src/build.sh` from `src/sup-capstl.S`; `images/SHA256SUMS` matches. M-mode,
no OpenSBI, JTAG load at 0x80000000; run with `../../rtl-smoke/sup-resume-2026-10-03/run_sup_bare_wedge.py`, which
reads the apertures after the timeout.
- The monitor arms a seal (`cssupervise`) with a 64-cycle quantum and resumes every preemption.
- Around every CALL it runs the sequence the FPGA monitor's compiler-generated `__domcallsaves` emits:
  - about 26 stores to a swap area (CPMP0..15 STCs, 8 more STCs, CSR saves, the cepc STC);
  - `ccsrrw x0, cscratch, sp; li sp, 0`, then the CALL;
  - the swap back after it.
- **`armdep-nt-d16-q64.bin`** (sha256 `0986d3946ef9abca`) has no trace prints. It hung before escape 16 in 2 of 2 runs.
- **`armdep-d16-q64.bin`** (sha256 `d09491cd969f2650`) is the same with trace prints for the first CALL and 15
  resumes. Those prints delay the hang: it hung between escapes 16 and 31 in 3 of 3 runs.

**The FPGA monitor** (capstone-sbi `monitor/supcall-fpga`, its classic-call test hook) ran the SQLite speedtest under
supervised CALL in boots supmon-c5q, supmon-c5u and supmon-c5f (`../../rtl-smoke/supmon-2026-10-03/`). All three read
224 = 0x1f and 225 = 0x88.
- C5q and C5u: the stale commit pc on `li sp, 0`, so the CALL side.
- C5f (with the fence workaround): a domain instruction, so the escape side.
- In C5u, after 212 good resumes, the resume CALL never came back.
- C5q was quiet, so whether it hung on its first supervised CALL or on a resume is unknown.
- Their 226-240 apertures were not read. That they share the localisation below is inferred from 224/225 and the code
  shape, not measured.

## What triggers it, measured on silicon
Every arm below is bare, part of the sup-resume ladder, and has its result lines there.
- **The number of stores issued right before the switch.**
  - With only `ccsrrw x0, cscratch, sp; li sp, 0` around the CALL plus N stores to an ordinary buffer: N = 12
    completes 8,552 escapes; N = 18 and N = 24 hang with the signature above; 24 plain `sd`s hang as well.
  - Each data-region store also writes a shadow tag, so the `sd` arm does not separate tag writes from plain stores.
  - Without the swap sequence, about 22,000 supervised resumes in this harness never hung, including runs whose walk
    reads missed L1 (one seal line timed at the first escape: 27 cycles hot, 83 after a sweep, against a 7-cycle hit).
- **Which switches: every kind, on silicon.**
  - The mis-checked write is the FIRST switcher write after an ordinary store, and the 4th write after it starves.
  - Where that first write falls depends on the request. `commit_stage.sv` at 36a641e0b sets `save_en` for exactly
    three: the armed CALL (:497), the supervised RETURN (:522) and the quantum escape (:746).
    - A switch that SAVEs first starts at id 3 and starves at **idx 7** (227 = 0x04).
    - A switch that starts with the exchange, i.e. a plain CALL or RETURN with `save_en` clear, starts at id 0 and
      starves at **idx 4** (227 = 0x06; an exchange write also sets the register).
  - Measured with bare arms of the RTL lane's `sup-s16-stores.S` (2026-10-04, sessions s16stores and s16next in
    `../../rtl-smoke/sup-resume-2026-10-03/`, each pre-registered). The switch is named from the stale slot-0 pc
    where that reading is exact:

  | arm (image) | the switch that hung | idx |
  |---|---|---|
  | 32 stores, then an armed CALL; one round (`s16st-n32-r1`) | the armed CALL (pc = the tail's `li sp, 0`) | 7 |
  | 4 stores, then an armed CALL to a domain that stores once and RETURNs; three rounds (`s16st-n4-r3`). One round completes (`s16st-n4-r1`) | the domain's supervised RETURN (pc = the domain's first instruction) | 7 |
  | the same with a `fence` before the CALL (`s16st-n4-r3-fence`) | the same RETURN | 7 |
  | 8-store bursts in the domain under a 150-cycle quantum, a `fence` before the domain's RETURN (`s16st-esc-n8-retfence`) | an escape (by design, the quantum) within the domain's store bursts or at the marker store that ends them (pc = a burst store, 11 instructions before the RETURN; a fence flushes, so a RETURN-side hang would read the RETURN) | 7 |
  | the same without that fence (`s16st-esc-n8`, twice, identical) | not resolved: the RETURN, or an escape near the end of the last burst (pc = the `addi` before the RETURN; a backed-up scoreboard can hold up to 7 instructions issued past the commit point) | 7 |
  | a PLAIN CALL (no `cssupervise`) after 32 stores, its only CALL (`s16st-plain-n32-r1`) | the plain CALL | **4** |
  | a PLAIN CALL after 4, 24 or 32 stores, three rounds (`s16st-plain-n4-r3`, `-n24-r3`, `-n32-r3`) | a plain CALL | **4** |

  - **A plain CALL is therefore exposed on silicon too.** The FPGA monitor's plain domcalls carry the same ~26 stores
    and swap-out tail before every CALL, and they have run on every boot without showing it. Why the bare plain CALL
    hangs and the monitor's does not has not been measured. N = 1 per image. The timing is not understood: silicon
    disagreed with the simulation on 2 of the 4 plain arms (cold-seal `-n32-r1` hangs, and `-n4-r3` hangs at a CALL
    rather than the RETURN).
- **A `fence` immediately BEFORE the CALL covers that CALL's own switch, and nothing else.**
  - The fast reproduction plus that fence COMPLETED: 8,551 escapes, every reading exact.
  - So did 24 STCs plus that fence: 8,552.
  - A fence AFTER the CALL does not help, because the switch has already started.
  - **It does not cover a short domain that stores before its RETURN** (`s16st-n4-r3-fence` above). The CALL's own
    switcher writes, plus that store, fill the queue when the RETURN starts.
  - **It does not cover an escape** (`s16st-esc-n8-retfence` above, and the FPGA monitor below).
- **A `fence` before the CALL AND before the domain's RETURN covers every switch without a quantum escape, in the
  measured shapes** (session s16fence, each arm the twin of an image that hung, differing only by those fences).
  - `s16st-plain-n32-r1-fence`, `s16st-plain-n4-r3-fence2` and `s16st-plain-n32-r3-fence2` (plain CALLs, idx 4 in
    their twins) COMPLETED.
  - So did `s16st-n4-r3-fence2` (armed CALL and supervised RETURN, idx 7 in its twin). Every reading was exact.
  - Each switch then starts on a drained commit queue. By the mechanism, a store between the fence and the switch
    reopens the window.
  - N = 1 per arm.
- **The ESCAPE side in the FPGA monitor.** With a fence before all 8 domcalls (boot supmon-c5f), the supervised
  speedtest made 552 preemptions (C5u: 212), then hung with the same 224/225.
  - Its stale slot-0 commit pc was a DOMAIN instruction: VA 0xe56bc, in `lookupName`, inside an -O0 local-init burst
    of `stc`s.
  - The domain had run far more than eight instructions since it was last resumed. So by the rule above, 0xe56bc is
    one of the last eight instructions issued before the stuck switch's flush, and the quantum ESCAPE landed within
    eight issued instructions of it, inside that store burst. No domain RETURN is that close to `lookupName`.
  - **Withdrawn (2026-10-04).** An earlier version of this paragraph read 0xe56bc as the resume point of the PREVIOUS
    preemption, and put the escape "up to one quantum later, at an unrecorded point". That reading ignored the issue
    pointer's wrap over the 8 slots (the RTL lane's own correction of the rule it had given).
  - That the escape's SAVE walk met a commit queue full of the domain's own stores is what the mechanism requires.
    The burst makes it likely; it was not observed directly.
  - No software placement covers an arbitrary preemption point. **Preemptive supervision needs the RTL fix.**

## The mechanism (the supervised-CALL RTL lane's simulation, 2026-10-04; it matches every silicon read above)
In simulation, at the 5th switch of their `sup-mswap-noploop` run, with store-path tracers:
1. The switcher's first SAVE write (idx 3) reaches the store unit while the store buffer's commit queue (depth 4)
   still holds 4 committed stores from before the CALL.
2. The store unit asks the buffer "ready?" keyed on the kind of the PREVIOUS accepted store
   (`store_unit.sv:443`, `.is_dom_switch(... is_dom_switch_q)`). It therefore gets the speculative queue's answer
   (`store_buffer.sv:170`), which is "yes".
3. The write is pushed into the full commit queue without a room check (`store_buffer.sv:245-257`). With the write
   pointer equal to the read pointer, it **overwrites the head: the oldest committed store is lost**.
4. The ring then drains past an invalid head and starves the 4th write after the overwrite: idx 3 + 4 = idx 7,
   `WAIT_STORE_READY`, count 4 (three live entries behind an invalid head).
- **A switch that starts with the exchange** takes the same path at its id 0 write, and id 4 starves. That is the
  RTL lane's simulation of the PLAIN arm (sup-call 57a9874c8), and idx 4 on the board.
- The trigger needs the commit queue full at the instant the first switcher write after an ordinary store arrives.
  That explains the store-count threshold, and why DDR's slower drain makes it common on silicon and rare in
  simulation.
- **A consequence beyond the hang, by the mechanism:** every S-16 occurrence also dropped one committed store,
  the oldest in the commit queue at that instant.
- **The RTL side** is the supervised-CALL RTL lane's registry entry R-49. Its fix is capstone-ariane 192a5e624 on branch
  `sup-call` (it was cited here as 1f56774bd, a lane-branch hash since rewritten): the store unit's room check is keyed on the switcher request being decided, so every switcher write is
  checked against the commit queue it enters. That covers every switcher write: the CALL's, the RETURN's and the
  escape's, SAVE or exchange. An assertion fires if such a push ever meets a full queue.

## Prior art
capstone-ariane `controller.sv:220-224` carries a comment that flushing EX during a domain switch "causes a deadlock
where req_en_q never clears". The code sets `flush_ex_o = 1'b1`: 901c45ce1 (2026-02-25) changed it, and 030378a66
(2026-04-21) restored it to fix "the instruction after CALL" being executed. Per the RTL lane, the deadlock that
comment names is now prevented by `load_unit.sv:649`. In S-16 the held flush is a symptom of the stuck switch, not
its cause.

## Controls
`call-retpc` (sup-bare-2026-10-02, a05ca464) ran first in every board session of this investigation and read PASS
exact each time.
