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
| 227 (..., data_req.write_en, ...) | **0x04** | that request is a **WRITE** |
| 228 / 239 / 240 | idx **7** / 7 / 7 | the switch is at walk index 7 |
| 193 store_buf_commit_cnt | **4** | the commit queue's count reads full |
| 194 store_state | **3** | `WAIT_STORE_READY` (store_unit.sv:155-160) |
| 195 / 229 load_state | 0 | the load unit is idle |
| 192 commit_instr[0].valid | 0 | |

**The commit-pc aperture (230..237) during a stuck switch is STALE scoreboard slot 0**, not a live instruction:
`scoreboard.sv:149` reads the slot whether valid or not, and the switch's held flush resets the pointer to 0. It shows
the first instruction issued after the last flush. Positive control: with a `nop` inserted after
`ccsrrw x0, cscratch, sp`, the reading moved from `li sp, 0` to that nop (`s16nop-nt`). The CALL itself had retired;
it retires when the switcher accepts it (commit_stage.sv:532).

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
- **A `fence` immediately BEFORE the CALL removes the CALL-side trigger, and only that.**
  - The fast reproduction plus that fence COMPLETED: 8,551 escapes, every reading exact.
  - So did 24 STCs plus that fence: 8,552.
  - A fence AFTER the CALL does not help, because the switch has already started.
- **The ESCAPE side is not covered.** In the FPGA monitor with a fence before all 8 domcalls (boot supmon-c5f), the
  supervised speedtest made 552 preemptions (C5u: 212), then hung with the same 224/225.
  - Its stale slot-0 commit pc was a DOMAIN instruction: VA 0xe56bc, inside an -O0 local-init burst of `stc`/`sw` in
    `lookupName`. So the stuck switch was the quantum escape, fired while the domain's own committed stores filled
    the queue.
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
- The trigger needs the commit queue full at the instant the first SAVE write arrives. That explains the store-count
  threshold, and why DDR's slower drain makes it common on silicon and rare in simulation.
- **A consequence beyond the hang:** every S-16 occurrence also dropped one committed store, the oldest of the
  stores issued before the CALL.
- The fix belongs in the RTL: consult the commit queue's readiness for a dom-switch request, and assert that such a
  push never meets a full queue. It is the RTL lane's to make.

## Prior art
capstone-ariane `controller.sv:220-224` carries a comment that flushing EX during a domain switch "causes a deadlock
where req_en_q never clears". The code sets `flush_ex_o = 1'b1`: 901c45ce1 (2026-02-25) changed it, and 030378a66
(2026-04-21) restored it to fix "the instruction after CALL" being executed. Per the RTL lane, the deadlock that
comment names is now prevented by `load_unit.sv:649`. In S-16 the held flush is a symptom of the stuck switch, not
its cause.

## Controls
`call-retpc` (sup-bare-2026-10-02, a05ca464) ran first in every board session of this investigation and read PASS
exact each time.
