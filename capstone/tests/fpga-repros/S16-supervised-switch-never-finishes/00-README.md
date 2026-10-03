# S-16 — a supervised CALL's switch never finishes on silicon: dom_switch_busy stays 1, the flush is held, commit stops

**Sibling, a different signature: [S-17](../S17-ldc-after-supervised-switch-lsu-stuck/).** There a capability load right
behind the post-CALL `ccsrrw sp <- cscratch` hangs with the LSU not ready (aperture 224 = `0x0d`, domsw 0). If your
aperture 225 reads `0x80` rather than `0x88`, you are in S-17, not here.

## The signature
Bitstream `caplifive_supcall_36a641e0b.bit`, the supervised-CALL bitstream, identified by csnodefree (CSR 0xFC4)
reading 0xFFCD. The LED-path apertures after the hang:

| aperture | value | meaning |
|---|---|---|
| 224 {excommit, ldsync, stsync, lsu_rdy, dyn_rdy, flu_rdy, flush, privM} | **0x1f** | nothing at commit; no load or store syncer pending; LSU, DYN and FLU ready; **flush asserted**; M-mode |
| 225 {tbe, wstore, wload, wrev, domsw, stall, memwr, memwait} | **0x88** | no wait condition, no memory wait; **dom_switch_busy = 1** |
| commit pc (230..237) | the `li sp, 0` that follows `ccsrrw x0, cscratch, sp` | just before the next supervised CALL |

The core is not faulting. A domain switch is in progress and never ends, and the flush it holds stops commit.

## Reproductions (`results/result-lines.txt`)
1. **The bare images.** Rebuild both with `src/build.sh` from `src/sup-capstl.S` and the committed ladder harness;
   `images/SHA256SUMS` matches. M-mode, no OpenSBI, JTAG load at 0x80000000.
   - **`armdep-nt-d16-q64.bin`** (sha256 `0986d3946ef9abca`) is the fastest. It hangs within its first 16 resumes,
     with signature A, and its commit pc is the `li sp, 0` directly after `ccsrrw x0, cscratch, sp` and directly
     before the CALL: exactly the FPGA monitor's `ccsrrw(x0, cscratch, sp); li sp, 0; domcall`.
   - **`armdep-d16-q64.bin`** (sha256 `d09491cd969f2650`) is the same with per-resume trace prints for the first 16
     resumes. Those prints DELAY the hang: it came between escapes 16 and 31 in two runs.
   - The monitor arms a seal (`cssupervise`) with a 64-cycle quantum and wraps every CALL in the sequence the FPGA
     monitor's compiler-generated `__domcallsaves` emits:
     - CPMP0..15 read, cleared and stored with STC, plus 8 more tag-setting STCs;
     - mcause, mtval, stvec, scause, stval, sepc, sscratch, satp, 0x803 and cepc swapped out;
     - `ccsrrw x0, cscratch, sp; li sp, 0`, then the CALL;
     - after it, `ccsrrw sp, cscratch, x0`, then an LDC THROUGH that sp, then everything swapped back.
   - It resumes every preemption.
   - With the prints it shows `ACR` for each of the first 16 resumes, then a `.` every 16 escapes, before hanging.
     A third run of the traced image (a dot every 256) hung before escape 256.
   - Run it with `capstone/tests/rtl-smoke/sup-resume-2026-10-03/run_sup_bare_wedge.py`
     (`SUP_IMG`, `SUP_OUT`, `SUP_REC_ADDR=80015000`). It reads the apertures above after the timeout.
2. **The FPGA monitor** (capstone-sbi `monitor/supcall-fpga`, its classic-call test hook) running the SQLite
   speedtest under supervised CALL. Boots supmon-c5q and supmon-c5u, in `../../rtl-smoke/supmon-2026-10-03/`:
   - C5u resumed 212 times, re-armed after escape 213 (`SUPA:0`), and its resume CALL never came back;
   - both boots read 224 = 0x1f, 225 = 0x88, and a commit pc on the `li sp, 0` after `ccsrrw(x0, cscratch, sp)`,
     just before the resume `domcall`.

## What was measured not to matter, or not to help (bare, same bitstream)
- **Without the swap sequence, no hang** in about 22,000 supervised resumes:
  - hot loops and cache-evicting loops;
  - a capability store-to-load at the escape;
  - every walk read missing L1, shown by a load-latency probe at the escape (27 cycles hot, 83 swept, against a
    7-cycle hit).
  The ladder `sup-resume-2026-10-03` holds every arm.
- **A `fence` or a UART print after every CALL** does not prevent it: both hang before escape 256.

## What is not known
- **The mechanism.** The supervised-CALL RTL lane's simulation (2026-10-04, ~700 resumes with every walk read
  missing L1, under the harness's in-order fixed-latency memory) did not reproduce this signature. That lane's
  candidate is a walk read held in the AXI adapter's TAG_WAIT under DDR. No aperture on this bitstream exposes the
  adapter's tag state (`tag_wr_pend`, `tag_rd_inflight`).
- **Whether "commit pc" is the last RETIRED instruction or the uncommitted HEAD.** The answer decides whether
  `li sp, 0` retired and the CALL's switch then hung, or `li sp, 0` itself cannot commit while the switch is busy.

## Controls in every session
`call-retpc` (sup-bare-2026-10-02, a05ca464) read PASS exact before every image: N = 15 on this bitstream.
