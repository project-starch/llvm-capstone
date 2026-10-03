# S-17 — on silicon, an LDC right behind the post-CALL `ccsrrw sp <- cscratch` never issues: the LSU stays not-ready

**Sibling, a different signature: [S-16](../S16-supervised-switch-never-finishes/).** There a supervised CALL's switch
never finishes (aperture 225 = `0x88`, dom_switch_busy 1). If your aperture 224 reads `0x1f` rather than `0x0d`, you
are in S-16, not here.

## The signature
Bitstream `caplifive_supcall_36a641e0b.bit` (csnodefree 0xFFCD). The LED-path apertures after the hang:

| aperture | value | meaning |
|---|---|---|
| 224 {excommit, ldsync, stsync, lsu_rdy, dyn_rdy, flu_rdy, flush, privM} | **0x0d** | **LSU not ready**; DYN and FLU ready; no load or store syncer pending; no flush; M-mode |
| 225 {tbe, wstore, wload, wrev, domsw, stall, memwr, memwait} | **0x80** | no wait condition; no domain switch in progress |
| commit pc (230..237) | the `ccsrrw sp, cscratch, x0` at CALL + 4 | the next instruction is the LDC |

## The matched pair (`images/`, `results/result-lines.txt`)
These are bare M-mode images (JTAG load at 0x80000000; rebuild them with `src/build.sh` from `src/sup-capstl.S`). A
supervised domain is preempted every 64 cycles and the monitor resumes it. After every CALL the monitor runs
`ccsrrw sp, cscratch, x0`, then a load from a swap area that it filled before the CALL.

| image | that load | result |
|---|---|---|
| `arm12-ld-q64.bin` (sha256 `8027a661a5972427`) | `ld t1, 464(s10)` | COMPLETED, 8,552 escapes, every reading exact |
| `arm12-ldc-q64.bin` (sha256 `07de9fb7af896dd5`) | `ldc t1, 464(s10)` | HANGS at the first escape, signature above |

The two images differ in exactly that one instruction, at both of its sites (and in the test-ID character).

**Not only supervised.** `mswapfix-plain-noploop.bin` (sha256 `ffd6f6db241db2e1`) runs the same pre- and post-CALL
sequence around ONE plain, un-armed CALL, and hangs with the identical signature (commit pc on the post-CALL ccsrrw).

## What the bisection of that sequence showed (images and scripts in `../../rtl-smoke/sup-resume-2026-10-03/`)
- **It needs the post-CALL `ccsrrw sp <- cscratch` followed by a capability or DRAM load.**
  - Without the ccsrrw, the post-CALL LDC completes.
  - With the ccsrrw followed only by a UART (MMIO) poll, it completes.
  - With an `ld` in place of the LDC, it completes (the pair above).
- **What masks it in the PLAIN case:**
  - a `fence` right after the CALL masks it;
  - 8 `nop`s do not (the RTL lane's simulation: 4 nops do not either);
  - a UART print between the CALL and the ccsrrw masks it.
- **In the ARMED case** a fence after every CALL does not mask it.
- **The FPGA monitor's own post-domcall code** is `ccsrrw sp, cscratch, x0; ldc ra, -16(sp)`. Its LDC depends on the
  ccsrrw's result, unlike the independent LDC here. The monitor's supervised hang is S-16, not this.

## Simulation (the supervised-CALL RTL lane, 2026-10-04)
The armed image hangs there too, deterministically at the first escape, at the same LDC. In simulation the DYN unit's
load syncer has its request SET and waits for an LSU result that never comes, and the load unit never receives the
request. On silicon the syncer flag reads 0 and the LSU reads not-ready. **The two agree on the instruction and
disagree on the state.** The simulation's mechanism is that lane's to report.

## Controls in every session
`call-retpc` (sup-bare-2026-10-02, a05ca464) read PASS exact before every image: N = 15 on this bitstream.
