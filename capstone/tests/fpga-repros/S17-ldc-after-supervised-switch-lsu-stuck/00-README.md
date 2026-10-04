# S-17 — on silicon, an LDC right behind the post-CALL `ccsrrw sp <- cscratch` does not complete, and the LSU stays not-ready

**S-17: NOT REPRODUCED on `caplifive_supcall_715bdd1fe.bit`.** arm12-ldc ran 3 of 3, 8,552 escapes each, identical
to its `ld` twin; it hung on 36a641e0b.
- **Mechanism unknown.** Neither R-49 nor R-50 produces the state read here. Their end states read WAIT_STORE_READY
  with count 4, and lsu_ready 1 with DYN waiting, respectively.
- A timing-marginal path moved by the rebuild cannot be excluded.
- Apertures 219..222 (the LSU bypass head, the commit queue, the adapter's tag FSM) remain armed for its return.

**Sibling, a different signature: [S-16](../S16-supervised-switch-never-finishes/).** There a supervised CALL's switch
never finishes: aperture 224 = `0x1f` with 225 = `0x88` (dom_switch_busy 1). This issue reads 224 = `0x0d` together
with 225 = `0x80`. Use both bytes to decide. 225 = `0x80` alone is the resting value of many unrelated wedges.

## The signature
Bitstream `caplifive_supcall_36a641e0b.bit`, the supervised-CALL bitstream. Its CSR 0xFC4 (csnodefree) exists: it read
0xFFCD in the Linux identification boot and reads 0xFFFC in these bare images. The LED-path apertures after the hang
(labels from `cva6.sv` at capstone-ariane 36a641e0b, MSB first):

| aperture | value | meaning |
|---|---|---|
| 224 {ex_commit.valid, load_syncer_req_set, store_syncer_req_set, lsu_ready, dyn_ready, flu_ready, flush, privM} | **0x0d** | **LSU not ready**; DYN and FLU ready; no syncer request; no flush; no exception at commit; M-mode |
| 225 {trace_buf_empty, wait_store_syncer, wait_load_syncer, wait_rev_res, dom_switch_busy, stall_issue, mem_write_flag, mem_wait_flag} | **0x80** | no domain switch in progress; no wait flags |
| commit pc (230..237) | the `ccsrrw sp, cscratch, x0` at CALL + 4 | the next instruction is the LDC |

The commit-pc aperture shows scoreboard slot 0 whether or not it holds a live instruction (`scoreboard.sv:149`), so
it names an instruction position, not a live head.

## The matched pair (`images/`, `results/result-lines.txt`)
Bare M-mode images. Rebuild them with `src/build.sh` from `src/sup-capstl.S`; `images/SHA256SUMS` matches. JTAG load
at 0x80000000.
- A supervised domain is preempted every 64 cycles, and the monitor resumes it.
- After every CALL the monitor runs `ccsrrw sp, cscratch, x0`, then a load from a swap area it filled before the CALL.

| image | that load | result (one run each) |
|---|---|---|
| `arm12-ld-q64.bin` (sha256 `8027a661a5972427`) | `ld t1, 464(s10)` | COMPLETED, 8,552 escapes, every reading exact |
| `arm12-ldc-q64.bin` (sha256 `07de9fb7af896dd5`) | `ldc t1, 464(s10)` | HANGS at the first escape, signature above |

The two images differ in exactly that one instruction, at both of its sites, plus the test-ID character in the
prologue: 3 words in all. Since the `ld` image completes, the LSU's not-ready state is caused by the LDC itself.

**Not only supervised.** `mswapfix-plain-noploop.bin` (sha256 `ffd6f6db241db2e1`) runs the full swap sequence (all four
parts, see below) around ONE plain, un-armed CALL. Its first two post-CALL instructions are the same (`ccsrrw sp,
cscratch`, then an LDC), and it hangs with the identical signature, commit pc on the post-CALL ccsrrw. The hang:
3 runs. The signature: read once.

## What the bisection of that sequence showed (images and scripts in `../../rtl-smoke/sup-resume-2026-10-03/`)
These are plain (un-armed) runs, one each unless stated.
- **It needs the post-CALL `ccsrrw sp <- cscratch` followed by an LDC.**
  - Without the ccsrrw, the post-CALL LDC completes.
  - With the ccsrrw followed only by a UART (MMIO) poll, it completes.
  - With an `ld` in place of the LDC, it completes (the pair above, armed).
- **A `fence` right after the CALL masks it.**
- **8 `nop`s there do not.** That hang's signature was not read.
- **A UART print between the CALL and the ccsrrw masks it.**
- **In the armed case, with a fence after every CALL, the run still hung.** Its signature was not read, so it may be
  S-16 rather than this.
- **The FPGA monitor's own post-domcall code** is `ccsrrw sp, cscratch, x0; ldc ra, -16(sp)`. That LDC depends on the
  ccsrrw's result, unlike the independent LDC here.

## Simulation
The supervised-CALL RTL lane ran this folder's source (`sup-capstl.S`, the armed independent-LDC shape) in RTL
simulation on 2026-10-04. It hangs there too, deterministically at the first escape, at the same LDC. In simulation
the DYN unit's load syncer is SET and waits for an LSU result that never comes. On silicon the syncer flag reads 0
and the LSU reads not-ready. **The two agree on the instruction and disagree on the state.** The mechanism is that
lane's to report.

## Controls
`call-retpc` (sup-bare-2026-10-02, a05ca464) ran first in every board session of this investigation and read PASS
exact each time.
