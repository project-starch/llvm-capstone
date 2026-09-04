# Single-step diagnosis: the gp-free measurement runs FULLY on silicon; only the domain teardown (domreturn) resets

**Date:** 2026-07-21
**Board:** captype-fixed, image `fw_payload_fpga_up_ctl.bin` (fe37ebdb), gdb-boot.
**Method:** stall-probe bisection (hardware breakpoints are unavailable on this
CVA6 debug module — "too many hardware breakpoints"; and a *resetting* domain can't
be caught by B's sleep-then-halt, which only works on a *stall*). Instead each
probe inserts a `j .` spin at a chosen point in a gp-free domain build and either
halt-reads `$pc` (parked-in-domain ⇒ reached the probe) or watches the UART for a
reset banner (banner ⇒ reset before the probe). Scratchpad drivers:
`/tmp/capstone/board_probe*.py`. Board left powered off + unlocked.

## Result — the whole measurement works on real silicon

| Probe | Spin point | Verdict |
|-------|-----------|---------|
| 1 | top of `test` (after `__test_entry`) | PARKED — `ccsrrw/lcc/**split**/scc/delin(sp)` all execute |
| 2 | `domain_main` entry | PARKED — the plain **`call domain_main`** (auipc+jalr) works in PRV_C |
| 3 | after `csrr mcycle` | PARKED — **`mcycle`** reads fine in the domain (PRV_C) |
| 4 | after `mrev/delin/revoke` | PARKED — SHRINK, all region access, copy loops, and the **borrow ops** execute |
| 5 | after `fpga_write_results` (full run) | HUNG, no banner — the **full measurement + writeback COMPLETE**; results ARE written to the region |
| 6 | glue, after `call` returns | HUNG, no banner — the plain **`ret`** returns to the glue |

**Conclusion:** every novel piece of the gp-free approach works on silicon — the
gp-free entry glue, `split`, the plain call/ret ABI, `mcycle`, `SHRINK`
(object-granularity narrowing), all shared-region access, the copy loops, and the
**revoke-at-free temporal-safety ops themselves (`mrev`/`delin`/`revoke`)**. The
borrow-cost measurement runs to completion and writes its 8 result slots to the
region. **This is the first time a Capstone temporal-safety domain measurement has
executed on hardware.**

The board reset in the real domain is localized to exactly one thing: the **domain
teardown** — the glue register-clear + `ccsrrw cscratch` + mcause/mtval save +
**`domreturn(t1, t2, x0)`** — which runs *after* the results are already written.
Probes 5 and 6 (which replace the ret/teardown with a spin) do NOT reset; the real
domain (which executes `domreturn`) does.

## Why this matters / why the number isn't captured yet

The controller reads the 8 results back only *after* the REGION_SHARE ioctl
returns, and that return happens via the domain's `domreturn`. Since `domreturn`
resets the board, the ioctl never returns and the controller (blocked in it) can't
read the already-written results. So: the number is **computed and written on
silicon**, but not yet **extractable**, purely because of the teardown reset.

Note: no domain in this lineage has ever cleanly `domreturn`ed on this board+image
(the old domain stalled at `delin gp` at entry; ours is the first to reach the
exit). So the teardown/`domreturn` path is newly exercised here and is the next
thing to fix.

## Next step (targeted — to extract the number)

1. Bisect the teardown: spin **before** `domreturn` (after the register-clear +
   cscratch/mcause/mtval save). Parks ⇒ the cleanup is fine and `domreturn` itself
   resets; resets ⇒ a cleanup op faults. (One board cycle, UART-banner method.)
2. If `domreturn` is the culprit: diff our glue's exit against a reference domain's
   (`capstone-test-domains` fib/thread/smode `.dom.S`) `domreturn` sequence — in
   particular the sealed-return-cap handling (`ldc t1, sp, 48`), the `cscratch`
   swap, and the fact we dropped the old `stc gp, sp, -16` save. Match the
   reference exit exactly.
3. Alternative extraction without fixing the exit: the results are in the region's
   physical pages while a probe5-style build spins; read them via gdb from the
   region's physical address (instrument the controller to print its
   `map_region` mmap offset / the monitor's region paddr).

## Update (2026-07-21, later): domreturn confirmed; extraction domain built

- **probe6**: the plain `ret` (jalr zero,0(ra)) returns to the glue fine (no reset).
- Adding the missing `stc(x0,sp,-16)` context slot to match the reference exit did
  **not** fix it — **`domreturn` itself resets this RTL**. Ours is the first domain
  in the lineage to actually reach `domreturn` on this board, so it is newly
  exercised (the reference domains' exit may never have run here either).
- **Extraction domain** (`/tmp/capstone/extract.c` → `nogp_extract.dom`): runs the
  full measurement, `fpga_write_results`, then loads the 8 slots into `s2..s9`
  (s2=iters..s9=copy2_bytes) and spins at paddr `0x819a03e0` — **no `domreturn`**.
  Confirmed on board: it **spins** (no reset), so the numbers are computed and held
  in registers on silicon. Reading them needs one clean `gdb halt` of the parked
  core (`p/x $s2..$s9`), then `raw=(s4-s3)/s2`, `borrow=(s5-s3)/s2`, etc.
- **Blocked on board infra**, not on our code: after ~16 cycles the console
  degraded — DTM "Examination failed" on gdb re-attach, then power-on timeout, then
  HTTPS connection timeouts (console unreachable). Board left off + unlocked.
  **Resume:** `/tmp/capstone/board_extract.py` (keep-gdb-attached + Ctrl-C halt)
  once the console recovers.

## Infra notes (for the next board session)

- Hardware breakpoints: unavailable ("too many hardware breakpoints/watchpoints").
  Use the `j .` stall-probe + UART-banner method, not `hbreak`.
- The websocket drops frequently mid-transfer; the driver needs the
  reconnect-resilient `_emit` wrapper (in `board_run_nogp.py`). gzip+base64 +
  per-chunk sha256 keeps transfers short and self-verifying.
- `monitor halt` on a *reset/rebooting* core fails ("Hart 0 failed to halt during
  examine") — that failure is itself a signal the domain reset. Prefer the
  UART-banner method to distinguish park vs reset.
