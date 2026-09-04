# Board run of the gp-free domain: binaries run, domain created, RESET at the cscall entry

**Date:** 2026-07-20
**Board:** Genesys2 CVA6 `working-caplifive-captype-fixed.bit`, image
`~/capstone-b-artifacts/fw_payload_fpga_up_ctl.bin` (sha `fe37ebdb`), gdb-boot.
**Driver:** scratchpad `/tmp/capstone/board_run_nogp.py` (reconnect-resilient;
gzip+base64 UART transfer, sha-verified per chunk; foreground run bracketed by
BEGIN/END markers). Board left powered off + unlocked.

## What ran

The gp-free / cjalr-free binaries (commit `aada422`) were shrunk (strip + gzip →
~4.4 KB base64), UART-transferred to `/tmp`, and verified on-board by sha256:
- `/tmp/nogp_ctl`  sha `c28e3285ea21b4aa` ✓
- `/tmp/nogp.dom`  sha `8c6d872c65b6f941` ✓

Running `/tmp/nogp_ctl /tmp/nogp.dom` reproducibly (4/4 attempts) reaches:

```
borrow-cost-fpga: created domain ID = 0
borrow-cost-fpga            <- start of the "region = " line, then:
Hit any key to enter update mode .. booting!   <- BOARD RESET (bootrom)
init SPI ... OpenSBI v1.3 for Capstone
```

i.e. `/dev/capstone` opens, `create_domain` succeeds (domain ID 0), and then the
board **resets** — the bootrom/OpenSBI banner appears — at the point the
controller shares the region, which is the domain's first `cscall` entry.

## Interpretation

- This is a **reset**, not the old `delin gp` **stall**. The gp fix is real: our
  domain no longer has `delin gp`, so it does not hang the pipeline the way the
  old one did (`history 20-07 *_fpga-borrow-cost-reproduction` §7). It gets
  *further conceptually* (no gp dependence) but the entry still fails on silicon.
- **Why a reset and not a cap-trap:** B's gp-seed experiment established that this
  RTL's **fast first-entry (c-effective) path does not restore `ctvec`** for the
  entered domain — `ctvec` arrives `0`. So any in-domain fault at entry has no
  handler (ctvec=0) and the core resets to the bootrom. The old domain *stalled*
  (delin gp never retires, no trap); ours apparently *traps*, and the trap with
  ctvec=0 → reset.
- **Prime suspect: `split`.** The old entry glue (`my_first_domain/start.S`) that
  reached `test`/`delin gp` used only `ccsrrw / lcc / scc / delin(sp)` at
  `__test_entry` — all of which demonstrably work on this RTL (the old domain got
  past them). Our `start-fpga-nogp.S` adds ONE new instruction there:
  `split(a2, sp, t3)` (carving the linear scratch cap for `measure_borrow`). If
  `split` traps on this RTL (or produces a cap that a later op faults on), the
  ctvec=0 path resets the board. Other candidates not yet excluded: the plain
  `call domain_main`, or an op inside `domain_main`, or the `cscall` mechanics
  themselves under ctvec=0.

## Next step (single-step diagnosis — the method B proved)

Pin the faulting instruction exactly (do NOT guess further):
1. gdb-boot the image, run the controller, `time.sleep` to the cscall, `monitor
   halt`, then `stepi` from the domain entry (`0x819a00xx`) reading `$pc` each
   step — find the instruction that resets/does-not-retire (B's `run_singlestep.py`
   / `run_gpprobe.py` pattern; see the reproduction runbook §7).
2. If it is `split`: try a `split`-free `measure_borrow` — carve the scratch with
   `scc`+`tighten` (no linearise) or measure the borrow op on a region-derived
   cap; re-verify on QEMU first (revoke cost is provenance-independent, so the
   measured op is unchanged).
3. Independently worth resolving: the **ctvec=0 first-entry** gap — even a correct
   domain has no fault handler on entry, so any fault resets. Restoring `ctvec`
   for the entered domain (monitor `create_domain` seal slot, or the RTL
   c-effective path) would turn resets into diagnosable cap-traps.

## Status

- Durable result (committed, QEMU-validated): the gp-free/cjalr-free domain runs
  end-to-end on QEMU — raw=2, borrow=6 (commits `aada422`, `adafbca`).
- Silicon: binaries transfer + run + create the domain; the measurement's first
  cscall resets the board. The cycle number is **not yet captured** — it needs the
  single-step diagnosis above. Board is a shared, flaky resource (frequent
  websocket drops); each boot+transfer cycle is ~10 min.
