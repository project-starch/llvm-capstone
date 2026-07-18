# Task 017 — fence.i board confirmation: domain CALL still resets the CVA6 (fix NOT sufficient)

**Date:** 2026-07-18
**Branch:** capstone-bootstrap-b
**Scope:** watched board run of the patched UP built-in `fw_payload_up_builtin_fence.bin`
(sha `9c53ffd8…`) over the FPGA web console (gdb-boot, non-persistent). No SPI/flash,
no submodule-source or RTL commits. Additive driver hardening + this note only.

## Headline

The `fence.i` monitor patch is real, correct, and regression-clean under QEMU — but
on the actual CVA6 it does **NOT** unblock the Capstone domain CALL. `create_dom`
succeeds; the `cscall` then **resets the core to the bootrom** before any measurement
prints. No `RESULT`/cycles line was ever produced on the board (0 in 4318 UART
records / ~20 boots). So the missing-`fence.i` hypothesis was necessary-but-not-
sufficient at best — there is a deeper fault in the domain-switch path on this HW.

## The decisive, attributable run (attempt 7)

Freshly **power-cycled** (off 8 s → on → 15 s DTM settle → clean JTAG attach, tap
`0x00000001`), clean gdb-boot of the fence image, **working root shell first try**:

```
buildroot login: root
# echo RDY''OK
RDYOK                                              <- shell confirmed (login is fine)
# ls -la /dev/capstone
crw-rw-rw- 1 root root 10, 127 ... /dev/capstone   <- built-in, present at boot
# ./revoke_cost_fpga.user ./revoke_cost_fpga_norevoke.dom
Ok, good file. ... Loadable executable segment found. Entry address = 10000
[262.77] Domain memory region vaddr = ffffffd801a00000, paddr = 81a00000   <- create_dom OK
[262.79] code size = 525472, tot_size = 100000, entry_offset = 0
revoke-cost-fpga^@Hello World!                     <- dom label, then BOOTROM banner
Hit any key to enter update mode .. booting!       <- board RESET
```

`^@Hello World!` + `Hit any key to enter update mode .. booting!` is the CapliFive
bootrom — i.e. the domain CALL transferred control to the reset vector. After the
reset the bootrom reloads the **662 KB SPI-resident** firmware and boot-loops; that
post-CALL reset is what made the driver's `login_root` intermittently report "could
not confirm a shell" on earlier attempts (it was racing the reset, not a login bug).

Reproduced identically 3× more in the same capture's history (all under the 222 KB
gdb-loaded firmware): borrow/revoke label → bootrom → reset. Never a RESULT.

## Why earlier attempts looked like flakiness (they weren't the real story)

Six attempts before the clean one surfaced four *separate*, real driver/board
fragilities (all now fixed in the driver, see below), which masked the underlying
result:
1. cold power-on → immediate JTAG attach → OpenOCD "scan chain … all ones" (DTM not
   ready). Fixed: `POWER_ON_SETTLE` after a cold power-on.
2. `monitor reset halt`'s JTAG-interrogation output lags its `(gdb)` prompt → the
   next gdb_cmd matched a stale prompt and fired the DTB load while the 15 MB image
   load was still running → timeout. Fixed: settle after reset-halt.
3. a crashed run orphaned the GDB/OpenOCD session ('running'), tying up the single-
   threaded console → lock handshake timeout. Fixed: reap stale session before lock
   + `gdb_stop` in the error path.
4. `login_root` sent `root`/`echo` as bulk writes → UART RX FIFO overrun. Fixed:
   throttle the login keystrokes. (Login then worked first try in attempt 7.)

Only after clearing all four did the true signal show through: the CALL resets.

## Attribution caveat (now closed)

The fence patch changes only OpenSBI, not the kernel, so banner/kernel-stamp/firmware
size (222 KB) are identical between the fence and non-fence built-in images — earlier
history-replayed dom runs could not be told apart. Attempt 7 closes this: a *clean,
freshly-power-cycled* boot of the fence image, shell confirmed, still resets on the
CALL. Attributable.

## Suggested next differential (not run — needs a call)

The 662 KB SPI-resident firmware the bootrom loads on reset is the board's own
reference monitor. Boot it (normal reset-board, no gdb) and run its borrow/revoke
dom: if it ALSO resets on the CALL → the fault is RTL/board (escalate); if it prints
a RESULT → the fault is specific to our monitor build (compare our sbi_capstone.S /
domain-switch asm against that image's monitor beyond just fence.i). This is the
cleanest way to split "RTL vs our monitor" and should precede any further monitor
patching.

## Artifacts

- `~/capstone-b-artifacts/board-run-fence-7.uart.txt` (clean attributable run, 227 KB)
- `~/capstone-b-artifacts/board-run-fence-6.uart.txt` (prior run, 225 KB)
- Driver hardening in `capstone/tests/rtl-smoke/fpga_driver/run_rtl_smoke.py`
  (`--power-cycle`, `POWER_ON_SETTLE`, stale-gdb reap, reset-halt settle, throttled
  login, UART-capture-on-failure). Dry-run green.

See also `18-07-2026_14-38-40_fpga-domain-call.md` (root-cause hypothesis) and
`18-07-2026_16-32-29_fpga-domain-call-rebuild.md` (build + QEMU validation).
