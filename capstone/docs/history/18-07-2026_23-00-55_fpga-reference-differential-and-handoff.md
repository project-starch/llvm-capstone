# Task 017 — reference differential inventory + RTL hand-off packet

**Date:** 2026-07-18
**Branch:** capstone-bootstrap-b
**Scope:** attempt the RTL-vs-monitor differential (boot the board's own SPI-resident
reference monitor via `reset-board`, run a dom); write a token-free, impersonal RTL
repro packet. Non-persistent (no SPI/flash), no submodule-source/RTL commits.

## Differential could NOT be run on this board — no self-booting reference

Booted the board with a plain `reset-board` and **no** JTAG load (new `--no-load`
driver flag), so the bootrom would boot whatever firmware is resident. Result, twice,
identical, from a clean power-cycle:

```
Hello World!
Hit any key to enter update mode .. booting!
init SPI
status: 0x0000000000000025
status: 0x0000000000000025
SPI initialized!
initializing SD... 
could not initialize sd... exiting          <- bootrom halts here; no OpenSBI/Linux
```

The board's autonomous boot path loads the OS image **from an SD card**, and SD init
fails every time (`could not initialize sd... exiting`), so the bootrom halts — no
OpenSBI, no Linux, no login (driver waited the full 180 s). **This is exactly why the
gdb-boot method exists**: the board only runs images we JTAG/gdb-load into DRAM.

Consequence: there is no self-booting reference monitor to point a dom at, and the
reference monitor (`genesys-testing`, OpenSBI `99aaffa8`) is not available here as a
loadable image file. So the differential cannot be completed on this board. Per the
task's own branch ("if it can't exercise a CALL at all, don't force it"), we hand off
on the fence-run evidence and leave the differential to be run on a board that can
boot the reference (working SD/boot path + the reference image).

The "662 KB firmware" seen earlier (in the fence gdb-runs, after a dom-CALL reset)
was a transient warm-reset residual in DRAM, not a reliably bootable reference — it
never appeared from a clean `reset-board`.

## What the escalation rests on (fence-run evidence, attempt 7, attributable)

Clean, freshly power-cycled gdb-boot of `fw_payload_up_builtin_fence.bin`
(sha `9c53ffd8…`): working root shell first try, `/dev/capstone` at boot, then:

```
# ./revoke_cost_fpga.user ./revoke_cost_fpga_norevoke.dom
[262.77] Domain memory region ... paddr=81a00000   <- create_dom OK
[262.79] code size = 525472, tot_size = 100000, entry_offset = 0
revoke-cost-fpga^@Hello World!                      <- dom label, then bootrom
Hit any key to enter update mode .. booting!        <- domain CALL resets the core
```

`create_dom` succeeds; the CALL resets to the bootrom before any `RESULT`. Reproduced
4× across the capture history; 0 `RESULT`/cycles lines in ~20 boots. fence.i is
correct and regression-clean under QEMU (borrow raw2/6, revoke 7/60/65) but does not
unblock the CALL on silicon.

## Driver change

`--no-load` (reset boot-method only): skip the JTAG image load, just `reset-board`,
so the bootrom boots the SPI-resident firmware. Threaded `load_via_jtag` through
`run_smoke`/`boot_board`. Additive, dry-run green.

## Deliverable

Token-free, impersonal, shareable RTL repro packet at `/tmp/capstone/RTL-ESCALATION-REPRO.md`
(kept out of the repo intentionally — not committed). Contains the image + sha, the
gdb-boot recipe and the reset-board note (the reference needs a working SD/boot path
or a gdb-load), the exact `.user`+`.dom` pair, the observed reset sequence, the QEMU
pass baseline, and what a pass looks like.

## Artifacts

- `~/capstone-b-artifacts/board-run-reference{,-2}.uart.txt` (reset-board SD-fail, 201 B each)
- `~/capstone-b-artifacts/board-run-fence-7.uart.txt` (clean attributable fence reset run)
- Prior trail: `18-07-2026_14-38-40_fpga-domain-call.md`,
  `18-07-2026_16-32-29_fpga-domain-call-rebuild.md`,
  `18-07-2026_21-44-37_fpga-domain-call-fence-board-result.md`.
