# How to launch something on the FPGA (short instruction)

Paste this to an agent (or follow it yourself). The board is a browser/websocket
CVA6+Capstone FPGA — **no SSH**; an agent drives it via the Python driver, a human
can also use the browser GUI. Every step and gotcha is in the KB files below.

## One-line instruction to give an agent

> Run `<domain/binary>` on the Capstone FPGA. Read
> `capstone/agent-handoff/ref/HOW-TO-LAUNCH-ON-FPGA.md` first, then the runbook
> `ref/gp-free-silicon-smoke-runbook.md` and KB
> `history/22-07-2026_18-05-00_gp-free-silicon-smoke-firmware-fixed-createdomain-hangs.md`.
> Build the firmware with the recipe in memory `project_fpga_fw_payload_build_recipe`,
> lock the board, power-cycle, boot, transfer + run, harvest the result, then
> **power off + unlock**. Board is flaky/slow (~2 min to JTAG-load 15 MB); one
> persistent write only (bitstream re-flash) and only if authorized.

## The 4 KB files (read in this order)

1. `ref/HOW-TO-LAUNCH-ON-FPGA.md` — this file (the map).
2. `ref/gp-free-silicon-smoke-runbook.md` — step-by-step runbook (build domain +
   controller, boot, run).
3. KB `history/22-07-2026_18-05-00_gp-free-silicon-smoke-firmware-fixed-createdomain-hangs.md`
   — full findings: firmware traps + fast build recipe, the create_domain root cause,
   exact reproduction, board etiquette.
4. Memory `project_fpga_fw_payload_build_recipe` — the exact firmware relink recipe
   (fpga/ariane from caplifive-system, embed FDT+kernel, regen `.c.S`). Plus
   `project_silicon_gp_delivery_boardowner_guidance`, `reference_fpga_rtl_platform`.

## Tooling (already on disk)

- Driver + venv: `/tmp/capstone/fpga-venv/bin/python`, drivers `board_run_*.py`
  (boot fw → UART-transfer gzip+base64 sha-verified → run → harvest),
  `board_reflash_only.py` (re-flash only). Protocol: `tests/rtl-smoke/socketio-api.md`,
  `tests/rtl-smoke/fpga_driver/`.
- Board URL token: `~/.config/capstone/fpga-board-url` (secret; never commit/echo).
- A local `.bit` is NOT needed — re-flash names the **server-side** bitstream
  `working-caplifive-captype-fixed.bit`.

## Non-negotiables

Lock → power-cycle → run → **power off + unlock in `finally`** (never leave it
locked/on). Verify the resident bitstream is `working-caplifive-captype-fixed.bit`
before measuring. `C_PRINT` (`csrw 0x800`) goes to the **RTL trace**, not the UART —
don't use it as a UART probe. Signal of a live domain = the controller prints its
first line (that's AFTER `IOCTL_DOM_CREATE` returns).
