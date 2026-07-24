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

## Running faster + running a suite (transfer tiers)

The old bottleneck was per-run UART transfer. Two levers, in order:

- **Tier-1 `fast_xfer` (DONE, board-validated ~3×).** Use
  `tests/rtl-smoke/fpga_driver/fast_xfer.py` `fast_put` for every domain transfer
  (direct-append base64 chunks, single final-sha guard, safe-retry on mismatch). A
  controller is now ~30 s vs ~4 min. Memory `project_board_transfer_tiers`.
- **Batch many domains in ONE session.** The firmware JTAG-load (~2 min, 15 MB)
  dominates — pay it **once**. Boot once, then loop `fast_put dom → run → read
  mcycle → next`. Domain binaries are tiny; only the firmware is big.
- **Tier-2b (JTAG `load_image` into reserved RAM + resident controller)** — the
  suite/SQLite scaling path: load → run → read `mcycle` → load next, many domains
  per session, no reflash, no UART transfer. Board owner's endorsed "domain in the
  image, loaded over JTAG" model. **Confirm the reserved-region address/size with
  the board owner before use — never guess a RAM address.** Not needed for a handful
  of tiny integer domains (Tier-1 in one session suffices); it's the on-ramp to
  SQLite-on-silicon. Design: `plans/sqlite-on-silicon-scoping.md` §"Delivery
  mechanism"; the UART-baked-controller variant (initramfs) is postponed
  (`project_board_transfer_tiers`).

## DO NOT rebuild the monitor / firmware

There is a confirmed **toolchain gap**: regenerating `fw_jump.elf` (QEMU) or the
FPGA firmware monitor from the current `capstone-c` **boot-hangs** (zero serial);
the working firmware is an unreproducible prebuilt (older compiler state, smaller
frames). **Use the existing working prebuilt as-is.** Fixing the regen is a
separate workstream (`plans/monitor-regen-audit-task-B.md`) that unblocks
large-`.rodata`/SQLite on silicon. Memory
`project_opensbi_monitor_rebuild_include_wrapper` (WARNING section).

## Non-negotiables

Lock → power-cycle → run → **power off + unlock in `finally`** (never leave it
locked/on). Verify the resident bitstream is `working-caplifive-captype-fixed.bit`
before measuring. `C_PRINT` (`csrw 0x800`) goes to the **RTL trace**, not the UART —
don't use it as a UART probe. Signal of a live domain = the controller prints its
first line (that's AFTER `IOCTL_DOM_CREATE` returns).
