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
- **Batch many domains in ONE session — BLOCKED today by the multi-domain hang.**
  The firmware JTAG-load (~2 min, 15 MB) dominates, so in principle you boot once
  and loop `fast_put dom → run → read mcycle → next`. **But a second domain reused
  at the same entry VA (`0x10000`) within one boot silently hangs its `cscall`** —
  the missing domain-boundary `fence.i` / icache-coherence gap (RTL does no icache
  invalidate on the switch). Until that monitor fix lands, each rung must run as the
  *first* domain of a clean boot ⇒ **one full power-cycle + firmware reload per
  rung (~2.5 min each)**. This per-rung cost is the real board-time bottleneck, and
  **no transfer tier removes it** — the unlock is the `fence.i` domain-boundary fix
  (`plans/curried-crunching-gizmo.md`), which is a monitor change gated on
  monitor-regen (see below).
- **Tier-2b / "route B" (JTAG `load_image` + resident/baked controller)** — the
  suite/SQLite scaling path. Two variants, and a key clarification learned 2026-07-25:
  - **Route A — live poke** (gdb `monitor load_image dom <addr>` into a reserved RAM
    region): **confirm the reserved-region address/size with the board owner before
    use — never guess a RAM address** (else it stomps the booted kernel). This is what
    the "never guess an address" rule guards.
  - **Route B — bake domains into the firmware initramfs and reload the whole image**
    ("recompile the image"): needs **no** reserved address (the `fw_payload` has a
    built-in initramfs), but requires a firmware-image rebuild ⇒ **gated on the
    monitor-regen boot-hang** (below).
  - **Neither route cuts the per-rung power-cycle cost** — both still reload the image
    per boot and both still hit the same-VA multi-domain hang. Tier-2b is the on-ramp
    for domains too big for Tier-1 transfer (SQLite-scale), **not** a speedup for a
    handful of tiny integer rungs (Tier-1 in one session is right for those). Design:
    `plans/sqlite-on-silicon-scoping.md` §"Delivery mechanism".

- **GOTCHA — never run stale domain binaries.** The ladder-perf runner reuses an
  existing `<rung>.dom` and does **not** rebuild it. On 2026-07-25 this made a sweep
  run pre-fix binaries and report 4 bogus "silicon miscompiles" that were actually an
  already-fixed compiler bug. **Delete `$OUT_DIR/ladder-fpga/*.dom` (or force-rebuild)
  before every sweep** so the current compiler is exercised.

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
