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
- Board URL token: **`FPGA_URL` environment variable, for the duration of ONE run.**
  The console URL embeds the access token in its path, so it is a credential: never
  commit it, never echo it into a capture, and **never persist it to disk** -- writing
  it to a dotfile (an older revision of this file suggested
  `~/.config/capstone/fpga-board-url`) is still a leak. Ask the user out-of-band each
  time; in committed text write `<FPGA-CONSOLE-URL>`.
- A local `.bit` is NOT needed — re-flash names the **server-side** bitstream
  `working-caplifive-captype-fixed.bit`.

## Running faster + running a suite (transfer tiers)

The old bottleneck was per-run UART transfer. Two levers, in order:

- **Tier-1 `fast_xfer` (DONE, board-validated ~3×).** Use
  `tests/rtl-smoke/fpga_driver/fast_xfer.py` `fast_put` for every domain transfer
  (direct-append base64 chunks, single final-sha guard, safe-retry on mismatch). A
  controller is now ~30 s vs ~4 min. Memory `project_board_transfer_tiers`.
- **Tier-1b BURSTING — do not regress to one emit per character (2026-07-28).**
  `fast_put`'s first tier now sends **16 characters per socket.io emit**, ~15× fewer
  round-trips. It had been emitting **one `uart_send` per character**, so a 6,032-char
  domain cost 6,032 HTTPS round-trips — and the round-trip, not the UART, was the
  wall clock.
  **The char-by-char throttle was solving a real problem, just not this one.** The
  board's ns16550a RX FIFO does overrun on a bulk write and silently drop characters
  (`fpga_console.run_command` records `borrow_cost` arriving as `row_cost`). But that
  bounds **bytes in flight before the UART drains**, not bytes per emit. 16 = the
  FIFO depth, so a burst fills it at most once and drains in ~1.4 ms at 115200 baud,
  against a ~20 ms inter-burst delay.
  Guarded exactly like the existing tiers: whole-file sha after every attempt,
  escalating `burst(16) → fast(1) → safe(1) → safest(1)`. The last three are the old
  behaviour verbatim, so a board that cannot take bursts costs one wasted attempt and
  nothing else. Verified offline that burst=16 emits a **byte-identical** stream to
  burst=1 and never exceeds 16 chars per emit.
  **If a transfer looks slow again, check this first** — the symptom is a log line
  with `burst=1` on the FIRST attempt rather than `burst=16`.
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

## Rebuilding the monitor — SOLVED 2026-07-28, but delete the stale objects first

The monitor **can** be rebuilt now. Every rebuild used to boot-hang with zero serial; the
cause was never the compiler. `fw_jump.o` in the shared build dir had been compiled for the
**FPGA** firmware, where embedding a device tree is mandatory, and `A=opensbi-rebuild` only
relinks — so the QEMU monitor inherited an FPGA DTB and `fw_base.S`'s
`#ifdef FW_FDT_PATH → lla a1, fw_fdt_bin` discarded the DTB QEMU passes. Wrong UART, no
console, no banner.

```bash
cd capstone/caplifive-buildroot
D=build/build/opensbi-custom/build/platform/generic/firmware
rm -f $D/fw_jump.o $D/fw_jump.elf $D/fw_jump.bin $D/fw_dynamic.o $D/fw_payload.o   # REQUIRED
make build A=opensbi-rebuild CAPSTONE_CC_PATH="$(realpath ../capstone-c)"
readelf -sW build/images/fw_jump.elf | grep -c fw_fdt_bin      # MUST be 0
readelf -SW build/images/fw_jump.elf | grep '\.rodata'         # MUST be 002de8, not 003a10
```

**This re-arms every time the FPGA firmware is built in this tree** — same build dir, and
the FPGA side genuinely needs `FW_FDT_PATH`. Treat the `rm` as part of the recipe.
Restore point if anything goes wrong: `~/capstone-b-artifacts/monitor-known-good/`
(`fw_jump.elf.good` md5 `6724bcb3`, plus the known-good `.c.S`). Trail: issue C-11.

## Superseded: "don't rebuild the monitor with our tree's `capstone-c`"

Regenerating `fw_jump.elf` (QEMU) or the FPGA firmware monitor from **our tree's**
`capstone-c` (`master`@`8cda52c`) **boot-hangs** (zero serial). **Known fix
(2026-07-25):** the working firmware is built by `caplifive-system`'s pinned
`capstone-c` = branch **`bugfix`@`508342a`** (divergent from our `master`; carries
a gct-alignment fix). So for any monitor rebuild, build with **that** compiler, not
our tree's. Recovering + pinning this is `plans/monitor-regen-audit-task-B.md` (fast
path); until it's pinned in-tree, **use the existing working prebuilt as-is** for
board runs. Memory `project_opensbi_monitor_rebuild_include_wrapper`.

## Large domains go in the BUILDROOT IMAGE over JTAG, not over UART (2026-07-28)

For anything bigger than a few tens of KB, build it into the FPGA buildroot image and
let it ride the firmware over JTAG. The board owner's answer when asked how to deliver
a large binary was, in substance, *"isn't it built into the buildroot image and loaded
through JTAG? Why do we need UART?"* -- and that is right.

    capstone/caplifive-system/sw/buildroot/overlay/test-domains/   <- put artifacts here
    cd capstone/caplifive-system/sw/buildroot
    make setup
    make build CAPSTONE_CC_PATH=$(realpath ../capstone-c)

The FPGA rootfs is a CPIO **initramfs compiled into the kernel**
(`fpga_defconfig`: `BR2_TARGET_ROOTFS_CPIO`, `BR2_TARGET_ROOTFS_INITRAMFS`;
`fpgakernel.config`: `CONFIG_INITRAMFS_SOURCE=.../rootfs.cpio`), so anything in the
overlay is present at boot with no transfer step at all. Note this is
**caplifive-system's** overlay -- caplifive-buildroot has a separate one that feeds the
QEMU image.

### Why UART is the wrong answer, and two mistakes made getting there

**Do not size a UART transfer from the raw file.** `fast_put` gzips first, so the
transfer is base64 of the COMPRESSED size. Sizing SQLite from its raw 2.27 MB gave
">= 63 min" and led to ruling UART out; the real figure was ~15 min (1.62 MB -> 529 KB
gzipped -> 705 K base64 chars at ~800 chars/s).

**But ~15 min was also wrong, in the other direction.** It assumes zero retries. At
703 K chars the transfer is 1,759 chunks, a dropped character is near-certain, and
`_put_once` **truncates and restarts the whole file** on any failure -- then falls back
to burst=1, which for that size is ~4 hours. Measured: the 529 KB domain wedged the
shell partway through and began exactly that fallback. So UART's limit at this scale is
**reliability, not throughput**, and no amount of pacing fixes a whole-file restart.

If UART ever has to carry something this big, make `fast_put` verify and retry
PER CHUNK instead of per file. Until then: use the image.

## Two things that LOOK like a hang and are not (2026-07-28)

Both cost time in one session; check them before debugging anything.

1. **The board powering off at the end of a run is CORRECT.** `powered off` / `unlocked`
   in the log is the runner's `finally`, and the process exits 0. It is not a crash.
2. **A reboot banner in the UART right after a rung's `BG<rung>` marker is usually the
   NEXT rung's power-cycle, not a wedge.** `Hello World! ... booting! ... OpenSBI v1.3`
   appearing after `domain ID = 0` reads like the domain took the board down; it is the
   runner power-cycling because that rung produced no END marker within 120 s. The actual
   failure is a hang (`cscall` never returns). To tell them apart, look at whether a
   `power-cycle + reload firmware` line precedes the banner.

**And one real hazard: an ad-hoc console script that never calls `disconnect()` stays
alive forever.** The socket.io client thread is non-daemon, so a script that connects,
does its job and falls off the end keeps running and holds a console session — it shows
up as an inflated `users connected:` count and looks like someone else is on the board.
One such script lived 49 minutes. Always `disconnect()`, or run it under `timeout`.

## The board-driver contract (2026-07-31) — read before waiting on ANY run

Three drivers follow it: `run_sqlite_baked_fpga.py`, `probe_sqlite_wedge.py`,
`probe_revnode.py`. All of it exists because "is the board still busy?" was answered
wrongly, repeatedly, and each wrong answer cost either board time or an idle session.

**Required environment.** Both, or the driver dies before touching the board:

| Var | What | Failure if unset |
|---|---|---|
| `FPGA_URL` | console URL **with token** — a credential | `FPGA_URL not set` |
| `FPGA_FW`  | absolute path to `fw_payload.bin` | `KeyError: 'FPGA_FW'` at import |

`FPGA_FW` has no default on purpose — an implicit one silently flashes whatever was built
last. The FPGA firmware is **not** in `build/images/`; it is
`caplifive-system/sw/buildroot/build/build/opensbi-custom/build/platform/fpga/ariane/firmware/fw_payload.bin`
(~17.4 MB). `build/images/` holds only `fw_jump.bin` (~569 KB), which is the QEMU monitor
and will not boot the board. Getting this wrong throws at *import*, i.e. before the run —
so a stale `sqlite-run-scoped.txt` from an earlier session is still sitting on disk and
reads exactly like a fresh result. **Always confirm the driver actually ran before reading
its output file.**

**Completion is signalled, never inferred.** Every driver prints, in this order:

```
RUN_DONE | PROBE_DONE     <- first statement in finally; survives a throwing teardown
BOARD_RELEASED            <- switches -> power off -> unlock -> disconnect, each time-boxed
```

Poll for those strings. **Never poll with `pgrep -f <pattern>`** — the polling loop's own
command line contains the pattern, so it matches itself and spins forever. Six such loops
ran here for up to 21 hours. Bound every wait with `for i in $(seq 1 N); do ...; done`.

**Drivers exit via `hard_exit()`, not `sys.exit()`.** `sys.exit` unwinds the main thread
and then *waits on every non-daemon thread*; socketio's survives `disconnect()` often
enough that it cannot be relied on. Measured 2026-07-31: a run printed `RUN_DONE` and
`BOARD_RELEASED` and then stayed alive emitting `user_count` events — the board was
genuinely free, but from outside it was indistinguishable from a session still in
progress, and it was reported as phantom board activity. `hard_exit` flushes the streams
and calls `os._exit`, preserving the exit code.

**Teardown is time-boxed and ordered least-important-first** (`safe_cleanup.py`): switches,
then power, then unlock, then disconnect. Each step waits on a board event and can block
forever while holding the flock — which strands every session queued behind it. A step
that will not complete is abandoned and logged rather than waited on, because releasing
the board matters more than tidying it. Precedent: `probe_revnode.py` once sat 16 minutes
inside `set_switches(console, 0)` with the board powered and the lock held, after all four
of its LED reads were already on disk.

## Non-negotiables

Lock → power-cycle → run → **power off + unlock in `finally`** (never leave it
locked/on). Verify the resident bitstream is `working-caplifive-captype-fixed.bit`
before measuring. `C_PRINT` (`csrw 0x800`) goes to the **RTL trace**, not the UART —
don't use it as a UART probe. Signal of a live domain = the controller prints its
first line (that's AFTER `IOCTL_DOM_CREATE` returns).
