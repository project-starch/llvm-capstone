# Task 017 phase 3 — DIY: protocol obtained, driver wired + validated on the real board

**Date:** 2026-07-16
**Branch:** capstone-bootstrap-b
**Scope:** additive test tooling only (`capstone/tests/rtl-smoke/fpga_driver/`)

Goal (authorized full autonomy): take the collaborator off the critical path —
get the Socket.IO protocol and the boot image ourselves, wire the driver, and
operate the physical board to run the perf sweep. Token'd board URL supplied at
runtime only; never committed or logged.

## Outcome in one line

Protocol obtained DIY and **verified against live hardware**; the driver drives
the real board (power / JTAG load / reset / UART all confirmed). The one
remaining gate for the cycle-accurate number is a **fresh `fw_payload.bin` with
our embedded-initramfs overlay** — the image already on the console is a stale
SD-rootfs build that does not boot to a shell.

## What was done

### 1. Protocol — Route B (fetch the live client JS ourselves). DONE.

The console serves its own **unminified** client JS at `<url>/static/app.js`
(found via `<script src>` in the page). `curl` + `extract_from_js.py` gave every
event; the call sites gave the payloads. No collaborator JS and no DevTools HAR
were needed. The DevTools-HAR fallback was not required.

**Key finding: the console is a HYBRID, not pure Socket.IO** (the earlier
placeholder assumed pure Socket.IO). Action verbs are HTTP POST to a REST API;
the live stream + a few controls are Socket.IO; completion is per-action state
events. Full map in `fpga_driver/PROTOCOL.md`. Highlights:

- Verbs (REST): `POST /api/images/upload` (multipart), `/api/load-image`
  (`{filename}`, completes via `load_state`), `/api/reset-board`,
  `/api/trace-start` (completes via `trace_result`).
- Socket emits: `power_toggle` (toggle), `uart_send {text}`, `switch_toggle
  {index}` (toggle), `set_auto_shutdown {timeout_seconds, locked}` (the Lock).
- Socket listens: `uart_data {seq,text}`, `load_state`, `trace_result`,
  `power_state`, `switch_state`, `auto_shutdown_state`, `user_count`, ...
- Connection: `io({ path: '/<token>/socket.io' })`; REST base `<url>/api`; the
  driver derives both from the URL path so the token never enters the tree.

### 2. Wiring — real code change, not just config. DONE, mock green.

Because the real protocol is a hybrid, the wire-up touched `fpga_console.py`
(added a `requests`-based HTTP layer; toggle-with-verify for power/switch;
per-action state waits; Lock + user-count good-citizen helpers), `config.py`
(new `HTTP` action table + real `EMIT`/`LISTEN`), `mock_server.py` (rewritten to
serve REST endpoints + the socket stream), `run_rtl_smoke.py` (multipart upload;
Lock + back-off), and `test_dryrun.py`. `PROTOCOL_SOURCE = "verified"`.
`test_dryrun.py` passes green against the rewritten mock (5 actions + end-to-end
sweep + `--parse-uart` reproduces bump 7 / norevoke 60 / revoke 65 → +5 O(1)).

`requests` is already a dependency of `python-socketio[client]`, so the HTTP
layer added no new requirement.

### 3. Live validation against the real board. DONE.

- **Transport check (non-destructive):** connected to `fpga.corank.info`, received
  every state event with the **exact** payload shapes wired, read live UART.
  `user_count = 1` (board free), power off.
- **Action chain (good-citizen: Lock → run → release):** power-on, `load-image`
  over JTAG (~2 min), and `reset` all work on hardware. Two real behaviours found:
  - `load-image` returns **409 `Already loading`** while a load is in flight, and
    a load issued immediately after power-on can transiently error — the driver
    now settles/serialises around this.
  - **`reset` drops the Socket.IO connection.** python-socketio reconnects, but
    the reconnected session must re-`request_history` or it misses the post-reset
    boot log. **Fixed:** a `connect` handler now re-emits `request_history` on
    every (re)connect (this had stalled the first recon).

### 4. Boot image. Domain binaries BUILT; `fw_payload.bin` is the remaining gate.

- The `.user`/`.dom` binaries built cleanly (`build-borrow-cost-fpga.sh`,
  `build-revoke-cost-fpga.sh`) → `$CAPSTONE_TMP_ROOT/capstone-rtl-smoke/`.
- **No gh-auth wall:** all five `caplifive-system` submodules (captainer-buildroot,
  capstone-c, caplifive-cva6, caplifive-qemu, anvil) clone **without auth** — the
  anticipated private-submodule gate does not exist.
- **The on-board `fw_payload.bin` (2026-05-25) does not boot to a shell:** after
  JTAG-load + reset the UART ends with `could not initialize sd... exiting` — it
  expects an SD-card rootfs. Our FPGA flow instead needs `LINUX_PAYLOAD=1` with an
  **embedded initramfs** carrying our `/root/rtl-smoke` overlay. So there is no
  runtime-push shortcut; we must build our own image.
- The FPGA image build lives in the umbrella's `sw/buildroot`
  (`captainer-buildroot`, `configs/fpga_defconfig`, `PLATFORM=fpga/ariane`): a
  deep nested-buildroot compile (`make setup && make build && make build
  LINUX_PAYLOAD=1`) producing `.../firmware/fw_payload.bin`, with our binaries in
  `BR2_ROOTFS_OVERLAY`. **Open risk:** the umbrella pins its own OpenSBI monitor
  (`components/opensbi`); it must be recent enough to support our domains
  (csdrop / LINEAR row-11/12) or the built image won't run them.

## The end-to-end run (once the image exists)

Board free, driver verified. The command is:

```
python fpga_driver/run_rtl_smoke.py \
    --url '<token URL, runtime only>' \
    --image <path>/fw_payload.bin
```

It takes the Lock, backs off if `user_count > 1`, uploads + loads + resets, runs
the borrow + 3 revoke configs over UART, harvests RESULT lines, feeds
`--parse-uart`, and releases the Lock.

## Guardrails honoured

Token never entered the tree (runtime `--url`; stashed only in a scratchpad file
outside the repo; redacted in all logs). Good citizen: Lock while running,
back-off check, released the Lock, left the board powered (not off) so
auto-shutdown reaps it. Additive test tooling only — no `llvm/`, monitor, RTL, or
submodule-bump changes.
