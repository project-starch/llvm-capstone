# FPGA web-console automation — DIY Socket.IO driver scaffold (task-017)

**Branch:** `capstone-bootstrap-b`. Additive test tooling only; no `llvm/`,
submodule, monitor, `start.S`, allocator, or RTL-tree changes.

## Goal

Remove the human from the RTL perf sweep. The `tests/rtl-smoke/` run (borrow-cost
+ the three revoke-cost configs) is fully staged and QEMU-validated, but the board
console (`fpga.corank.info`) is a browser Socket.IO GUI with no scriptable API, so
each run needs ~5 button clicks. Build a headless `python-socketio` driver that
performs the five board actions and drives the sweep into the existing
`run-revoke-cost-fpga-qemu.sh --parse-uart`.

Time-sensitive framing: the collaborator sends the console's client JS **Thursday
evening BST (2026-07-16)** then is unreachable. So the deliverable now is a fully
built scaffold whose only remaining gap is the wire event names — a ~10-minute
`config.py` edit once the JS lands.

## What was built (`tests/rtl-smoke/fpga_driver/`)

- `config.py` — **the single wire-up point.** All Socket.IO event names, payload
  builders, completion signals (`DONE_WHEN`), connection settings, and the (real,
  verified) UART markers. Carries a `PROTOCOL_SOURCE` guard = `"placeholder"`.
- `fpga_console.py` — `FpgaConsole`: sync `socketio.Client` wrapper. Catch-all
  handler records every server→client event under a `Condition`; `wait_event` /
  `wait_status` / `wait_uart` (whole-buffer regex with a `search_from` offset)
  are the sync primitives. The five actions — `upload_boot_image`,
  `load_boot_image`, `reset` (+await prompt), `set_switch`, `trace_dump` — plus
  `run_command` (types a command, returns UART up to a marker) and `power`.
  Transport only; nothing board-specific beyond what it reads from `config`.
- `run_rtl_smoke.py` — end-to-end: connect → power → upload → load → reset(await
  shell) → run borrow + 3 revoke `.user`/`.dom` pairs over UART → harvest RESULT
  lines → `--parse-uart`. Flags: `--url`, `--image`, `--no-upload`, `--remote-dir`,
  `--capture-out`, `--allow-unverified`, `--parse-only`.
- `mock_server.py` — aiohttp + `socketio.AsyncServer` implementing exactly the
  placeholder protocol; emits UART in small chunks (marker lines deliberately
  split across chunks) with the reference RESULT numbers (bump 7 / norevoke 60 /
  revoke 65).
- `test_dryrun.py` — launches the mock as a subprocess, drives the whole scaffold,
  asserts the five actions + sweep + that `--parse-uart` reproduces the reference
  breakdown (+5 O(1) revoke-at-free). **Passes.**
- `extract_from_js.py` — greps the console client JS for `emit`/`on`/`io` → event
  names + payload/auth/path hints (survives minification), to turn into `config.py`.
- `PROTOCOL.md` — the (placeholder) protocol map + the three routes to the real
  protocol (collaborator JS Thursday; site-JS WebFetch with the user's OK; a
  DevTools WS/HAR capture checklist). `README.md`, `requirements.txt`.

## Validation (offline, no board)

`python test_dryrun.py` → "OK: dry-run passed (5 actions + end-to-end sweep +
parse)." Confirmed:
- connect/handshake; `power`, upload (ack), load (status `Loading`→`Done`),
  reset (waits for the shell prompt), `switch_set`, `trace_dump` (awaits
  `trace_complete`), `terminal_input`.
- UART marker matching across chunk boundaries.
- Full sweep capture → `run-revoke-cost-fpga-qemu.sh --parse-uart` → the paper
  breakdown (alloc-side +53, revoke-at-free **+5** O(1), total +58 / 9.29×).

Safety rails verified: the driver **refuses a real board** while
`PROTOCOL_SOURCE="placeholder"` (raises `ProtocolNotVerified` unless
`--allow-unverified`, which is for the mock only); `--parse-only` works without
`--url`/`--image`. `extract_from_js.py` smoke-tested on a synthetic bundle.

One bug found+fixed during the dry-run: `wait_uart` matched a *previous*
command's `measurement complete` marker (whole-buffer search); added a
`search_from` offset so `run_command` only matches its own output.

## Bug caught for free (a real defect the scaffold prevents)

The whole-buffer-marker bug would have mis-attributed every revoke-cost run's
RESULT to the prior config on real hardware (silent wrong numbers, not a crash) —
exactly the kind of thing the offline mock + reference-number assertion exists to
catch before a scarce board slot.

## Dependencies

Runtime: `python-socketio[client]` only (declared in `requirements.txt`). The
mock/test additionally need the server extra + `aiohttp`. Neither is installed in
the repo's system Python; validated in a throwaway venv (not committed).

## What remains (the one gap)

Wire the real Socket.IO events into `config.py` from the client JS (or a fallback
capture), set `PROTOCOL_SOURCE="verified"`, update `mock_server.py` to match, and
re-run `test_dryrun.py`. Then an agent-driven perf run is a single
`run_rtl_smoke.py --url … --image fw_payload.bin`. See `PROTOCOL.md`.

Not attempted (out of scope / by rule): operating the board, fetching the private
token'd URL (needs the user's explicit OK), obtaining credentials.
