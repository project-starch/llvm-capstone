# fpga_driver — headless Socket.IO driver for the CapliFive FPGA web console

Removes the human from the RTL perf sweep. The board's console
(`the FPGA web console`) is a browser Socket.IO GUI with no scriptable API, so the
`tests/rtl-smoke/` run currently needs a person clicking ~5 buttons per run
(README.md "Run (human-driven)"). This is a `python-socketio` client that
performs those actions headlessly and drives the sweep end-to-end into the
existing `--parse-uart` parser.

**The human-driven run already works — this is the parallel automation, not the
critical path.**

## Status

- **Protocol: VERIFIED** (`config.PROTOCOL_SOURCE = "verified"`). Obtained DIY
  from the console's own client JS (`static/app.js`, fetched live 2026-07-16);
  see `PROTOCOL.md`. The console is a **hybrid**: the action verbs
  (upload/load/reset/trace) are HTTP POST to a REST API, and the live UART stream
  + power/switch/Lock controls are Socket.IO.
- **Validated offline and against the real board.** `test_dryrun.py` drives the
  full flow against a mock that implements the real hybrid protocol (REST + socket);
  and the wired driver has connected to the live board, received every state event
  with the exact payload shapes, and read live UART.
- **Board note:** a board `reset` drops the Socket.IO connection briefly; the
  client reconnects (`reconnection=True`), but see `PROTOCOL.md` / the history note
  for the reconnect handling around reset.

## Files

| File | Role |
|------|------|
| `config.py` | **The single wire-up point.** All Socket.IO event names, payloads, completion signals, connection settings, and UART markers. Editing this = wiring the real protocol. |
| `fpga_console.py` | `FpgaConsole`: the Socket.IO client — connect/handshake, event-wait + UART helpers, the five actions. Transport only; no board specifics. |
| `run_rtl_smoke.py` | End-to-end runner: upload → load → reset → run the `.user`/`.dom` pairs over UART → harvest RESULT lines → `run-revoke-cost-fpga-qemu.sh --parse-uart`. Also `--parse-only` and `--parse`-style reuse. |
| `mock_server.py` | Mock console implementing the **placeholder** protocol; lets the flow run offline. Not the real board. |
| `test_dryrun.py` | Offline dry-run: drives the whole scaffold against the mock and asserts the parse reproduces the reference numbers. |
| `extract_from_js.py` | Greps the console client JS for `emit`/`on`/`io` → event names + payload hints, to turn into `config.py`. |
| `PROTOCOL.md` | The protocol map (placeholder) + the three ways to get the real one, incl. the DevTools capture checklist. |
| `requirements.txt` | `python-socketio` (client). The mock/test also need the server extra + aiohttp. |

## Install (required before first use)

Neither dependency ships with the system Python here — install them first or the
driver fails at import:

```sh
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # python-socketio[client] + aiohttp
```

`requirements.txt` covers both the driver's runtime need
(`python-socketio[client]`) and the offline dry-run / mock server (`aiohttp`), so
a single install runs everything below. If you only want the driver (no mock),
`pip install 'python-socketio[client]'` is enough; the dry-run additionally needs
`aiohttp`.

## Offline test (no board, no network beyond localhost)

```sh
python test_dryrun.py
# -> "OK: dry-run passed (5 actions + end-to-end sweep + parse)."
```

## Wiring the real protocol (when the JS arrives)

```sh
python extract_from_js.py <console-bundle>.js
```

Then follow `PROTOCOL.md` §"Getting the real protocol" step 1 (map the names into
`config.py`, set `PROTOCOL_SOURCE = "verified"`, update `mock_server.py`, re-run
`test_dryrun.py`).

## Real run (only after the protocol is verified)

```sh
# build the artifacts + assemble fw_payload.bin per ../README.md, then:
python run_rtl_smoke.py \
    --url '<FPGA-CONSOLE-URL>' \
    --image /path/to/fw_payload.bin
# -> uploads, runs the sweep over UART, prints the temporal-safety breakdown.
```

Parse a capture without touching the board:

```sh
python run_rtl_smoke.py --parse-only <uart-capture.txt>
```

## Guarantees / lane rules

- Additive test tooling; touches nothing in `llvm/`, the monitor, `start.S`, the
  allocators, or the RTL tree.
- Never fetches a URL or seeks credentials on its own; the token'd URL is a CLI
  argument, kept out of the repo and never logged.
- Refuses to drive a real board while the protocol is a placeholder.
