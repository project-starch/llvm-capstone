# FPGA console Socket.IO protocol map

Status: **PLACEHOLDER** (`config.PROTOCOL_SOURCE = "placeholder"`). The event
names and payloads below are **inferred from the user manual**
(`/tmp/capstone/FPGA_Remote/FPGA_Remote/FPGA_Remote_Manual.md`), not observed on
the wire. The console's client JS is **not** in our local capture, so the real
protocol has to come from one of the three sources in "Getting the real protocol"
below. Until then the driver refuses to touch a real board (it runs only against
`mock_server.py`, which implements exactly these placeholders).

**The whole point of the scaffold:** every wire detail lives in `config.py`.
Filling this map in = editing that one file. Nothing else should change.

## The five actions the driver needs (task goal)

| # | Action | Placeholder emit event | Payload (placeholder) | Completion (placeholder) |
|---|--------|------------------------|-----------------------|--------------------------|
| 1a | Upload boot image (`.bin`) | `boot_image_upload` | `{name, size, data(base64)}` | Socket.IO **ack** callback |
| 1b | Load image → JTAG `0x80000000` | `boot_image_load` | `{name}` | `status` → `Loading` → `Done` |
| 2 | Reset | `reset` | *(none)* | UART shows the Linux prompt |
| 3 | Read UART until marker | *(listen only)* | — | regex over the UART stream |
| 4 | Set virtual switch N | `switch_set` | `{index, on}` | `switches` broadcast |
| 5 | Trace Dump | `trace_dump` (+ `switch_set` 0 then 1) | *(none)* | `trace_complete` event |

Plus the helper the sweep depends on:

| Helper | Placeholder emit event | Payload |
|--------|------------------------|---------|
| Terminal keystrokes (run `.user`/`.dom`) | `terminal_input` | `{data: "<text>"}` |
| Power on | `power` | `{on: true}` |

## Server → client events (LISTEN, placeholder)

| Logical | Placeholder event | Assumed payload |
|---------|-------------------|-----------------|
| UART bytes | `terminal_output` | `{data: "<chunk>"}` (driver also tries `text/bytes/chunk/output`, or a bare string) |
| Status | `status` | `{state: "Idle\|Loading\|Flashing\|Capturing\|Done\|Error"}` |
| Trace frame | `trace_frame` | `{seq, event, data}` |
| End-of-dump | `trace_complete` | `{frames, truncated}` |
| LED / switch state | `leds` / `switches` | live bar state (informational) |

## Connection (placeholder)

- URL: `https://fpga.corank.info/<token>/` — the token is in the URL **path**.
  Passed to the driver on the CLI (`--url`), never committed.
- `socketio_path`: `socket.io` (default). The console may customise it.
- Namespace: `/` (default).
- Auth: assumed the token in the URL path is sufficient for the handshake. If the
  server wants it echoed in the connect `auth` payload, set `config.CONNECT.auth_key`
  (e.g. `"token"`) and pass `--token`.

## The UART contract (VERIFIED — not part of the web protocol)

These come from the guest software and the existing parser, and are already
correct (they are what the human run and the QEMU validation use):

- Shell prompt after reset: `config.UART["login_prompt"]`.
- Per-run completion markers: `revoke-cost-fpga: measurement complete` and the
  borrow-cost `RESULT vs-raw` line.
- RESULT line format consumed by `run-revoke-cost-fpga-qemu.sh --parse-uart`:
  `revoke-cost-fpga: RESULT cycles/op  mode=<m>  alloc_free=<n>`.

## Getting the real protocol

Three routes; any one yields the events. Primary is (1).

### 1. Collaborator's client JS (primary; Thursday 2026-07-16 evening BST)

The moment the JS lands:

```sh
python extract_from_js.py <the-bundle>.js [more.js ...]
```

It prints every `socket.emit(...)` / `socket.on(...)` event name (+ `io(...)`
setup: path, auth) — event names are string literals and survive minification.
Then, by hand:

1. Map the emit names → `config.EMIT` (the five actions + `terminal_input`,
   `power`). Read each call site for the payload shape.
2. Map the on names → `config.LISTEN` (`uart_output`, `status`, `trace_*`).
   Confirm which key holds the UART text → `config.UART_TEXT_KEYS`, and the
   status field → `config.STATUS_STATE_KEY`.
3. Fill the completion signals → `config.DONE_WHEN` (status string vs dedicated
   event vs ack). Watch especially how **upload** completes (ack? chunked?
   progress events?) and how **load** signals done.
4. Set `config.CONNECT` (path, namespace, `auth_key`).
5. Set `PROTOCOL_SOURCE = "verified"`.
6. Update `mock_server.py` to the real names and re-run `python test_dryrun.py`.
   Then a real run is `run_rtl_smoke.py --url … --image fw_payload.bin`.

Raise any ambiguity (chunked upload, ack vs event, unexpected namespace) **the
same evening** — he's unreachable afterwards.

### 2. Fetch the site JS (fallback; needs the user's OK on the private URL)

Ask the user first — the URL carries a private token. Then WebFetch the console
URL, find the referenced JS bundle(s) (`<script src=…>`), fetch those, and run
`extract_from_js.py` on them exactly as in (1).

### 3. Live DevTools WebSocket capture (fallback the user can do in one pass)

Hand the user this checklist; it captures the real event stream directly:

1. Open the console URL in Chrome/Firefox; open **DevTools → Network**.
2. Filter to **WS** (WebSocket). Reload so the Socket.IO connection is captured.
3. Click the socket.io entry → **Messages** tab (Chrome) / **Response** (FF).
   Socket.IO frames look like `42["event_name",{…payload…}]` (`42` = Engine.IO
   message + Socket.IO event; `430…` = ack). The array is `[event, payload]`.
4. **Click each control once**, noting the frame each produces, in this order:
   Power, Boot Images→Upload, Boot Images→Load, Reset, type a char in Terminal,
   toggle a Switch, Trace Dump. Also capture a few **inbound** frames (UART
   output, status).
5. Export: right-click the WS entry → **Save all as HAR**, or copy the Messages
   list. Send the HAR/paste back.

From the HAR, each `42[...]` outbound frame gives an emit name+payload and each
inbound gives a listen name+payload — same mapping as (1), then same steps 1–6.

## Offline validation already done

`test_dryrun.py` drives the whole scaffold against `mock_server.py`: connect →
upload(ack) → load(status) → reset(await prompt) → run borrow + 3 revoke configs
over UART (markers split across chunks) → harvest RESULT lines → switch 0/1 +
Trace Dump (await `trace_complete`) → `--parse-uart`. It asserts the parse
reproduces the reference breakdown (bump 7 / norevoke 60 / revoke 65 → +5 O(1)).
So the transport, the five actions, and the integration are proven; only the wire
names in `config.py` remain.
