# FPGA console protocol map

Status: **VERIFIED** (`config.PROTOCOL_SOURCE = "verified"`). Observed from the
console's own client JS (`static/app.js`), fetched from the live site on
2026-07-16 and read directly (it is unminified — event names and payload shapes
are read straight off the `socket.emit(...)` / `socket.on(...)` / `fetch(...)`
call sites). No collaborator hand-off and no DevTools HAR was needed.

**Key finding — the console is a HYBRID, not pure Socket.IO.** The earlier
placeholder assumed every action was a Socket.IO emit. In reality the action
*verbs* (upload / load / reset / trace) are **HTTP POST to a REST API**, while the
live UART stream, the power/switch controls, terminal keystrokes, and the
auto-shutdown Lock are **Socket.IO events**; each long action reports completion
through a per-action Socket.IO *state* event. Wiring this therefore touched
`fpga_console.py` (added an HTTP layer + toggle-with-verify semantics), not just
`config.py`.

Board Socket.IO server: **v4.7.5** (Engine.IO 4); `python-socketio` 5.x is
compatible.

## Connection

- URL: `https://fpga.corank.info/<token>/` — the access token is in the URL
  **path**. Passed to the driver on the CLI (`--url`), never committed or logged.
- The page sets `SOCKET_PATH = '/<token>/socket.io'` and `URL_PREFIX = '/<token>'`
  (inline `<script>`), and connects with `io({ path: SOCKET_PATH })`.
- `FpgaConsole` derives both from the URL path at runtime, so no token lives in
  the repo: `socketio_path = <url-path>/socket.io`, REST base = `<url>/api`.
- Namespace `/` (default). No extra auth payload — the path token is sufficient
  for the handshake (`config.CONNECT.auth_key = None`).

## Action verbs — HTTP POST (`<url>/api/...`)

| # | Action | Method + path | Body | Completion |
|---|--------|---------------|------|------------|
| 1a | Upload boot image | `POST /api/images/upload` | multipart form `name` + `file` | HTTP response `{name}` (200) / `{error}` (non-200); synchronous |
| 1b | Load image → JTAG `0x80000000` | `POST /api/load-image` | `{filename}` | `load_state` event `{state}`: `loading` → `done` \| `error` (409 `Already loading` if one is in flight) |
| 2 | Reset | `POST /api/reset-board` | *(none)* | no state event — wait for the Linux prompt on the UART stream |
| 5 | Trace dump | `POST /api/trace-start` (cancel: `/api/trace-cancel`) | *(none)* | `trace_result` event `{text}` (the finished dump); `trace_state` also goes `capturing` → `idle` |

Bitstream flashing (`POST /api/flash-bitstream {filename[, volatile]}`, completion
via `flash_state`) and file management (`GET/DELETE/PATCH /api/images`,
`/api/bitstreams`, `.../upload`) are present too but not on the rtl-smoke path.

## Socket.IO client → server (EMIT)

| Logical | Event | Payload | Notes |
|---------|-------|---------|-------|
| Power | `power_toggle` | *(none)* | **toggle** — driver reads `power_state` and only toggles if it differs from the target |
| Terminal keystrokes | `uart_send` | `{text}` | how the `.user`/`.dom` commands are typed (append `\r`) |
| Terminal clear | `uart_clear` | *(none)* | |
| Set switch N | `switch_toggle` | `{index}` | **toggle** — driver reads `switch_state` and only toggles if the bit differs |
| Reset all switches | `switch_reset_all` | *(none)* | |
| Lock (auto-shutdown) | `set_auto_shutdown` | `{timeout_seconds, locked}` | the good-citizen hold on the shared board |
| History replay | `request_history` | `{last_seq}` | client sends on connect; resyncs UART + current states |

(GDB/JTAG console events — `gdb_start` / `gdb_stop` / `gdb_input` / `gdb_output` /
`gdb_state` — also exist for manual bring-up; unused by the sweep.)

## Socket.IO server → client (LISTEN)

| Logical | Event | Payload |
|---------|-------|---------|
| UART bytes | `uart_data` | `{seq, text}` (text carries the bytes → `config.UART_TEXT_KEYS = ["text"]`) |
| Load state | `load_state` | `{state, loaded_image_name}` |
| Flash state | `flash_state` | `{state, nv_bitstream_name}` |
| Trace state | `trace_state` | `{state}`: `idle` \| `capturing` |
| Trace result | `trace_result` | `{text}` (the finished dump) |
| Power state | `power_state` | `{state}`: `on` \| `off` |
| GDB state | `gdb_state` | `{state}`: `idle` \| `starting` \| `running` \| `error` |
| Switch state | `switch_state` | `{states: [0/1 × 8]}` |
| LED state | `led_state` | `{states: [0/1 × 8]}` |
| Lock/auto-shutdown | `auto_shutdown_state` | `{timeout, locked}` |
| User count | `user_count` | `{count}` (used to back off if the shared board is in use) |

On connect the server pushes an initial burst of these (power/switch/lock/led/
load/user_count), so the driver knows current state without waiting for a change.

## The UART contract (VERIFIED — guest software, not the web protocol)

Unchanged from the placeholder; these come from the guest + the existing parser:

- Shell prompt after reset: `config.UART["login_prompt"]`.
- Per-run completion markers: `revoke-cost-fpga: measurement complete` and the
  borrow-cost `RESULT vs-raw` line.
- RESULT line format consumed by `run-revoke-cost-fpga-qemu.sh --parse-uart`:
  `revoke-cost-fpga: RESULT cycles/op  mode=<m>  alloc_free=<n>`.

## What differed from the placeholder (for the record)

1. **Action verbs are REST, not Socket.IO emits.** upload/load/reset/trace are
   HTTP POST; only the live stream + a few controls are Socket.IO.
2. **Upload is a multipart HTTP form**, not a base64 Socket.IO payload / ack — and
   it is synchronous (no chunking, no progress events).
3. **Power and switches are toggles**, not set-value calls; the driver reads the
   state event first and only toggles on a mismatch.
4. **Completion is per-action state events** (`load_state`, `trace_result`), not a
   single shared `status` stream.
5. **The dump is a REST call + one `trace_result`**, not the manual's
   switch-0/switch-1 sequence.

## How the protocol was obtained (Route B, DIY)

```sh
curl -sS "<url>/"               # find <script src=".../static/app.js">
curl -sS "<url>/static/app.js"  # unminified client JS
python extract_from_js.py app.js   # lists every emit/on/io name
# then read the fetch()/emit() call sites for payload shapes + REST paths
```

The DevTools-HAR route (below) was the documented fallback and was **not needed**.

<details><summary>DevTools WebSocket/HAR capture checklist (fallback, unused)</summary>

1. Open the console URL; DevTools → Network → filter **WS**; reload.
2. Socket.IO frames are `42["event",payload]` (`42` = message+event; `43…` = ack).
3. Also watch the **Fetch/XHR** panel — the action verbs are REST POSTs, not WS
   frames, so a WS-only capture would miss upload/load/reset/trace.
4. Right-click → **Save all as HAR with content**; read `[event,payload]` pairs
   and the `/api/...` POST bodies.
</details>

## Offline validation + live validation

- `test_dryrun.py` drives the whole scaffold against `mock_server.py` (which now
  implements the real hybrid: REST endpoints + Socket.IO stream/state): connect →
  Lock → power → upload(multipart) → load(`load_state` done) → reset(await prompt)
  → borrow + 3 revoke configs over UART (markers split across chunks) → harvest
  RESULT lines → switch toggle → trace dump (`trace_result`) → `--parse-uart`.
  Asserts the parse reproduces bump 7 / norevoke 60 / revoke 65 → +5 O(1).
- **Live transport check against the real board** (2026-07-16): the wired driver
  connected to `fpga.corank.info`, received every state event with the exact
  payload shapes above, and read live UART — confirming the map against hardware,
  not just the mock.
