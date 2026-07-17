# FPGA console protocol map

Status: **VERIFIED** (`config.PROTOCOL_SOURCE = "verified"`). Observed from the
console's own client JS (`static/app.js`), fetched from the live site on
2026-07-16 and read directly (it is unminified — event names and payload shapes
are read straight off the `socket.emit(...)` / `socket.on(...)` / `fetch(...)`
call sites). No collaborator hand-off and no DevTools HAR was needed.

**Cross-checked (2026-07-17) against the authoritative
`capstone/tests/rtl-smoke/socketio-api.md`** — the console `app.py` author's own
Socket.IO reference. It confirms the reverse-engineered map byte-for-byte (event
names, payloads, toggle semantics, the Lock, `request_history{last_seq}`). Three
things it made precise, now folded in below: `uart_data.seq` must be **threaded**
into `request_history` on reconnect (was hardcoded `-1`); `POST /api/reset-board`
also emits `load_state{state:'idle'}`; and `trace_state` has a `done` state and
is **mutually exclusive with `uart_data`** (bytes route to the trace parser while
capturing). The doc covers Socket.IO only; the REST verbs stay as mapped from
`app.js`.

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
| 2 | Reset | `POST /api/reset-board` | *(none)* | emits `load_state{state:'idle', loaded_image_name:null}`; the driver waits for the Linux prompt on the UART stream (more robust than the state event) |
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
| History replay | `request_history` | `{last_seq}` | sent on every (re)connect. Server replays only `uart_data` chunks with `seq > last_seq` (to the requesting sid, as one `uart_data{seq,text}`); `-1` = full replay. The driver tracks the max `seq` seen (`config.UART_SEQ_KEY`) and threads it here, so a reconnect fills only the gap instead of re-injecting the ≤512 KB buffer and duplicating RESULT lines |

| GDB start/stop | `gdb_start` / `gdb_stop` | *(none)* | open/close the OpenOCD + `gdb-multiarch` session (`gdb_state` → `starting`→`running`→`idle`) |
| GDB keystrokes | `gdb_input` | `{text}` | raw PTY input — how the driver types `monitor reset halt` / `set $pc` / `continue` |

### GDB-driven boot (`--boot-method=gdb`)

`reset-board` makes the bootrom reload the **SPI-resident** firmware, clobbering
our JTAG-loaded DRAM image — so on a board that boots resident firmware, the
`reset` boot method never runs our image. The GDB path avoids `reset-board`
entirely: `gdb_start` → `monitor reset halt` (halts the hart at the reset vector
via the debug module, **not** the reset button) → `set $pc = 0x80000000` →
`continue`, then watch UART for OpenSBI → Linux → shell. Non-persistent (flashes
nothing). `gdb_output` is accumulated separately from UART (`console.gdb_text`).
The exact OpenOCD ordering is verified against the live session — see the
`fpga-gdb-boot` history note.

### Module load (both boot methods)

The `.doms` open `/dev/capstone` (libcapstone `capstone_init`), created by
`capstone.ko`. The fpga image does **not** auto-load it at boot, so the sweep
`insmod /capstone.ko` first (else `failed to initialize Capstone`). A UP
(`CONFIG_SMP=n`) image must ship a `capstone.ko` built against the **same** SMP
setting — an SMP-built module refuses to load into a UP kernel
(`version magic '… SMP …' should be '…'`).

## Socket.IO server → client (LISTEN)

| Logical | Event | Payload |
|---------|-------|---------|
| UART bytes | `uart_data` | `{seq, text}` (`text` → `config.UART_TEXT_KEYS=["text"]`; `seq` → `config.UART_SEQ_KEY="seq"`, tracked for `request_history`). Stops flowing while `trace_state=='capturing'` (bytes route to the trace parser) |
| Load state | `load_state` | `{state, loaded_image_name}`; also `idle` after `reset-board` |
| Flash state | `flash_state` | `{state, nv_bitstream_name}` |
| Trace state | `trace_state` | `{state}`: `idle` \| `capturing` \| `done` (`done` fires just before `trace_result`) |
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
  Asserts the parse reproduces bump 7 / norevoke 60 / revoke 65 → +5 O(1). It also
  asserts the **history-seq threading**: a reconnect that re-requests history with
  the tracked `seq` re-injects nothing (no duplication), while a `last_seq=-1`
  replay does re-deliver the buffer (proving the mock — and the real server — gate
  on `last_seq`).
- **Live transport check against the real board** (2026-07-16): the wired driver
  connected to `fpga.corank.info`, received every state event with the exact
  payload shapes above, and read live UART — confirming the map against hardware,
  not just the mock.
