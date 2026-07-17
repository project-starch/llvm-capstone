# Socket.IO API

This document describes the real-time Socket.IO interface exposed by `app.py`. It complements the REST endpoints (`/api/load-image`, `/api/flash-bitstream`, `/api/reset-board`, `/api/trace-start`, `/api/trace-cancel`, and the file-manager routes under `/api/images` and `/api/bitstreams`), which are not covered here.

## Connecting

The client connects with a custom path so the transport works correctly when the app is hosted under `--url-prefix`:

```js
const socket = io({ path: SOCKET_PATH });
```

`SOCKET_PATH` is `<url_prefix>/socket.io`, injected into the page as a JS global (see `URL_PREFIX` / `SOCKET_PATH` in `templates/index.html`). If `--allowed-origin` is passed on the server, cross-origin connections are restricted to that origin.

All events use the default namespace (`/`). There is exactly one room — every client receives every broadcast (aside from a few events sent only to the requesting `sid` on connect / on-demand history).

---

## Connection lifecycle

### `connect` (server → client, on-demand handshake)

Fired automatically by Socket.IO when a client connects. The server does **not** push UART history at this point — the client must request it separately (see `request_history`). Instead it emits a snapshot of every other piece of server state, addressed only to the new client's `sid`:

| Event | Payload |
|---|---|
| `load_state` | `{state: 'idle'\|'loading'\|'done'\|'error', loaded_image_name: string\|null}` |
| `flash_state` | `{state: 'idle'\|'loading'\|'done'\|'error', nv_bitstream_name: string\|null}` |
| `power_state` | `{state: 'on'\|'off'}` |
| `led_state` | `{states: [0\|1, ...]}` (8 entries) |
| `switch_state` | `{states: [0\|1, ...]}` (8 entries) |
| `trace_state` | `{state: 'idle'\|'capturing'\|'done'}` |
| `auto_shutdown_state` | `{timeout: number (seconds), locked: boolean}` |
| `gdb_state` | `{state: 'idle'\|'starting'\|'running'\|'error'}` |

Then, broadcast to **all** clients:

| Event | Payload |
|---|---|
| `user_count` | `{count: number}` — total connected clients, after increment |

### `disconnect` (implicit)

No client action required. On disconnect the server broadcasts an updated `user_count` to all remaining clients.

```
socketio.emit('user_count', {count: number})
```

---

## UART terminal

### `uart_send` (client → server)

Sends keystrokes/text to the board over the serial connection.

```js
socket.emit('uart_send', { text: 'ls\r' });
```

- Resets the auto-shutdown inactivity timer.
- Writes `text` (UTF-8 encoded) directly to the open serial port. Silently no-ops if the serial port isn't open.

### `uart_data` (server → client)

Streamed terminal output. Emitted by the `serial_emitter` background task roughly every 10 ms whenever new bytes have arrived from the UART (and trace capture is not active).

```json
{ "seq": 1234, "text": "decoded chunk of UART output" }
```

- `seq` is a monotonically increasing counter (`_uart_history_seq`), incremented once per emitted chunk.
- Every emitted chunk is also appended to a server-side history ring buffer (`_uart_history`), capped at 512 KB total (oldest chunks evicted once the cap is exceeded).

### `request_history` (client → server)

Requests replay of UART history newer than a given sequence number. Typically sent by the client immediately after `connect` (or on reconnect) so history isn't duplicated.

```js
socket.emit('request_history', { last_seq: -1 }); // -1 for a fresh page load
```

- Server gathers all buffered chunks with `seq > last_seq`, concatenates their decoded text, and replies **only to the requesting client**:

```json
// uart_data, addressed to the requesting sid only
{ "seq": <seq of last chunk>, "text": "concatenated history text" }
```

- If there are no newer chunks, nothing is emitted.

### `uart_clear` (client → server)

Clears the server-side UART history buffer (used by the terminal's "Clear" button). Does not affect other clients' currently rendered terminal content, only future `request_history` replays.

```js
socket.emit('uart_clear');
```

No response event.

---

## Load state (image loading)

Triggered via `POST /api/load-image`, but state transitions are broadcast over Socket.IO so all clients stay in sync.

### `load_state` (server → client, broadcast)

```json
{ "state": "idle" | "loading" | "done" | "error", "loaded_image_name": "kernel.bin" | null }
```

- `loading` is emitted when the load begins.
- `done` / `error` is emitted when the GDB/OpenOCD load sequence finishes; `loaded_image_name` is set to the basename of the loaded file on success.
- Also emitted (state `idle`, `loaded_image_name: null`) after a board reset (`POST /api/reset-board`).
- Cleared to `loaded_image_name: null` internally on power-off, but no dedicated event is emitted for that — clients see it via the next `power_state` broadcast plus whatever `load_state` follows.

No client-emitted event exists for load state directly — use the REST endpoint `POST /api/load-image`.

---

## Flash state (bitstream flashing)

Same pattern as load state, triggered via `POST /api/flash-bitstream`.

### `flash_state` (server → client, broadcast)

```json
{ "state": "idle" | "loading" | "done" | "error", "nv_bitstream_name": "top.bit" | null }
```

- `nv_bitstream_name` is set only when a **non-volatile** (SPI flash) write succeeds; volatile (JTAG-only) flashes leave it unchanged.
- Unlike `loaded_image_name`, this is never cleared on power-off (non-volatile bitstreams survive power cycles).

---

## Power control

### `power_toggle` (client → server)

Flips board power (GPIO pin 17).

```js
socket.emit('power_toggle');
```

- Broadcasts the new state:

```json
{ "state": "on" | "off" }
```

- Resets (if turning on) or clears (if turning off) the auto-shutdown deadline.
- On power-off: clears `_loaded_image_name` (so the next `load_state` snapshot will show `null`) and, if a GDB session is `starting`/`running`, kills it (no explicit event beyond the eventual `gdb_state` transition performed by `_gdb_kill_procs`'s cleanup path).

### `power_state` (server → client, broadcast)

```json
{ "state": "on" | "off" }
```

Emitted on: manual `power_toggle`, connect (addressed to the new client only), and automatic shutdown by the inactivity watcher.

---

## Virtual LEDs (read-only GPIO inputs)

No client-emitted event — LEDs are sampled server-side.

### `led_state` (server → client, broadcast)

```json
{ "states": [0, 1, 0, 0, 1, 0, 0, 0] }
```

8 entries, index = LED number, pins `[9, 10, 22, 27, 6, 5, 0, 11]`. Emitted by the `led_emitter` background task at 10 Hz, but **only when the sampled state differs from the last emitted state** (i.e., not a fixed heartbeat).

---

## Virtual switches (writable GPIO outputs)

### `switch_toggle` (client → server)

Flips a single switch.

```js
socket.emit('switch_toggle', { index: 3 }); // 0-based, must be 0..7
```

- Resets the auto-shutdown timer.
- Ignored if `index` is missing or out of range.
- Broadcasts the full updated state (see below).

### `switch_reset_all` (client → server)

Resets all 8 switches to low.

```js
socket.emit('switch_reset_all');
```

- Resets the auto-shutdown timer.
- Broadcasts the full updated state.

### `switch_state` (server → client, broadcast)

```json
{ "states": [0, 0, 1, 0, 0, 0, 0, 0] }
```

8 entries, index = switch number, pins `[21, 20, 16, 12, 1, 7, 8, 25]`. Emitted after any `switch_toggle` / `switch_reset_all`, and to each client individually on `connect`.

---

## Auto-shutdown timer

### `set_auto_shutdown` (client → server)

Configures (or locks) the inactivity auto-power-off timer.

```js
socket.emit('set_auto_shutdown', { timeout_seconds: 1200, locked: false });
```

- `timeout_seconds` is clamped to `[60, 86400]`; defaults to `600` if omitted.
- `locked: true` disables automatic shutdown regardless of inactivity.
- If a deadline is currently active, it is rescheduled using the new timeout (`now + timeout_seconds`).
- Broadcasts the new config to all clients:

```json
{ "timeout": 1200, "locked": false }
```

Any of the following resets the deadline to `now + timeout` while power is on: `uart_send`, `switch_toggle`, `switch_reset_all`, `power_toggle` (on), starting an image load, starting a bitstream flash, or an active `load_state == 'loading'` / `flash_state == 'loading'` / `gdb_state == 'running'` observed by the watcher.

When the deadline passes with power on and the timer unlocked, the server cuts GPIO power, clears the loaded image name, kills any running GDB session, and emits `power_state: {state: 'off'}` (no separate "auto-shutdown fired" event — infer it from `power_state` turning off without a preceding `power_toggle`).

---

## Trace capture

Triggered via REST (`POST /api/trace-start`, `POST /api/trace-cancel`), not a client-emitted Socket.IO event, but state and results stream back over Socket.IO.

### `trace_state` (server → client, broadcast)

```json
{ "state": "idle" | "capturing" | "done" }
```

- `capturing`: emitted when `/api/trace-start` succeeds; incoming UART bytes are now routed to the trace parser instead of the terminal (`uart_data` stops flowing while this is active).
- `done`: emitted once an end-of-dump frame is detected in the captured stream.
- `idle`: emitted after `/api/trace-cancel`.

### `trace_result` (server → client, broadcast)

Emitted once, immediately after `trace_state: {state: 'done'}`.

```json
{ "text": "formatted trace entries..." }
```

`text` is the human-readable output of `trace_parser.format_entries`, built from parsed CVA6 Capstone tracer frames.

---

## GDB debugging session

### `gdb_start` (client → server)

Starts an interactive GDB session (OpenOCD on the host + `gdb-multiarch` inside the `fpga-gdb` Podman container).

```js
socket.emit('gdb_start');
```

- No-ops if `gdb_state` is not `idle` or `error`.
- Refuses (transitions to `error` with an explanatory `gdb_output`, sent only to the requesting client) if an image load or bitstream flash is in progress.
- On success, transitions `gdb_state` through `starting` → `running`.

### `gdb_stop` (client → server)

Ends the current GDB session.

```js
socket.emit('gdb_stop');
```

- Resumes the target CPU (`continue`), sends `SIGTERM` to the GDB container, closes the PTY, and terminates OpenOCD.
- Results in `gdb_state` transitioning to `idle` once cleanup completes (via the output reader's stop sentinel).

### `gdb_input` (client → server)

Sends raw keystrokes to the GDB session's PTY (interactive terminal input).

```js
socket.emit('gdb_input', { text: 'bt\n' });
```

- Silently no-ops if no session is active (`_gdb_master_fd` is `None`).

### `gdb_output` (server → client, broadcast)

Streamed GDB/terminal output, drained from the PTY every 10 ms.

```json
{ "data": "raw terminal text, may include ANSI escapes" }
```

Also used to deliver one-off diagnostic messages addressed only to the requesting client, e.g.:
```json
{ "data": "\r\n[GDB] Cannot start: image load in progress\r\n" }
```
```json
{ "data": "\r\n[GDB] Start failed: <exception message>\r\n" }
```

### `gdb_state` (server → client)

```json
{ "state": "idle" | "starting" | "running" | "error" }
```

- Broadcast to all clients on every transition, **except** the two guard-rejection cases above (start refused due to load/flash in progress), which are addressed only to the requesting client.
- State machine: `idle → starting → running → idle`, or `→ error` on failure (from `error`, a further `gdb_start` is allowed to retry).

---

## User presence

### `user_count` (server → client, broadcast)

```json
{ "count": 2 }
```

Emitted on every `connect` and `disconnect`, reflecting the current number of connected Socket.IO clients.

---

## Event summary

| Event | Direction | Broadcast scope |
|---|---|---|
| `connect` | implicit | — |
| `disconnect` | implicit | — |
| `uart_send` | C → S | — |
| `uart_data` | S → C | all, except history replay (sid only) |
| `request_history` | C → S | — |
| `uart_clear` | C → S | — |
| `load_state` | S → C | all |
| `flash_state` | S → C | all |
| `power_toggle` | C → S | — |
| `power_state` | S → C | all |
| `led_state` | S → C | all (on change only) |
| `switch_toggle` | C → S | — |
| `switch_reset_all` | C → S | — |
| `switch_state` | S → C | all |
| `set_auto_shutdown` | C → S | — |
| `auto_shutdown_state` | S → C | all |
| `trace_state` | S → C | all |
| `trace_result` | S → C | all |
| `gdb_start` | C → S | — |
| `gdb_stop` | C → S | — |
| `gdb_input` | C → S | — |
| `gdb_output` | S → C | all, except start-guard errors (sid only) |
| `gdb_state` | S → C | all, except start-guard errors (sid only) |
| `user_count` | S → C | all |
