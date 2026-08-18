You are Agent-B, continuing task 017 in `/home/alexey/dev/llvm-capstone-b`,
branch `capstone-bootstrap-b`.

```bash
cd /home/alexey/dev/llvm-capstone-b && git fetch origin && git pull --rebase
source capstone/tests/capstone-test-env.sh
```

## What arrived

The RTL collaborator sent the **authoritative** Socket.IO API doc (the console's own
`app.py` author). It is committed on `capstone-bootstrap` at
**`capstone/tests/rtl-smoke/socketio-api.md`** — read it:

```bash
git show origin/capstone-bootstrap:capstone/tests/rtl-smoke/socketio-api.md
```

Good news: it **confirms** what you reverse-engineered from `app.js` — the hybrid
model (REST verbs + Socket.IO stream/state), every EMIT/LISTEN event name, the
payload shapes, the toggle semantics (`power_toggle`, `switch_toggle{index}`), the
Lock = `set_auto_shutdown{locked}`, and `request_history{last_seq}`. The doc covers
**only Socket.IO**; the REST endpoints (`/api/load-image`, `/api/reset-board`,
`/api/trace-start`, …) stay as you already have them from `app.js`. So this is a
reconciliation, not a rewrite.

## The one substantive check — history-seq threading

The doc specifies: `uart_data` carries `{seq, text}` with a monotonic `seq`;
`request_history{last_seq}` replays **only** chunks with `seq > last_seq` (`-1` =
full replay), addressed to the requesting sid. Your reconnect fix re-requests
history — **verify it threads the real last-seen `seq`, not a hardcoded `-1`.** If it
always sends `-1`, every reconnect re-injects the entire 512 KB history and can
duplicate/garble the `RESULT` lines the parser reads. Confirm the driver tracks the
latest `seq` from incoming `uart_data` and passes it as `last_seq` on reconnect;
fix if not, and cover it in the mock/dry-run.

## Quick confirmations (tighten if they differ; none are likely blockers)

1. **Reset completion.** The doc says `POST /api/reset-board` emits
   `load_state{state:'idle', loaded_image_name:null}`. Your config notes "no state
   event" for reset. You already wait on the UART Linux prompt, so this is optional,
   but you *may* use `load_state→idle` as a cleaner reset-done signal — your call.
2. **Trace vs UART are mutually exclusive.** While `trace_state=='capturing'`,
   `uart_data` stops flowing (bytes route to the trace parser). The revoke-cost perf
   run uses UART `RESULT` lines and **not** the tracer, so this shouldn't bite — just
   don't interleave a trace capture with UART reads in the same phase.
3. **`trace_state` has a `done` state** (before `trace_result`); you list
   idle|capturing. Harmless since you wait on `trace_result`, but note it.
4. **Connection path** = `<url_prefix>/socket.io` (the doc's `SOCKET_PATH`). Confirm
   `FpgaConsole` still builds `/<token>/socket.io` correctly (it connected live, so
   this is just a re-confirm against the authoritative wording).

## Deliverables

- `mock_server.py` updated if any check above changed an event/payload; **re-run
  `python fpga_driver/test_dryrun.py` → stays green.**
- `PROTOCOL.md`: add a one-line note that it is now cross-checked against the
  authoritative `capstone/tests/rtl-smoke/socketio-api.md`, and fix any spot where
  your reverse-engineered map differed. Keep `PROTOCOL_SOURCE="verified"`.
- Short report: what matched (expected: essentially all of it), whether the
  history-seq threading needed a fix, and anything you tightened.

## Still gated on the console

This does **not** unblock the perf run — that is still waiting on the board console
(HTTP 503, lab-side backend restart), not on the protocol. After this pass you are
ready to fire `run_rtl_smoke.py` the moment the console serves; nothing else about
the protocol is outstanding.

## Guardrails (unchanged)

- Additive **test tooling only** — no `llvm/`, monitor, submodule, or RTL changes.
- The board-operation and URL-fetch authorization from the phase-3 brief still
  stands; token never enters the tree or any log; good-citizen on the shared board.
- Commit on `capstone-bootstrap-b`, exact paths, **no `Co-Authored-By:`**, no
  worker/agent identity in commit messages, no debug/report files.
