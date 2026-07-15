# Follow-up prompt for Agent-B — task 017 phase 2 (wire the real protocol)

*Paste everything below the line into `claude-b`. Self-contained. This is the
follow-on to the scaffold you already built and pushed.*

---

You are Agent-B, continuing task 017 in `/home/alexey/dev/llvm-capstone-b`,
branch `capstone-bootstrap-b`.

## Where things stand

Your scaffold from phase 1 is **done and independently verified green** — the
main lane overlaid your `capstone/tests/rtl-smoke/fpga_driver/` onto its tree and
ran `test_dryrun.py` against the real `run-revoke-cost-fpga-qemu.sh --parse-uart`
+ env: EXIT 0, "dry-run passed (5 actions + end-to-end sweep + parse)". The
design is right: one wire-up file (`config.py`), the `PROTOCOL_SOURCE`
placeholder guard, the DevTools-HAR fallback, and the cross-chunk marker fix.

Two things to fix now, then you're in a holding pattern until the real protocol
arrives (Thursday evening BST, or sooner via a DevTools capture the user may
provide).

```bash
cd /home/alexey/dev/llvm-capstone-b && git fetch origin && git pull --rebase
```

## Do now (small, ~10 min)

1. **Fix `fpga_driver/requirements.txt`.** `aiohttp` is currently only in a
   comment, so `pip install -r requirements.txt` does not install it and the
   dry-run/mock cannot run on a clean host. Make `aiohttp>=3.8` an actual
   installable line (or add a `requirements-dev.txt` for the mock/test deps and
   note it in `README.md`). Verify from a fresh venv:
   `python -m venv /tmp/v && /tmp/v/bin/pip install -r fpga_driver/requirements.txt`
   then `/tmp/v/bin/python fpga_driver/test_dryrun.py` → must print the green line.
2. **README install note.** The runtime host needs
   `pip install python-socketio[client] aiohttp` before the driver imports — make
   that explicit in `fpga_driver/README.md` (it is a real prerequisite; the host
   we checked had neither installed).

Commit these on `capstone-bootstrap-b` (imperative subject, no worker/agent
identity in the message, no `Co-Authored-By:`).

## When the real protocol arrives — the wire-up (time-critical Thursday)

The collaborator sends the web UI **client JS** Thursday evening BST, then is
**unreachable**. Or the user may hand you a **DevTools WebSocket HAR** sooner.
Either yields the events. The whole job is editing `config.py` — nothing else
should change. Procedure (also in `fpga_driver/PROTOCOL.md`):

1. **Extract the events.**
   - From client JS: `python fpga_driver/extract_from_js.py <bundle>.js [...]` —
     prints every `socket.emit(...)` / `socket.on(...)` name + `io(...)` setup
     (path, auth). Names are string literals, so they survive minification.
   - From a HAR: each outbound Socket.IO frame is `42["<event>",<payload>]` and
     each inbound is the same shape (`430...` = ack). Read the `[event, payload]`
     pairs directly. (If useful, extend `extract_from_js.py` to also parse HAR
     `42[...]` frames — optional.)
2. **Map emit names → `config.EMIT`** for the five actions + `terminal_input` +
   `power`. Read each call site for the real **payload shape**. Watch especially:
   - **upload** — is it an **ack** callback, or chunked over multiple events, or
     progress events? Set `expects_ack` / `DONE_WHEN` accordingly.
   - **load** — how does it signal done (a status string vs a dedicated event)?
3. **Map on names → `config.LISTEN`** (`uart_output`, `status`, `trace_*`).
   Confirm which payload key carries the UART text → `config.UART_TEXT_KEYS`
   (order matters, first present wins), and the status field →
   `config.STATUS_STATE_KEY`.
4. **Completion signals → `config.DONE_WHEN`** (status string vs dedicated event
   vs ack) for upload / load / reset / trace-dump.
5. **Connection → `config.CONNECT`**: `socketio_path` (default `socket.io`),
   `namespace` (default `/`), and how the URL-path access token is presented —
   if the server wants it echoed, set `auth_key` (e.g. `"token"`).
6. **Flip `PROTOCOL_SOURCE = "verified"`** in `config.py`.
7. **Update `mock_server.py`** to the real event names and **re-run
   `python fpga_driver/test_dryrun.py`** — it must stay green. That proves the
   wiring is internally consistent before any board contact.
8. **Raise every ambiguity the same evening** (chunked upload, ack-vs-event,
   unexpected namespace/auth) while the collaborator is still reachable — after
   Thursday evening he is gone.

## Real-board run — needs the user, do NOT self-initiate

The guard (`PROTOCOL_SOURCE`) plus the lane rules mean: after wiring + green
mock, a real run requires **the user's explicit go-ahead and the token'd URL**
(never commit the token). The intended command is:

```
python fpga_driver/run_rtl_smoke.py --url 'https://fpga.corank.info/<token>/' \
    --image <path-to>/fw_payload.bin
```

but do not run it against the board without the user saying go. Report readiness
and let the user trigger it (they may also want to drive the first run manually
via the web console per `tests/rtl-smoke/README.md`).

## Guardrails (unchanged)

- Additive **test tooling only** — no `llvm/`, no submodule bumps, no monitor /
  `start.S` / allocator / RTL changes.
- Do not operate the board or seek credentials; WebFetch the private URL only
  with the user's explicit OK.
- Commit on `capstone-bootstrap-b`, exact paths, **no `Co-Authored-By:`**, **no
  worker/agent identity in commit messages**, no debug/report files.
- If you boot QEMU for any reason, claim the shared `rootfs.ext2` lock
  (`COORDINATION.md`) — the main lane may be running suites.

## Deliverables

- `config.py` filled + `PROTOCOL_SOURCE="verified"`, `mock_server.py` matching,
  `test_dryrun.py` green.
- `PROTOCOL.md` updated from PLACEHOLDER to the observed map (events + payloads +
  handshake), noting the source (JS bundle / HAR).
- History note appended, and a short report: what the protocol turned out to be,
  anything that differed from the placeholders, and whether a real-board run is
  ready to trigger.

## Framing

This is still parallel, not critical-path — the human-driven RTL run already
works. The value is a hands-off driver + being ready to wire + sanity-check the
protocol the same evening it arrives. Fast and low-risk; don't touch the board
without the user.
