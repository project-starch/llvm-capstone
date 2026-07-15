# Follow-up prompt for Agent-B — task 017 phase 3 (DIY: take Jason off the critical path)

*Paste everything below the line into `claude-b`. Self-contained. This is the
follow-on to the scaffold (phase 1) and the wire-up brief (phase 2,
`agentB-017-wireup-PROMPT.md`). New direction: stop waiting on the collaborator's
Thursday JS — get the protocol and the boot image ourselves, so an agent-driven
run is ready the moment the user authorizes board contact.*

---

You are Agent-B, continuing task 017 in `/home/alexey/dev/llvm-capstone-b`,
branch `capstone-bootstrap-b`.

```bash
cd /home/alexey/dev/llvm-capstone-b && git fetch origin && git pull --rebase
source capstone/tests/capstone-test-env.sh
```

## The reframe (why this task)

Your `fpga_driver/` scaffold is **done and verified green**; it is in a holding
pattern waiting for the Socket.IO protocol. The plan was to get that protocol from
the collaborator's client JS on **Thursday evening BST**, after which he is
unreachable. That is fragile ("will try"). This task removes him from the critical
path: **the protocol and the boot image are both obtainable DIY, without him.**

Only **three** things gate an agent-driven RTL perf run, and only the last needs a
human — and that human is the **user**, not the collaborator:

| Piece | Needs collaborator? | DIY path (this task) |
|---|---|---|
| Socket.IO protocol (event names + payloads) | **No** | fetch the live console's own client JS, or the user's DevTools HAR |
| Boot image `fw_payload.bin` (our test overlay) | **No** | build via `caplifive-system` — no board |
| Operating the physical board | **No** | needs the **user's** explicit go-ahead + the token'd URL |

So: do the two DIY unblocks below, wire the driver, and report readiness. Do **not**
touch the board.

## Task 1 — build the boot image (pure build, no board)

Build `fw_payload.bin` (OpenSBI + Linux payload, single `.bin` → `0x80000000`) with
**our** rootfs overlay carrying the rtl-smoke test binaries, via the umbrella repo
`github.com/project-starch/caplifive-system`:

- `scripts/build-software.sh --mode fpga` → `.../firmware/fw_payload.bin`.
- Overlay: point `BR2_ROOTFS_OVERLAY` at our own dir containing the built
  `.user`+`.dom` pairs from `run-revoke-cost-fpga-qemu.sh` (build them if not
  present; do **not** need the board to build them).
- See `reference` details in `capstone/tests/rtl-smoke/README.md` /`RESULTS.md`
  and the RTL platform notes (boot image = OpenSBI `fw_payload.bin`, payload-linked
  kernel + embedded initramfs — NOT our QEMU `fw_jump.elf` + separate `Image` +
  `rootfs.ext2`).

**Known wall:** `caplifive-system` likely pulls **private** submodules and `gh` is
**not authed** in the sandbox. If a clone/submodule fetch fails on auth, **stop
there and report exactly which repo/submodule needs auth** — do not try to work
around it or seek credentials. Getting as far as "cloned the public parts, blocked
at <X> needing gh auth" is a perfectly good result the user can unblock.

## Task 2 — get the Socket.IO protocol DIY (no collaborator)

Two routes. **Route A (DevTools HAR) is the reliable one; prefer it.**

- **Route A — user's DevTools capture (most reliable).** The user is authenticated
  in-browser, so the captured frames are guaranteed real. The capture checklist is
  in `fpga_driver/PROTOCOL.md` (Network → filter WS → reload → click each control
  once: Power, Boot-image Upload, Load, Reset, Terminal keypress, Switch toggle,
  Trace Dump → "Save all as HAR with content"). If the checklist there is thin,
  tighten it. When the user hands you a `.har`, read the `42["<event>",<payload>]`
  outbound frames and the inbound frames directly (`43…` = ack).
- **Route B — fetch the live client JS ourselves (needs the user's explicit OK).**
  The app's JS bundle is served from the **same origin as the token'd URL**
  (`https://fpga.corank.info/<token>/`). **Only with the user's explicit go-ahead**,
  WebFetch the console page, find the referenced JS bundle(s), fetch them, and run
  `python fpga_driver/extract_from_js.py <bundle>.js` → emit/on/io names survive
  minification. **Expect this may fail:** the fetch can be auth-gated or return
  rendered/opaque content instead of raw JS — if so, fall back to Route A. **Never
  WebFetch the private URL without the user saying go, and never commit the token.**

## Task 3 — wire the driver from whatever protocol you obtained

Exactly the phase-2 procedure (`agentB-017-wireup-PROMPT.md`, also `PROTOCOL.md`),
condensed:

1. Map emit names → `config.EMIT` for the five actions + `terminal_input` +
   `power`; read each call site for the real payload shape. Watch **upload**
   (ack-callback vs chunked vs progress events → set `expects_ack`/`DONE_WHEN`)
   and **load** (status string vs dedicated done-event).
2. Map on names → `config.LISTEN` (`uart_output`, `status`, `trace_*`); confirm the
   UART-text key → `config.UART_TEXT_KEYS` (first-present wins) and status field →
   `config.STATUS_STATE_KEY`.
3. Completion signals → `config.DONE_WHEN` (status vs event vs ack) for upload /
   load / reset / trace-dump.
4. Connection → `config.CONNECT`: `socketio_path`, `namespace`, and how the
   URL-path token is presented (`auth_key` if the server wants it echoed).
5. **Flip `PROTOCOL_SOURCE = "verified"`**, update `mock_server.py` to the real
   event names, **re-run `python fpga_driver/test_dryrun.py` → must stay green.**
   That proves the wiring is internally consistent before any board contact.
6. Note every ambiguity in your report (chunked upload, ack-vs-event, unexpected
   namespace/auth).

## Real-board run — needs the USER, do NOT self-initiate

After wiring + green mock, a real run requires the **user's explicit go-ahead and
the token'd URL** (never commit the token). Intended command, for reference only:

```
python fpga_driver/run_rtl_smoke.py --url 'https://fpga.corank.info/<token>/' \
    --image <path>/fw_payload.bin
```

Do **not** run it against the board without the user saying go. Report readiness.

## Guardrails (unchanged)

- Additive **test tooling only** — no `llvm/`, no submodule bumps, no monitor /
  `start.S` / allocator / RTL changes. (Building the boot image is an external
  build in `caplifive-system`, not a change to our tree.)
- **Do not operate the board**; do not seek credentials. WebFetch the private URL
  only with the user's explicit OK; the token is **never** committed.
- Commit on `capstone-bootstrap-b`, exact paths, **no `Co-Authored-By:`**, **no
  worker/agent identity in commit messages** (imperative subject), no debug/report
  files. Committing is not gated on an explicit ask — commit validated work as a
  checkpoint; if genuinely in doubt, ask.
- If you boot QEMU for any reason, claim the shared `rootfs.ext2` lock
  (`COORDINATION.md`) — the main lane may be running suites.

## Deliverables

- **Boot image:** either `fw_payload.bin` built (path + how), or a precise report
  of the exact repo/submodule that needs `gh` auth to proceed.
- **Protocol:** `config.py` filled + `PROTOCOL_SOURCE="verified"`, `mock_server.py`
  matching, `test_dryrun.py` green — OR, if neither route yielded it yet, a crisp
  statement of which route you attempted, what it returned, and what the user must
  do (authorize the URL fetch, or run the DevTools capture).
- `PROTOCOL.md` updated from PLACEHOLDER to the observed map (events + payloads +
  handshake), noting the source (client JS / HAR).
- History note → `capstone/agent-handoff/history/DD-MM-YYYY_HH-MM-SS_fpga-diy-experiments.md`.
- Short report: image state, protocol state, and exactly what remains before a
  real-board run can be triggered.

## Framing

Still parallel, not critical-path — the human-driven RTL run already works. The
prize here is that **nothing about an agent-driven run depends on the collaborator
anymore**: we build the image and extract the protocol ourselves, and the only
remaining gate is the user authorizing board contact. Fast, offline, low-risk;
don't touch the board without the user.
