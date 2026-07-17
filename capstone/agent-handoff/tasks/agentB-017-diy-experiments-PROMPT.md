# Follow-up prompt for Agent-B — task 017 phase 3 (DIY: take Jason off the critical path)

*Paste everything below the line into `claude-b`. Self-contained. This is the
follow-on to the scaffold (phase 1) and the wire-up brief (phase 2,
`agentB-017-wireup-PROMPT.md`). New direction: stop waiting on the collaborator's
Thursday JS — get the protocol and the boot image ourselves and close the whole
loop autonomously (the user has authorized end-to-end board operation).*

---

You are Agent-B, continuing task 017 in `/home/alexey/dev/llvm-capstone-b`,
branch `capstone-bootstrap-b`.

```bash
cd /home/alexey/dev/llvm-capstone-b && git fetch origin && git pull --rebase
source capstone/tests/capstone-test-env.sh
```

## Authorization (2026-07-15) — run the whole flow autonomously

The user has authorized you to do **everything end-to-end without them in the
loop**: fetch the private token'd console URL, extract the protocol, build the
boot image, wire the driver, **and operate the physical board** to run the perf
sweep — all on your own judgement, no per-step go-ahead needed. The board URL
(with its access token) is supplied to you separately at paste time; use it at
runtime only.

Two constraints remain absolute and are NOT relaxed by this authorization:
- **Never commit or log the token.** It goes nowhere in the tree — not
  `config.py`, not `PROTOCOL.md`, not a history note, not a commit message. Pass it
  via `--url` at runtime; if you must persist it, put it in a **gitignored** local
  file and confirm `git status` never shows it.
- **Be a good citizen on the shared board.** It is shared hardware (all users see
  one state, ~10-min idle timeout). Take the **Lock** while running and release it
  when done; keep runs short and targeted; leave the board at a clean prompt (do
  not power it off — others may need it); and if you arrive to find it in an
  unexpected/in-use state, **back off** rather than stomp on someone's session.

Autonomy means no approval gate — it does **not** mean silence: still report
readiness, results, and any ambiguity, and still obey every lane rule below.

## The reframe (why this task)

Your `fpga_driver/` scaffold is **done and verified green**; it is in a holding
pattern waiting for the Socket.IO protocol. The plan was to get that protocol from
the collaborator's client JS on **Thursday evening BST**, after which he is
unreachable. That is fragile ("will try"). This task removes him from the critical
path: **the protocol and the boot image are both obtainable DIY, without him.**

Only **three** things gate an agent-driven RTL perf run, and none of them needs the
collaborator — and the user has now authorized you to do all three yourself:

| Piece | Needs collaborator? | DIY path (this task) |
|---|---|---|
| Socket.IO protocol (event names + payloads) | **No** | fetch the live console's own client JS (or DevTools HAR fallback) |
| Boot image `fw_payload.bin` (our test overlay) | **No** | build via `caplifive-system` — no board |
| Operating the physical board | **No** | drive it via the wired `python-socketio` driver + the token'd URL you were handed |

So: build the image, extract the protocol, wire the driver, and — once the mock is
green — run the board and report the numbers.

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
- **Route B — fetch the live client JS ourselves (authorized; do it).** The app's
  JS bundle is served from the **same origin as the token'd URL** (supplied at paste
  time). WebFetch the console page, find the referenced JS bundle(s), fetch them,
  and run `python fpga_driver/extract_from_js.py <bundle>.js` → emit/on/io names
  survive minification. **Expect this may fail:** the fetch can be auth-gated or
  return rendered/opaque content instead of raw JS — if so, fall back to Route A
  (produce the tightened DevTools checklist and flag that you need a human-captured
  HAR). Never commit or log the token, wherever it appears in the URL.

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

## Real-board run — authorized; run it yourself

Once the driver is wired (`PROTOCOL_SOURCE="verified"`) **and the mock dry-run is
green**, run the real perf sweep against the board on your own — the green mock is
your gate, not a human. Command:

```
python fpga_driver/run_rtl_smoke.py --url '<token-URL supplied at paste time>' \
    --image <path>/fw_payload.bin
```

Take the Lock first, run the `.user`+`.dom` sweep, capture the UART `RESULT` lines,
feed them to `run-revoke-cost-fpga-qemu.sh --parse-uart`, release the Lock, and
report the breakdown (bump / norevoke / revoke → revoke-at-free delta) next to the
QEMU reference. Keep the token out of every committed/logged artifact. If anything
about the live protocol differs from what you wired (upload chunking, ack-vs-event,
namespace/auth), adapt `config.py`, re-green the mock, and note it in your report.

## Guardrails (unchanged)

- Additive **test tooling only** — no `llvm/`, no submodule bumps, no monitor /
  `start.S` / allocator / RTL changes. (Building the boot image is an external
  build in `caplifive-system`, not a change to our tree.)
- Operating the board and fetching the token'd URL are **authorized** (see the
  Authorization section) — but the **token is never committed or logged**, and be a
  good citizen on the shared board (Lock, short runs, clean state, back off if it's
  in use). Do not seek any credentials beyond the URL you were handed.
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
  matching, `test_dryrun.py` green — OR, if Route B's fetch came back opaque and you
  genuinely cannot reach the events, the tightened DevTools checklist + a crisp
  statement of what the fetch returned (this is the one case where a human capture
  is the only remaining option).
- **Board results:** the perf sweep run against the board — the parsed breakdown
  (bump / norevoke / revoke → revoke-at-free delta) beside the QEMU reference — or,
  if a real gate stopped you (gh-auth for the image, opaque protocol), exactly what
  blocked and where.
- `PROTOCOL.md` updated from PLACEHOLDER to the observed map (events + payloads +
  handshake), noting the source (client JS / HAR). **No token anywhere in it.**
- History note → `capstone/agent-handoff/history/DD-MM-YYYY_HH-MM-SS_fpga-diy-experiments.md`.
- Short report: image state, protocol state, board-run results, and anything that
  blocked full end-to-end autonomy.

## Framing

The human-driven RTL run already works, so nothing here is on the paper's critical
path — but the user has authorized you to close the whole loop yourself: build the
image, extract the protocol, wire the driver, and run the board, no human in the
loop. Deliver the real cycle-accurate number if you can reach it; where a genuine
gate stops you (gh-auth for the image, opaque protocol fetch), report precisely
what blocked. Move fast, keep the token out of the tree, and be a considerate
tenant of the shared board.
