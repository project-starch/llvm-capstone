# Agent-B task 017 — FPGA web-console automation (DIY Socket.IO extraction)

*Hand this whole file to Agent-B (`claude-b`), clone `/home/alexey/dev/llvm-capstone-b`,
branch `capstone-bootstrap-b`. Obey `./CLAUDE.md` and the workflow docs. Roles are
not strict now — this is web/tooling work, fine for either lane.*

## Why this task, and why now

The paper's performance storyline needs the **RTL cycle-accurate** number. The
**human-driven** run is now fully staged and QEMU-validated (`tests/rtl-smoke/`,
commit `f509b31`: `mcycle`-in-domain works, borrow-cost + revoke-cost FPGA ports
reproduce the reference, `run-revoke-cost-fpga-qemu.sh --parse-uart` turns pasted
UART into the paper breakdown). What is **still blocked is AGENT-driven** runs:
the board's console is a **browser Socket.IO GUI with no scriptable API**. The
collaborator agreed to expose the Socket.IO protocol but gave **no timeline**, so
this task attacks the blocker ourselves rather than waiting.

**This is parallel, not critical-path.** The human-driven run needs nothing from
this task. So: do the offline scaffold + protocol map now; the board wire-up
waits on protocol access (below). Good use of spare tokens, not a gate.

**TIME-SENSITIVE (2026-07-15).** The collaborator committed to **sending us the
web UI's client JS on Thursday evening BST (2026-07-16)**, then is **flying out
and unreachable**. So Thursday evening is the last easy window to ask follow-ups.
Readiness is the goal: **have the `python-socketio` driver scaffold fully built
BEFORE Thursday** so that the moment the JS arrives it is a ~10-minute wire-up +
a same-evening sanity check, and any gaps can be raised while he is still
reachable. Do NOT wait for the JS to start — build the scaffold now.

## The reality (know before planning)

- The console is `<FPGA-CONSOLE-URL>` (get the exact token'd URL
  from the user; it is in the RTL platform memory / Slack). It drives a Genesys 2
  Capstone core over **Socket.IO** (WebSocket). Controls: Power, Bitstreams,
  **Boot Images** (upload `.bin` → JTAG to `0x80000000`), Reset, **Terminal**
  (UART), Trace Dump, Virtual Switches/LEDs, Lock.
- The **event names + payloads live in the web app's client JS** — which is NOT
  in our local capture. `/tmp/capstone/FPGA_Remote/` has only the Manual (`.md`)
  and a Website PDF; neither documents the Socket.IO API. So the protocol must be
  obtained from the live site (below).
- No browser automation in the agent sandbox — you cannot click the GUI. The end
  goal is a **headless `python-socketio` driver** the agent (or a human) runs.

## Goal

A `python-socketio` driver that performs the five board actions headlessly, each
mapped to the real Socket.IO event(s):

1. upload a boot image (`fw_payload.bin` → `0x80000000`) and await load-complete;
2. reset and wait for the Linux prompt on the UART stream;
3. read UART until a marker line (e.g. `measurement complete`);
4. set virtual switch N (for Trace Dump);
5. trigger Trace Dump and receive the end-of-dump frame.

Then it drives the `tests/rtl-smoke/` run end-to-end: upload → run the
`.user`+`.dom` pairs over the UART → capture `RESULT` lines → feed them to
`run-revoke-cost-fpga-qemu.sh --parse-uart`.

## Steps

0. Read `tests/rtl-smoke/README.md` + `RESULTS.md` (what the run needs, the two
   ports, the parser). Read `plans/perf-cheri-vs-capstone-qemu.md` for context.
1. **Get the protocol.** Primary path — the **collaborator is sending the client
   JS Thursday evening BST**; the moment it lands, grep it for `socket.emit(` /
   `.on(` / `io(` and extract event names, payload shapes, and any auth/handshake
   token or namespace, then wire the scaffold (step 2) to it. Same-day hedges, in
   case his fly-out "will try" slips: (a) **ask the user before fetching the
   private token'd URL**, then WebFetch the console URL + its referenced JS
   bundle(s) and extract the same way; (b) a precise **DevTools capture checklist**
   (Network → WS filter → click each control once → export frames/HAR) so the user
   captures the live event stream in one pass. Any of the three yields the events.
2. **Build the driver scaffold NOW (needs no JS):** a `python-socketio` client
   with the connection/handshake, the five action stubs, event-wait helpers, and
   a single config block naming the events to fill in. So when step 1 lands, wiring
   is ~10 minutes.
3. **Validate what's validatable offline:** arg parsing, the run flow, a dry-run
   against a mock Socket.IO server. Real board validation needs a human slot.
4. **Integrate** with `run-revoke-cost-fpga-qemu.sh` (its `--parse-uart` already
   parses UART) so a full agent-driven perf run is one command once the events
   are known.

## Deliverables

- Driver + scaffold under `tests/rtl-smoke/fpga_driver/` (Python; keep it self-
  contained, `python-socketio` the only dep).
- A **protocol map** doc: event names + payloads + handshake (or, if step 1 was
  fallback-only, the DevTools checklist + the mock the scaffold was tested against).
- History trail → `history/DD-MM-YYYY_HH-MM-SS_fpga-automation-diy.md`.
- Report: what the WebFetch attempt yielded, the scaffold state, and exactly what
  is still needed to finish (the events, or the user's capture).

## Scope / lane rules

- Additive **test tooling only** — no `llvm/` or submodule changes; do not touch
  the monitor, `start.S`, allocators, or the RTL tree.
- **Do not operate the board** and do not attempt to obtain credentials; WebFetch
  the client JS only with the user's go-ahead on the private URL.
- Commit on `capstone-bootstrap-b`, exact paths, **no `Co-Authored-By:`**, no
  worker/agent identity in commit messages, no debug/report files.

## Closing note

The prize is a headless driver so the perf sweep can run without a human clicking
five buttons. But the human path already works — so deliver the offline scaffold +
protocol map fast, and don't block on the collaborator or on board access to make
progress.
