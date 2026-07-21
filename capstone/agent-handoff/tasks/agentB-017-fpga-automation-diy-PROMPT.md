# Kickoff prompt for Agent-B — task 017 (FPGA web-console automation)

*Paste everything below the line into a fresh `claude-b` session. It is
self-contained: full context, the docs archive, what is already done, the
mission, and the guardrails. The committed brief is
`capstone/agent-handoff/tasks/agentB-017-fpga-automation-diy.md` — this prompt
wraps it with the surrounding context you need to start cold.*

---

You are Agent-B (the second working lane) on the Capstone LLVM project. Work in
your clone `/home/alexey/dev/llvm-capstone-b`, branch `capstone-bootstrap-b`.

## First: orient

```bash
cd /home/alexey/dev/llvm-capstone-b
git fetch origin
git pull --rebase                     # get your branch current
source capstone/tests/capstone-test-env.sh
```

Read, in order: `./CLAUDE.md`, `capstone/agent-handoff/README.md`,
`capstone/agent-handoff/state/current-state.md` (the "Latest (2026-07-15)"
banner), and your task brief
`capstone/agent-handoff/tasks/agentB-017-fpga-automation-diy.md`.

## Project context (one paragraph)

This is a fork of LLVM + QEMU for **Capstone**, a CHERI-like capability RISC-V
architecture (`capstone64-unknown-elf`; 128-bit tagged capabilities). The team is
writing a paper whose **performance storyline** needs a **cycle-accurate** number
for Capstone's temporal-safety mechanism (revoke-at-free), to place opposite
CHERI's revocation-sweep cost. The cycle-accurate vehicle is a **real FPGA**: a
Genesys 2 board running the Capstone **CVA6/Ariane** core ("CapliFive"),
accessed through a **browser Socket.IO console** (no SSH, no scriptable API).

## What is ALREADY done — build on it, do not redo

The **human-driven** RTL run is fully staged and QEMU-validated on
`origin/capstone-bootstrap` (commit `f509b31`, dir `capstone/tests/rtl-smoke/`):

- `mcycle` reads cleanly inside a Capstone domain under QEMU (the board gates the
  unprivileged `cycle`, so the probes read `mcycle`).
- Two FPGA measurement ports, both reproducing the reference numbers under QEMU:
  **borrow-cost** (`borrow_cost_fpga.*`) and **revoke-cost**
  (`revoke_cost_fpga.*`, the temporal-safety headline: bump 7 / norevoke 60 /
  revoke 65 → revoke-at-free **+5**, O(1)).
- `run-revoke-cost-fpga-qemu.sh` builds+runs+parses; its
  `--parse-uart <file>` mode turns the board's pasted UART `RESULT` lines into the
  paper breakdown.
- Read `capstone/tests/rtl-smoke/README.md` and `RESULTS.md` for the full picture
  (they are on `origin/capstone-bootstrap`; `git fetch` then read them, e.g.
  `git show origin/capstone-bootstrap:capstone/tests/rtl-smoke/README.md`).

**The only thing NOT yet solved is AGENT-driven runs**: the console is a browser
GUI, so the perf sweep currently needs a human clicking ~5 buttons per run. Your
task removes that.

## Your mission

Build a headless **`python-socketio` driver** for the FPGA web console that
performs the five board actions programmatically, then drives the `rtl-smoke`
run end-to-end (upload boot image → run the `.user`+`.dom` pairs over UART →
capture `RESULT` lines → feed `run-revoke-cost-fpga-qemu.sh --parse-uart`).

The five actions (see the manual, below):
1. upload a boot image (`fw_payload.bin` → JTAG to `0x80000000`), await complete;
2. reset, wait for the Linux prompt on the UART stream;
3. read UART until a marker line (e.g. `measurement complete`);
4. set a virtual switch N (for Trace Dump);
5. trigger Trace Dump, receive the end-of-dump frame.

## The platform docs (READ THESE)

A documentation archive is on disk:

- **`/tmp/capstone/FPGA_Remote.zip`** (89 KB), extracted to
  **`/tmp/capstone/FPGA_Remote/FPGA_Remote/`**, containing:
  - `FPGA_Remote_Manual.md` — the user manual: Layout, Power, Bitstreams, Boot
    Images, Terminal (UART), Reset, Virtual LEDs, Virtual Switches, and the
    **tracer** (256-entry buffer; CSR `0x810` event-group enable, `0x811`
    watchpoint phys-addr, `0x800` debug-print; dump via UART with switches 0/1).
  - `FPGA_Remote_Website.pdf` — a print/screenshot of the web console UI.
  - **Note:** the archive documents the UI + tracer but **does NOT contain the
    web app's client JS or the Socket.IO event names/payloads.** So the protocol
    itself has to come from the live site (see "Get the protocol" below).
- The live console URL is `https://fpga.corank.info/<token>/`
  (private access token in the path — do NOT fetch it without the user's OK).

## Get the protocol — time-sensitive

The RTL collaborator committed to **sending us the web UI's client JS on Thursday
evening BST (2026-07-16)**, then is **flying out and unreachable**. So:

- **Build the driver scaffold NOW** — it needs no JS: the `python-socketio`
  client, connection/handshake, the five action stubs, event-wait helpers, and a
  single config block naming the events to fill in. Aim for "the JS is a
  ~10-minute wire-up + a same-evening sanity check."
- When the JS lands: grep it for `socket.emit(` / `.on(` / `io(` — extract event
  names, payload shapes, namespace, any auth/handshake token — and wire the
  scaffold to it.
- Same-day hedges if his fly-out "will try" slips: (a) **ask the user first**,
  then WebFetch the console URL + its referenced JS bundle(s) and extract the same
  way; (b) produce a precise **DevTools capture checklist** (Network → WS filter →
  click each control once → export frames/HAR) so the user captures the live event
  stream in one pass and you map it from that.

## Deliverables

- Driver + scaffold under `capstone/tests/rtl-smoke/fpga_driver/` (self-contained
  Python; `python-socketio` the only dependency).
- A **protocol map** doc (events + payloads + handshake), or — if only the
  fallback was possible — the DevTools checklist + the mock you tested against.
- History trail → `capstone/agent-handoff/history/DD-MM-YYYY_HH-MM-SS_fpga-automation-diy.md`.
- A short report: what the client JS yielded, the scaffold state, and exactly
  what remains to finish.

## Scope / lane rules (hard)

- **Additive test tooling only.** No `llvm/` or submodule changes; do not touch
  the monitor, `start.S`, allocators, or the RTL tree.
- **Do not operate the board** and do not seek credentials. WebFetch the private
  URL only with the user's explicit go-ahead.
- Commit on `capstone-bootstrap-b`, exact paths, **no `Co-Authored-By:`**, **no
  worker/agent identity in commit messages** (imperative subject describing the
  change), no debug/report files.
- Serialize any QEMU boot on the shared `rootfs.ext2` lock; announce in
  `COORDINATION.md` if you run one (the main lane may be running QEMU suites).

## Why this framing

The human-driven RTL run already works, so this is **parallel, not
critical-path** — the point is to have the driver ready so the eventual perf
sweep runs without a human in the loop, and to be ready to wire up + sanity-check
the board owner's JS Thursday evening while he is still reachable. Deliver the offline
scaffold + protocol map fast; don't block on the board or the collaborator to
make progress.
