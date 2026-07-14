# Agent-B task 016 — RTL/FPGA smoke test (time-sensitive)

*Hand this whole file to Agent-B (`claude-b`), clone `/home/alexey/dev/llvm-capstone-b`,
branch `capstone-bootstrap-b`. Obey `./CLAUDE.md` and the workflow docs.*

## Why this task, and why now

The 13 July meeting established that **real RTL exists** (built by the hardware/RTL
collaborator) and that *"if something works on QEMU it should pretty much work on the
RTL — not a major gap."* The PI's advice was to **test this while the collaborator is
physically reachable this week** — if a port/config tweak is needed, it is far cheaper to
resolve now. This unblocks the paper's **performance** storyline: the RTL/FPGA is the
**cycle-accurate vehicle** that B's task-014 instruction-count proxy explicitly could not be.

This is a **smoke test**, not the full perf eval (that is T7 in `plans/ndss-pivot-master-plan.md`).
Two questions only: **(1) does our Capstone-compiled binary run on the RTL?** and **(2) can we
get a first real number off it?**

## The platform (know this before you plan)

Access is a **browser GUI, not SSH** — a WebSocket console at
`https://fpga.corank.info/<token>/` controlling a **Genesys 2** board running the Capstone
**CVA6/Ariane** core ("CapliFive"). Manual + website capture: `/tmp/capstone/FPGA_Remote.zip`
(RTL repo `github.com/project-starch/capstone-ariane` is **private**; needs `gh auth`/a local
clone to read `tracer.sv`/`cva6.sv`). Salient facts that shape the task:

- **Human-in-the-loop.** Power / Flash bitstream (`.bit`) / Load boot image (`.bin` → JTAG to
  `0x80000000`, ~2 min for 15 MB) / Reset / Trace Dump are **manual clicks**; the Terminal tab
  is UART. An agent cannot drive this directly — **the human operates the ~5 clicks; you
  prepare every artifact and analyse the pasted UART/trace output.**
- **Shared board, 10-min idle timeout.** All users see one state; use **Lock** to hold the
  board during a run. Book a window; batch the runs.
- **Instruments.** Standard RISC-V **`mcycle`/`rdcycle` CSR** for cycle timing; a **hardware
  tracer** (256-entry buffer; CSR `0x810` enable event groups, `0x811` watchpoint phys-addr,
  `0x800` debug-print; dump via UART with switches 0/1) for fine-grained event traces.

## Step 0 — resolve the boot-image question FIRST (blocker)

Before building anything, confirm with the collaborator: **does our existing OpenSBI-monitor +
rootfs image boot on this bootloader, or do we need a bare-metal `.bin` loaded at
`0x80000000`?** (The platform loads a single `.bin` into an "update mode" bootloader — this may
not match our QEMU OpenSBI+rootfs boot.) Also confirm: which **bitstream is preloaded** (is it
the tracer-enabled Capstone build?) and whether a **scriptable API** exists behind the web UI.
Coordinate on Slack; note the answers in COORDINATION. If access/boot path is not yet settled,
report that and stop — do not sink time into an image that won't boot.

## Steps

1. **Boot path settled** (Step 0). Boot-image format confirmed and access reachable.
2. **Functional parity — start minimal.** Take one known-good artifact that already runs on
   *our* QEMU and run it on the RTL (as whatever `.bin` the boot path requires):
   - simplest first: a trivial Capstone binary (a capability materialise + bounded access),
   - then one **corpus repro** — confirm it **faults on the RTL at the same point** it faults
     under QEMU (this is the "works on QEMU ⇒ works on RTL" check).
3. **First real number — via the `mcycle` CSR.** Port the **task-014 borrow-cost-probe**
   (`tests/runtime-qemu/borrow-cost-probe/`) to read **`mcycle`/`rdcycle`** in place of the
   QEMU `csrdicount` instruction-count proxy, bracketing the same raw / borrow / copy loops, and
   have it **print the cycle deltas over UART**. Load + run on the RTL → the three numbers are
   the deliverable (this is the **real cycle-accurate** figure the proxy stood in for; it fills
   the paper's `evaluation.tex` perf placeholder). Keep an instruction-count readout too, for a
   direct cross-check against the QEMU proxy.
4. **Record the gap.** Note anything that differed between QEMU and RTL (boot format, a config,
   a port, a missing instruction, `mcycle` availability in the Capstone build) — that list is
   exactly what the meeting wanted surfaced while the collaborator is reachable.

*(Deferred to T7/T10, not this task: the hardware **tracer** — a watchpoint at the borrowed
address showing the fault fires at the contract point (security demo), and the feature-disable
**overhead breakdown**. Note whether the preloaded bitstream exposes the tracer, but do not
build the breakdown here.)*

## Deliverables

- `capstone/tests/rtl-smoke/RESULTS.md` — access path, what ran, functional-parity outcome
  (repro faults at same point Y/N), any first number (cycles or instr, labelled), and the
  QEMU↔RTL gap list.
- History trail → `history/DD-MM-YYYY_HH-MM-SS_rtl-fpga-smoke.md`.
- Report: does it run, first number, and the gap list — flag anything needing the
  collaborator's help **now** while reachable.

## Scope / lane rules

- **The RTL is external** (the collaborator's tree / the deploy interface) — do **not** vendor
  it into `llvm/` or bump any submodule for it. Additive test artifacts only.
- Reuse existing repros/probes; do not modify their semantics, `start.S`, the monitor, the
  allocators, or `capstone-c`.
- Commit on `capstone-bootstrap-b`, exact paths, **no `Co-Authored-By:`**, no debug/report
  files. If you also boot our QEMU for a side-by-side, claim the rootfs lock and release it.

## Closing note

Priority is **parity + one real number + the gap list**, delivered fast while access and the
collaborator are available — not a complete perf harness. The full end-to-end perf eval
(micro-benchmarks + SQLite CVE benchmark + breakdown) is the follow-on (T7/T10) once the RTL
path is proven here.
