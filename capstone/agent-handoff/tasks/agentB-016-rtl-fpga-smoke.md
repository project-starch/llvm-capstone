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

## Step 0 — secure hardware access first (coordinate)

Access is via the **hardware-access contact**, who has opened a **web deploy interface** to the
RTL/FPGA. You do not control this — **coordinate on the shared Slack channel before building
anything**, confirm you can reach the deploy interface, and note the access path in
COORDINATION. If access is not yet available, report that immediately and stop — do not
sink time into a harness you cannot run.

## Steps

1. **Access** (Step 0). Confirm you can deploy/run on the RTL via the web interface.
2. **Functional parity — start minimal.** Take one known-good artifact that already runs on
   *our* QEMU and run it on the RTL:
   - simplest first: a trivial Capstone binary (a capability materialise + bounded access),
   - then one **corpus repro** — confirm it **faults on the RTL at the same point** it faults
     under QEMU (this is the "works on QEMU ⇒ works on RTL" check).
3. **First real number.** Run the **task-014 borrow-cost-probe** (`tests/runtime-qemu/borrow-cost-probe/`)
   on the RTL and capture whatever the RTL exposes — **cycle count if available** (the real
   timing the proxy stood in for), else retired-instruction count for cross-check against the
   QEMU proxy. Even a single clean number for the three variants (raw / borrow / copy) is the
   deliverable.
4. **Record the gap.** Note anything that differed between QEMU and RTL (a config, a port, an
   opened interface, a missing instruction) — that list is exactly what the meeting wanted
   surfaced while the collaborator is reachable.

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
