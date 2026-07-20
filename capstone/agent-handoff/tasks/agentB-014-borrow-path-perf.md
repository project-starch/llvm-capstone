# Agent-B task 014 — borrow-path cost measurement (paper deliverable 2)

*Hand this whole file to Agent-B (`claude-b`), clone `/home/alexey/dev/llvm-capstone-b`,
branch `capstone-bootstrap-b`. **Gated:** confirm with the human before starting —
see "Read this first" — because the vehicle question below may change the task.*

---

You are **Agent-B** (compiler/codegen + emulator + measurement lane). Obey
`./CLAUDE.md` and `capstone/agent-handoff/{MULTI-AGENT-WORKFLOW,COORDINATION}.md`.

## Why this task

The paper's conclusion defers two deliverables; this is the second: *"a
measurement showing the capability-mediated borrow path stays close to raw
pointers and well below the copy baseline."* The preliminary-validation subsection
(`solution.tex` §Preliminary validation, Table `tab:valid`) now shows the
mechanisms *work* on real SQLite; this task is meant to show they are *cheap*.

## Read this first — the vehicle problem (why this is gated)

Our "after" runs on the **QEMU functional model** of Capstone. QEMU is a
functional/ISA emulator: it executes the right instructions and faults, but it
**does not model microarchitectural timing** (no pipeline, cache, or cycle
model). Therefore:

- On QEMU we can honestly measure a **dynamic-instruction-count / operation-count
  proxy** — how many extra instructions the borrow path costs vs a raw pointer
  and vs a copy — and argue overhead from that.
- We **cannot** produce real cycle-accurate timing on QEMU. A "stays close to raw
  pointers" *timing* claim needs a cycle-accurate vehicle (the Capstone hardware
  RTL / an FPGA or gem5-class model), which is **not in our tree**.

**So before building anything, the human/lead must confirm which claim we are
making:** (a) an instruction-count proxy on the functional model (achievable
now, must be labelled as a proxy, not silicon timing), or (b) a real timing
number that requires a cycle-accurate vehicle we would first have to obtain. Do
not start until this is answered — building an instruction-count harness is wasted
if the target is (b) on hardware we do not yet have.

## Scope — measure, do not build the co-designed binding

- This is **deliverable 2 only**. Do **not** build the full engine-minted binding
  (deliverable 1). Reuse the existing literal repros and the `runtime-qemu`
  harness as the substrate.
- Measure three variants of the same boundary operation (borrow one result value
  across the host/engine boundary and use it):
  1. **raw pointer** — today's zero-copy path, no capability;
  2. **capability borrow** — mint a revocation cap + delegate a linear cap +
     access + revoke (the paper's mechanism);
  3. **copy baseline** — the `TRANSIENT`-style defensive copy the paper says the
     mechanism replaces.
- Report the **dynamic instruction count** (and, if cheap, capability-op count)
  per operation for each variant, over enough iterations to be stable, at
  `-O2`. Present overhead as ratios (borrow vs raw, copy vs raw).

## Strict scope (lane rules)

- Additive only. A codegen/emulator change **may** be legitimate here (e.g. an
  instruction-count readout) — if you bump the `capstone-qemu` gitlink, that is
  your lane; **log it** in COORDINATION and push the submodule branch before the
  gitlink bump. If you touch `llvm/`, coordinate first (shared tree).
- Do not touch A's row2/3/5/7/9/11/14 repros, `start.S`, the monitor, or
  `capstone-c` beyond what a measurement harness strictly needs.
- Commit on `capstone-bootstrap-b`, exact paths, **no `Co-Authored-By:`**, no
  debug/report files. Claim the QEMU rootfs lock if you boot; release it after.

## Steps

1. **Confirm vehicle (a) vs (b) with the human** before building anything.
2. Build a microbenchmark harness for the three variants above on the existing
   `runtime-qemu` substrate; make the borrow variant use the real primitives
   (SPLIT/MREV/REVOKE) the validation used.
3. Collect dynamic-instruction counts, enough iterations for stability, `-O2`;
   compute the ratios.
4. Write the numbers + the **explicit methodology caveat** (functional-model
   proxy, not cycle-accurate) into a results note and a small table the paper can
   cite. Do **not** put a number in the paper that is not labelled as a
   functional-model proxy.
5. **Report** with the ratios, the vehicle used, and whether any `llvm/` /
   `capstone-qemu` gitlink changed. Trail →
   `history/DD-MM-YYYY_HH-MM-SS_borrow-path-perf.md`.

## Closing note

A clean instruction-count proxy is a legitimate, publishable overhead argument for
a design/characterization paper *if labelled honestly*; a mislabelled "timing"
number on a functional model is a reviewer trap. The vehicle decision in step 1 is
the whole ballgame — that is why this task is gated, not fired.
