# Tooling for silicon-only miscompute: differential trace + a claim ledger

**Status:** proposal, nothing built. The hardware half needs a bitstream rebuild, which is
ask-first. Written 2026-08-07 after the nested-loop capability-index defect (see
`history/07-08-2026_02-30-00_nested-loop-capability-index-iteration-loss.md`).

## TL;DR

Two tools, not one.

* **Mechanism** — a differential retire-trace between reference model and silicon, answering
  *"first divergence at instruction N, pc=X, the load returned A instead of B."*
* **Reasoning** — a claim-and-confound ledger that re-checks stored conclusions against every
  new measurement.

The mechanism half is largely off-the-shelf, and **two of its three components are already in
`capstone-ariane` and simply not connected to each other**. The reasoning half has no product.
The zero-cost version of both: **never record one number when the harness gave you four** — the
single biggest discriminator of the investigation (that the failures split into two distinct
faults) sat unread in cycle counts printed on every `RESULT` line since the ladder existed.

## 1. Why a debugger is not the answer

The investigation always knew the *value* — "QEMU says 576, silicon says 567" — and never the
*instruction*. Every technique that could have named the instruction was blocked:

| Constraint | Evidence |
|---|---|
| **Instruments perturb the fault away** | A sentinel array (+32 B) and taking `&qc` (+16 B) each *cured* the failure by shifting frame slots. Anything that adds a local, changes frame size, or moves code layout measures a different program. |
| **A wedged core cannot be interrogated** | The debug mux returns byte-identical values on passing, wedging and entry-stalled runs; the debug register path has twice returned AXI error-slave junk (`0xca11ab1ebadcab1e`); `trace_dump()` hangs the board hard. Capture must happen *before* the wedge and survive it. |
| **Experiments are expensive and low-bandwidth** | A board boot is ~4–6 min, dominated by a ~148 s JTAG upload, capped at 4 domains, yielding one number each. |
| **The cheap, high-visibility model does not reproduce it** | Verilator turns around in ~13 s with full internal visibility — and the defect does not appear there. |

The tool has to live in the gap between those last two rows: **simulation's visibility at the
board's fidelity.**

Two capabilities would have collapsed the whole effort:

1. **A retire trace on the FPGA** — circular capture of `(pc, load addr, load data, store addr,
   store data)` for the last N retired instructions, dumped over the existing debug path on a
   trigger. The fault is deterministic and periodic, so a few hundred entries around a trigger
   are enough.
2. **A hardware data watchpoint** — *"halt when the word at `sp+0x1c` deviates from the expected
   sequence."* That one question cost roughly ten board boots of manual bisection.

## 2. What already exists — including in this repo

### 2.1 Differential trace is a solved problem, aimed at the wrong target

Model-vs-RTL divergence checking is standard RISC-V verification practice:

* **RVFI** (RISC-V Formal Interface) — the retire-trace signal bundle. **Present in our tree:**
  `core/cva6_rvfi.sv`, `core/cva6_rvfi_probes.sv`, `core/include/rvfi_types.svh`,
  `corev_apu/tb/rvfi_tracer.sv`. Critically, `rvfi_probes_o` is a **real output port on the
  core** (`core/cva6.sv:437`), not a testbench-only construct — the signals already leave
  `cva6`.
* **Reference-model scoreboard scaffolding**, also in tree:
  `verif/tb/core/uvmc_rvfi_reference_model_pkg.sv`, `verif/tb/core/uvmc_rvfi_scoreboard_pkg.sv`.
* **RVVI + ImperasDV** (OpenHW ecosystem) — asynchronous lockstep RTL-vs-model comparison whose
  output format is literally "first divergence".
* **Dromajo** — a RISC-V reference model built for co-simulation against RTL with checkpointing.
* **Spike `--log-commits`**, `riscv-dv`, `core-v-verif` step-and-compare.

**The catch, and it is the entire problem:** all of these compare **RTL simulation** against an
ISS. Our defect does not reproduce under Verilator. The mature tooling therefore covers exactly
the case we do not have, and says nothing about **FPGA-vs-model**, which is the only comparison
that would have answered this.

### 2.2 On-FPGA capture — vendor IP, in tree, never instantiated

This is the actionable finding. `corev_apu/fpga/xilinx/xlnx_ila/` contains a **Vivado ILA IP
wrapper with its own build recipe** (`tcl/run.tcl`): 8 probes, `C_DATA_DEPTH 16384`, triggered
capture, read back over the JTAG path we already drive.

That is precisely the "circular trace buffer dumped on a trigger" described in §1 — as vendor IP,
already packaged for this board. **Nothing instantiates it:** grepping the FPGA build, `xlnx_ila`
is referenced only by its own Makefile.

So the honest statement is: *RVFI probes already exit the core, and a 16,384-deep triggered
capture buffer already has a build recipe. Connecting them is a wiring exercise, not research.*

Real but ordinary caveats: BRAM cost, timing closure at the probe width, and a likely need to
narrow the captured set to `(pc, addr, data)` to fit 8 probes. And it needs a **new bitstream —
ask-first, always.**

Productised equivalents if we ever want them: the ratified **RISC-V E-Trace/Nexus** spec, as
implemented by SiFive Insight and Siemens Tessent Embedded Analytics.

### 2.3 Hardware watchpoint — standard in the ISA, apparently absent here

The **RISC-V Debug spec trigger module** (`tdata1`/`mcontrol`) natively supports address- and
data-match watchpoints; "trap when this location takes this value" is a spec feature, not a
custom build.

A grep found **no trigger-module implementation under `core/`** — the `mcontrol`/`tdata1` hits
were all in `cv32e40s`/`cv32e40x` testbenches under `verif/core-v-verif/`. This is a negative
result from a single search and should be confirmed before being relied on. If it holds, the
absence is directly why the stack-slot question needed ten boots.

### 2.4 The reasoning half — no product, but old primitives

Half the cost of the investigation was not measurement but bookkeeping: **five conclusions were
retracted in one session, two of them already recorded as root causes.** No integrated tool
exists, but every piece has a mature analogue we are not using:

| Need | Existing analogue |
|---|---|
| Record everything the harness emitted, not the one number being looked for | Experiment trackers — MLflow, Weights & Biases, DVC, Sacred. Solved, zero-cost to adopt. |
| "These two variables were never varied independently" | **Alias structure in design of experiments** — a ~70-year-old formalism. JMP, Minitab, R's `FrF2` compute which effects a design cannot separate. |
| Automated minimisation of a failing case | Delta debugging (`ddmin`), C-Reduce, `llvm-reduce`, `git bisect`. |
| Claim-plus-evidence ledger with review | Assurance-case tooling (GSN, Adelard ASCE) — right shape, far too heavyweight, re-checking is manual. |

What does not exist is the **join**: a ledger wired into a hardware experiment loop that re-runs
stored claims against each new data point automatically.

## 3. What the ledger would have caught

Concretely, from data already collected at the time:

* **The bits-[3:2] law** would have been flagged the moment two builds with byte-identical frame
  geometry returned 906 and 909. That contradiction was in hand before the law was ever
  formulated.
* **`qc == k+8`** held in every build ever made, and **`p == k+4`** still does. A confound check
  reporting *"variables X and Y have never been varied independently across your sample"* kills
  both overfits before they reach a commit message.

Requirements, in order of value:

1. Every claim stored with the builds supporting it **and the geometry of each build**.
2. Automatic re-check of all stored claims on each new measurement; flag any that has become
   non-functional.
3. Mechanical confound detection over the sample of builds.

A JSON claim file plus a re-check script captures most of this. It does not need to be good to
be worth more than it costs.

## 4. Recommendation

Ordered by value per unit of risk:

1. **Adopt "log the whole instrument output" immediately.** No new tooling, no board time, no
   bitstream. This is the change that would have surfaced the two-fault split months earlier.
2. **Build the claim/confound ledger as a script.** Cheap, offline, no hardware dependency.
3. **Confirm whether the debug trigger module is implemented.** If it is, a data watchpoint is
   available today and replaces the most expensive class of bisection we run.
4. **Scope the ILA + RVFI wiring.** Highest mechanism value; requires a bitstream rebuild, so it
   does not start without an explicit go-ahead.

Item 4 is the only one that touches hardware, and it is deliberately last for that reason.
