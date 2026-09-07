# What makes a timing-failing bitstream usable here: the census, not the slack

**Status:** the acceptance criterion actually in force on this design, replacing the one
`run.tcl` states. Derived 2026-08 to 2026-09 across five routed builds.

## The problem with the stated criterion

`run.tcl` says: negative post-route WNS means DO NOT FLASH. **No bitstream this project has ever
produced meets it.** Five routed builds, five negative, each from its own post-route report on
`clk_out1_xlnx_clk_gen`:

    39b21639d  -10.629 ns   96,727 failing     <- least bad
    76b7f2afc  -12.084       93,200
    84ed6eafb  -13.516      103,197
    52fa06b9d  -14.125      104,238
    80843404c  -16.400      102,769            <- worst, and it is the RESIDENT image on which
                                                  every S-12 board result rests

A criterion that forbids every flash already performed is a mis-stated premise, not a rule. And
"restore retiming" is not the remedy it names: retiming-ON builds are negative too.

## What actually licenses these bitstreams

**Every failing endpoint originates from `dom_switcher/cur_idx_q_reg`.** Verified as a CENSUS —
by-startpoint and by-endpoint sums each equal to the design's own failing-endpoint count, with
`dom_switcher` as a positive control in the same query as the `s07_ldc0` / `load_unit` / `lsu_i`
ones:

    84ed6eafb   103,197 / 103,197   (103,193 on bit [5], 4 on [4], 2 DDR)
    52fa06b9d   104,457 / 104,457   (all on bit [0]) -- every one of the +1,260 over base landed
                                     in the same cone
    80843404c   102,769 / 102,769   (bit [3])

`cur_idx` toggles **only** during a domain switch, and `commit_stage.sv:494` forces
`flush_commit_o` high for the whole of one while the controller flushes the frontend throughout.
So switcher activity and domain-body execution are mutually exclusive: **every failing path in
the design is inert while the code under test runs.**

That is a STRUCTURAL property, not a margin. It is why WNS -16.400 is tolerable, and it is the
reason board results on these images mean anything.

## Why this is the criterion to gate on

`run.tcl`'s own warning is that a timing-failing bitstream "behaves intermittently and
data-dependently — the exact signature of the defect under investigation, with no way to separate
the two afterwards." That hazard is real. The census is what excludes it, and nothing else does:

* per-image clustering, ~54% wedge rates and data-dependence are ALSO what a timing-marginal
  design produces. The clustering alone could never have separated instrument from subject.
* the census can. It settled that question for the entire 46-draw S-12 corpus, not just the next
  boot, and it settled it by measurement rather than argument.

## AMENDED 2026-09-04: name the PROPERTY, not the module

The criterion below was written as "every failing endpoint originates from
`dom_switcher/cur_idx_q_reg`". That conflated two things which had coincided on every build up to
that point:

    what it SAID    the launch register is in the dom_switcher module
    what it MEANT   the launch register is INERT while a domain body executes

`cur_idx` satisfied both, so nothing separated them. **Arm 1 of the S-12 fix separated them.** Its
census is 100% within `dom_switcher` — the criterion as written is literally satisfied — but the
registers are not `cur_idx`:

    101,573   dom_switcher/_thread_0_event_reg_87_q_reg[0]
        209   dom_switcher/_init_0_reg

`_thread_0_event_reg_87_q` does not appear in the `.anvil` source at all. It is a
compiler-generated event-join register in the switcher's thread machinery, a rendezvous flag for
two predecessor events, and the `cur_idx` argument says nothing about it.

**Both are inert, established by DIFFERENT methods, neither of them inheritance:**

* `_init_0_reg` — RTL: set at reset, cleared once when `EVENTS0[4]` first fires, static
  thereafter. Inert by construction.
* `_thread_0_event_reg_87_q_reg` — MEASURED. It can only change via
  `_q ^ {EVENTS0[86], EVENTS0[80]} ^ {EVENTS0[87], EVENTS0[87]}`, and across a 1057-timestamp
  domain-body trace each of those three events shows exactly **one** transition, the settle to 0
  at t=0. Every driving term constant, so the register cannot toggle.

**A source reading gave the OPPOSITE answer and was wrong.** The idle path at
`capstone_dom_switcher.anvil:99-107` is `try recv … else { cycle 1 }`, which reads as "cycles every
cycle while idle" and would make 99.8% of the failing endpoints live during execution. Measured,
106 of 106 dom-switcher events toggled a combined 112 times across the run, about two apiece — the
Anvil scheduler parks the thread rather than spinning. **This is the second time reading Anvil
control flow as if it were software produced the opposite of the hardware's behaviour** (the first
being "separate `.anvil` registers imply mutual exclusion", also wrong). Treat that source as
non-authoritative for timing and activity questions; measure instead.

**LIMIT ON THE MEASUREMENT, stated because it is load-bearing:** taken on a run where the switcher
stays IDLE throughout. It establishes inertness for the case that matters — body execution with no
switch in progress — but does not show the thread returning to the static state *after* a
completed switch. A trace of a workload that performs a `capenter`/`domcall` and then keeps
executing would close that, and should be run before this is relied on for a flash.

### The criterion, restated

> **The launch register of every failing endpoint must be shown INERT while a domain body
> executes.** Membership of `dom_switcher` is evidence toward that, not the test itself. When a
> build's failing paths launch from a register no previous build used, the inertness argument does
> not transfer — it has to be made again for that register, by RTL for a structurally static
> signal or by measurement for anything else.

## The gate, for any future bitstream on this design

**Run the launch census before trusting a build, and verify it IS a census before reading it.**

    census is 100% dom_switcher-originating     -> usable. Negative WNS is not disqualifying.
    ANY originating register outside that cone  -> NOT usable, regardless of WNS or routability.
                                                   The build routes, the board boots, and the
                                                   reason its measurements meant anything is gone
                                                   SILENTLY.

The second branch is the dangerous one and it has no other detector. A timing-marginal path inside
issue or LSU logic is indistinguishable from the defect class under investigation.

Discipline that makes the reading valid, each of which has failed here at least once:

* **verify it is a census, not a sample** — both axis sums against the design's own count, BEFORE
  reading anything from it. A worst-N tail cannot answer "is any failing path in cone X".
* **positive control in the same command** — a zero for `s07_ldc0` means nothing without
  `dom_switcher` returning six figures alongside it.
* **invert the query where you can** — asking for the worst path THROUGH a cone answers with n=1
  and full power; searching a failing set for that cone leaves the remainder unexamined.

## Worked application: the S-12 fix, 2026-09-04

Two functionally equivalent fixes for the same defect, both validated on `80843404c`:

    A  add commit_ack_i to the two stall_waw clauses   83/85 rows cycle-identical to base;
                                                       imports a deep cross-module signal into the
                                                       scoreboard:129 cone for the first time
    B  delete the clause entirely, defer to clause 1   functionally identical (suite 72/13/3, sweep
                                                       0/4, UNOPTFLAT set unchanged); costs up to
                                                       +8.95% on 31 store-heavy rows

Neither WNS nor routability discriminates them. **The census does**: A's risk is precisely that
its new path adds failing endpoints originating in the ISSUE cone, which would break the property
above. B adds no signal and cannot.

Pre-registered reading, agreed before the build:

    A census still 100% dom_switcher      -> ship A; validated and performance-neutral
    A census gains a non-dom_switcher
      originating register                -> ship B; the ~9% is the price of believable results
    B census also moves                   -> neither is safe on this base

Build A first: **A's risk is measurable in one run; B's advantage is not measurable here at all.**
One A build either clears it or says switch. One B build leaves A's question open and spends 9%
against a risk nobody measured.

Note also that the 9% falls on store-heavy workloads, which are where the paper's silicon figures
come from. If the census forces B, whether the affected figures are re-measured or caveated is a
project-lead decision and should be made deliberately rather than discovered in a table later.

## Related

- `plans/s12-fix-synthesis-request.md` — the fix, its validation table, and what its in-code
  comment deliberately does not overclaim.
- `history/26-08-2026_16-00-00_s12-recorder-bitstream-built-and-collector-exposure.md` — the census
  that first established this for 84ed6eafb and 52fa06b9d.

---

## QUALIFIED 2026-09-04: the census has only ever been validated on INSTRUMENTED builds

**What changed.** `6f8345fdb` is the first build ever synthesised on this project **without the
debug instrumentation** (same S-12 fix as `5097eb166`, debug tree tied off, same base
`80843404c`). Its census is the **mirror image** of every build above:

    6f8345fdb   99,879 / 99,879 launching from `issue_read_operands`
                dom_switcher: ZERO

**The mechanism, verified in the base's own source, not inferred.** The debug mux consumes
dom-switch state heavily — all five `dom_switch_*_log_q` logging registers are present in
`cva6.sv` — and the instrumented and tied-off builds have **identical RTL**, 200 `dom_switch`
references each. The removal happens in **synthesis**, via a single `debug_led_o` tie-off. So a
on the **fixed** design, removing the debug tree moves the cone entirely. The exposed cone sits
in **issue logic, which is NOT inert during body execution**.

**AMENDED 2026-09-04, same day: the causal half of this is WITHDRAWN.** `5097eb166` vs
`6f8345fdb` is a one-variable comparison — both carry the fix, they differ only by the tie-off —
so the inversion is established **on the fixed design**. Extending it to the historical
pre-fix builds is a **two-variable** inference (fix *and* instrumentation), and the competing
reading is not excluded: **the fix may have created the issue-cone paths, with the mux merely
masking them** in the instrumented arm — under which the historical builds' inertness is
**genuine** and the instrument is not "why" at all. `6f8345fdb` is the **only instrument-free
build in existence** on this project; the other thirteen all carry the debug tree. The control
that would settle it — base `80843404c` with the same tie-off and no fix — **has not been
run**.

**What survives.** Every census recorded above is still correct for the build it was taken on, and
the inertness argument still licenses those specific bitstreams and the board results resting on
them. What does not survive is the word **"STRUCTURAL"** in §"What actually licenses these
bitstreams". The honest statement is narrower:

> **The inertness argument has only ever been validated on INSTRUMENTED configurations. Whether
> the property depends on the instrumentation is UNMEASURED — no instrument-free build of any
> pre-fix commit exists. It should not be inherited by an instrument-free build without being
> remade.**

**The gate below needs restating, because as written it misclassifies the better build.**
`6f8345fdb` has an originating register outside the `dom_switcher` cone, so the gate as phrased
says NOT usable — yet it is the *cleaner* build (−13.491 vs the base's −16.400, 750 fewer LUTs,
closer to a production configuration). The gate encodes an **instance** where it means a
**principle**:

    AS WRITTEN:  census is 100% dom_switcher-originating   -> usable
    AS MEANT:    every failing path is provably INERT during body execution -> usable

`dom_switcher` satisfies the principle because `cur_idx` toggles only during a switch with the
frontend flushed. `issue_read_operands` does not satisfy it at all — so the gate reaches the right
verdict on `6f8345fdb` for the wrong stated reason. Restate it as the principle, and keep
`dom_switcher` as the one cone known to satisfy it.

**Consequence for any write-up.** If the census is used to argue that timing failure is benign on
this processor, that argument is about the **debug configuration**, not about the CVA6-Capstone
design as such. Do not carry it into a paper unqualified.

Source: synthesis lane, 2026-09-04; artifacts retained on that machine (13 tarballs, three
directories). See `fpga-silicon-measurements-for-paper.md` §7/§7a for the routed-build table and
the measured cost of the instrumentation itself.

## MEASURED 2026-09-07: the first build without the S-07 layer, and the edge the criterion never covered

`947327f6d` (`fpga-testing-dev-clean`: the shared base `7e4dc440f` plus S-06, S-08, S-07, S-10, S-12
and the mtval cursor; the S-07 recorder/aperture layer never added; upstream's UART tracer and the
base's 12 debug-bank apertures retained) routed at **WNS −11.717 ns**, **97,438** failing endpoints
on the CPU clock plus one in the DDR controller on `clk_pll_i`. Census verified on both axes before
being read (by-startpoint 97,439 = by-endpoint 97,439 = the report's own 97,439):

    947327f6d   97,438 / 97,439 launching from `dom_switcher/req_en_q_reg[0]`
                1 from the DDR controller · issue_read_operands ZERO · cur_idx ZERO

Third distinct `dom_switcher` launch register in four builds (`cur_idx` → `_thread_0_event_reg_87`
→ `req_en_q`). **As the gate is written it passes. It does not license the bitstream**, and the
reason is a clause the criterion never stated. Audited adversarially against the RTL at the commit
(claim-auditor, 2026-09-07; every citation re-read):

* `req_en_q` **is** `dom_switch_busy`: `capstone_dom_switcher.anvil.sv` exports it combinationally
  (`_busy_ch_busy_0 = req_en_q`). One set — when the switcher accepts commit's request
  (`EVENTS0[103]` requires `~req_en_q && _commit_ch_req_valid`) — and one clear at the end of the
  switch (four inlined copies). Constant 0 for the whole of a body. **Steady-state inertness:
  SUPPORTED.**
* `cur_idx` was licensed because it "toggles only during a switch **with the frontend flushed**".
  `req_en_q` is the signal that **causes** the flush. At its rising edge nothing is flushed yet: the
  switch instruction retired at T0 with busy still 0 (the ack needs `~req_en_q`), so at T0+1 the
  **next** instruction sits at the commit head and the only things stopping it are
  `!dom_switch_busy_i` at `commit_stage.sv:303` — which is commit `030378a66`'s fix for precisely
  that bug, "instruction after CALL executed" — and `flush_commit_o = flush_commit | dom_switch_busy_i`
  at `:494`. Both are **consumers of the late launch register**: every failing path in this census runs
  from `req_en_q` *through* the protective logic. An endpoint that captures late evaluates T0+1 with
  busy = 0. No independent backstop: the scoreboard's `dom_switch_*` ports are dead (declared
  `scoreboard.sv:89-90`, zero uses), the second-port guard at `commit_stage.sv:517` is compiled out
  (`NrCommitPorts = 1` in the FPGA config), and issue does not serialise behind CALL/RETURN
  (`issue_read_operands.sv:307` feeds only the pc/branch latch). **Rising-edge benignity: REFUTED as
  argued.**
* The commit-class cone **physically fails on this build**. By-endpoint, from the same forensics:
  `csr_regfile_i` `mepc_q` 128, `mcause_q` 128, `mtval_q` 128, `sepc_q` 128, `scause_q` 128,
  `stval_q` 128, `dpc_q` 122, `instret_q` 52, `mstatus_q` 7, `priv_lvl_q` 1 — the trap-entry CSRs
  and the retired-instruction counter, all written from the busy-gated commit block. Plus 12,874
  endpoints in `i_scoreboard`, 4,742 in `i_issue_read_operands` (the register file lives there) and
  5,230 in `lsu_i`, which the forensics script buckets at four path components and cannot resolve
  further. And 65,562 in `i_tracer`, whose only outputs are a UART pin and one LED-mux bit
  (**observation-only: SUPPORTED**) — which also means the tracer can record a phantom commit exactly
  at a switch boundary, where domain switches are being studied.
* Falling edge: the late value is the conservative one (keep flushing, hold `npc`). One reconvergence
  hazard named, **UNRESOLVED**: the controller seeing `flush_commit_i` still old (1) while its direct
  `dom_switch_busy_i` is already new (0) takes `set_pc_commit_o = 1` with a stale `pc_commit_i`; it
  needs the long path to fail and the short one to pass at the same endpoint, a placement fact.
* The slacks at those CSR endpoints are outside the worst-2,000 sample, so how much margin silicon has
  at typical conditions is **unmeasured**; static timing at the slow corner says the edge can be missed.

**Verdict: NOT usable as a board bitstream.** A result taken on it would be read against a possible
phantom commit at every domain switch.

**The criterion, restated once more.** The launch register must be inert while a body executes
**and** its transitions must be benign: either they occur only while the pipeline is flushed
(`cur_idx`), or every endpoint that acts on them is shown to tolerate a one-cycle-late capture. A
level that gates commit fails the second clause by construction, whatever its steady state.

**What would change the verdict** (design decisions, recorded, not taken):
1. Give `commit_stage`, `controller` and `frontend` a locally registered switch-in-progress flag, set
   from the handshake they perform themselves (`dom_switch_valid_o && dom_switch_ack_i`) and cleared
   when busy falls, so the commit gate is a one-level local path and `req_en_q`'s cone stops being
   commit-class. Any such change re-runs synthesis and this census.
2. Experiment B, no board and no synthesis: in Verilator, a one-cycle delay register on
   `dom_switch_busy` at one consumer at a time (three arms), on a workload that performs a switch and
   keeps executing. A phantom commit at the boundary in the `commit_stage` arm confirms the mechanism
   independent of any bitstream; a wrong first PC in the `controller` arm confirms the falling-edge
   hazard.
3. `timing-forensics.tcl` should bucket endpoints at six path components or emit the raw endpoint
   list; at four it cannot separate the register file from the operand latches.

Artifact: `synth-947327f6d-exit0.tar.gz` (403,665,530 bytes) on the synthesis machine, next to the
two S-12 arms.

### Experiment B, run the same day: the mechanism is measured, on both edges

Simulation only (Verilator, host), no board, no synthesis. Three worktrees at `947327f6d`, each with
`dom_switch_busy` captured one cycle late at **exactly one** consumer — a flop inserted in `cva6.sv`
in front of that instance's `dom_switch_busy_i` port — then the 18 tests that contain a domain
switch, compared row by row (verdict, cycles, RVFI trace hash, exception count) with the tip's own
record from the identical `core/` tree. Models verified freshly built from the patched source
for all three arms (the patch log names the instance, the arm diff shows the flop on that instance's
port, and each model's generated C++ carries it).

| arm (late consumer) | rows identical | changed | what the RVFI trace shows |
|---|---:|---:|---|
| `commit_stage_i` | 16 / 18 | `revocation`, `s06sec-ctx-scalar-roundtrip` | **the instruction after the CALL retires** — `li gp,4` (the test's own "falling through means the switch never happened" code) → FAIL; in `revocation` the successor `lui a0` clobbers a0, the CALL re-executes and traps UNEXPECTED_OPERAND, the core ends at pc 0 → TIMEOUT |
| `controller_i` | 18 / 18 | — | no change in the two informative tests: **untested at useful power, not exonerated** |
| `i_frontend` | 16 / 18 | the same two | **measured:** after the switch the callee's instruction words retire with PCs shifted by +4 (`0x30302373` retired at `…b0` and again at `…b4`, then `…b4`'s word at `…b8`, …); PC-relative control flow goes wrong, both tests spin at `j pc+0` → TIMEOUT, zero exceptions. *Interpretation:* the late release of the `npc` hold after the restart mis-associates PCs with fetched words |

**Read the population before the fractions.** Only three of the 18 tests contain a CALL or RETURN;
`CAPENTER` is detected at commit as its own mechanism (`commit_stage.sv:187-191`) and executes in the
FLU unit, while the switch request (`dom_switch_en`) is a dyn-unit result raised by CALL/RETURN
(`capstone_dyn_unit.anvil:519-520`), so the eight CAPENTER tests never raise the busy level and are
uninformative about its edges. Of the three, `call-ctx-save` traps at cycle 507 on the tip and never
retires its CALL. **The informative population is two tests, and both
changed in both the commit-stage and the frontend arm.** The 16 identical rows are not evidence of
safety; they never raise busy.

What this settles: the rising-edge hazard (C2) is a measured phantom commit, not an argument; and
the falling edge has a measured hazard of its own at the frontend (the PC shift; the `npc`-release
reading is interpretation). The controller reconvergence path named above is **not** what the controller
arm tests: that path needs `flush_commit_i` late with the direct busy on time, which is arm C's
configuration, where the rising-edge divergence happens first and masks it. The controller arm delays
the direct input instead; every switch has a falling edge, so it did see two, and showed no change —
a negative at n = 2, untested at useful power. A directed CALL/RETURN test built for the controller
path would give that arm the power the other two have. What it does not settle: how much timing margin silicon
has at the failing endpoints in `commit_stage`'s and the frontend's cones — the experiment forces a
full-cycle lateness at a whole port, whereas in silicon only the endpoints on failing paths are late.

One more intersection worth having in writing: the S-12 board verdict is read out of `mcause`/`mepc`
packed into the domain return word, and on **this** build those CSRs lie on the busy-launched cone.
That does not touch the recorded S-12 result — the flashed `5097eb166` has a different launch
register, and the S-12 trap fires during the body, not at a switch edge — but on any build with this
census the readout path and the hazard cone intersect.

Records: `expB-{C,K,F}.txt`, the diverging traces, the patch and the arm diff are kept with the
rebuild records outside the repo (`~/dev/llvm-capstone-rebuild/records/expB*`).
