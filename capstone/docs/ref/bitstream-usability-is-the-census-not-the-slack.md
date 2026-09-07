# What makes a timing-failing bitstream usable here: the census, not the slack

> **RETRACTED 2026-09-08 — read the last section first.** The census verified the WORST launch of each
> failing endpoint only. On the flashed `5097eb166`, 101,604 of the 101,784 failing endpoints also have a
> failing path from a live register (`lsu_bypass_i/status_cnt_q_reg[0]`, −15.157 ns, 154 ps behind the worst
> launch). "Every failing path in the design is inert while the code under test runs" was never true of
> that build. The census is not a licence; the resident's board record is the evidence, and the reason the
> board works is unmeasured. The second cone, and the exact mechanism by which a one-launch-per-endpoint
> enumeration would lose it, were written down in this repository on 2026-08-21, fourteen days before this
> document existed, and never consulted. The gate below stands as history and is amended in the last section.

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

> *2026-09-08: the two sentences above are RETRACTED — the census checked one launch per endpoint; the
> second launch on the flashed build is live. See the last section.*

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

> *2026-09-08: "the launch register", singular, is the error. Every register with a failing path to the
> endpoint must be inert, and the worst-launch census enumerates only one of them. See the last section.*

## The gate, for any future bitstream on this design

> *2026-09-08: this gate is kept as the record of what was run. It is NOT a licence — see the last
> section: the flashed build passes it and still fails from a live launch. Any future reading of a
> timing-failing build needs the second-launch count (C in that section) alongside the census, and even
> C = 0 licenses nothing by itself until the third launch is asked for.*

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

### The fix, and the census reading pre-registered for it (2026-09-07, later)

Commit `ef5a8eaf2` on `fpga-testing-dev-clean` adds one top-level flop, `dom_switch_active_q`, set by
the commit handshake (the switcher's ack, which already contains `~req_en`) and held by `busy`, so it
rises on the same edge as `req_en` and falls one cycle later; `commit_stage`, `controller` and
`frontend` take it in place of the raw wire. Measured: 87 of 88 rows identical to `947327f6d`, the
one mover `revocation` 932 → 934 cycles with an identical trace; the acceptance arm (busy one cycle
late at the flag's only busy input) changes nothing but two more cycles on the same test, which paired
with Experiment B isolates the old commit-stage failure to the rising edge; lint counts identical;
adversarial audit: rising edge unchanged SUPPORTED, later release harmless SUPPORTED. Details in the
commit message.

**What the audit refuted, and what the census must therefore show.** `req_en` still reaches
commit-class logic combinationally through the ack: `commit_stage.sv:357` `commit_ack_o[0] =
dom_switch_ack_i`. That leg is benign only because `req_en` is stable at 0 in the handshake cycle
and, at its two transitions, `:357` sits inside the gate now driven by `dom_switch_active_q` —
**conditional on the flag meeting timing at `commit_stage`**. The flag's own D input is a
core → switcher → core round trip; if it misses setup the flag rises a cycle late and the phantom
commit returns at a new launch site. The logic after the consumer ports is unchanged, so whether the
failing paths disappear is a placement outcome (the worst path was route-dominated, 43.3 of 51.4 ns,
which is the one reason to expect improvement from a dedicated, lightly loaded launch register).

**Pre-registered reading for the synthesis of `ef5a8eaf2`, written before the run:**

1. **Zero** failing endpoints launched from `dom_switch_active_q`. Any → the hazard is re-rooted,
   not closed, and Experiment B's frontend PC shift can return as route skew between the three
   consumers.
2. The flag's D endpoint has **positive slack**.
3. `req_en`-launched failing paths only via the ack leg (`commit_stage.sv:357`), the log register or
   the aperture — licensed by the conditional argument above, and only if (1) and (2) hold.

Anything else is a NOT-usable verdict for that build too. **Board-side shape, from the board lane:**
a committed successor or a +4 first-PC shift would present on the board as R-3's silent hang at entry
or as the trap-vector-zero behaviour (a domain enters with `mtvec = 0`, so any trap looks like a hang);
no such observation exists on the resident `5097eb166`, whose launch register is a different one and
which switched cleanly seven times in one boot on 2026-09-07 (sw30). The `CAPSTONE_DOMAIN_TRAP_VECTOR`
acceptance test in the monitor is the instrument to run with the fixed bitstream, when one exists.


### MEASURED 2026-09-07 (read 2026-09-08): `ef5a8eaf2` synthesised — the three pre-registered lines hold, and the build is still NOT usable

Same machine, container, guard and flow as `947327f6d`, ceiling 100 GB: exit 0, 3h09m wall (the box ran at
load 75–85 on 80 cores for the whole run, so the wall time is not comparable), synthesis peak 20.96 GB.
Artifact `synth-ef5a8eaf2-exit0.tar.gz` (404,402,489 bytes) with all five checkpoints; provenance worktree
`capstone-clean-flag` on the synth machine.

| build | WNS (ns) | failing / total (CPU clock, flow report) | placed LUTs | launch census (collector, per-endpoint worst launch) |
|---|---:|---:|---:|---|
| `ef5a8eaf2` fix | −12.733 | 101,143 / 174,188 | 170,410 | 101,143 / 101,143 from `issue_stage_i/i_issue_read_operands` |
| `947327f6d` base | −11.717 | 97,438 / 174,756 | 170,481 | 97,438 `dom_switcher/req_en_q_reg[0]`, 1 DDR |
| `6f8345fdb` arm 2 | −13.491 | 99,879 / 173,789 | 168,944 | 99,879 `issue_read_operands` |

**The pre-registered reading, line by line** (census verified: the startpoint axis and the endpoint axis
both sum to the report's 101,143):

1. Failing endpoints launched from `dom_switch_active_q`: **0**. The register does not appear on the
   startpoint axis at all.
2. The flag's D input is **not a failing endpoint** (absent from the endpoint axis): its slack is
   non-negative, so the flag rises on the edge it was designed to.
3. Failing paths launched from `req_en_q`: **none** — not through the ack leg (`commit_stage.sv:357`),
   not through the log register, not through the aperture. The busy level launches nothing that fails
   on this placement.

All three hold. **The busy-edge hazard that Experiment B measured has no failing path on this build.**
That is the result the fix was built for, and it stands regardless of what the placer did next; the
fix stays in the branch.

**The census, and why the verdict is nevertheless NOT usable.** All 101,143 failing endpoints have their
worst launch in `issue_read_operands`. Worst path: `i_issue_read_operands/lsu_valid_q_reg[0]_rep` →
`i_scoreboard/mem_q_reg[1][sbe][rd][4]/D`, slack −12.733 (the flow's report and the re-opened checkpoint agree;
TNS −769,201), data path 52.5 ns (logic 8.0, route 44.6), 115 logic levels; the `_rep` suffix is Vivado's fan-out replica of the register.
`lsu_valid_q` is the issue-to-LSU valid — `assign lsu_valid_o = lsu_valid_q`
(`core/issue_read_operands.sv:388`), set in the issue `case` for `LOAD, STORE` and for every capability
load/store (`:1284`, `:1302-1320`), cleared otherwise — so it toggles on every memory instruction of a
body. There is no inert-launch argument and no benign-transition argument for it. This is the shape of
`6f8345fdb` (arm 2: `fu_data_q[operand_a]` → scoreboard, −13.491, 94 levels, census 100%
`issue_read_operands`). **Verdict: NOT usable as a board bitstream.**

Endpoint axis, for the record: the UART tracer 65,562 (the same count to the endpoint as on
`947327f6d`), issue stage 21,731, ex stage 8,415, cache subsystem 2,460, `csr_regfile` 978 (`mcause`,
`mepc`, `mtval`, `scause`, `sepc`, `stval`, `dpc` at 128 each, `instret` 64, `mstatus` 7), frontend 451,
`dom_switcher` 358 — the same endpoint population as the base build, now failing from a different worst
launch. Interior (section 4 of the enumeration): 100,926 of the 101,143 failing paths pass *through*
`dom_switcher` cells (worst −12.505) although the switcher launches none of them; the S-07 fix's nets
carry 4,597 failing paths, worst −10.646, not the critical path. Area, from the placed utilization
reports (the post-synthesis hierarchical report says 170,415 and 170,507 — wrong stage, same trap as
before): 170,410 LUTs against the base's 170,481, and **92,817 slice registers against 93,101 — 284 fewer
on a change that adds exactly one flop.** Placement noise, downstream simplification once the signal is
registered, or something unseen: not separable while the flow's run-to-run variance is unmeasured; recorded
so it is not passed over silently.

**Measured, and inferred — kept apart.** MEASURED: two placements of RTL that differs only at three
consumer ports fail the same endpoint population under different worst launches, `req_en_q` on
`947327f6d` and `lsu_valid_q` on `ef5a8eaf2`. INFERRED, and unmeasured: the census attributes each
failing endpoint to its worst launch only, so it cannot say whether on `947327f6d`, or on the resident
`5097eb166`, those same endpoints also failed from live registers through their second-worst launches.
Nothing here shows that a historical licence was wrong. The resident `5097eb166`'s licence rests today
on its board record — seven clean domain switches in one boot on 2026-09-07 — which is evidence of a
different kind and stands. *(Superseded four hours later by the query below: the resident's checkpoint
shows the second launch is live. The board record stands; the census licence does not. Last section.)*

**The query that settles it, without another synthesis.** Three routed checkpoints are retained on the
synth machine (`5097eb166`, `947327f6d`, `ef5a8eaf2`). `second-launch.tcl` (kept with the rebuild scripts,
outside the repo) opens each and counts, against the report's failing total A and the census number B:
**C**, the failing endpoints that have a failing path from a launch OUTSIDE the census's launch module,
with a control D (the same query restricted to the inside, which must be ≥ B) and E (input-port
launches, which a `-from <cells>` query cannot see). Reading, written before the result: if `947327f6d`
shows C in the tens of thousands, the busy cone was never the only failing one, and no re-placement of
this RTL passes the census at 40 ns; if it shows C near zero, the fix's placement is what created the
issue-cone failures, and the fix itself needs a different placement or a different form. Status at the time of writing: the query is running on the synth machine on all three checkpoints, and its numbers go into this entry when they land.
If C is large on the resident build too, the gate in this document needs the second-launch count added
to the census — that amendment waits for the number.

**Options for the lead** (none taken here). *Slower CPU clock.* The worst data path is 52.5 ns, so under
the shift rule (slack at period P ≈ slack at 40 + (P − 40)) a 60 ns period clears all three builds by
5–8 ns and licenses a bitstream by timing rather than by argument; `mcycle`-based measurements are
unaffected, wall clock is 1.5× slower. It is not one variable: the 25 MHz lives in
`corev_apu/fpga/xilinx/xlnx_clk_gen/tcl/run.tcl` (`CLKOUT1_REQUESTED_OUT_FREQ`, which drives both the
MMCM and the derived constraint; `CLK_PERIOD_NS=40` is passed to the FPGA Makefile but nothing reads it
there), in the bootrom's `CLOCK_FREQUENCY` (UART divisor), and in the board firmware's DTS
(`caplifive.dts`: `timebase-frequency` 12.5 MHz, two `clock-frequency = <25000000>` nodes), so it is a
flow edit plus a firmware rebuild, and anything read through `rdtime` changes scale. *Keep the resident
`5097eb166`* for all board work, as now. *A base+tie-off control build* — less informative than the
checkpoint query above, which asks the same question of the existing checkpoints. Not proposed: timing
work on the scoreboard forwarding cone.

## RETRACTED 2026-09-08: the census verified one launch per endpoint; on the flashed build the second launch is live

**The instrument.** `second-launch.tcl` (with the rebuild scripts, outside the repo) opens a routed checkpoint
and counts: **A**, failing endpoints over all launches (per-endpoint worst path, `-max_paths 1000000
-nworst 1 -slack_lesser_than 0`); **B**, those whose worst launch matches the census's launch pattern;
**C**, failing endpoints with a failing path from a launch OUTSIDE the pattern (`-from` every sequential
cell not matching it); **D**, the same restricted to the inside (a control that must equal B); **E**,
failing endpoints launched from input ports (what a `-from <cells>` query cannot see). Every count prints
its denominator; C's endpoints are checked to be a subset of A's. Run on COPIES of the retained routed
checkpoints, one Vivado at a time, 12 GB peak, about twelve minutes each.

| checkpoint | pattern excluded | A | B (census) | C (fail from outside) | D (control) | worst C | C's worst launch (per-endpoint, one register for all) |
|---|---|---:|---:|---:|---:|---:|---|
| `5097eb166` flashed | `*/dom_switcher/*` | 101,784 | 101,782 | **101,604** | 101,782 | −15.157 | `ex_stage_i/lsu_i/lsu_bypass_i/status_cnt_q_reg[0]` |
| `947327f6d` base | `*/dom_switcher/*` | 97,439 | 97,438 | **96,494** | 97,438 | −11.561 | `lsu_bypass_i/status_cnt_q_reg[1]` 96,493; 1 DDR |
| `947327f6d` base | `*/dom_switcher/req_en_q_reg*` | 97,439 | 97,438 | **96,510** | 97,438 | −11.561 | `status_cnt_q_reg[1]` 93,159; `dom_switcher/_thread_0_event_reg_63_q_reg[1]` 3,350; 1 DDR |
| `ef5a8eaf2` fix | `*/i_issue_read_operands/*` | 101,143 | 101,143 | **100,900** | 101,143 | −12.505 | `dom_switcher/cur_idx_q_reg[3]` 100,892; `status_cnt_q_reg[1]` 8 |

A exceeds the CPU-clock count only by the DDR endpoints on `clk_pll_i` (2, 1 and 0). E = 0 on every checkpoint. Every D
equals its B, and every B equals the census recorded for that build, so the `-from` query sees exactly what
the census saw; C is not a narrowed view.

The symmetry is the result:

| build | worst launch (the census) | second launch (this query) |
|---|---|---|
| `5097eb166` flashed | switcher, inert | bypass counter, **live** |
| `947327f6d` base | switcher, inert | bypass counter, **live** |
| `ef5a8eaf2` fix | issue valid, **live** | switcher `cur_idx_q_reg[3]`, inert |

Every build fails at 40 ns from a live launch, by its first cone or its second. The licence was never
"this build has no live failing launch"; it was "the one launch the instrument reported happened to be
inert", and which cone the instrument reported was decided by picoseconds of placement. On `ef5a8eaf2`
the second launch is `cur_idx_q_reg[3]` — the exact register, and the exact bit, that `80843404c`
reported as its *first* cone when this doctrine was written.

**The resident build.** 101,604 of its 101,784 failing endpoints have a failing path from
`lsu_bypass_i/status_cnt_q_reg[0]`, 154 ps behind the census's worst launch. That register is the LSU
bypass FIFO's occupancy counter (`core/lsu_bypass.sv:57-126`): incremented on every push, decremented on
every pop, cleared on flush, and `empty = (status_cnt_q == 0)` gates the unit. It moves on every memory
instruction of a domain body. So for essentially the whole failing population of the flashed bitstream a
failing path launches from a register that is live while the code under test runs.

**What this retracts, by sentence.**

* "Every failing path in the design is inert while the code under test runs" and "That is a STRUCTURAL
  property, not a margin" (the section *What actually licenses these bitstreams*). The census verified the
  worst launch per endpoint; the second launch was never asked for. The sufficient condition the doctrine
  rested on does not hold for the flashed build.
* The restated criterion's "the launch register", singular. The condition is every register with a failing
  path to the endpoint, and the worst-launch census enumerates one.
* The 2026-09-04 worked application, "a census still 100% dom_switcher → ship A": its outcome was validated
  empirically; its mechanism was wrong.
* The same night's sentence above, "nothing here shows that a historical licence was wrong". It did not;
  this query does.

**This cone was in the project's own record before the doctrine was written, and the doctrine's
instrument could not show it.** The history note
`21-08-2026_09-30-00_s10-exonerated-and-a-second-single-bit-cone.md` (committed 2026-08-21, build
`76b7f2afc`), section 3, enumerated the failing paths NOT launched from the switcher and found 78,790 of
them launched from `ex_stage_i/lsu_i/lsu_bypass_i/status_cnt_q_reg[0]` — "a second single-bit
startpoint, and it is not the switcher" — and said why it had been invisible: "`-nworst 1` reports one
worst path per endpoint, so while the switcher cone was worse, these paths never surfaced." Its
conclusion was "the remedy is RTL on **two** cones", with "nothing equivalent known" for the bypass
counter. This document was first committed on 2026-09-04, fourteen days later, never cited that note,
and its census — the same `-nworst 1` enumeration — reported one launch per endpoint from then on. The
second cone did not leave the design; it left the instrument's field of view, and its absence from every
census afterwards was read as its absence. The synth lane, whose starting briefing named both registers,
raised this on 2026-09-08 when the resident's number landed; the note is the primary source. Section 4 of
every forensics run since has in fact shown it — `store_unit` and `load_unit` carrying the whole failing
count at the design WNS — and it was read each time as the switcher cone passing *through* the LSU rather
than the LSU cone showing through. So the error is not "the census overstated": the project knew of two
cones, built an instrument that could see one, and reasoned from the instrument until the other was gone.
That is the lesson worth more than the retraction: an enumeration that returns one cause per effect
cannot report a second cause, and a cone that a prior note named does not need re-discovering — it needs
a query that can still see it.

**A hypothesis the record supports, not a finding.** The census has named a different single bit on every
build: `cur_idx_q_reg[3]` (80843404c), `[5]` (84ed6eafb), `_thread_0_event_reg_87_q_reg[0]` (5097eb166),
`req_en_q_reg[0]` (947327f6d); and the second launch is `status_cnt_q_reg[0]` on the resident but
`status_cnt_q_reg[1]` on 947327f6d — the same two-bit counter, a different bit. A per-endpoint worst-launch
enumeration must name exactly one register; when several bits of one counter drive near-identical cones,
which bit wins by a few hundred picoseconds is placement, not structure. The 154 ps between the switcher's
worst launch and the bypass counter's on the resident is of the same order: two cones separated by less
than a routing decision. If that is right, "a single-bit startpoint" was an artefact of the reporting all
along, and any RTL remedy is about the counter and the switcher's control state, not about one bit of
either. The bit index moving while everything else stays put is measured; the rest is the hypothesis.

**What this does NOT retract.** The board results on `5097eb166`. They rest on their own record — the
S-12 corpus, the SQLite logic-test corpus identical to native across 10,807 records, seven clean domain
switches in one boot on 2026-09-07 — and that record is untouched. What changed is the *explanation* of
why the bitstream works: it is no longer the census argument, and it is now unmeasured. Candidates, none
chosen here: functional masking — a live launch whose transition cannot propagate through the intermediate
logic during a body is still a failing STA path but never captures late data; margin between the
slow-corner timing model and the actual silicon; or both. **C says a structural failing path exists from a
live launch; it does not say the endpoint ever captures late data.** This instrument cannot separate those,
and no result in this document does.

**The reframing.** The gate never separated the flashed build from the three it rejected. Queried: on all three checkpoints essentially every failing endpoint has failing paths from at least two launch cones, and on every build at least one of them is live during a body — the worst launch on `6f8345fdb` (by its own census) and on `ef5a8eaf2` (the issue-to-LSU valid; its second launch is the switcher's `cur_idx_q_reg[3]`, 100,892 endpoints at −12.505), the second launch on `5097eb166` and `947327f6d` (the bypass counter). All four builds in this family fail at 40 ns from live launches. Excluding only the census's named register on `947327f6d` still leaves 3,350 endpoints failing from a second switcher register, `_thread_0_event_reg_63_q_reg[1]`: the same artefact one level down — the cone is the module, the named bit is whichever won by picoseconds. The
only difference between them is that one has a board record. So a flash decision on this design is not a
census verdict any more; it is an empirical risk decision, made with acceptance tests, by the project lead
with the board lane. Nothing is recommended for flashing here.

**The slower clock is now the only route to a licence by timing.** The resident's slack histogram
(1 ns bins of A; an endpoint in bin −k needs a period ≥ 40 + k + 1 ns under the shift rule):

| bin (ns) | −16 | −15 | −14 | −13 | −12 | −11 | −10 | −9 | −8 | −7 | −6 | −5 | −4 | −3 | −2 | −1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| endpoints | 4 | 100 | 1,535 | 5,758 | 4,984 | 18,898 | 17,423 | 20,269 | 15,249 | 3,357 | 1,815 | 2,070 | 5,934 | 2,975 | 671 | 742 |

Under the shift rule 52 ns still leaves 12,381 failing, 56 ns leaves 4, 57 ns clears every endpoint, and
60 ns leaves 3–4 ns of margin on the resident. On `947327f6d` the histogram tops out at bin −12 (31 endpoints): 53 ns clears every endpoint and 60 ns leaves 7 ns; on `ef5a8eaf2` it tops out at −13 (122): 54 ns clears all, 60 ns leaves 6 ns. The three builds differ by 4 ns in what would clear them, a more useful comparison than their WNS alone. By that measure the resident is the worst of the three, 3–4 ns further from closing than either
build declared NOT usable: "safe" was never the property being measured. The edit sites are the three named in the entry
above.

**Open, for whoever takes the explanation:** count C's paths on the resident checkpoint that pass through
`dom_switcher` cells. If nearly all do, a single gating argument for the masking hypothesis may exist; if
not, none does. Five minutes of Vivado; not tonight's commit.
