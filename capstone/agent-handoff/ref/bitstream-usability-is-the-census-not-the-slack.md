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
