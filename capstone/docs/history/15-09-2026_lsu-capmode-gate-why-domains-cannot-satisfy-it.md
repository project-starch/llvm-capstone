# Why the LSU capability check is inert in a domain: the privilege half, not the capmode half — and the one arm that would make this airtight

> # AMENDED THE SAME DAY — READ THIS FIRST
>
> **The central question this note answers is still answered correctly: a domain cannot satisfy the
> gate, and that is a property of the design rather than a configuration error.** Two things in it
> are wrong and are struck in place rather than deleted.
>
> 1. **"The check is live for the trusted monitor."** Wrong. **R-34**: the block fires and the load
>    unit **discards** the exception, so plain data accesses are unenforced at *every* privilege, not
>    only in domains. It is also **not capability-specific** — stock `rv64mi-p-ma_addr` fails with
>    capmode never set.
> 2. **"The one arm that would make this airtight" is presented as owed.** It was **run**, and it came
>    back on the second of the two branches this note pre-registered — the one meaning a second,
>    independent defect.
>
> **My first correction was also wrong** ("lost on an immediate cache grant") and is recorded as such.
> The accurate wording is **raised and dropped**.
>
> **Sequencing that came out of it:** R-24 must be ruled **before or with** R-34's fix, never after.

**RTL lane, 2026-09-15.** Question routed by the lead via the paper lane, open since August and
blocking a framing decision: `cap_violation_detection` in `load_store_unit.sv` is gated on
`capmode_i && ld_st_priv_lvl_i == riscv::PRIV_LVL_M`. **What sets those two signals for a domain
after CAPENTER, and can a domain be made to satisfy the gate?**

**Answer: the capmode half IS satisfied; the privilege half is not, and a domain cannot be made to
satisfy it by configuration.** This is a property of the deployed design, not a setup error. Read
from the RTL and monitor sources; the confirming measurement is named at the end and has not been
taken.

## The capmode half is satisfied, and is not the reason

`capmode` is **sticky**, not scoped to a domain:

    csr_regfile.sv:295   assign capmode_d = capmode_q | capmode_set_i;  // set by CAPENTER, sticky
    csr_regfile.sv:3005  capmode_q <= 1'b0;                             // only at reset

It is driven by `capenter_commit` (`cva6.sv:1972`), and the board's boot path issues one CAPENTER in
M-mode before anything else:

    sbi_capstone_init.S:3   #define CAPSTONE_ENTER_C      <- the guard is defined in the same file
    sbi_capstone_init.S:34  CAPENTER(s0, s1)

So from early boot onward `capmode_i == 1` **everywhere, forever**, monitor and domain alike. An
earlier guess of mine — that capmode and M-mode were close to mutually exclusive — was **wrong**, and
is retracted here: stickiness is exactly what makes them co-satisfiable.

## The privilege half is not satisfied, and that is the whole reason

    csr_regfile.sv:2284   ld_st_priv_lvl_o = (mprv) ? mstatus_q.mpp : priv_lvl_o;

**Domains run in S-mode.** The monitor `mret`s into it, and says so in its own comments:

    sbi_capstone.S:177  # right before returning to S mode:
    sbi_capstone.S:181  mret
    sbi_capstone.S:183  call_into_smode:
    sbi_capstone.S:192  mret
    sbi_capstone.S:194  resume_smode:

**CAPENTER does not change privilege.** It appears nowhere in `csr_regfile.sv` except as the
`capmode_set_i` input, and nothing in the commit path touches `priv_lvl`. So entering a domain leaves
the privilege level to the `mret`, which targets S.

With `priv_lvl_o == S` the gate needs `mprv` to rescue it, and it cannot:

> **The very `mret` that enters the domain clears MPRV.**
> `csr_regfile.sv:2331` — `if (mstatus_q.mpp != riscv::PRIV_LVL_M) mstatus_d.mprv = 1'b0;`
> which is the architectural behaviour, MPP being S here.

And a domain cannot set it afterwards: `mstatus.mprv` is an M-mode field and the domain is in S-mode.

**So `ld_st_priv_lvl_i == S` in a domain, unconditionally, and the block is skipped in its entirety.**
That is why `obn` — a store through a base with no capability metadata, which violates the block's
*first* clause and should raise cause 24 — completes silently. The clause is never reached.

## Can a domain be made to satisfy it? No, not by configuration

Only two routes exist and neither is a setting:

1. **Set MPRV with MPP=M.** Closed twice over: the entering `mret` clears MPRV because MPP is S, and
   an S-mode domain cannot write the field to restore it.
2. **Run the domain at M-mode.** This would satisfy the gate, and it hands the domain full machine
   privilege — abandoning the isolation the check exists to enforce. That is not a configuration fix;
   it is discarding the threat model to make the checker run.

> ## ~~Where the check IS live: the monitor.~~ **RETRACTED 2026-09-15 — SEE R-34. AND MY FIRST CORRECTION WAS WRONG TOO.**
>
> The struck sentence read: *"capmode is set and privilege is M while M-mode code runs, so the block
> is enabled for the trusted monitor's own plain loads and stores… present exactly where it is least
> needed and absent exactly where it matters."* **It is not enabled anywhere in any useful sense.**
>
> **My second attempt was also wrong** and is recorded because two wrong wordings are worth more to
> the next reader than one. I proposed *"the exceptions are lost on an immediate cache grant"*. The
> board lane's auditor refuted that framing as well.
>
> **The correct wording is RAISED AND DROPPED.** The block **fires** — `cap_exception.valid` rises 21
> times in the audited run, causes 24/27/28 — and the **load unit discards it**. Verified here by
> content rather than by line number: the emit is guarded on
> `if (ex_i.valid && (state_q inside {SEND_TAG, SEND_TAG_LDC}) …)` in `load_unit.sv`, and the code's
> **own comment** states the assumption that breaks — *"An exception arrives one cycle after
> `dtlb_hit_i` is asserted, i.e. when we are in SEND_TAG."* On an immediate grant the request is
> popped before that, the state is no longer SEND_TAG, and the exception is silently discarded. The
> store unit drops it the same way, and the MMU forwards `misaligned_ex_i` unregistered because
> upstream #2528 removed `misaligned_ex_q`.
>
> **It is not capability-specific.** Stock `rv64mi-p-ma_addr` fails on this RTL with capmode never
> set. This is a general exception-delivery defect inherited from upstream, which changes who should
> care about it.
>
> So the privilege gate keeps **domains** off the block, and R-34 discards the block's exceptions
> **everywhere else**: plain data accesses are unenforced at **every** privilege. The monitor's
> `rdtime` emulation stores through an untagged base at every Linux clock read and does not halt —
> that is why. Full account: `docs/ref/ISSUES.md` R-34 and
> `tests/fpga-repros/R34-lsu-exception-lost-on-immediate-grant/`.

## What this settles for the manuscript, and the limit on it

The fork resolves to the second branch: **the deployed configuration does not enforce spatial or
temporal safety on plain `LD`/`SD` inside a domain, and no configuration change reaches it.** The
four `tab:safety` rows claiming *"Stops at access"* are not supported for plain data accesses on this
configuration. Capability accesses are a separate path and are unaffected — `LDC`/`STC` carry their
node-validity query in the DYN unit, which has **no privilege gate at all**.

> **AMENDED 2026-09-15 (R-34): the conclusion holds and the reason is now TWO independent ones.**
> The privilege gate keeps domains off the block; R-34 discards the block's exceptions everywhere
> else. So those four rows fail **twice over**, and **"satisfy the gate" would not have enforced them
> either** — which is the part this note originally got wrong by implying the gate was the whole
> story. Anyone rebutting one reason still has the other to answer.

**Do not restate this as "the silicon does not enforce capability bounds".** That form was filed and
retracted once because two halves had been measured in different domains. The accurate claim is
narrower and is about one gated block on the plain load/store path.

## ~~The one arm that would make this airtight~~ — IT WAS RUN, AND IT ANSWERED ON THE SECOND BRANCH

This section asked for one arm and pre-registered both readings. **The board lane ran it** (R-34,
`lsu-mmode-gate.S`), and the answer was the second branch, verbatim as written here:

> * Traps → the gate is the whole story.
> * **Does not trap → there is a second defect**, the block is broken independently of the gate, and
>   "satisfy the gate and re-measure" would not have closed it either.

**It did not trap.** In M-mode with capmode set and `MPRV = 0` — both witnessed, so the gate is
demonstrably satisfied and the block's own revnode tracking updates — a load through a write-only
tagged capability, a load at exactly `bound_end`, a store through a read-only tagged capability, a
misaligned load and store, and a load through an untagged base **all retire with no trap**, and the
two stores land, the misaligned one corrupting its neighbours.

**The one exception that was delivered proves the R-24 entanglement rather than weakening it:** it
came back as cause 24, which is this core's `DEBUG_REQUEST`, so the core entered the debug ROM, and
on `dret` the same instruction re-ran, was granted at once, and completed silently.

**Consequence for any fix, and it is a sequencing constraint rather than a caveat.** Once delivery is
repaired, cause 24 enters debug mode on **every** M-mode plain access through an integer base —
including `RVTEST_PASS`'s own store to `tohost` and the debug ROM's own accesses. **R-24 must be ruled
before or with R-34's fix, never after.**

**What remains source-derived rather than measured**, and is this lane's next work: the
translation-on path (S-mode with `satp` set) is argued from `cva6_mmu.sv` alone, and there is no store
bounds arm. Both are simulation work; only the post-fix confirmation needs a boot.
