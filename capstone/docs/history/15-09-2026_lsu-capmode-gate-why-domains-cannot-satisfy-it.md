# Why the LSU capability check is inert in a domain: the privilege half, not the capmode half — and the one arm that would make this airtight

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

**Where the check IS live: the monitor.** capmode is set and privilege is M while M-mode code runs,
so the block is enabled for the trusted monitor's own plain loads and stores, and disabled for
untrusted domain code. **The enforcement is present exactly where it is least needed and absent
exactly where it matters.** That is the finding, and it is sharper than "the block is inert".

## What this settles for the manuscript, and the limit on it

The fork resolves to the second branch: **the deployed configuration does not enforce spatial or
temporal safety on plain `LD`/`SD` inside a domain, and no configuration change reaches it.** The
four `tab:safety` rows claiming *"Stops at access"* are not supported for plain data accesses on this
configuration. Capability accesses are a separate path and are unaffected — `LDC`/`STC` carry their
node-validity query in the DYN unit, which has **no privilege gate at all**.

**Do not restate this as "the silicon does not enforce capability bounds".** That form was filed and
retracted once because two halves had been measured in different domains. The accurate claim is
narrower and is about one gated block on the plain load/store path.

## The one arm that would make this airtight, and why it is owed

Everything above explains why the block is **not reached**. Nobody has shown the block **works when
it is reached**. Those are different failures and the current evidence cannot separate them:

> **Run the `obn` probe from M-mode with capmode already set. Predicted: it traps, cause 24.**

* **Traps → the gate is the whole story.** The block is functional and unreachable from a domain,
  which is what this note argues.
* **Does not trap → there is a second defect**, the block is broken independently of the gate, and
  "satisfy the gate and re-measure" would not have closed it either.

That is the positive control for a claim that currently rests on a negative, and it needs one M-mode
arm rather than a board boot in a domain. It goes through the board lane. Until it exists, this note
states a source-derived explanation and not a measured one, and the distinction should survive into
whatever the paper says.
