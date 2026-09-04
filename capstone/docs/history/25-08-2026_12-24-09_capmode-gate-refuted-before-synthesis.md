# The capmode gate was refuted before synthesis: it would have filtered nothing

**Proposed by me, refuted by audit, never built.** Cost: one auditor run. Cost avoided: a ~90-minute
synthesis plus a board reflash, for a change that closes on an empty set.

## What was proposed

The S-07 LDC recorder is first-wins and one-shot; boot software consumes its single slot before any
domain runs, because an `ldc` over a zeroed stack slot is *legitimately* untagged. The proposal was
to AND `capmode_i` into the capture condition in `core/load_unit.sv`, so that only a load executed
inside a capability domain could take the slot.

**Stated premise:** a domain runs at M with capmode ON; Linux and the monitor do not.

## Why it is wrong

**Half that premise is false, and it is the half that mattered.**

`core/csr_regfile.sv:295`:

```systemverilog
assign capmode_d = capmode_q | capmode_set_i;  // set by CAPENTER, sticky
```

An OR and nothing else. The only writers of `capmode_q` are reset (`:2994`) and `:3083`. **There is
no clear on domain exit, RETURN, exception, trap or wedge.**

And `CAPENTER` executes during OpenSBI bring-up — `sbi_capstone_init.S`, a few instructions in,
straight-line, before the branch to `sbi_capstone_init_cap` and long before the `mret` that starts
Linux.

**So capmode is 1 for the entire life of the machine, Linux included.** The gate has no
capmode==0 window to exclude. It filters nothing.

## What is actually consuming the slot

The monitor's trap entry issues `LDC(gp, sp, -16)` at `_cap_trap_entry`, installed as `ctvec`
*after* the CAPENTER. Every S-mode ecall, every timer interrupt under Linux, every trap runs it —
with capmode already set. The granule recorded at boot 31 (`0x81170`) falls inside the monitor's
`dom_stack`, consistent with that slot.

This is worse than "some kernel activity might steal the record": timer ticks make it a
millisecond-cadence certainty.

## Consequences

- **The existing debug-mux clear (switch 160, already in `84ed6eafb`) is the only instrument, and
  no gate improves on it.**
- **The release-to-arm window is continuously vulnerable**, not occasionally. There is no ordering
  of operations that makes it safe.
- **Therefore the paddr attribution cross-check is MANDATORY.** It is not a backstop for the
  recorder; it is the thing that makes any reading of it trustworthy. A reading whose granule is
  not the subject's must be discarded, not interpreted.

## Two things worth keeping

**A trap for whoever edits `load_unit.sv` next.** `ldbuf_t` is filled by a **positional
concatenation** at `core/load_unit.sv:322-325`. Adding a field to the struct without updating that
literal silently shifts `trans_id` and `operation` — the "positional struct literals whose field
count changed" hazard, invisible under the suppressed-warning build.

**An unestablished sidebar.** If the recorded granule really is the monitor's own saved-`gp` slot,
then an `stc` of `gp` came back untagged on a later `ldc` **inside the monitor** — an S-07
signature in the monitor rather than in a domain. Not claimed, not investigated; recorded so the
next attribution hit on a monitor granule is not read as noise.

## The process point, which is the reusable part

The change was **correct** — it did what it said, was lint-clean, two files, and its own nets would
not have been on the critical path. It was simply **insufficient**, in a way no lint gate, no
simulation and no synthesis would ever have revealed: the bitstream would have built, flashed,
worked, and changed nothing.

**A correct-but-insufficient RTL change costs exactly the same cycle as a wrong one.** The only
check that catches it is asking an auditor *"is this enough, and what would stop it from firing"*
before committing the cycle — not *"is this right"*.
