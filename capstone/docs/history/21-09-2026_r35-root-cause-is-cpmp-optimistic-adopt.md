# R-35's root cause is the CPMP per-entry revnode tracker adopting an unseen revnode as VALID — not the LSU block, which a domain cannot reach

**2026-09-21.** R-35 is *"on silicon a revoked capability still reads and writes the storage its object
has given up, at every age, and the access does not trap"*, reproduced four times on the board with
controls. This note replaces its root cause. Two of my own intermediate claims are retracted below.

## The answer

The S-mode data path enforces capabilities in **CPMP**, `core/pmp/src/pmp_data_if.sv`, and that block
**does** check revocation — `cpmp_check` requires `cpmp_revnode_valid_q[i]` before it will allow an
access. What defeats it is how that bit is maintained, in `cpmp_revnode_tracking` (board RTL
`054cea69b`, `pmp_data_if.sv:82-102`), quoted verbatim:

```systemverilog
// When the CPMP entry gains a new revnode_id (entry was (re)written),
// assume the revnode is live until proven otherwise.
if (cpmp_tag_i[i]
    && cap_type_t'(cpmp_i[i].comp_metadata.cap_type) != NOT_CAP
    && cpmp_i[i].comp_metadata.revnode_id != cpmp_tracked_revnode_id_q[i]) begin
  cpmp_tracked_revnode_id_d[i] = cpmp_i[i].comp_metadata.revnode_id;
  cpmp_revnode_valid_d[i]      = 1'b1;          // <-- an UNSEEN revnode is assumed VALID
end
```

There is one tracked id per CPMP entry (16 entries) and invalidation is by exact index match:
`revnode_invalidation_id_i[15:0] == cpmp_tracked_revnode_id_q[i][15:0]`. So a **revoked** capability
installed into an entry whose tracked id is anything else is **re-adopted as valid on arrival**, and
`cpmp_check` then permits the access. Bounds and permissions still hold, because those are read from
the capability itself and a stale reference's own bounds are genuinely its old storage.

**That is exactly the folder's observed scope — "the same path enforces BOUNDS and not TAGS" — and it
is now a mechanism rather than a description.**

**The approved M1 specification predicted this in these words:** *"the same stale capability installed
into a different CPMP entry is re-adopted until the next broadcast of that index — a property of the
tracker, not the reclaimer."* It says **CPMP entry**. The prediction was on file and named the right
module.

## Why the M1 harness is the worst case by construction

The harness keeps 16 live slots rotating. Every `take()` installs a capability into a CPMP entry, so
consecutive installs into one entry almost always carry different revnode ids, the tracker is thrashed,
and nearly every install re-adopts itself as valid. That is why the board saw it at **every** age —
`k=0` (2,706 storage reuses later), `k=21648`, and `k=43295` alike.

## RETRACTED: "the root cause is load_store_unit.sv:984-989"

I read the optimistic re-validation in `load_store_unit.sv`'s `cap_violation_detection` and attributed
R-35 to it. **The mechanism is identical and the module is wrong.** That block is gated on

    capmode_i && ld_st_priv_lvl_i == riscv::PRIV_LVL_M

and **a domain runs in S-mode**, so it never evaluates for the harness at all. This was already
recorded, six days earlier, in `docs/history/15-09-2026_lsu-capmode-gate-why-domains-cannot-satisfy-it.md`,
whose own measured conclusion is that in M-mode with the gate demonstrably satisfied a load through a
write-only capability, a load at `bound_end`, a store through a read-only capability and a load through
an untagged base **all retire with no trap**. CLAUDE.md's "search prior art before investigating" rule
names this exact cost; the note existed and I re-derived it instead of reading it.

## RETRACTED: "R-35 is therefore a symptom of R-34 plus the capmode privilege gate"

My immediate next inference was also wrong. The LSU block being unreachable in S-mode does not make
R-35 a duplicate: the S-mode path is **CPMP**, which *is* reachable and *does* check revocation. R-35
is a standalone defect in CPMP's tracker. The folder's existing exclusion of R-34 and R-24 stands.

## A second instance, and one that is NOT a duplicate of it

`load_store_unit.sv:984-989` and `commit_stage.sv:239` carry the same optimistic-adopt shape for the
M-mode data path and the PC capability respectively. They are real and should be fixed with this one,
but they are not what the board measured.

## The fix is tractable here, unlike in the LSU, because a query path already exists

The LSU has no revnode query port, which is what made a fix there architectural. The **rev-node unit
already exposes one**: `capstone_rev_node.anvil`'s `IDLE_STAGE` serves `ep.query_req` and answers
`ep.query_res(node_in.valid)`. So the candidate fix for CPMP is to **query on adopt instead of
assuming**, i.e. replace `cpmp_revnode_valid_d[i] = 1'b1` with a lookup, stalling the access until the
answer returns.

Costs and risks, written down rather than waved at:

- **Adopt is on the access path**, so a query turns a first touch of a re-installed entry into a stall
  of the rev-node unit's read latency. The M1 rotation makes that the common case, not the rare one.
- **Failing closed instead (`1'b0`) is not an option**: every rotation would then raise on first touch.
- **Who else uses `query_req` today** must be established before proposing it — if nothing does, this
  would be its first consumer and the channel's arbitration is untested.
- `pmp_data_if.sv` feeds the address path; CLAUDE.md's rule about adding signals into a cone that
  already carries a combinational loop applies, and **only synthesis proves synthesizability**.

**No RTL is changed by this note.** The fix belongs to the RTL lane and to the lead's decision on a
bitstream respin.

## What the directed reproducer does and does not do

`capstone-ariane/verif/tests/custom/capstone/r35-stale-deref.S` on `board/r35-directed-repro` targets
the **LSU** block and therefore cannot reproduce this. Measured, not assumed: it exits 16 (its control
did not trap) because it never executes CAPENTER, so `capmode_i` is 0 and the whole block is inert; a
64-nop barrier after `REVOKE` did not change the reading, ruling out walk timing. Its `CAPPRINT` dump
also proved the test could never have worked anyway — `CAPCREATE` hardcodes `revnode_id = 2`
(`capstone_flu_unit.anvil:337`), so both of its regions shared one revnode and neither could displace
the other's tracker entry.

Two further facts it established, both worth keeping:

- **`MREV` inserts the new node immediately before its parent in DFS order and pushes the parent one
  level deeper**, so `REVOKE(handle)` does reach the parent. Confirmed against the log: `s1`=node 3,
  `s3`=node 4, `a5`=`s2`=node 2.
- **The branch the reproducer sits on is not the board's RTL.** `054cea69b` is *not* an ancestor of
  `board/r35-directed-repro`; `load_store_unit.sv` differs by 22 lines and `capstone_rev_node.anvil` by
  278. Any simulation of R-35 must be done on the `m1-reclaimer` line. A sim reproducer would also have
  to run **in a domain** to reach CPMP at all — `cpmp-su-mode.S` and `domain-switch-S-mode.S` are the
  templates, not `revocation.S`.
