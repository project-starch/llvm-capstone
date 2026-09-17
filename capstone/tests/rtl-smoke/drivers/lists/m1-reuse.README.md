# `m1-reuse.txt` / `m1-reuse-split.txt` — M1's permitted-reuse demonstration

This sits beside the lists rather than inside them because **the list format has no comments**: the
driver parses every line as an invocation (`board-r1e4.sh:114`) and slices by line number
(`:112`), so a `#` line would both be run as a boot and shift every subsequent line's slice. The
warnings below are load-bearing, so they live in a file a reader will find rather than in a commit
message, which is where reasoning goes to stop being checked.

## Two hazards, read these first

**NEVER RUN THIS ON THE DEPLOYED BITSTREAM (`1bfff7776`).** It has no reclaimer, so the run exhausts
the node pool and **R-12 makes exhaustion a wedge that takes the core with it** — losing the rest of
the boot. On a reclaiming build the same exhaustion *traps* instead, returning an allocator status;
creating that difference is what the exhaustion path was changed for. This list is only for a
bitstream that contains the reclaimer.

**A RUN THAT EXHAUSTS WILL NOT REACH ITS END LINE.** It traps, so the allocation count has to come
from the **last snapshot**, not from `R1 m1 end`. That is why `--cap 20000` is chosen: it sets
`snap_every` to 1250, which keeps a full 200,000-allocation transcript at roughly 32 KB, inside the
64 KiB readback region. Raising the target without re-checking that arithmetic silently truncates the
transcript — `out()` drops characters at `out_limit` and says nothing.

## What it measures, and why it needs no node-id read

The reclaiming bitstream **exposes no way for software to read a revocation-node id** — no LCC
selector (they remain type, cursor, base, end), no new opcode; ids are readable only through
`CAPPRINT`, which lowers to `$display` and exists in simulation only. So the identifier-turnover
witness M1's protocol asks for cannot be taken directly, and any pre-registration of the form
"predict the knee from distinct-indices-per-allocation" is circular, because the only way to obtain
that coefficient on this bitstream is from the knee itself.

**The pool is used as the counter instead.** It hands out **65,532** distinct indices — the head runs
3..65,535 with the sentinel excluded. That number is measured on the reclaiming bitstream rather than
inferred from the head width: `r12-pool-exhaust` does not revoke, so it exhausts by the bump path
alone, and reads exactly 65,532 mints then cause 30 (confirmed three times). So, with `c` =
distinct indices consumed per allocation and `A` = allocations at the trap:

    exhausts at A            ->  c = 65,532 / A          measured directly
    reaches 200,000 target   ->  c < 0.3277              a bound, AND the reuse demonstration:
                                                         more allocations than the pool holds cannot
                                                         have happened without reuse

A 200,000 target measures `c` anywhere from 1.0 down to 0.33, which covers the whole range either
lane is willing to assume — which is none.

**Non-circular by construction:** `c` comes from the exhaustion point and the cost knee comes from the
take curve. They are different parts of the transcript and neither is computed from the other.

## What bounds the coefficient — the two rules that decide it

1. **A node is reclaimable only in the walk that invalidates it.** The free-list push lives inside the
   `node_in.valid == 1'd1` branch of the walk; `free` is set there and only there, never in
   `change_rev_node_validity`, which DROP shares and whose nodes stay linked. Anything invalidated
   earlier — by DROP, or by a previous walk that has already spliced it out — is unreachable to every
   later walk and leaks for the life of the boot.
2. **The handle a revoke uses is not freed by that revoke.** Inherent: the handle stays usable after
   the revoke, so its node must stay live.

For a workload that never DROPs, on a spliced tree, rule 1 means **every node a walk visits is valid**
and the valid fraction is 1.0 by construction — so it is not worth measuring, and the coefficient is
decided entirely by rule 2. `sublet_take` is LDC + MREV + STC + DELIN and DELIN does not invalidate,
so this harness qualifies.

**Assert it rather than measure it.** A transcript showing a walk visiting a node it does not free
means either a DROP nobody knew about or an index that has retired after 16,384 reclaims of itself.
Both are findings; the constant is not.

## Expectations

**None between 1.0 and 0.33, deliberately.** `c = 1.0` — every allocation consuming a fresh index —
would say the reclaimer works exactly as designed and still cannot help a workload whose allocations
are all handles. That is a decision-level answer for M1, whose claim is about whether reclamation
bounds this cost, and it is a result rather than a null.
