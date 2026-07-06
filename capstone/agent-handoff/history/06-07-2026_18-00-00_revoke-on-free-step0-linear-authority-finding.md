# Revoke-on-free step-0 smoke: the linear-authority finding (#78 phase 2)

**Date:** 2026-07-06
**Outcome:** The mrev/revoke primitive path is wired and domain-callable, and the
rev-tree semantics support per-allocation revocation. **But** the central gating
constraint is now *empirically confirmed*: `csmrev` requires a **LINEAR** source
capability, and a linear cap cannot be fabricated from a NONLIN one — so
intra-domain revoke-on-free requires capturing a linear authority cap from the
domain's boot `sp`/`gp` **before** `start.S` delinearizes them. Step 1 (linear
authority) must therefore modify the domain entry.

## What was validated

- **Primitives wired** (grounded in code): `__builtin_capstone_cap_mrev` /
  `cap_revoke` (`clang/.../BuiltinsCapstone.td` 188/196) → `int_capstone_cap_{mrev,
  revoke}` → `selectMrev`/`selectRevoke` → `helper_csmrev` (op_helper.c 726) /
  `helper_csrevoke` (703). No privilege gate — domain-mode code can execute them.
- **Rev-tree model** (`cap_rev_tree.c`): a depth-ordered doubly-linked list, not a
  general tree. `mrev(C)` = `_dup_node_before(C)` + `C.depth++`: inserts a senior
  REV handle *before* C's node and pushes C one level junior; returns the REV
  handle. `revoke(H)` walks H's `.next` (junior) run while `depth > H.depth`,
  setting `.valid=false`, then splices the run out (nodes recyclable via
  `free_list` — `_cap_rev_tree_alloc_node` reuses them, so **node capacity is
  bounded by live+in-flight, not total-ever** — good for the scalability risk).
  `capstone_cap_revoked` (967) + the load path (1045) enforce: a revoked cap
  faults on use and returns untagged when reloaded.
- **Per-allocation revocation is expressible:** take a LINEAR alloc cap C, `H =
  mrev(C)` (keep H, C stays LIN), `delin(C)` → NONLIN copyable, hand C to the user;
  copies share C's node; `revoke(H)` invalidates C and every copy. Reuse re-mints
  a fresh node so stale copies can't alias.

## The blocker, empirically confirmed

`csmrev` asserts `rs1.type == CAP_TYPE_LIN` (op_helper.c 731). A throwaway domain
that `mrev`s a stack local (NONLIN after `delin sp`) aborts QEMU with exactly:

```
helper_csmrev: Assertion `rs1_v->val.cap.type == CAP_TYPE_LIN' failed.
```

And linearity is conserved: there is no instruction/builtin to mint a LINEAR (or
UNINIT) cap from a NONLIN one (`csinit` needs UNINIT; UNINIT/LIN originate only
from the boot caps or a revoke). `start.S` shows the domain boots with `sp` and
`gp` **linear**, then `delin(sp)` (at `_start`) and `delin(gp)` (at `test:`)
before `domain_main`. So by the time `cap_heap` runs, no linear cap over the BSS
arena survives.

**Conclusion:** intra-domain revoke-on-free needs `start.S` to carve a linear
sub-capability for a dedicated heap-authority region from `gp` (or `sp`) *before*
delin and hand it to `cap_heap` (e.g. `stc` it to a known global slot the
allocator `ldc`s). This is the step-1 shape. Linear caps survive store/reload
(the stack save/restore already relies on this), so a stashed authority slot is
viable. `gp` covers the whole data segment including `cap_arena`, so a linear
sub-cap of `gp` can be an ancestor of every allocation (required: `revoke` kills a
node's junior run, so the alloc cap must derive from the mrev'd authority).

## Also observed (not investigated)

A first probe that `mrev`'d the address of a *static* with globals-SHRINK on
crashed earlier in `helper_csshrink` (`base >= cap.base && end <= cap.end`),
i.e. a globals-narrowing codegen assertion, before reaching mrev. Flag for step 2:
the allocation cap will be **both** SHRINK-narrowed (C1 spatial) **and** mrev'd
(temporal); verify the ordering (narrow the delin'd user cap, mrev the wider
authority-derived cap) so the two do not fight. Not a blocker; just an ordering
constraint to respect in `cap_heap`.

## Spike follow-up (2026-07-06, later) — a THIRD blocker: `gp` doesn't cover the arena

Built a custom domain entry (`start_linauth.S`, via `START_SRC=`, so shared
`start.S` untouched) that tried to `cssplit` a linear authority off `gp` before
`delin`. Two empirical results:

1. `cssplit t2, gp, (gp.end-4096)` aborted: `helper_cssplit` assert
   `mid > base && mid < end` — i.e. `gp` spans **< 4096 bytes**.
2. A diagnostic that stores `gp`'s measured range to a global OOB-faulted while
   storing, printing `bounds = (0x101560000, 0x101560250)` — **`gp` spans ~592
   bytes**. It is a genuine RISC-V **small-data global pointer**, not a whole-image
   data capability. `cap_arena` (tens of KB of BSS) is **not** covered by `gp`;
   large globals/BSS are reached by another capability (PC-relative via the code/
   PC capability under medany, most likely), whose linearity and capture path are
   unknown.

**Consequence:** the "split `gp` for arena authority" plan is void. Obtaining a
linear authority *over the heap arena* intra-domain requires understanding the
domain boot capability model — specifically which capability covers BSS and
whether a linear form of it is capturable before it is delin'd/used. This is a
real investigation, not a quick edit.

## Feasibility verdict (answers "is phase 2 a quick, low-risk finish?")

**No.** Phase-2 revoke-on-free has (at least) three independent, substantial
blockers, each now evidenced:
1. **Per-allocation revocation vs coalescing.** Independent per-allocation nodes
   need per-allocation LINEAR caps (via non-mergeable `split`), which fragments
   and cannot coalesce → fights umm. Points to a **slab/fixed-slot** substrate.
2. **`mrev` needs LINEAR** (confirmed abort on NONLIN); linearity can't be
   fabricated from NONLIN.
3. **Arena linear authority is not `gp`.** `gp` is a ~592-byte small-data pointer;
   the arena-covering capability and its linear-capture path are unknown and need
   a domain-boot-capability-model investigation.

Each is tractable, but together they make phase 2 a genuine multi-step research +
engineering effort (new slab allocator + domain-entry linear-authority sourcing +
integration + probes + regression), **not** a low-risk one-shot finish. Phase-1
(spatial, umm, RV8 7/7 + reuse/coalesce probes) is complete and shippable; phase-2
should be its own scoped effort.

## Next step (needs sign-off: touches shared domain entry)

Modify `my_first_domain/start.S` to carve+preserve a linear heap-authority cap
before `delin(gp)`, guarded so existing domains are unaffected (they simply ignore
the extra global). Then the step-0 smoke can `mrev`/`revoke` the captured
authority and confirm a post-revoke access tag-faults, before any `cap_heap`
integration. See `design/heap-temporal-safety-revoke-on-free-proposal.md` §3/§5.
