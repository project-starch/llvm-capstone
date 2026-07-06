# Heap temporal safety via revoke-on-free (#78 phase 2) — PROPOSAL

*Status: DESIGN PROPOSAL for review before implementation (standing rule for new
research directions). Builds on #78 phase-1 (the real reusing umm allocator, RV8
7/7 + `heap_free_reuse`/`heap_coalesce` authority probes) and revocation #70
(record → enforce → clean fault delivery → linear re-lend, end-to-end). Paths
against `capstone-bootstrap` as of 2026-07-06. Tracked under task #78.*

## 0. Summary

Phase 1 gave a real allocator with **spatial** safety (each allocation SHRINK-
narrowed to its bounds) and **reuse** (free → coalesce → re-allocate). Reuse is
precisely what creates the **temporal** hole: after `free(p)`, the domain may
still hold `p` (or copies), and a later `malloc` can hand that same memory to a
different object. A stale `*p` then reads/writes another live allocation — a
use-after-free (UAF) — and `free(p)` twice is a double-free. Phase-1's spatial
bounds do **not** catch this (the stale pointer is in-bounds of its *original*
object, which now belongs to someone else).

**Proposal:** make each allocation a distinct **revocation-tree node**; on
`free`, `revoke` that node. Every capability the domain derived from the freed
pointer shares the node, so all of them become invalid — a subsequent use faults
(revoked cap → untagged on reload → `Cap mem access` trap), and a double-free
faults on the already-revoked handle. Reuse is safe: the reclaimed memory is
re-handed under a **fresh** node, and stale caps under the old (revoked) node
cannot touch it.

**Why this is a paper-worthy result, not just an engineering fix:** Capstone's
revocation is a **tree-node invalidation with a lazy per-use validity check**
(`cap_rev_tree_revoke` + `capstone_cap_revoked`), *not* a memory sweep. So
revoke-on-free is **O(1) at free time with no stop-the-world scan of memory and
registers** — the exact cost that forces CHERI temporal-safety systems
(CHERIvoke, Cornucopia) into periodic quarantine-and-sweep. Demonstrating
sweep-free, immediate revoke-on-free on a real allocator is the contribution.

## 1. Threat model (what phase 2 adds over phase 1)

| class | phase-1 (spatial) | phase-2 goal |
|-------|-------------------|--------------|
| over-read/-write past an allocation | **caught** (SHRINK bounds) | unchanged |
| use-after-free (stale `*p` after `free(p)`, memory reused) | **not caught** | **caught** → tag-fault |
| double-free (`free(p)` twice) | **not caught** (umm corrupts) | **caught** → fault on revoked handle |
| dangling copy escapes (stored, returned) then used | **not caught** | **caught** (copies share the node) |

Non-goals: cross-domain revocation (that is #70, already done); sub-object
temporal safety; concurrency (domain is single-threaded).

## 2. Mechanism

The primitives (all confirmed present and domain-callable, unprivileged):

| C builtin | intrinsic | ISel | QEMU helper | semantics |
|-----------|-----------|------|-------------|-----------|
| `__builtin_capstone_cap_mrev(p)` | `int_capstone_cap_mrev` | `selectMrev` | `helper_csmrev` (726) | rs1 **LIN** (asserted), *not consumed*; returns a **REV** handle carrying a fresh child revocation node |
| `__builtin_capstone_cap_revoke(h)` | `int_capstone_cap_revoke` | `selectRevoke` | `helper_csrevoke` (703) | h **REV**; invalidates the node's whole subtree; returns the handle as LIN/UNINIT for reuse |
| (enforcement) | — | — | `capstone_cap_revoked` (967) + load path (1045) | a revoked cap **faults on use** and **comes back untagged when reloaded from memory** |

Per-allocation flow (conceptual — exact derivation sequence is the step-1 spike):

- **`malloc(n)`**: `umm_malloc` picks a block at address `p` (as today). Mint a
  fresh revocation node from the allocator's **linear arena-authority cap** and
  derive the user's allocation capability *under that node*, narrowed to
  `[p, p+n)` (phase-1 SHRINK). Record the REV handle in a side table keyed by the
  block. Hand the narrowed (copyable, NONLIN) cap to the user.
- **`free(user_p)`**: recover the block (as today), look up its REV handle,
  `__builtin_capstone_cap_revoke(handle)` → the node's subtree (every copy the
  domain made of `user_p`) is now invalid. Then return the block to umm's free
  list for immediate reuse. No quarantine, no sweep.
- **reuse**: a later `malloc` over the same memory mints a **new** node; stale
  caps under the revoked node stay dead and cannot alias the new object.

Enforcement is automatic in QEMU: any load/store through a revoked cap traps, and
a revoked cap reloaded from memory is delivered untagged (so even spilled/stored
dangling copies are caught on reload).

## 3. The central open question — the linear mrev-authority cap

> **Step-0 update (2026-07-06, `history/06-07-2026_18-00-00_...`).** Empirically
> confirmed: `mrev` on a NONLIN pointer aborts QEMU (`helper_csmrev` LIN assert),
> and linearity cannot be fabricated from NONLIN. `start.S` boots `sp`/`gp`
> **linear** then `delin`s both before `domain_main`. So step 1 **must** modify
> `start.S` to carve+preserve a linear heap-authority cap (a sub-cap of `gp`,
> which covers the BSS arena) before `delin(gp)`, stashed in a global the
> allocator reloads (linear caps survive store/reload). Chosen path: intra-domain.
> Also good news: rev-tree nodes are **recycled** via a free list, so node
> capacity is bounded by live+in-flight allocations, not total-ever (§4 risk
> reduced).
>
> **Spike follow-up (2026-07-06, later): `gp` is NOT the arena authority.**
> Measured `gp` at runtime — it spans **~592 bytes** (a small-data pointer), so it
> does not cover `cap_arena`; large BSS is reached by another (PC-derived)
> capability. So "split `gp`" is void; sourcing a linear authority *over the heap
> arena* needs a domain-boot-capability-model investigation. Combined with the
> per-allocation-vs-coalescing tension (which points to a **slab** substrate, not
> umm), **phase 2 is confirmed to be a genuine multi-step effort, not a low-risk
> finish** (see `history/06-07-2026_18-00-00_...`). Recommend scoping phase 2 as
> its own project: (i) domain-boot linear-authority investigation, (ii) slab
> allocator, (iii) integration + `use_after_free`/`double_free` probes + RV8
> regression.

`csmrev` **asserts its source is `CAP_TYPE_LIN`.** The allocator therefore needs a
**linear** capability covering the arena to mint per-allocation nodes. Good news:
`csmrev` does *not* consume its source (op_helper.c 733–738: rs1 is untouched), so
**one** retained linear authority cap can mint an unbounded number of
per-allocation nodes. Also, the domain boots with linear caps — the stack pointer
is linear and `delin`'d at entry (op_helper.c 839 comment), so linear authority is
available at domain start in principle.

**Must verify (step-1 spike, gating):**
1. How does the allocator obtain and *retain* a linear cap over its static
   `cap_arena[]` (BSS) as the mrev authority? Candidates: (a) carve it from the
   domain's initial linear memory cap before anything `delin`s it; (b) a small
   start-up shim that hands `cap_heap.c` a linear arena cap. `delin` is one-way
   (LIN→NONLIN), so the linear cap must be captured *before* the arena is used as
   ordinary NONLIN data.
2. The exact instruction sequence to derive the user's revocable allocation cap
   **under** the minted node (so `revoke(handle)` invalidates it). This must
   mirror the **#70 lend path** (monitor `shared_region_annotated` borrow
   derivation) — lift it from there and validate intra-domain. Node hierarchy
   matters: `revoke` kills a node's *subtree*, so the user cap must be a
   *descendant* of the handle's node, not its parent.

If (1) proves impossible (domain data caps are irreversibly NONLIN with no linear
root reachable for the heap), the fallback is a monitor-assisted mint (the domain
asks the monitor to mint revocable heap), which is heavier and reintroduces an
SBI round-trip per allocation — to be avoided if the intra-domain path works.

## 4. Other design points

- **REV-handle storage.** One REV handle (a capability) per *live* allocation,
  held in allocator TCB memory (a side array indexed by umm block number),
  **not** in-band in user memory. Cap-aligned slots (16 B). This is allocator
  authority, like the arena cap — it never reaches untrusted user code. Sizing:
  `arena_blocks` handles worst-case; or a smaller hash keyed by block number.
- **Revocation-node capacity / recycling.** `cap_rev_tree` has finite nodes; a
  long-running allocator mints one per `malloc`. Must confirm `cap_rev_tree_revoke`
  **frees/recycles** the node (or that capacity comfortably exceeds live+freed
  allocation counts for our workloads). This is the main scalability risk; measure
  it. If nodes are not recycled, a batched-revoke (quarantine N frees, revoke
  together) reduces node churn at the cost of a bounded UAF window.
- **Interaction with the narrow/re-widen shim.** `free`'s `cap_rewiden`
  (`cincoffset` off the arena cap for umm's bookkeeping) derives under the
  *arena's* node, which is never revoked — so umm's free-list arithmetic is
  unaffected. Only the per-allocation node is revoked. No conflict.
- **Flagged, default-off.** Gate behind `-DCAPSTONE_HEAP_REVOKE` (or a cap_heap
  runtime flag) so the phase-1 spatial allocator stays the default and RV8/BEEBS
  remain green independent of this work.

## 5. Validation plan (phased; each gates the next)

0. **Smoke**: a minimal domain that `mrev`s a linear cap, derives a cap under the
   node, `revoke`s, and confirms a subsequent access **tag-faults** (proves the
   builtin→ISel→QEMU path works intra-domain, decoupled from the allocator).
1. **Linear-authority spike** (§3): obtain+retain a linear arena cap; derive one
   revocable allocation under a node; revoke; access faults. Gating.
2. **cap_heap integration**: mint-on-malloc + revoke-on-free + REV-handle table,
   behind the flag. umm unchanged.
3. **Authority probes** (the clean oracles phase-1 lacked):
   `heap_use_after_free` (write `p`, free, `malloc` reuses slot, read `p` →
   **tag-fault**) and `heap_double_free` (`free(p)` twice → fault on revoked
   handle). Oracle: `tag-fault`.
4. **Regression**: RV8 7/7 with the flag ON (revoke-on-free must not break
   functional reuse — every freed-then-reallocated benchmark still passes).
5. **Measure**: node-table + mrev/revoke overhead vs phase-1 (per-malloc/free
   cost; node high-water mark); confirm no sweep.

## 6. Deliverables

- `cap_heap.c` mint/revoke + REV-handle table (behind the flag); umm untouched.
- (maybe) a tiny domain start-up shim to capture the linear arena authority.
- authority domains `heap_use_after_free.c`, `heap_double_free.c` + oracle/README.
- a `history/` root-cause note if the spike surfaces ISA/QEMU gaps (as #70 did).
- coverage-matrix + this doc updated with measured results.

## 7. Pointers
- Phase-1: `bounded-heap-allocator-proposal.md` §7a; `benchmarks/rv8/adapted/{cap_heap.c,umm/}`.
- Revocation #70: `revocation-enforcement-proposal.md`,
  `history/03-07-2026_00-00-07_step-b-clean-in-domain-fault-delivery.md`.
- Primitives: `op_helper.c` `helper_csmrev` (726) / `helper_csrevoke` (703) /
  `capstone_cap_revoked` (967); builtins in `clang/.../BuiltinsCapstone.td`
  (`cap_mrev` 188, `cap_revoke` 196); mapping table in
  `sqlite-marshalling-feasibility.md`.
