# Proposal: enforce capability revocation on the capability-use path (QEMU)

*Status: DESIGN PROPOSAL for review with the QEMU/runtime author before any
implementation. No `capstone-qemu` changes have been made. Grew out of the M0
spike for the SQLite marshalling direction
(`sqlite-marshalling-feasibility.md`), which showed revocation is currently
**recorded but not enforced**. Paths/line numbers are against `capstone-bootstrap`
as of 2026-06-30.*

## 0. Summary

`csrevoke` marks revoked rev-tree nodes `valid = false`, but **no
capability-use path ever reads that bit**, so a revoked capability that is
already materialized (in a register, in memory, or freshly loaded) keeps
working. We propose making revocation bite by a **lazy per-use validity check**:
the load/store and cap-load paths consult the revoking capability's rev-tree node
and fault if it is invalid. This is a `capstone-qemu` (submodule) change only; the
compiler/toolchain are unaffected.

## 1. Minimal reproducer (borrow → revoke → use-after-revoke still succeeds)

Committed: `tests/runtime-qemu/borrow-revoke-uaf-probe/` (build + run wrappers in
`tests/runtime-qemu/`). Essence:

- **Lender** (`*_guest.c`, ordinary guest Linux via `libcapstone`): create a
  region, lend it as a **revocable borrow**
  (`shared_region_annotated(.., PERM_OUT, REV_BORROWED)`), then between two domain
  calls `revoke_region(region_id)`.
- **Borrower** (`*.smode.c`, runs in the domain): round 1 caches the delegated
  pointer and writes a stage-1 sentinel; round 2 (after the lender revoked)
  dereferences the **cached** pointer.

Observed (serial log): round 1 writes `0x1111…`; `revoke_region` succeeds; round 2
**store lands** and the lender reads `0x2222…`. No fault. The run wrapper is green
on this (gap) behaviour and is the regression artifact; the success criterion
*after* the fix is that round 2 instead raises a capability fault.

## 2. Exact current access path, and why rev-tree `valid` is not consulted

- **Load/store authorization** — `_helper_access_with_cap`
  (`target/riscv/op_helper.c:943`): faults only if `!rs1_v->tag`
  ("Cap mem access requires capability"), then computes
  `addr = cap->bounds.cursor + imm` and checks `cap_in_bounds`. **No rev-tree
  lookup.** So a tagged, in-bounds — but revoked — base register passes.
- **Capability load** — `helper_load_with_cap` sets
  `env->load_is_cap = cap_mem_map_query(&env->cm_map, addr, …)`; then
  `helper_reg_set_cap_compressed` (`op_helper.c:999`) sets `rd_v->tag =
  env->load_is_cap`. The reloaded tag therefore reflects "is a capability stored
  at this address" (`cm_map`), **not** rev-tree validity.
- **Revoke** — `helper_csrevoke` (`op_helper.c:703`) calls `cap_rev_tree_revoke`
  and retypes the REV cap to LIN/UNINIT. It does **not** scrub register tags or
  `cm_map` entries.
- **The `valid` bit** (`cap_rev_tree.c`): written `true` on node create, asserted
  in `_cap_rev_tree_dup_node_before`, set `false` in `cap_rev_tree_revoke:81` —
  and **read nowhere else** in `target/riscv`. Every capability already carries
  its `rev_node_id` (`cap.h:74`) and compression preserves it
  (`cap_compress.c:70,130`), so the link a check would need is present.

Net: the data structure for revocation exists end-to-end; only the *check at use*
is missing.

## 3. Intended architectural semantics

A **revoked derived capability must fault on any use** — load base, store base,
capability load that would reconstruct it, and (for completeness) any cap-op that
consumes it as input. The fault must be deterministic and trappable (not a host
abort; see the separate fault-delivery gap in the feasibility doc). Required
properties:

- **Senior/hierarchical revoke cascades:** revoking a senior (earlier) revocation
  node invalidates all overlapping/derived descendants (the `close`-cascade,
  proposal Table 4 H rows).
- **Re-share after revoke still works:** after `csrevoke`, the retained node
  becomes LIN/UNINIT and may be delegated again (the existing known-good probe).
- **Non-revoked capabilities are unaffected:** no false positives on live caps;
  benchmarks must stay green.

## 4. Lazy per-use valid check vs eager tag scrub

**Option A — lazy per-use validity check (recommended).** On each capability use,
look up `tree[cap.rev_node_id].valid`; if false, fault.

- *Pros:* O(1) array index on the node id; matches the very reason the rev tree
  exists (revoke is O(subtree) marking, use is O(1) check, no need to find all
  copies of a cap); uniformly covers register-held, memory-stored, and
  freshly-loaded caps because all of them carry `rev_node_id`; small, localized
  change.
- *Cons:* adds a load + branch to the hot memory path; needs the rev-tree pointer
  reachable from the access helper (it is — `env->cr_tree`); relies on revoked
  nodes not being recycled out from under a stale cap (see safety note).
- *Recycling safety:* `cap_rev_tree_revoke` does **not** free the revoked nodes
  (it unlinks the subtree but never calls `cap_rev_tree_release`), so their
  `valid=false` persists. `cap_rev_tree_release` is refcount-gated
  (`_CAP_REV_NODE_REUSABLE`), so a node with an outstanding cap reference is not
  reused. A lazy check is therefore safe against id aliasing **provided** that
  invariant holds; confirm with the author (and consider a per-node generation
  counter if refcounting can ever drop a still-referenced node).

**Option B — eager tag scrub on revoke.** `csrevoke` walks and clears `tag` on
every materialized copy of the revoked caps.

- *Pros:* zero hot-path cost; access path unchanged.
- *Cons:* copies live anywhere — all GPRs, spilled stack slots, arbitrary memory,
  other domains. The `cm_map` tracks in-memory cap *locations* but eager scrub
  would have to (a) enumerate `cm_map` entries within the revoked bounds and (b)
  scrub register files, and even then it conflates *address overlap* with
  *lineage* — two distinct revocation subtrees over the same address range would
  be wrongly scrubbed. This is precisely the global sweep the rev tree was
  designed to avoid.

**Recommendation: Option A (lazy).** It is the design the rev tree implies, it is
O(1), and it is the minimal correct change. Measure the hot-path cost; if it
matters, gate behind a fast "any revocation has ever happened" flag or a
per-page dirty hint.

## 5. Minimal QEMU change recommended

1. Add `bool cap_rev_node_is_valid(cap_rev_tree_t *tree, cap_rev_node_id_t id)`
   in `cap_rev_tree.c` (return `id == NULL || _CAP_REV_NODE(tree, id).valid`;
   NULL id = a cap with no revocation lineage = always valid).
2. In `_helper_access_with_cap` (`op_helper.c:943`), after the `!tag` check, add:
   `if (!cap_rev_node_is_valid(&env->cr_tree, cap->rev_node_id)) raise
   RISCV_EXCP_INVALID_CAP;` (`0x19` already exists in `cpu_bits.h:694`). This is
   the single change that makes the register-held and memory-stored (reloaded
   into a base reg) use-after-revoke fault.
3. For capability loads, derive the result tag from validity too:
   in `helper_reg_set_cap_compressed` (`op_helper.c:999`) set
   `rd_v->tag = env->load_is_cap && cap_rev_node_is_valid(&env->cr_tree,
   rd_v->val.cap.rev_node_id);` so a revoked cap reloaded from memory comes back
   untagged.
4. (Completeness, optional in v1) gate cap-op inputs that consume `$cap_in`
   (shrink/tighten/seal/mrev/…) on the same check, so a revoked cap can't be
   laundered into a fresh derivation.

No compiler or `libcapstone`/kernel-module change is required for the core fix;
the borrow/revoke ABI already carries the lineage. Confirm the revoke
subtree-selection (`cap_rev_tree_revoke`'s `depth` loop condition reads
`node_id.depth` inside the loop, which looks constant — verify it actually walks
the intended descendants) with the author as part of this change.

## 6. Test matrix (before = gap reproduced; after = fault / still-works)

| # | Case | Before fix | After fix |
|---|------|-----------|-----------|
| 1 | **Register-held cap** after revoke (cached pointer, no reload) — the M0 probe | store lands | capability fault on the round-2 access |
| 2 | **Memory-stored cap** after revoke (spill the borrowed cap to a domain stack/global, reload into a base reg, deref) | store lands | fault on the access via the reloaded base |
| 3 | **Cap loaded from cap-memory-map** after revoke (`ldc` a stored capability whose region was revoked) | loads back tagged | loads back untagged ⇒ next deref tag-faults |
| 4 | **Derived sub-cap after senior revoke** (mrev a senior node, derive/delegate a sub-region, revoke senior, use sub-cap) | sub-cap usable | sub-cap faults (cascade) |
| 5 | **Re-share after revoke** still succeeds (existing `…payload-revoke-probe`) | succeeds | still succeeds (no regression) |
| 6 | **Non-revoked caps still work** (authority suite 20/20; CoreMark + BEEBS-82 + RV8-7) | green | still green (no false positives) |

Cases 1–4 are new probes in the `borrow-revoke-uaf-probe` family (cases 2–3 add a
spill/`stc`+`ldc` step; case 4 adds a senior/sub delegation). 5 reuses the
known-good probe; 6 is the existing regression gate. Each new case ships with the
oracle-classified runner so the matrix is mechanically checkable.

## 7. Implementation status (2026-06-30)

Implemented and tested the enforcement half; the recording half is blocked on an
author design decision.

- **Enforcement patch (kept, in working tree, `op_helper.c`):** added
  `capstone_cap_revoked()` (NULL/out-of-range node id ⇒ live) gated by
  `CAPSTONE_REVOCATION_ENFORCE` (default 1, flip to 0 to revert); a check in
  `_helper_access_with_cap` raising `RISCV_EXCP_INVALID_CAP`; and an untag of a
  revoked cap reloaded in `helper_reg_set_cap_compressed`. Minimal and reversible.
  **Verified non-regressing:** with this patch alone, the re-share-after-revoke
  probe still passes and M0 stays at the no-trap gap — because it is **dormant**:
  nothing currently sets a node invalid (see next point), so the check never fires.

- **Recording bug (found, fix reverted):** `cap_rev_tree_revoke`'s loop condition
  is `_CAP_REV_NODE(tree, node_id).depth > depth` where `depth` was just read from
  the *same* node — always false on entry, so the loop body never runs and
  revocation marks **nothing** invalid. The intended condition almost certainly
  tests the current node (`cur`).

- **Why the recording fix is not landed:** changing the condition to `cur` does
  make the M0 **use-after-revoke fault as desired** (safe-fail achieved end to
  end — the borrower's cached cap is reloaded untagged and the round-2 store no
  longer lands). **But it breaks the known-good re-share-after-revoke flow**: once
  `csrevoke` walks the subtree and retypes the node (LIN→UNINIT via `retain_data`),
  the subsequent re-share's `helper_csmrev` asserts `CAP_TYPE_LIN`. So the
  recording-side bug is **entangled with the revoke/re-share lifecycle** (the
  `retain_data` / LIN-vs-UNINIT retyping and the mrev precondition) and a correct
  fix needs the author's intended semantics, not a one-token edit.

**Net:** the enforcement *mechanism* for #70 is in place and proven (when revoke
records an invalidation, a use-after-revoke faults), with no regression. The
remaining work is the **recording-side semantics**, which is the author's call.

## Next step

Raise this note with the QEMU/runtime author (lazy-vs-eager decision, the
recycling-safety invariant, and the `depth`-loop question) **before**
implementation. Tracked as task #70.

## Pointers
- M0 finding + probe: `sqlite-marshalling-feasibility.md` (M0 result),
  `../../tests/runtime-qemu/borrow-revoke-uaf-probe/`.
- Code: `target/riscv/op_helper.c` (`_helper_access_with_cap:943`,
  `helper_csrevoke:703`, `helper_reg_set_cap_compressed:999`),
  `target/riscv/cap_rev_tree.c` (`cap_rev_tree_revoke`, `cap_rev_tree_release`),
  `target/riscv/cap.h:74` (`rev_node_id`), `cpu_bits.h:694` (`RISCV_EXCP_INVALID_CAP`).
