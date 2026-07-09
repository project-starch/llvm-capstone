# row3 B2 — the LITERAL matched pair: a revoke-on-free linear allocator

*2026-07-10, B-lane, branch `capstone-b-008-revoke-on-free` (a separate branch,
per the task: this is emulator + allocator + codegen work). Builds on task-007
(held-cap delivery), task-006 (C1/C2 codegen), and A's B1 matched pair
(`09-07-2026_23-15-00_row3-b1-matched-pair.md`).*

## What this closes

row3's "after" was, until now, either an abstract probe or A's **B1**: real
SQLite in one domain, but the revoked pointer was a **carved copy** of the column
name, because memsys5 allocations are not independently revocable. A reviewer
could fairly object: *"you protected a copy you made, not SQLite's real memory."*

**B2 removes the objection.** SQLite's ENTIRE heap is a revoke-on-free linear
allocator (`benchmarks/sqlite/revoke_on_free_alloc.h`) installed via
`SQLITE_CONFIG_MALLOC`. The pointer that faults post-`finalize` is the exact value
`sqlite3_column_name` returned — SQLite's own pointer, into SQLite's own
allocation, revoked by the `xFree` on SQLite's own `finalize` path. No wrapper, no
copy, no driver-fired revoke.

## The allocator (Route 1: non-coalescing, no emulator change)

Every allocation is an independently revocable capability:

- **xMalloc(n)**: carve a fresh `SPLIT` sub-capability off the arena tail (its own
  revocation-tree node, tight bounds), `MREV` it **while still LIN** (store the rev
  handle in a side-table), hand SQLite the **`delin`'d** NONLIN alias. MREV-before-
  delin is forced: `csmrev` asserts LIN, and a LIN cap is consumed by copy (C3), so
  the delin must come after the mrev and before SQLite copies the pointer.
- **xFree(p)**: find `p`'s slot (keyed by `cap_get_base`, unique and monotone) and
  `REVOKE` its handle. Every cached copy of `p` — including one the caller stashed
  across `finalize` — now faults. The arena is **not** reclaimed (SPLIT is one-way;
  there is no merge op), only the slot is marked reusable.
- **xRealloc** = malloc + tag-preserving copy + free (the free revokes the old
  node, modelling realloc-invalidates-old).

This is the shape task-007's Step-3 verdict identified as the only reachable one:
per-allocation SPLIT nodes, never coalescing. The cost is fragmentation (Phase 2).

## The codegen fix this forced (the real prize)

A left B1 green at `-O0` but QEMU-blocked at `-O1/-O2` with `helper_csdelin:
Assertion rd_v->tag failed`, before any revoke — it aborted the no-revoke control
too, so it was a codegen robustness bug, not a mechanism flaw. A's hypothesis was
a `.insn`-split cap losing its tag across a call spill. **That hypothesis was
wrong.** The reproducer is 12 lines with no `.insn` and no SPLIT:

```c
static void *g_a, *g_b; static unsigned g_count;
void domain_main(void *arg, unsigned func) {
  if (func == 1) { if (g_count==0) g_a=arg; else if (g_count==1) g_b=arg; ++g_count; return; }
  *(unsigned*)arg = g_count;
}
```

Root cause, and it is mine (task-006 C2): a global's address is materialised as
`cincoffset rd, gp, off` (a LINEAR capability) followed by a tied, in-place
`delin` to make it NONLIN. C2 set `hasSideEffects = 1` on DELIN — correctly, so it
is not DCE'd — but the `cincoffset` is pure. MachineCSE hoists the shared
`_MergedGlobals` base out of the branches that use it while leaving one DELIN
behind in each. That leaves a **LINEAR value with several uses**, and the ISA
consumes a non-NONLIN source on copy (`movc rd, rs1` nulls rs1, C3): the two-
address pass gives each use its own `movc` copy, and the FIRST nulls the shared
register. Every later `delin`/deref then hits an untagged operand. `-O0` never
noticed because it re-materialises the base for every access.

Fix (`CapstoneISelDAGToDAG.cpp` selectLGA + `PseudoCapGlobalBase` in
`CapstoneInstrInfo.td` + its post-RA expander): emit the `cincoffset`+`delin` pair
as ONE pseudo. Inside a pseudo, no LINEAR value is ever an SSA value with multiple
uses — the operand is the scalar offset and the result is already NONLIN, so it is
freely copyable and freely CSE-able. lit: `global-base-nonlinear.ll`. This helps
B1 too, and any domain TU with a global used across branches at `-O1+`.

## The other bug B2 forced: tag-preserving realloc

A first cut of `xRealloc` copied the old block byte-by-byte. That **strips the tag
off any capability the block holds**, so a pointer SQLite stored in a realloc'd
structure came back untagged and faulted (cause 24) deep inside SQLite setup —
before the row3 read even ran. The domain's own libc `memcpy` is already
capability-preserving (copies `void**` when aligned); `xRealloc` now copies
capability-wide via `rof_copy_caps` (ldc/stc, which move a cap with its tag and a
scalar as a scalar). This is a general lesson for any custom Capstone allocator:
**realloc must copy tag-preserving.**

## Phases and results

**Phase 0 — allocator primitive, no SQLite** (`tests/runtime-qemu/revoke-on-free-probe/`,
3 probes × `-O0/-O1/-O2`, 9/9):

| Probe | `-O0` | `-O1` | `-O2` |
|---|---|---|---|
| `alloc_use_after_free_fault` | FAULT 24 | FAULT 25 | FAULT 25 |
| `alloc_no_free_ok` | `0x0812005e` | `0x0812005e` | `0x0812005e` |
| `alloc_sibling_survives_ok` | `0x0813003c` | `0x0813003c` | `0x0813003c` |

**Phase 1 — literal row3 on real SQLite** (`benchmarks/sqlite/run-sqlite-row3-b2.sh`,
fault + no-revoke control × `-O0/-O1/-O2`, 6/6):

| Domain-TU opt | control | fault variant |
|---|---|---|
| `-O0` | RETURNS, colname `'c'` live + post-finalize | FAULT **cause 24** |
| `-O1` | RETURNS, colname `'c'` | FAULT **cause 25** (self-proving) |
| `-O2` | RETURNS, colname `'c'` | FAULT **cause 25** (self-proving) |

The `-O1/-O2` self-proving boots are new: B1 could not boot at `-O1+` at all
(the DELIN-tag abort above), so this is the first time the intended cause-25 form
of row3 has run on real SQLite. SQLite (the engine) is built `-O0`; only the
domain TU opt varies.

The no-revoke control runs the whole program on the allocator and returns colname
`'c'` live and post-finalize; the fault variant's post-finalize `name[0]` read
faults, confirmed to be the read in `run_row3_b2` (the fault PC maps into that
function, badaddr in the arena — the revoked column-name buffer), not an internal
SQLite free-then-use. At `-O0` the cause is 24 (the control disambiguates); at
`-O1/-O2` `name` stays register-held across `finalize` so it is cause 25, self-
proving. SQLite (the engine) is built `-O0`; only the domain TU opt varies.

## Scope

- `llvm/` (in-tree): the `PseudoCapGlobalBase` global-base codegen fix + lit.
- `capstone/benchmarks/sqlite/`: `revoke_on_free_alloc.h`, `sqlite_row3_b2_domain.c`,
  `sqlite_host_row3_b2.c`, `run-sqlite-row3-b2.sh` (all NEW; A's `sqlite_row3_domain.c`
  untouched).
- `capstone/tests/runtime-qemu/revoke-on-free-probe/` + `build-`/`run-revoke-on-free-probe.sh`.
- Untouched: `start.S`, the monitor, `capstone-c`, `caplifive-buildroot`,
  `capstone-qemu` (Route 1 needs no emulator change).

## Phase 2 — how far does it generalize? (honest limits)

**B2 is correct for bounded-lifetime workloads, and is NOT a general heap.** The
churn probe (`-DROW3_B2_CHURN=N`, the row3 prepare/step/column_name/finalize cycle
in a loop with revoke on at every finalize; returns cleanly so the host flushes
the report):

- **N=100 completes:** `carved_total=583696  peak_live=30144  arena_left=3610608`.
  So a real 30 KiB working set carved 570 KiB — **≈19× overhead**, because SPLIT
  never coalesces: every allocation shrinks the arena from the top and a free can
  only mark the slot reusable, never return arena space. The 4 MiB arena depletes
  at ~700 cycles on this trajectory.
- **N=100000 aborts** — not on the arena, but on the emulator's **fixed
  10000-entry revocation-node pool**: `cap_rev_tree.c:20 _cap_rev_tree_dup_node_before:
  Assertion new_node != CAP_REV_NODE_ID_NULL failed`. Each allocation mints two
  nodes (SPLIT + MREV); revoke invalidates the junior run but does not reliably
  reclaim nodes to the free list while the owner still holds the rev handle, so
  the pool exhausts at a few hundred cycles — before the arena does.

So there are **two non-coalescing ceilings**, arena bytes and rev-tree nodes, and
the node ceiling is hit first and as a hard QEMU **assert**, not a graceful
`SQLITE_NOMEM`. For row3 (a handful of allocations) neither is remotely
approached; for a sustained workload both bite.

**Verdict:** B2 is the right artifact for the paper's row3 fidelity claim — the
literal matched pair, SQLite's own pointer revoked by SQLite's own free — but it
is a row3-scale / bounded-lifetime demonstration, not a drop-in general heap. The
general answer is **Route 2**: a rev-tree merge/unsplit op so a coalescing
allocator can reunify freed neighbours' nodes and reclaim both arena and nodes.
That is a real emulator-semantics change (rev-tree depth/seniority invariants +
its own probe and authority regression) and is left as a design direction, not
implemented — per the task's propose-before-big-directions rule.

## Flags for A / the PI (not fixed here)

- **The rev-node pool exhaustion is an ASSERT, not a graceful failure.** Under
  sustained revoke-on-free churn the emulator aborts (`_cap_rev_tree_dup_node_before`)
  instead of the allocator seeing a failed derivation and returning NOMEM. A
  robustness fix (return a NULL node id and let `helper_cssplit`/`helper_csmrev`
  raise a clean fault) is B's lane (emulator) but a semantics call; not done.
- The task-007 revoked-region **host landmine** still applies: after the domain
  revokes a REV_TRANSFERRED lineage the host must not touch that region. The B2
  host reads only the host-call payload region (live), never the arena.
- The `helper_csmrev` `rd == 0` guard (task-006) is now heavily exercised (one
  MREV per allocation) but stays unreachable from the compiler (`$rd` is
  `GPRNoX0`); still worth a guard if anyone hand-writes the op.
