# Faithful real-SQLite matched pairs, one per Stage-2 shape (task 010)

*2026-07-10, compiler-lane, branch `capstone-bootstrap-b`. Builds on the row3 B2
revoke-on-free allocator (task-008), the held-cap linear grant (task-007),
`csdrop` (task-002) and the UNINIT-from-revoke derivation (task-009).*

## What this closes

Before this task the SQLite Stage-2 corpus was 17/17 validated on RTL, but only
**one row (row 3)** was a faithful, literal real-SQLite repro (task-008 B2). The
other 16 were mechanism probes: the capability primitive was shown to exist, but
not that it catches the real bug in the real program. A reviewer could say so.

This task delivers **one LITERAL real-SQLite matched pair per Stage-2 shape**, so
the paper can claim a faithful repro for each defect class. Each is the SAME
program as the row's `before.c`, real SQLite linked, run two ways: native host →
ASan reports the defect; real-SQLite Capstone domain → capability fault on
SQLite's own handle/pointer, with a no-fault control on identical source.

| Shape | Row | Fault | Cause | Self-proving? |
|---|---|---|---|---|
| LINEAR (L) | 11 double-finalize | 2nd `sqlite3_finalize` derefs revoked stmt | 24 (O0+O2) | no — stmt spills across the finalize call; control disambiguates |
| UNINIT (U) | 14 uninit connection | pre-open read of UNINIT `db` | 26 (O0+O2) | yes — UNINIT keeps its tag, faults on TYPE |
| HIERARCHICAL (H) | 7 cursor dealloc | post-close `sqlite3_step` of revoked child stmt | 24 (O0+O2) | no on real SQLite (stmt spills across `close_v2`); cause-25 self-proving reached in the Phase-0 primitive |
| SEALED-CALLBACK (S) | 2 UDF context UAF | callback derefs revoked `pApp` context | 24 (O0+O2) | no — SQLite reloads `pApp` from its function table on each call |
| BORROW-REVOKE (R) | 3 | (already faithful, task-008 B2) | 24/25 | — |

Result: **every in-scope shape now has a literal real-SQLite repro** — 5 faithful
(one per class), the rest mechanism probes.

## The rows

### L — row 11 (double-finalize), `benchmarks/sqlite/sqlite_row11_domain.c`

SQLite's whole heap is the revoke-on-free allocator (`revoke_on_free_alloc.h`,
`SQLITE_CONFIG_MALLOC`). `sqlite3_prepare_v2` hands back a statement handle into an
`rof_malloc` allocation (the Vdbe block). The first `sqlite3_finalize` frees →
REVOKEs it; the second `sqlite3_finalize` dereferences SQLite's own revoked handle
and faults. `run-sqlite-row11.sh`, 4/4 -O0/-O2: control returns rc=0 (double
finalize with revoke suppressed runs clean — the freed-but-intact Vdbe is walked
again harmlessly); fault variant faults cause 24. The handle must cross the first
finalize call, so it is spilled and the reload comes back untagged (cause 24) at
every opt level; the control disambiguates.

### U — row 14 (uninit connection), `benchmarks/sqlite/sqlite_row14_domain.c`

`db` is a genuine UNINIT capability (revoke a still-linear lineage → UNINIT,
task-009), carved off a 64-byte SPLIT sub-cap of the arena tail so the rest stays
SQLite's heap. The matched pair is the ORDER: fault variant reads `db` then opens
(pre-open read faults cause 26); control opens FIRST with real `sqlite3_open`
(which overwrites `db` with a valid handle) then reads and returns.
`run-sqlite-row14.sh`, 4/4 -O0/-O2, cause 26 opt-independent and self-proving.
**Residual:** real `sqlite3_open` allocates a fresh object and writes its pointer
into `&db`; it does not initialise a caller-provided UNINIT region. So we MINT the
UNINIT `db` to model the uninitialised connection; the correct path runs real
SQLite. The fault is on a genuine UNINIT capability.

### H — row 7 (cursor dealloc), `benchmarks/sqlite/sqlite_row7_domain.c` +
`revoke_on_free_hier_alloc.h` + Phase-0 `tests/runtime-qemu/hier-revoke-probe/`

**The hard one, and the mechanism answer is YES.** Per-connection SUB-ARENAS: each
connection gets its own SPLIT sub-arena off the main grant, MREV'd to a senior
revocation node; every allocation SQLite makes while that connection is active is
a SPLIT descendant of the senior node. `hier_close` REVOKEs the senior node,
sweeping the whole subtree (connection + live statement). A sibling connection's
sub-arena is an independent SPLIT and SURVIVES — the scoping that makes it
hierarchical, not a global heap wipe. Intra-domain (SPLIT + MREV + REVOKE), no
monitor/region/`start.S` change.

Two mechanism facts the checkpoint had to establish (and did, in Phase 0):
1. **REVOKE of a connection's senior MREV handle DOES sweep a child SPLIT off the
   sub-arena AFTER the MREV.** `cssplit` makes siblings independently revocable OF
   EACH OTHER, but a child derived from the sub-arena is a DESCENDANT of the
   sub-arena's node, hence of the senior MREV node; revoking the common ancestor
   sweeps it. Confirmed: `hier_child_revoked_fault` faults (cause 24 -O0, **cause
   25 self-proving -O2**).
2. **Scoping.** `hier_sibling_conn_survives_ok`: closing connection A leaves
   connection B and its child fully usable. Phase-0 probe 6/6 -O0/-O2.

**Why the revoke is fired at close, not by SQLite freeing memory:** SQLite's
`sqlite3_close_v2` is a ZOMBIE close — with a live statement it does NOT free the
statement, so there is no xFree for the flat allocator to turn into a revoke. The
cascade comes from the capability tree this layer builds. Phase-1 real SQLite
(`run-sqlite-row7.sh`) 4/4 -O0/-O2: control returns SQLITE_ROW (zombie close
leaves the statement usable — this is exactly why the host UAF exists); fault
variant faults cause 24 (the statement handle spills across `close_v2`). Row 7 is
the literal representative for all HIERARCHICAL rows (4/5/7/8/9/10/12).

### S — row 2 (UDF context UAF), `benchmarks/sqlite/sqlite_row2_domain.c`

A SQL function is registered with a context pointer `app` (an rof allocation); the
host frees `app` (REVOKE) while the function stays registered; `sqlite3_exec`
drives the callback, which reads the revoked context via `sqlite3_user_data` and
faults. `run-sqlite-row2.sh`, -O0/-O2: control reads 42 and exec returns
(SQLite invokes the real callback on the allocator); fault variant faults cause 24
(SQLite keeps `pApp` in its function table and reloads it on each call, so the
revoked context is always reached through a memory reload — cause 24 at every opt
level; the control disambiguates).

**Mechanism question — the two halves of "sealed callback":**
- The CONTEXT lifetime (the `pApp` UAF) is demonstrated faithfully: real SQLite
  invokes the real registered callback and it faults on the revoked context. This
  is the actual memory-safety defect in rows 2/6/16.
- The SEAL proper — a sealed capability guarding a DOMAIN-CROSSING call boundary —
  is **NOT** exercised single-domain: SQLite invokes the callback through an
  ordinary C function pointer within this one domain, so no domain crossing and no
  unseal happens. A fully-faithful sealed-callback would need SQLite's callback
  dispatch to cross into a separate callback domain (`__seal`/`__domcall` on that
  boundary), which a single domain cannot express. This is the residual to report
  to the PI: the callback-context UAF is faithful today; the domain-crossing seal
  needs a second domain and a monitor call boundary.

### R — row 19 (result lifecycle): NOT built, documented why

row 19's borrowed `sqlite3_column_text` pointer is reused IN PLACE by SQLite on
the next `sqlite3_step` (its own NOTE.md: "SQLite reuses or retains addressable
storage without an allocator-level invalid access"). There is no free event for
the revoke-on-free allocator to hook, so a faithful row19 would need a driver-fired
revoke bolted onto `sqlite3_step` — which violates the fidelity bar. Row 3 already
covers BORROW-REVOKE faithfully. Skipped per the task's "optional, skip if not
cheap."

## Scope / discipline

- Additive only: new files under `benchmarks/sqlite/` and `tests/runtime-qemu/`.
  A's `sqlite_row3{,_b2}_domain.c` untouched. No `start.S`, monitor, `capstone-c`,
  buildroot or LLVM change. **No `capstone-qemu` gitlink bump** — every mechanism
  reused ops that already exist (SPLIT, MREV, REVOKE, csdrop, the UNINIT-load
  check and DROP diagnostic from task-009). The generic B2 host
  (`sqlite_host_row3_b2.c`, 3 regions) is reused by rows 11/14/7/2.
- Cause honesty: only row 14 (U) is self-proving on real SQLite (cause 26,
  opt-independent). Rows 11/7/2 are cause 24 because the faulting handle crosses a
  SQLite call (11, 7) or lives in SQLite's function table (2) and is reloaded
  untagged; each carries a no-fault control that runs the identical program with
  the revoke suppressed. The self-proving cause-25 form of the hierarchical cascade
  is reached in the Phase-0 primitive (no intervening call).

## For A / the PI

- **Sealed-callback (S) residual:** the callback-context UAF is faithful in one
  domain; the SEAL PROPER (domain-crossing entry) needs SQLite's callback dispatch
  to cross into a separate callback domain (`__seal`/`__domcall`) — a real
  domain-crossing primitive, not expressible single-domain. Scoped, not built.
- **Hierarchical (H) allocator** (`revoke_on_free_hier_alloc.h`) is a row-scale
  demonstration on top of the non-coalescing B2 allocator; it inherits B2's
  Phase-2 ceilings (arena fragmentation, the 10000-node rev-tree pool). A general
  hierarchical heap wants the same Route 2 (rev-tree merge op) B2 flagged.
