# Task 012 — promote the HIERARCHICAL family to literal across languages

**Date:** 2026-07-10
**Branch:** `capstone-bootstrap-b`
**Outcome:** DONE. Two more HIERARCHICAL-REVOKE rows promoted to literal
real-SQLite matched pairs, in two more language bindings. The family now has
THREE literal repr(CPython row 7 from task-010, Ruby row 9, PHP row 5), all on
the same intra-domain sub-arena cascade (`revoke_on_free_hier_alloc.h`).

## What this closes

Before: the HIERARCHICAL-REVOKE family (rows 4, 5, 7, 8, 9, 10, 12) had exactly
one literal repro — row 7 (CPython cursor dealloc, task-010). A reviewer could say
"you showed one binding." This task adds two more, in DIFFERENT languages, so the
corpus shows the same mechanism catching distinct real CVEs across CPython + Ruby +
PHP. This is **generalization breadth, not a new mechanism** — the deliberate scope
was ~3 languages; 4→7 would be padding.

| Row | Binding | Real defect | Result |
|---|---|---|---|
| 7 (task-010) | CPython | cursor dealloc → stmt UAF | literal, 4/4 |
| **9** | **Ruby** (sqlite3-ruby #49) | finalize-after-dbfree | **literal, 5/5** |
| **5** | **PHP** (bug #69971) | destruction-order UAF | **literal, 5/5** |

## The shape, and how each row instantiates it

All three are the connection⊃statement ownership tree. Each connection gets its own
MREV'd SUB-ARENA off the one linear grant; SQLite's own allocations for that
connection (the `sqlite3` object and the Vdbe statement) are SPLIT descendants of
the connection's senior revocation node. The parent teardown REVOKEs the senior
node, sweeping the connection AND its live statement; a later `sqlite3_finalize` /
`sqlite3_step` on the child statement dereferences SQLite's own revoked handle and
faults. A sibling connection is an independent SPLIT and survives. Only SPLIT +
MREV + REVOKE — no monitor/region/`start.S` change (same finding as row 7).

- **Row 9 (Ruby).** Faithful sqlite3-ruby wrappers: `sqlite3Ruby { sqlite3 *db }`
  owns `sqlite3StmtRuby { sqlite3_stmt *st }`, `REQUIRE_OPEN_STMT` only checks
  `st != NULL`. The Database teardown (`clear_cache!`/close) revokes the sub-arena;
  the child wrapper still holds a non-NULL `st`; its second `sqlite3_finalize(st)`
  faults. The wrapper is a domain stack local (survives), so the fault lands on the
  `sqlite3_stmt` — SQLite's own handle.
- **Row 5 (PHP).** Faithful ext/sqlite3 wrappers: `php_sqlite3_db_object` with its
  `zend_llist free_list` owns `php_sqlite3_stmt_object`; #69971 wrong order (db
  object freed before the stmt object). The Zend wrapper objects model a SEPARATE
  heap and live on the surviving global arena; only SQLite's connection + statement
  are in the revoked sub-arena. The db-object free handler revokes; the statement
  free handler's `sqlite3_finalize(stmt->stmt)` faults inside finalize on the
  revoked `sqlite3_stmt` — one step before the real crash's `db_obj->free_list` read.

### Fidelity note (important, and a mid-course correction)

The task's bar is "the faulting handle is the child statement SQLite's own API
returned." My first row-5 cut allocated the PHP wrapper objects INSIDE the revoked
sub-arena, so at `-O2` the fault landed on the revoked `stmt` WRAPPER field
(`stmt->initialised`) in the domain code — cause 25, but the WRONG object (a
wrapper read, not SQLite's handle). Forensics: fault `lw a0, 0x20(s2)` at
domain_main+0x95c (capabilities are 16 bytes here, so `initialised` sits at
offset 32); the fault preceded the `sqlite3_finalize` call. That is faithful to
#69971's actual crash (a wrapper-field read) but NOT to the fidelity bar.

Fixed by moving the Zend wrapper objects to the surviving global arena (they model
Zend's heap, which is genuinely separate from SQLite's), leaving only the SQLite
connection + statement in the revoked sub-arena. Now row 5 faults INSIDE
`sqlite3_finalize` on the `sqlite3_stmt`, at the identical PC as row 9
(`pc=0x10200edd8`, vaddr `0x1edd8`, inside `sqlite3_finalize` `[0x1ed7c,0x1ef90)`),
`badaddr` in the connection sub-arena — SQLite's own handle. Consistent with row 9
(whose wrappers were already stack locals) and with the bar.

## Cause honesty

Both rows fault **cause 24 at -O0 and -O2**. The faulting deref is INSIDE
`sqlite3_finalize`, and the SQLite engine is always built `-O0`, so the statement
handle is spilled there and the reload comes back untagged → cause 24 regardless of
the domain TU's opt level. This matches row 7's real-SQLite behaviour (cause 24
because the statement crosses the SQLite call). Each row carries a no-revoke control
(zombie close leaves the statement usable → finalize/step returns) that runs the
identical program and disambiguates cause 24. The self-proving cause-25 form of the
hierarchical cascade is reached in the Phase-0 primitive
(`tests/runtime-qemu/hier-revoke-probe`, task-010) where there is no intervening
call, at -O2.

## Scoping proven on real SQLite (new vs task-010)

Task-010 proved sibling survival only at the primitive level (the Phase-0 probe).
This task adds a `-DROW{5,9}_SIBLING` variant that proves it on REAL SQLite: open
two connections/Databases A and B, revoke A's subtree, then `sqlite3_step` B's child
statement — B returns SQLITE_ROW (100) and the domain returns cleanly. The revoke is
scoped to A's subtree, not a global heap wipe. Validated at -O2 for both rows.

## Validation matrix

`OPT_LEVELS="-O0 -O2" SIBLING_OPT_LEVELS="-O2"` (per-row, one guest boot each):

| Row | -O0 control | -O0 fault | -O2 control | -O2 fault | -O2 sibling |
|---|---|---|---|---|---|
| 9 | PASS | PASS cause 24 | PASS | PASS cause 24 | PASS (B survives) |
| 5 | PASS | PASS cause 24 | PASS | PASS cause 24 | PASS (B survives) |

Fault evidence (both rows): `Cap mem access requires capability … cause = 24,
pc = 0x10200edd8` (inside `sqlite3_finalize`), `badaddr` in the connection
sub-arena. `-O2` was run rather than `-O1` to stay economical under a heavily
loaded machine (load ~10) while still reaching the `-O1/-O2` bar; `OPT_LEVELS`
overrides re-run `-O1` at will.

## Discipline / regressions

- **Additive only.** New files: `benchmarks/sqlite/sqlite_row{5,9}_domain.c` +
  `run-sqlite-row{5,9}.sh`. Reuses the generic B2 host `sqlite_host_row3_b2.c`.
  **No `llvm/` change, no `capstone/capstone-qemu` change (no gitlink bump).**
  `revoke_on_free_hier_alloc.h`, `sqlite_row7_domain.c`, A's `sqlite_row3*` /
  `sqlite_row2*`, `start.S`, the monitor, `capstone-c` all untouched.
- Docs: `stage2-mapping.md` H literal column now lists rows 7/9/5 across three
  bindings; each row's `README.md` "after" section added; this note.
- **A's provenance files** (`cve-repros/row*/before*.c`, `oracle*`,
  `run-host-asan-repros.sh`) left alone — only the READMEs' "after" sections touched.
- Regressions: Capstone lit (`llvm/test/CodeGen/Capstone`) **39/39**. The shared
  hierarchical allocator header is unmodified, so row 7 and the Phase-0
  `hier-revoke-probe` (6/6, task-010) run on identical unchanged code and are
  unaffected.

## Nothing needed from A / the lead

No emulator or monitor primitive was required — pure intra-domain SPLIT + MREV +
REVOKE, as task-010 predicted for the whole family. The rows inherit the same
Phase-2 non-coalescing ceilings the B2 allocator already flagged (arena
fragmentation, fixed rev-tree node pool); a general hierarchical heap still wants
Route 2 (a rev-tree merge op), unchanged from the task-008 note.
