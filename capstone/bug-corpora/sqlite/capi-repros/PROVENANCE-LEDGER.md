# SQLite corpus — consolidated provenance ledger

**Purpose.** One paper-ready table collapsing the 19 per-row `PROVENANCE.md`
files into a single view: upstream artifact, defect class, provenance verdict,
observed host fault (ASan), Capstone primitive family, and RTL status. This is
the source of truth behind Table `tab:scope` / `tab:fix`.

**Why this exists.** The table caption "All links verified" is *not* accurate as
written: three rows cite an artifact that describes a **different or adjacent**
defect than the row's essence (7, 12, 16), and one row is a real bug that is **not
convertible** to an SQLite-pointer capability (13). This ledger states the honest
per-row verdict so the caption can be reworded to match (see "Caption" below).
Row 12's citation-mismatch was resolved in task-013 by re-anchoring it from the
leak-fix PR to the actual crash report (still △ modeled — see the row line and
`row12_.../PROVENANCE.md`).

## Provenance verdicts

- **✓ literal-traceable** — the cited artifact is a real SQLite-pointer
  memory-safety defect and the row's repro is a faithful lowering of it.
- **△ modeled** — the cited artifact resolves but describes a *different or
  adjacent* defect; the row's Capstone essence is a constructed model, not a
  lowering of that artifact.
- **○ out-of-scope** — a real upstream bug, but with no SQLite pointer lifetime
  to revoke (stale-state / host-language glue).

## Ledger

| Row | Binding / lang | Upstream artifact | Class | Verdict | Host fault (ASan) | Primitive | RTL status |
|---|---|---|---|---|---|---|---|
| 1 | CPython | [cpython#142830](https://github.com/python/cpython/issues/142830) | UAF | ✓ literal | heap-use-after-free (`print_or_clear_traceback`) | S (L,R,H,S) | family (S via row 2) |
| 2 | rusqlite | [RUSTSEC-2021-0128](https://rustsec.org/advisories/RUSTSEC-2021-0128.html) | UAF | ✓ literal | use-after-free (non-`'static` hook closure) | S | **literal RTL rep** `run-sqlite-row2.sh` |
| 3 | diesel | [RUSTSEC-2021-0037](https://rustsec.org/advisories/RUSTSEC-2021-0037.html) | UAF | ✓ literal | use-after-free (cached colname ptr) | R | **literal RTL rep** `run-sqlite-row3-b2.sh` |
| 4 | PHP | [php#66550](https://bugs.php.net/bug.php?id=66550) | use-after-close | ✓ literal | use-after-free (stmt after `$db->close()`) | H,R | family (H) |
| 5 | PHP | [php#69971](https://bugs.php.net/bug.php?id=69971) (dup #53626) | UAF | ✓ literal | heap-use-after-free (shutdown destruction order) | H,R | **literal RTL rep** `run-sqlite-row5.sh` |
| 6 | PHP | [php#77977](https://bugs.php.net/bug.php?id=77977) | UAF | ✓ literal | heap-use-after-free (UDF `pApp`) | S | family (S) |
| 7 | CPython | [cpython#99886](https://github.com/python/cpython/issues/99886) | UAF | **△ modeled** | real crash in `_PyObject_ClearInstanceAttributes` — **managed-dict GC, NOT a stmt-lifecycle UAF** | H,R | `run-sqlite-row7.sh` runs, but provenance is a model |
| 8 | CPython | [cpython#85981](https://github.com/python/cpython/issues/85981) (bpo-41815) | use-after-close | ✓ literal | null-deref (faithful crash class; essence fabricates UAF) | H | family (H) |
| 9 | Ruby | [sqlite3-ruby#49](https://github.com/sparklemotion/sqlite3-ruby/issues/49) | UAF | ✓ literal | use-after-free (finalize after db-free) | H,R | **literal RTL rep** `run-sqlite-row9.sh` |
| 10 | Ruby | [sqlite3-ruby list](https://groups.google.com/g/sqlite3-ruby/c/SGRQE_2MZ8I) | use-after-close | ✓ literal | use-after-free (stmt reuse after close) | H,R | family (H) |
| 11 | Go | [go-sqlite3#1303](https://github.com/mattn/go-sqlite3/pull/1303) | double-free | ✓ literal | double-free (double `sqlite3_finalize`) | L | **literal RTL rep** `run-sqlite-row11.sh` |
| 12 | expo-sqlite | [expo#34990](https://github.com/expo/expo/issues/34990) (fixed by [#34992](https://github.com/expo/expo/pull/34992)) | null-deref | **△ modeled** | re-anchored (task-013): the real crash is a **Java `NullPointerException`** on close-while-busy — adjacent, not a C SQLite-pointer null-deref (see negative search) | H,R | model only |
| 13 | CPython | [cpython#149738](https://github.com/python/cpython/issues/149738) | null-deref | **△ modeled** | real null-deref, but a **host-language event** (`row_factory` deletion, no SQLite C call) — not literal-convertible | R | not RTL-convertible |
| 14 | CPython | [cpython#27472](https://github.com/python/cpython/pull/27472) | null-deref | ✓ literal | uninitialised connection use before `sqlite3_open` | U | **literal RTL rep** `run-sqlite-row14.sh` |
| 15 | CPython | [cpython#101767](https://github.com/python/cpython/issues/101767) | stale-state | ○ out-of-scope | subinterpreter/GIL — no SQLite pointer lifetime | — | out of scope |
| 16 | Datasette | [datasette-sqlite-authorizer#3](https://github.com/datasette/datasette-sqlite-authorizer/issues/3) | UAF | **△ modeled** | cited issue is a read-only-protection test failure — **no use-after-free PoC** | S | model only |
| 17 | node-sqlite3 | [node-sqlite3#1449](https://github.com/TryGhost/node-sqlite3/issues/1449) | stale-state | ○ out-of-scope | crash in Node/napi glue, memory-safe w.r.t. SQLite (`SQLITE_RANGE`) | — | out of scope |
| 18 | PHP | [php#79294](https://bugs.php.net/bug.php?id=79294) | stale-state | ○ out-of-scope | logic/stale-state, ASan-clean (oracle `none`) | R (weak) | N/A-for-literal |
| 19 | PHP | [php-src#5204](https://github.com/php/php-src/pull/5204) (#64531, #79293) | stale-state | ○ out-of-scope | logic/stale-state, ASan-clean (oracle `none`) | R | N/A-for-literal |

## Tally

| Verdict | Rows | Count |
|---|---|---|
| ✓ literal-traceable | 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 14 | **11** |
| △ modeled | 7, 12, 13, 16 | **4** |
| ○ out-of-scope | 15, 17, 18, 19 | **4** |

**Literal RTL representatives (one faithful matched pair per in-scope shape):**
row 2 (S), row 3 (R), row 5 + 9 + 7 (H — three bindings), row 11 (L), row 14 (U).

## Caption

Replace the `tab:scope` caption's "All links verified" with the accurate:

> Each row cites its upstream artifact. 11 rows are literal-traceable
> memory-safety defects with faithful reproducers; 4 are modeled on an adjacent
> or non-convertible artifact (rows 7, 12, 13, 16); 4 are out of scope for the
> pointer-capability model (rows 15, 17, 18, 19). See the per-row provenance
> ledger for verdicts.

## Resolved item (task-013, 2026-07-13)

Row 12's hard citation-mismatch is closed. A thorough upstream search (expo,
cordova-sqlite-storage, better-sqlite3, node-sqlite3, PHP sqlite3, go-sqlite3, and
the SQLite CVE database) found **no** binding that ignores `SQLITE_BUSY`, nulls its
C `sqlite3 *` handle, and later dereferences it in C. The citation was re-anchored
from the leak-fix PR #34992 to the real crash report **#34990** (a Java
`NullPointerException` on close-while-busy — the artifact #34992 fixes), which
documents this row's shape. Because #34990's fault is a **managed-language** NPE,
not a C SQLite-pointer null-deref, row 12 stays honestly **△ modeled** (same
category as rows 7 and 13), and its C repro is a behavioral translation of that
crash. Full negative search: `row12_expo_unfinalized_close/PROVENANCE.md`. No row
graduates to ✓ literal; the tally is unchanged. The "All links verified" exposure
is removed for row 12 — its citation now points at the artifact that documents the
crash.
