# PROVENANCE — row12 (expo-sqlite close-while-busy null handle)

- Real bug report : https://github.com/expo/expo/issues/34990 (the crash)
- Cited fix       : https://github.com/expo/expo/pull/34992 (fixes #34990; a
  statement-*leak* fix — see below)
- Tier            : **△ modeled** (was MODEL / CITATION-MISMATCH; the false
  "link verified" exposure is now removed — see resolution)
- Class           : null-deref (oracle `null-deref`) — unchanged.

## What the sweep flagged

The consolidated provenance sweep flagged row 12 because the row cited only the
**fix PR #34992**, which is a **statement-leak fix**: it replaces expo-sqlite's
manual `cachedStatements` map with SQLite's native `sqlite3_next_stmt()` +
`sqlite3_finalize()` to finalize all open statements before closing —
resource-management, *not* a null-dereference safety fix. Citing a leak-fix PR for
a null-deref row is the mismatch behind the paper's inaccurate "All links verified."

## Resolution (task-013): re-anchor to the real bug report, keep △ modeled

**Citation fixed.** #34992's fix closes **issue #34990**, which *is* the actual
crash report for this row's shape: a `java.lang.NullPointerException` from
`NativeDatabase.closeSync()` after tab navigation left prepared statements live, so
that "the database instance is either uninitialised, already closed, or otherwise
invalid at the time of closure." That is exactly row 12's essence — a close path
that runs into a live-statement (`SQLITE_BUSY`) condition and then dereferences a
handle it treats as gone. So the row now points at the artifact that documents the
crash, not the leak-fix PR. The false-"verified" exposure is removed.

**Still △ modeled, not ✓ literal — and honestly so.** #34990's observed fault is a
**managed-language `NullPointerException`** on the JS/Java `NativeDatabase` wrapper,
not a C-level dereference of SQLite's own `sqlite3 *` handle. By the ledger's own
bar ("✓ literal-traceable = the cited artifact is a real *SQLite-pointer*
memory-safety defect"), a Java NPE is *adjacent*, not a lowering — the same
honest call already made for **row 7** (real crash is managed-dict GC, not a
stmt-lifecycle UAF) and **row 13** (real null-deref is a host `row_factory`
deletion with no SQLite C call). Row 12's C repro (`before.c` /
`before-faithful.c`, `sqlite3_get_autocommit(NULL)` on the cleared handle) is a
*behavioral translation* of #34990 into the SQLite-pointer domain — a faithful
MODEL of the shape, not a literal lowering of a C SQLite-pointer defect.

## Negative search — no C-level SQLite-pointer null-deref of this shape exists

Searched for a real "ignore `SQLITE_BUSY` on close → NULL/stale `sqlite3 *` handle
→ later C-level deref" bug across bindings (task-013, 2026-07-13):

| Binding | What exists upstream | Why it is not this row's C-level defect |
|---|---|---|
| expo-sqlite | #34990 close-while-busy NPE; #34992 leak fix | crash is a **Java** NPE, not a C `sqlite3 *` deref (this row's re-anchor) |
| cordova-sqlite-storage | #702 "database might be busy when close is called" | feature request about orphaned connections; **no crash / no deref** documented |
| better-sqlite3 | #155 "how does it handle SQLITE_BUSY" | Q&A; better-sqlite3 throws on live statements — no null-handle deref |
| node-sqlite3 | #16 / #9 / #372 / #1764 "unable to close due to unfinalised statements" | the `SQLITE_BUSY` string is *surfaced* to JS; no handle is nulled-then-derefed in C |
| go-sqlite3 | #192 / golang #18429 nil-pointer panics | **Go-runtime** nil derefs from double-close / connection-state, not a C `sqlite3 *` deref on ignore-BUSY |
| PHP sqlite3 | #47145 "unable to close … unfinalised statements" | the BUSY condition; PHP surfaces it, does not null-then-deref |
| SQLite CVE DB | CVE-2018-8740 / -2019-9937 / -2019-19880 / -2020-9327 / -2020-13435 / -2020-35525 | **engine-internal** null-derefs (corrupt schema, fts5, ORDER BY, generated cols) — not a binding close-while-busy handle deref |

**Conclusion.** The close-while-busy hazard is real and widely reported, but every
real artifact either (a) surfaces `SQLITE_BUSY` as a *handled return code* (no
deref), (b) crashes in the *managed/runtime* layer (expo Java NPE, go nil panic),
or (c) is an unrelated engine-internal null-deref. No upstream artifact documents a
binding that ignores `SQLITE_BUSY`, nulls its C `sqlite3 *`, and later dereferences
it in C. Row 12 therefore stays an honest **△ modeled** row — the C null-deref is a
faithful behavioral translation of the real #34990 crash, not a lowering of a C
SQLite-pointer defect. Do not force it to ✓ literal.

## Capstone framing (unchanged; see NOTE.md)

Row 12 is the HIERARCHICAL null-deref flavor. The fidelity point is the
**return-code path**: a *failed* `sqlite3_close` (returns `SQLITE_BUSY`) does **not**
revoke the live connection — so this is *not* a successful-teardown cascade. The
wrapper nulls its handle anyway, and the fault is a genuine null/stale-handle deref.
Under Capstone the statement is a child capability of the connection: closing with a
live child either fails loudly or the child's authority is revoked, so neither a
stranded statement nor a resurrected NULL handle is silently usable. No new
primitive is needed; no RTL matched pair is built for a MODEL row (RTL status:
model only).
