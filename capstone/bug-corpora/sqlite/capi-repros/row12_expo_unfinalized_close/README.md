# row12_expo_unfinalized_close

Upstream (crash): https://github.com/expo/expo/issues/34990 — Java `NullPointerException` on `closeSync()` after live statements leave the db invalid on close-while-busy.
Upstream (fix): https://github.com/expo/expo/pull/34992 — finalizes all statements before close (a stmt-leak fix; closes #34990).
Table 3 row: 12
Class: null-deref
Provenance: **△ modeled** (task-013). #34990's fault is a managed-language NPE, not a C SQLite-pointer null-deref, so this row is a faithful behavioral translation of that crash, not a literal lowering. See `PROVENANCE.md` for the negative search across bindings.
Essence: A host ignores SQLITE_BUSY from closing a connection with a live statement, clears its database handle, and later dereferences that NULL handle.
Observed: ERROR: AddressSanitizer: SEGV on unknown address 0x000000000000

## Stage-2 "after" (Capstone) — modeled (HIERARCHICAL null-deref flavor)

No RTL matched pair is built (RTL status: model only). The fidelity point is the
**return-code path**: a *failed* `sqlite3_close` (`SQLITE_BUSY`) does **not** revoke
the live connection, so this is not a successful-teardown cascade — the wrapper
nulls its handle anyway and a later use is a genuine null/stale-handle deref. Under
Capstone the statement is a child capability of the connection, so closing with a
live child either fails loudly or the child's authority is revoked; neither a
stranded statement nor a resurrected NULL handle is silently usable. No new
primitive is required (same finding as the rest of the HIERARCHICAL family).
