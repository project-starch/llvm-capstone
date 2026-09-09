# row7 — fidelity note: stays ESSENCE-level (deliberate)

**Decision: do NOT upgrade row7 to a binding-faithful repro.** Verified 2026-07-09.

The real bug (CPython gh-99886) is **not a binding-glue pointer-lifetime defect**.
It is a CPython *interpreter* dealloc-order bug: a `Cursor` subclass with a managed
instance dict crashes in `subtype_clear` / `_PyObject_ClearInstanceAttributes`,
doing `Py_DECREF` on already-freed memory (`0xdddddddddddddddd`). Introduced by
commit `a8b9350`, fixed by `de388c0` (PR #99902). The `pysqlite_Cursor` struct's
`statement`/`connection` fields are incidental — the fault is in the managed-dict
teardown machinery, not in how the `_sqlite3` glue handles a SQLite pointer.

A "binding-faithful" repro would therefore have to model CPython's managed-dict +
`subtype_clear` + `tp_dealloc` ordering — deep interpreter internals with no SQLite
pointer-lifetime content — which:
1. cannot run in the bare Capstone domain (no CPython runtime), and
2. would not exercise the paper's mechanism (hierarchical revoke of
   connection ⊃ cursor ⊃ statement) at all.

So row7 keeps its **essence** `before.c`, which models the *class* the paper cares
about — a cursor wrapper that owns a statement field, freed then used — as a clean,
domain-buildable stand-in. This is the tiered-fidelity policy working as intended:
upgrade to faithful only where the binding's *glue* carries the lifetime bug; for
rows whose real defect lives in the host runtime, the essence is the honest model
and faithful is neither achievable nor apt.

## Sources
- Issue: https://github.com/python/cpython/issues/99886
- Struct: `python/cpython` `Modules/_sqlite/cursor.h` (`pysqlite_Cursor`)
