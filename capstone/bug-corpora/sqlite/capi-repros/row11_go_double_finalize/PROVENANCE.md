# PROVENANCE — row11 (mattn/go-sqlite3 double-finalize)

- Upstream / fix : https://github.com/mattn/go-sqlite3/pull/1303
- Tier           : **LITERAL-traceable**

**Real memory-safety bug: YES.** `SQLiteStmt.Close()` finalized the statement and
only then removed the Go runtime finalizer — but returned early on a finalize
error *before* clearing it. The still-registered runtime finalizer could then call
`sqlite3_finalize()` a second time on the already-finalized handle (double-finalize).

Verbatim buggy code (`SQLiteStmt.Close`):
```go
rv := C.sqlite3_finalize(s.s)
if rv != C.SQLITE_OK {
    return s.c.lastError()          // early return: finalizer NOT removed
}
s.c = nil
runtime.SetFinalizer(s, nil)
```
Verbatim fix (remove finalizer first; snapshot + clear the handle):
```go
stmt := s.s; s.s = nil
runtime.SetFinalizer(s, nil)
rv := C.sqlite3_finalize(stmt)
if rv != C.SQLITE_OK {
    return conn.lastError(int(rv))
}
```
Verbatim regression test (idempotent double Close):
```go
if err := stmt.Close(); err != nil { t.Fatal(err) }
if err := stmt.Close(); err != nil { t.Fatal(err) }
```

**How our repro maps:** row11 (literal L representative, task-010) consumes the
move-only statement handle at the first `sqlite3_finalize`; a second finalize of
the same handle faults (cause 24 + no-drop control). Faithful to the double-finalize
of one SQLite statement handle.
