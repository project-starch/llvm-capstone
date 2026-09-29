# row12 — fidelity note (essence vs binding-faithful)

| File | Level | What it models |
|---|---|---|
| `before.c` | **Essence** | flat `struct connection{db;stmt}`; prepares a stmt, `sqlite3_close(db)` returns `SQLITE_BUSY`, sets `db = 0`, then raw-derefs `*db` — a NULL deref. |
| `before-faithful.c` | **Binding-faithful (behavioral)** | expo-sqlite `NativeDatabaseBinding{db; closed}`. `closeDatabase()` calls `sqlite3_close(db)` (returns `SQLITE_BUSY` because a prepared statement is still live), **ignores the return code**, clears the handle (`db = NULL`, `closed = 1`), and reports success. A later `isInTransaction()` → `sqlite3_get_autocommit(db)` dereferences `db->autoCommit` on the cleared handle → SEGV. |

## Why "behavioral translation" and not full-faithful

expo-sqlite's native binding is **C++/JNI** (`NativeDatabaseBinding.cpp` on Android,
`SQLiteModule.swift` on iOS) and cannot compile into a bare Capstone domain (no
libstdc++, no JVM). This is the tier the policy calls *behavioral translation for
non-C bindings*: the SQLite-facing C slice is reproduced exactly — `sqlite3_close`
returning `SQLITE_BUSY` on a live statement, the pre-fix "ignore BUSY + clear the
handle anyway" defect, and a later native state-read that dereferences the cleared
handle — while the C++/JNI object and the Java handle are rendered as a C struct.

Confirmed on host: `sqlite3_close` with a live prepared statement returns
`SQLITE_BUSY (5)` and frees nothing — so clearing the handle strands the live
statement and NULLs a handle the module still uses. The fix (PR #34992) adds
`sqlite3_finalize_all_statement()` (iterate `sqlite3_next_stmt` + `sqlite3_finalize`)
so `close()` returns `SQLITE_OK` before the handle is cleared.

The later deref goes through a **real SQLite API**: `sqlite3_get_autocommit(NULL)`
dereferences `db->autoCommit` with no NULL guard (API armor off) — a truer model of
"a later native op touches the cleared handle" than a raw `*db`.

## Capstone framing

Row12 is the null-deref class: the C mitigation is manual handle-nulling, and the
bug is that a later path forgot to re-check it (and that close was allowed to
"succeed" while a child statement was still live). Under Capstone the statement is a
child capability of the connection; closing with a live child either fails loudly or
the child's authority is revoked, so neither a stranded statement nor a resurrected
NULL handle is silently usable.

## Verified (host)

```
gcc -std=c11 -g -O0 -fsanitize=address -I<sqlite-src> \
    before-faithful.c <sqlite-src>/sqlite3.c -o row12-faithful -lpthread -ldl -lm \
    -DSQLITE_OMIT_LOAD_EXTENSION
ASAN_OPTIONS=detect_leaks=0 ./row12-faithful
# => AddressSanitizer: SEGV on unknown address 0x18 (READ)
#    #0 sqlite3_get_autocommit (sqlite3.c)   db == NULL   (SQLite 3.53.3)
```

Oracle unchanged: `null-deref`.

## Sources
- Crash (re-anchored, task-013): https://github.com/expo/expo/issues/34990 —
  Java `NullPointerException` on `closeSync()` when live statements leave the db
  invalid at close-while-busy. This is the artifact that documents the fault; the
  row is △ modeled because it is a managed NPE, not a C SQLite-pointer null-deref
  (see `PROVENANCE.md`).
- Fix: https://github.com/expo/expo/pull/34992 (closes #34990; adds
  `sqlite3_finalize_all_statement` — a stmt-*leak* fix, not a null-deref safety fix)
- Glue: `expo/expo` `packages/expo-sqlite/.../NativeDatabaseBinding.cpp`
  (`sqlite3_finalize_all_statement`, `sqlite3_next_stmt` loop).
