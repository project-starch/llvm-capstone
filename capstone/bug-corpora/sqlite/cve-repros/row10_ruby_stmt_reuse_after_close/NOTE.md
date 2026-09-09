# row10 — fidelity note (essence vs binding-faithful)

| File | Level | What it models |
|---|---|---|
| `before.c` | **Essence** | flat `struct ruby_statement{database;statement}`; `sqlite3_close_v2(db)`, `free(wrapper)`, then read `wrapper->statement` from the freed wrapper and `sqlite3_reset` it — a UAF on the freed *wrapper* block. |
| `before-faithful.c` | **Binding-faithful** | sqlite3-ruby `sqlite3Ruby{db; stmts[]}` + `sqlite3StmtRuby{st; done_p}`. Database teardown finalizes each tracked `st` (freeing the `sqlite3_stmt`) but leaves the child wrapper's `st` non-NULL; `Statement#reset!` — whose `REQUIRE_OPEN_STMT` guard checks only `st != NULL` — then calls `sqlite3_reset(ctx->st)` on the freed statement. |

## Why the faithful version matters here

This is the direct analog of **row4** (PHP `reset()` without an `initialised` guard):
a liveness check that tests only for **NULL**, not for **revocation**, so a
freed-but-retained handle slips straight through. The faithful file lands the UAF
**inside `sqlite3_reset`** (the same crash site as row4), reproducing the real
"reuse a closed statement" mechanism rather than a synthetic read of a freed malloc
block. HIERARCHICAL-REVOKE shape: the Database owns the Statement, so revoking the
Database's authority on close should invalidate the child's `st` capability and make
`reset!` fault locally — instead of the pointer-value NULL check that the binding
actually shipped.

## Faithful to / approximated

- **Faithful:** the wrapper structs + field names, the parent→child ownership,
  `REQUIRE_OPEN_STMT` checking only `st != NULL`, `reset_bang → sqlite3_reset(ctx->st)`,
  and the retained dangling `st` after db teardown.
- **Approximated:** Ruby GC + `@connection` ivar → explicit child list. `xfree` →
  `free`. Self-contained: links real SQLite; no Ruby VM.

## Verified (host)

```
gcc -std=c11 -g -O0 -fsanitize=address -I<sqlite-src> \
    before-faithful.c <sqlite-src>/sqlite3.c -o row10-faithful -lpthread -ldl -lm \
    -DSQLITE_OMIT_LOAD_EXTENSION
ASAN_OPTIONS=detect_leaks=0 ./row10-faithful
# => heap-use-after-free in sqlite3_reset (sqlite3.c:93822),
#    freed by the db-teardown sqlite3_finalize -> sqlite3_free  (SQLite 3.53.3)
```

Oracle unchanged: `heap-use-after-free`.

## Sources
- Thread: https://groups.google.com/g/sqlite3-ruby/c/SGRQE_2MZ8I
- Structs/glue: `sparklemotion/sqlite3-ruby` `ext/sqlite3/statement.c`
  (`sqlite3StmtRuby`, `REQUIRE_OPEN_STMT`, `reset_bang`), `ext/sqlite3/database.c`.
