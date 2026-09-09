# row9 — fidelity note (essence vs binding-faithful)

| File | Level | What it models |
|---|---|---|
| `before.c` | **Essence** | flat `struct ruby_statement{database;statement}`; `sqlite3_close_v2(db)`, `free(wrapper)`, then read `wrapper->statement` from the freed wrapper and finalize it — a UAF on the freed *wrapper* block. |
| `before-faithful.c` | **Binding-faithful** | sqlite3-ruby `sqlite3Ruby{db; stmts[]}` + `sqlite3StmtRuby{st; done_p}`. Database teardown finalizes each tracked statement's `st` (freeing the `sqlite3_stmt`) but leaves the child wrapper's `st` non-NULL; the child's `Statement#close` — whose `REQUIRE_OPEN_STMT` guard checks only `st != NULL` — then finalizes the freed stmt a **second time**. |

## Why the faithful version matters here

Row9 is a HIERARCHICAL-REVOKE shape and the defect *is* the ownership relationship:
the Database owns its Statements, and on unlucky GC-finalizer order the parent's
teardown frees a statement the child still believes it owns. The faithful file lands
the UAF **inside `sqlite3_finalize`** — exactly the observed backtrace
(`sqlite3_finalize+0x23`) — because it is a genuine double-finalize driven by the
parent/child ownership confusion, not a synthetic read of a freed malloc block. This
is precisely what senior/hierarchical revocation is meant to make safe: revoking the
Database's authority on close invalidates the child's `st` capability, so the child's
finalize faults locally instead of double-freeing.

## Faithful to / approximated

- **Faithful:** the two wrapper structs + field names (`ext/sqlite3/{database,statement}.c`),
  the parent→child ownership (`@statements` cache), `REQUIRE_OPEN_STMT` checking only
  `st != NULL`, `Statement#close` doing `sqlite3_finalize(ctx->st); ctx->st = NULL`,
  and the double `sqlite3_finalize` the real crash lands in.
- **Approximated:** Ruby GC + `@connection` ivar → an explicit child list + retained
  pointer. `xfree` → `free`. Self-contained: links real SQLite; no Ruby VM.

## Verified (host)

```
gcc -std=c11 -g -O0 -fsanitize=address -I<sqlite-src> \
    before-faithful.c <sqlite-src>/sqlite3.c -o row9-faithful -lpthread -ldl -lm \
    -DSQLITE_OMIT_LOAD_EXTENSION
ASAN_OPTIONS=detect_leaks=0 ./row9-faithful
# => heap-use-after-free in sqlite3_finalize (sqlite3.c),
#    freed by the prior sqlite3_finalize -> sqlite3_free  (SQLite 3.53.3)
```

Oracle unchanged: `heap-use-after-free`.

## Sources
- Issue + backtrace: https://github.com/sparklemotion/sqlite3-ruby/issues/49
- Structs/glue: `sparklemotion/sqlite3-ruby` `ext/sqlite3/statement.c`
  (`sqlite3StmtRuby`, `REQUIRE_OPEN_STMT`, `Statement#close`), `ext/sqlite3/database.c`.
