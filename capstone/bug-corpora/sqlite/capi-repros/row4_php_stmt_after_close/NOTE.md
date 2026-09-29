# row4 — fidelity note (essence vs binding-faithful)

| File | Level | What it models |
|---|---|---|
| `before.c` | **Essence** | flat `struct host_stmt{db;stmt}`; close the db, `free(host)`, then `step(host->stmt)` — a UAF on the freed *wrapper*. |
| `before-faithful.c` | **Binding-faithful** | PHP `ext/sqlite3`: `SQLite3::close()` → `php_sqlite3_free_list_dtor()` finalizes each statement (frees the `sqlite3_stmt`) and sets `initialised=0`, leaving the wrapper's dangling `stmt`; `SQLite3Stmt::reset()` (pre-fix, no `SQLITE3_CHECK_INITIALIZED`) then dereferences it. |

The faithful version reproduces the *real* mechanism: the UAF is inside `sqlite3_reset`
on a `sqlite3_stmt` that `close()` already finalized — not a UAF on the host wrapper.
This matters for the paper because the fix PHP shipped is an `initialised` guard, and
the Capstone story is that a revoked statement capability makes the guardless `reset()`
fault deterministically.

- **Faithful:** `free_list` teardown, `php_sqlite3_free_list_dtor` finalize+mark-stale,
  the retained dangling pointer, the missing `initialised` guard on `reset()`.
- **Approximated:** `zend_llist` → minimal list. Self-contained; links real SQLite.

## Verified (host)

```
gcc -std=c11 -g -O0 -fsanitize=address -I<sqlite-src> \
    before-faithful.c <sqlite-src>/sqlite3.c -o row4-faithful -lpthread -ldl -lm
# => heap-use-after-free in sqlite3_reset (sqlite3.c), freed by sqlite3_finalize
#    (SQLite 3.53.3)
```

Oracle unchanged: `heap-use-after-free`.

## Sources

- Bug + repro: https://bugs.php.net/bug.php?id=66550
- Structs: `php/php-src` `ext/sqlite3/php_sqlite3_structs.h`
