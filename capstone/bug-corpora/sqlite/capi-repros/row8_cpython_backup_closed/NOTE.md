# row8 — fidelity note (essence vs binding-faithful)

| File | Level | What it models |
|---|---|---|
| `before.c` | **Essence** | flat `struct connection{db}`; `sqlite3_close_v2(src->db)`, `free(src)`, then read `src->db` from the freed wrapper and pass it to `sqlite3_backup_init` — a heap-use-after-free. |
| `before-faithful.c` | **Binding-faithful** | CPython `_sqlite`: `pysqlite_Connection{initialized; db}`; `close()` does `sqlite3_close_v2(self->db); self->db = NULL;`; `backup(target)` (pre-fix, no `pysqlite_check_connection`) passes `self->db` (now NULL) as the source to `sqlite3_backup_init(target->db, "main", NULL, "main")`. |

## Crash-class divergence (deliberate, documented)

The real defect is a **NULL / closed-handle dereference**, *not* a heap-use-after-free.
pysqlite **nulls `self->db` on close** (defensive), so `backup()` reads a NULL source
handle and real SQLite dereferences `pSrcDb->mutex` with no NULL check (API armor off)
→ SEGV inside `sqlite3_backup_init`. The essence `before.c` fabricates a UAF instead
(it `free()`s the wrapper and re-reads the field); that is a convenient ASan oracle but
**not** the real mechanism. The faithful file is honest about the real crash class.

This is a good paper point: the C binding's *manual* mitigation is nulling the handle on
close, and the bug is that `backup()` forgot to *check* it. Under Capstone the connection's
`db` capability is **revoked** on close automatically, so any later read of it faults
deterministically — no reliance on a hand-written per-entry-point `check_connection` guard.

## Verified (host)

```
gcc -std=c11 -g -O0 -fsanitize=address -I<sqlite-src> \
    before-faithful.c <sqlite-src>/sqlite3.c -o row8-faithful -lpthread -ldl -lm \
    -DSQLITE_OMIT_LOAD_EXTENSION
ASAN_OPTIONS=detect_leaks=0 ./row8-faithful
# => AddressSanitizer: SEGV on unknown address 0x18 (READ)
#    #0 sqlite3_backup_init (sqlite3.c)
#    #1 pysqlite_connection_backup  (self->db == NULL)   (SQLite 3.53.3)
```

Oracle for the faithful file: **null-deref (SEGV)** — differs from the essence
`before.c` oracle (`heap-use-after-free`). Both are "use of a closed connection"; the
faithful file reflects the true fault.

## Sources
- Issue: https://github.com/python/cpython/issues/85981 (bpo-41815)
- Fix: https://github.com/python/cpython/pull/22322 (adds `pysqlite_check_connection`
  to `pysqlite_connection_backup`)
- Struct/glue: `python/cpython` `Modules/_sqlite/connection.c` (`pysqlite_Connection`,
  `pysqlite_connection_close` sets `self->db = NULL`, `pysqlite_connection_backup`).
