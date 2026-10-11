# PROVENANCE — row8 (CPython bpo-41815 / gh-85981 backup-after-close)

- Upstream : https://github.com/python/cpython/issues/85981 (bpo-41815)
- Fix      : GH-22322 (backports GH-22344 3.9, GH-22345 3.8)
- Tier     : **LITERAL-traceable** (real crash; faithful crash class = null-deref)

**Real memory-safety bug: YES.** `Connection.backup()` on a closed source passed a
NULL `sqlite3 *` to `sqlite3_backup_init` → segfault. The verbatim regression test
`test_bad_source_closed_connection` and the exact fix hunk
(`pysqlite_check_thread || pysqlite_check_connection`) are quoted in the repro
header: `before-faithful.c`. NOTE: the faithful crash class is **null-deref**
(essence `before.c` fabricates a UAF); scored via `oracle-faithful = null-deref`.
