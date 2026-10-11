# fts3destroyoom -- 129371553c

Upstream fix `129371553c`. Collected in round R1 (NVD and upstream history).

The reduction is argued in the case source's own header, reproduced here so the
claim travels with the case rather than living only in the build tree.

```
new-4 / sqlite-129371553c -- fts3DestroyMethod() dereferences a freed Fts3Table
 * after a nested OOM.
 *
 * fts3DestroyMethod() drops the five shadow tables through fts3DbExec():
 *
 *     fts3DbExec(&rc, db, "DROP TABLE IF EXISTS %Q.'%q_content'", zDb, p->zName);
 *     fts3DbExec(&rc, db, "DROP TABLE IF EXISTS %Q.'%q_segments'", zDb, p->zName);
 *     fts3DbExec(&rc, db, "DROP TABLE IF EXISTS %Q.'%q_segdir'",   zDb, p->zName);
 *     ...
 *
 * If one of those statements hits OOM, sqlite3VdbeHalt() -> sqlite3RollbackAll()
 * -> sqlite3ResetAllSchemasOfConnection() -> sqlite3VtabUnlockList() ->
 * fts3DisconnectMethod() sqlite3_free()s the Fts3Table -- i.e. a schema reset runs
 * underneath the object's own destructor. Control then returns to
 * fts3DestroyMethod, which goes on to the NEXT fts3DbExec. Its body returns early
 * because rc!=SQLITE_OK, but `p->zName` is still evaluated at the call site, and p
 * is freed. Fixed 3.27.0 by collapsing the five calls into one.
 *
 * Host ASan oracle on 3.22.0 (lookaside off, default page size, upstream's minimal
 * CREATE-then-DROP shape, counted OOM injection):
 *   READ of size 8, 40 bytes into a 592-byte region   (Fts3Table.zName)
 *     use   fts3DestroyMethod:149364 / :149365
 *     free  fts3DisconnectMethod <- sqlite3VtabUnlock <- sqlite3VtabUnlockList
 *             <- sqlite3ResetAllSchemasOfConnection <- sqlite3RollbackAll
 *             <- sqlite3VdbeHalt
 *     alloc fts3InitVtab
 *   40 of 300 injection points hit, in two contiguous runs: 167-184 and 259-280.
 *   The two runs are the two distinct fts3DbExec call sites that can be the use.
 *
 * WHY THIS DOMAIN CANNOT OBSERVE THE BUG ITSELF, and what it establishes instead.
 * The dangling read is Fts3Table.zName at offset 40. memsys5 overwrites only the
 * first 8 bytes of a freed block (its in-band Mem5Link freelist ints), so offset 40
 * keeps both its bytes and its capability tag: the read is necessarily SILENT on
 * base Capstone. It is also unobservable from inside the domain -- the value is
 * discarded (fts3DbExec returns before using it), and the post-DROP state is
 * identical whether or not the injection landed in the window: drop_rc=7 and the
 * table still present for EVERY injection point from 1 to 292 on the host. An
 * allocator-level witness does not separate them either, because distinguishing
 * in-window from out-of-window means knowing whether fts3DestroyMethod was still on
 * the stack when the free happened, which the allocator cannot see.
 *
 * So this domain does not claim a per-point correspondence. It establishes that its
 * allocation sequence is the SAME sequence the host oracle swept, by reporting two
 * numbers that are fixed by SQLite's code path:
 *     - the DROP's total allocation count   (host: 292)
 *     - the transition point where drop_rc stops being SQLITE_NOMEM and becomes
 *       SQLITE_OK because the injection point is past the end (host: last 7 at 292,
 *       first 0 at 293)
 * If those match, the host's window indices transfer. The domain then sweeps every
 * injection point and runs to NOTRAP, which is the control-arm result: the
 * use-after-free is silent on unprotected Capstone.
 *
 * SQLITE_UNTESTABLE removes sqlite3_test_control, so OOM comes from a counting
 * wrapper installed over memsys5. SQLITE_CONFIG_HEAP sets sqlite3GlobalConfig.m to
 * memsys5's methods, SQLITE_CONFIG_GETMALLOC copies them out, and
 * SQLITE_CONFIG_MALLOC installs the wrapper -- so the order below matters and this
 * case does its own init instead of calling repro_init().
 * NOTE: -DSQLITE_DQS=0, so SQL string literals must be single-quoted.
```
