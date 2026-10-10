# staticbind -- eab0e10304

Upstream fix `eab0e10304`. Collected in round R1 (NVD and upstream history).

The reduction is argued in the case source's own header, reproduced here so the
claim travels with the case rather than living only in the build tree.

```
new-3 / sqlite-eab0e10304 -- fts3/fts5/rtree leave a freed heap buffer bound
 * SQLITE_STATIC to a PERSISTENT cached statement.
 *
 * fts3's segment writer binds its own buffer into the cached segdir statement and
 * then frees it, with no bind_null in between:
 *     sqlite3_bind_blob(pStmt, 6, zRoot, nRoot, SQLITE_STATIC);
 *     sqlite3_step(pStmt);
 *     rc = sqlite3_reset(pStmt);
 * sqlite3_reset() does NOT clear bindings, so parameter 6 keeps pointing at the
 * buffer after fts3SegWriterFree() releases it. Fixed 3.23.0 -- 16 days after
 * 3.22.0 shipped -- by adding the bind_null. The fix's own commit message states
 * the threat model: "a user may obtain a pointer to the persistent statement using
 * sqlite3_next_stmt() and attempt to access the freed buffer using
 * sqlite3_expanded_sql() or similar".
 *
 * WHY THIS DOMAIN DOES NOT USE sqlite3_expanded_sql().
 * It cannot: this port compiles with -DSQLITE_OMIT_FLOATING_POINT=1, and 3.22.0's
 * sqliteInt.h reacts to that with
 *     #define SQLITE_OMIT_DATETIME_FUNCS 1
 *     #define SQLITE_OMIT_TRACE 1
 * so sqlite3_expanded_sql() is compiled as `return 0;` in EVERY group of this port.
 * A command-line -USQLITE_OMIT_TRACE does not help, because the #define happens
 * inside the translation unit. (This was verified with a dedicated probe: every
 * expanded_sql call returns NULL even for a no-parameter statement in the smallest
 * image with a completely idle arena, and the arena serves all size classes from 64
 * to 16384 bytes. An earlier session blamed memsys5 fragmentation; that was wrong.)
 *
 * So this domain takes the "or similar" route the fix names: re-STEP the cached
 * statement without rebinding. Parameter 6 still points at the freed buffer, so
 * OP_MakeRecord reads it and REPLACEs the %_segdir row with whatever is there now.
 *
 * Host ASan oracle on 3.22.0, same shape:
 *   re-step  : READ of size 1492, 0 bytes into a freed 4064-byte region
 *                use   sqlite3VdbeSerialPut <- sqlite3VdbeExec <- sqlite3_step
 *   expanded : READ of size 1    (upstream's route, for comparison)
 *   free  fts3SegWriterFree <- fts3SegmentMerge <- fts3DoOptimize
 *           <- fts3SpecialInsert <- sqlite3Fts3UpdateMethod
 *   alloc fts3SegWriterAdd <- fts3SegmentMerge
 *
 * This route is also OBSERVABLE without any memory-safety tool, which matters
 * because on base Capstone the read is silent (the buffer is read as bytes, not as
 * a pointer). On the host the re-step rewrites the row with DIFFERENT content at the
 * same length, and the index is corrupt afterwards:
 *     before len=1492 sum=3696465335 first4=0006616c
 *     after  len=1492 sum=951463155  first4=60ed69d4   <- looks like a heap pointer
 *     integrity-check -> SQLITE_CORRUPT
 * In the domain memsys5 writes its in-band Mem5Link freelist ints over the start of
 * the freed block, so a changed hash here is the freed block's new contents being
 * copied into the database.
 * NOTE: -DSQLITE_DQS=0, so SQL string literals must be single-quoted.
```
