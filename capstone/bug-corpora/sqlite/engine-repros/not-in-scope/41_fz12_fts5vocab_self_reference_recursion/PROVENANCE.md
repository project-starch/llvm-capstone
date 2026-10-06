# fz12 -- fz12

Upstream fix `FZ12`. Collected in round R2 (fuzz-corpus diff).

The reduction is argued in the case source's own header, reproduced here so the
claim travels with the case rather than living only in the build tree.

```
FZ12 — unbounded recursion (stack overflow) on a self-referential fts5vocab table.
 *
 * Collected by replaying SQLite's post-3.22.0 fuzz corpus against 3.22.0. Host ASan on
 * stock 3.22.0 reports a stack-overflow unwinding through the expression deleter:
 *
 *   #0 __interceptor_free
 *   #1 sqlite3MemFree       sqlite3.c:21551
 *   #2 sqlite3_free         sqlite3.c:25461
 *   #3 sqlite3DbFreeNN      sqlite3.c:25504
 *   #4 sqlite3ExprDeleteNN  sqlite3.c:93756
 *   #5 sqlite3ExprDelete    sqlite3.c:93760
 *
 * i.e. the recursion is in name resolution and the stack is already exhausted by the time
 * the parse tree is torn down -- the free path is where it finally tips over, not the cause.
 *
 * Minimised from a 277-byte fuzz case to 86 bytes:
 *
 *   CREATE VIRTUAL TABLE rowid USING fts5vocab( rowid , 'instance');
 *   SELECT * FROM rowid;
 *
 * fts5vocab's first argument names the FTS5 table it reads, so this declares a vocab table
 * that reads ITSELF. Resolving it recurses without a depth limit.
 *
 * The name `rowid` is NOT incidental, and this was verified: the same self-reference under
 * an ordinary name (CREATE VIRTUAL TABLE v USING fts5vocab(v,'col'); SELECT * FROM v)
 * runs CLEAN. `rowid` is special in the resolver, which is what sends it down the
 * recursive path instead of a clean "no such fts5 table" error.
 *
 * This is a denial-of-service bug, not a heap memory-safety bug -- it is in the corpus as
 * one, and should not be counted with the spatial set.
 * NOTE: -DSQLITE_DQS=0, so SQL string literals must be single-quoted.
```
