# fz09 -- fz09

Upstream fix `FZ09`. Collected in round R2 (fuzz-corpus diff).

The reduction is argued in the case source's own header, reproduced here so the
claim travels with the case rather than living only in the build tree.

```
FZ09 — NULL dereference in sqlite3StrICmp, reached through searchWith().
 *
 * Collected by replaying SQLite's post-3.22.0 fuzz corpus against 3.22.0 (see
 * task-checkpoints/LADYBUG_FUZZDIFF_HAUL.md). Host ASan on stock 3.22.0:
 *
 *   #0 sqlite3StrICmp  sqlite3.c:28771      c = UpperToLower[*a] - UpperToLower[*b];
 *   #1 searchWith      sqlite3.c:122793
 *   #2 withExpand      sqlite3.c:122849
 *   #3 selectExpander  sqlite3.c:123041
 *
 * Minimised from a 320-byte fuzz case to 96 bytes. THREE ingredients, each verified
 * necessary by removing it and watching the crash go away:
 *
 *   1. the view carries a column list whose name is the EMPTY STRING:  CREATE VIEW t3 ('')
 *   2. the view body has a WITH clause (a CTE to search)
 *   3. the view body's FROM clause is malformed with a MISSING LEFT TABLE:
 *        SELECT x FROM CROSS JOIN t4
 *
 * With (1) removed the case runs clean; with (2) removed it runs clean; with (3) made
 * well-formed it runs clean. The malformed FROM leaves a SrcList item whose zName is NULL,
 * and searchWith() hands that NULL straight to sqlite3StrICmp against the CTE's name.
 *
 * The view body is NOT parsed at CREATE time -- only its text is stored -- so the CREATE
 * succeeds and the fault happens on first use. That is why the probe below reports both.
 * NOTE: -DSQLITE_DQS=0, so SQL string literals must be single-quoted.
```
