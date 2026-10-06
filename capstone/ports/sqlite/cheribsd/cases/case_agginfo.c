/* row17 / sqlite-5e4233a9e4 -- Use-after-free on AggInfo function expressions:
 * an aggregate inside a FROM sub-select is attributed to the wrong AggInfo (op2
 * miscount at resolve.c:598); countOfViewOptimization() (select.c) deletes the
 * ExprList owning those Expr nodes (5069 sqlite3ExprListDelete) while AggInfo
 * still points into them, and resetAccumulator() (4752) reads pF->pExpr after
 * the free. Needs -DSQLITE_COUNTOFVIEW_OPTIMIZATION for the free site. Fixed 3.45.0.
 *
 * CONTROL arm: run the SELECT that triggers countOfViewOptimization over a view
 * with a count() so the optimizer rewrites it. On unprotected Capstone the freed
 * Expr read is not caught, so the query returns rows / an error but the domain
 * RETURNS. Host ASan build flags the heap-use-after-free.
 *
 * Trigger shape (from the forum post c9970a37ed / chromium 41487453): a
 * SELECT count(*) FROM (SELECT ...) that the count-of-view optimization rewrites,
 * with an aggregate that the resolver mis-attributes across the sub-select level.
 */
#include "repro322_common.h"

static int run_case(void) {
  if (repro_init()) return 1;
  sqlite3 *db = 0;
  int rc = sqlite3_open(":memory:", &db);
  if (rc != SQLITE_OK) return FAILRC("open", rc);

  rc = sqlite3_exec(db,
    "CREATE TABLE t1(a,b);"
    "INSERT INTO t1 VALUES(1,2),(3,4),(5,6);"
    "CREATE VIEW v1 AS SELECT a, b FROM t1;",
    0, 0, 0);
  if (rc != SQLITE_OK) return FAILRC("setup", rc);

  /* count-of-view optimization: count(*) over a view is rewritten; the nested
   * aggregate is the mis-attributed one. Step it fully. */
  sqlite3_stmt *st = 0;
  const char *sql =
    "SELECT (SELECT count(*) FROM v1), max(a), sum(b) FROM v1 "
    "GROUP BY b HAVING count(*)>0";
  rc = sqlite3_prepare_v2(db, sql, -1, &st, 0);
  if (rc != SQLITE_OK) { out_text("agginfo prepare rc="); out_uint((unsigned)rc);
    out_text(" ("); out_text(sqlite3_errmsg(db)); out_text(")\n"); }
  else {
    int n = 0;
    while (sqlite3_step(st) == SQLITE_ROW) n++;
    out_text("agginfo rows="); out_uint((unsigned)n); out_text("\n");
    sqlite3_finalize(st);
  }
  sqlite3_close(db);
  out_text("agginfo NOTRAP done\n");
  return 0;
}

REPRO322_MAIN("agginfo")
