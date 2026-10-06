/* FZ09 — NULL dereference in sqlite3StrICmp, reached through searchWith().
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
 * NOTE: -DSQLITE_DQS=0, so SQL string literals must be single-quoted. */
#include "repro322_common.h"

static void kv(const char *k, long v){
  out_text("fz09 "); out_text(k); out_text("=");
  if(v < 0){ out_text("-"); v = -v; }
  out_uint((unsigned long)v); out_text("\n");
}

static int run_case(void){
  sqlite3 *db = 0; char *e = 0; int rc;
  if (repro_init()) return 1;

  rc = sqlite3_open(":memory:", &db);
  kv("open_rc", rc);
  if(rc != SQLITE_OK){ sqlite3_close(db); return 1; }

  /* Ingredient 1 + 2 + 3, all in one statement. */
  rc = sqlite3_exec(db,
        "CREATE VIEW t3 ('') AS WITH t4(a) AS (VALUES(1)) SELECT x FROM CROSS JOIN t4;",
        0, 0, &e);
  kv("create_view_rc", rc);
  if(e){ out_text("fz09 create_err: "); out_text(e); out_text("\n"); sqlite3_free(e); e = 0; }

  /* REACHABILITY PROBE. create_view_rc must be 0: the body is stored unparsed, so a
   * non-zero rc here means the malformed FROM was rejected up front and the bug site is
   * unreachable in this build. Only then is the SELECT below meaningful. */
  if(rc != SQLITE_OK){
    out_text("fz09 view was REJECTED at create; bug site not reached\n");
    sqlite3_close(db); out_text("fz09 NOTRAP done\n"); return 0;
  }

  out_text("fz09 about to expand the view (searchWith runs here)\n");
  rc = sqlite3_exec(db, "SELECT * FROM t3;", 0, 0, &e);
  kv("select_rc", rc);
  if(e){ out_text("fz09 select_err: "); out_text(e); out_text("\n"); sqlite3_free(e); e = 0; }

  sqlite3_close(db);
  out_text("fz09 NOTRAP done\n");
  return 0;
}
REPRO322_MAIN("fz09")
