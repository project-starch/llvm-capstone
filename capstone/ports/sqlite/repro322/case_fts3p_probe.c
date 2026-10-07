/* fts3P group baseline probe -- the differential control for case_fts3_snippet_or.c.
 *
 * Exercises everything row 6 needs except the nested OR: the same CREATE VIRTUAL TABLE
 * with the same four columns, the same row, snippet() and offsets() over a plain term
 * match, and a FLAT OR (no parentheses). If this PASSes while fts3snipor FAULTs, the
 * fault belongs to the nested-OR path and not to the fts3P group or to
 * SQLITE_ENABLE_FTS3_PARENTHESIS itself.
 *
 * It also confirms the option is actually in effect: with parentheses enabled a
 * parenthesised query parses and matches, so `paren_rows` is 1. Without the option the
 * parentheses are ordinary characters; that is the state in which row 6 returned 0 rows
 * and passed vacuously for weeks.
 * NOTE: -DSQLITE_DQS=0, so SQL string literals must be single-quoted. */
#include "repro322_common.h"

static void rcline(const char *what, int rc){
  out_text("fts3pprobe "); out_text(what); out_text(" rc=");
  out_uint((unsigned)(rc<0?-rc:rc)); out_text("\n");
}

static int q(sqlite3 *db, const char *label, const char *sql){
  sqlite3_stmt *st=0; int n=0;
  int rc = sqlite3_prepare_v2(db,sql,-1,&st,0);
  if(rc!=SQLITE_OK){
    out_text("fts3pprobe "); out_text(label); out_text(" prepare rc=");
    out_uint((unsigned)(rc<0?-rc:rc));
    out_text(" ("); out_text(sqlite3_errmsg(db)); out_text(")\n");
    return 0;
  }
  while(sqlite3_step(st)==SQLITE_ROW){
    if(n==0){
      const char *t=(const char*)sqlite3_column_text(st,0);
      out_text("fts3pprobe "); out_text(label); out_text(" value=[");
      out_text(t?t:"NULL"); out_text("]\n");
    }
    n++;
  }
  sqlite3_finalize(st);
  out_text("fts3pprobe "); out_text(label); out_text(" rows="); out_uint((unsigned)n);
  out_text("\n");
  return n;
}

static int run_case(void){
  if (repro_init()) return 1;
  sqlite3 *db=0; char *e=0; int rc;

  rc = sqlite3_open(":memory:", &db);
  rcline("open", rc);
  if(rc!=SQLITE_OK){ sqlite3_close(db); return 1; }

  rc = sqlite3_exec(db,
      "CREATE VIRTUAL TABLE t0 USING fts3("
      "col0 INTEGER PRIMARY KEY,col1 VARCHAR(8),col2 BINARY,col3 BINARY);", 0,0,&e);
  rcline("create", rc);
  if(e){ out_text("fts3pprobe create err ("); out_text(e); out_text(")\n"); sqlite3_free(e); e=0; }
  if(rc!=SQLITE_OK){ sqlite3_close(db); return 1; }

  rc = sqlite3_exec(db,"INSERT INTO t0 VALUES ('one', '1234','aaaa','bbbb');",0,0,&e);
  rcline("insert", rc);
  if(e){ sqlite3_free(e); e=0; }

  /* plain term match through the same auxiliary functions */
  q(db,"snippet plain",  "SELECT snippet(t0) FROM t0 WHERE t0 MATCH 'one'");
  q(db,"offsets plain",  "SELECT offsets(t0) FROM t0 WHERE t0 MATCH 'one'");
  /* a FLAT OR -- the operator without nesting */
  q(db,"snippet flat-or","SELECT snippet(t0) FROM t0 WHERE t0 MATCH 'one OR aaaa'");
  /* a parenthesised query: proves SQLITE_ENABLE_FTS3_PARENTHESIS is in effect */
  int paren = q(db,"snippet paren","SELECT snippet(t0) FROM t0 WHERE t0 MATCH '(one OR aaaa)'");
  out_text("fts3pprobe paren_rows="); out_uint((unsigned)paren); out_text("\n");
  if(paren==0)
    out_text("fts3pprobe WARNING parenthesised query matched nothing --"
             " SQLITE_ENABLE_FTS3_PARENTHESIS may not be defined for this group\n");

  sqlite3_close(db);
  out_text("fts3pprobe NOTRAP done\n"); return 0;
}
REPRO322_MAIN("fts3pprobe")
