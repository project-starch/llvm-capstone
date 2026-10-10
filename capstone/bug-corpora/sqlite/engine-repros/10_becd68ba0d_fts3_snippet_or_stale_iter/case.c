/* row 6 / becd68ba0d -- fts3 snippet() use-after-free via a nested OR.
 *
 * fts3EvalNextRow()'s nested-OR branch advances the exhausted side without setting
 * bEof, so the phrase keeps being evaluated after its doclist was freed. The 3.32.0
 * fix is one line: `pRight->bEof = pLeft->bEof = 1;`.
 *
 * THIS CASE NEEDS ITS OWN GROUP. The bug lives in the *nested* OR branch, and nested
 * query syntax exists only with -DSQLITE_ENABLE_FTS3_PARENTHESIS. Without it the
 * parentheses in the MATCH string are ordinary characters, a flat query runs instead,
 * and the case returns ZERO ROWS -- which is exactly how this row sat in the corpus as
 * a PASS that established nothing. Measured on the host, same SQL, 3.22.0:
 *
 *   with    -DSQLITE_ENABLE_FTS3_PARENTHESIS : rows=1, snippet "<b>one</b>",
 *                                              offsets "0 1 0 3 0 3 0 3", ASan fires
 *   without                                   : rows=0, ASan silent
 *
 * Host ASan oracle (3.22.0, with the option), upstream's fts3snippet2.test 2.1/2.2
 * verbatim:
 *   READ of size 1, 180 bytes into a freed 208-byte region
 *     use   fts3SnippetAdvance <- fts3SnippetNextCandidate <- fts3BestSnippet
 *             <- sqlite3Fts3Snippet
 *     free  sqlite3Fts3SegReaderFree <- sqlite3Fts3SegReaderFinish
 *             <- fts3SegReaderCursorFree <- fts3TermSelect <- fts3EvalPhraseLoad
 *             <- fts3EvalPhraseStart
 * offsets() reaches the same freed block through sqlite3Fts3Offsets, so the case runs
 * both and reports each.
 *
 * REACHABILITY PROBE, so this can never again pass vacuously: rows must be 1 and the
 * snippet must be exactly "<b>one</b>". The domain compares the string itself and
 * prints a WARNING if either differs -- a 0-row result now reports itself.
 * NOTE: -DSQLITE_DQS=0, so SQL string literals must be single-quoted. */
#include "repro322_common.h"

static void rcline(const char *what, int rc){
  out_text("fts3snipor "); out_text(what); out_text(" rc=");
  out_uint((unsigned)(rc<0?-rc:rc)); out_text("\n");
}

static int streq(const char *a, const char *b){
  if(!a || !b) return 0;
  while(*a && *a==*b){ a++; b++; }
  return *a==0 && *b==0;
}

/* Run one MATCH query that calls an fts3 auxiliary function; report rows and the
 * first column's text, and whether it equals the expected value. */
static int run_q(sqlite3 *db, const char *label, const char *sql, const char *expect){
  sqlite3_stmt *st = 0;
  int rc = sqlite3_prepare_v2(db, sql, -1, &st, 0);
  out_text("fts3snipor "); out_text(label); out_text(" prepare rc=");
  out_uint((unsigned)(rc<0?-rc:rc));
  if(rc!=SQLITE_OK){ out_text(" ("); out_text(sqlite3_errmsg(db)); out_text(")\n"); return 0; }
  out_text("\n");
  int n = 0, matched = 0;
  while(sqlite3_step(st)==SQLITE_ROW){
    const char *t = (const char*)sqlite3_column_text(st, 0);
    n++;
    if(n==1){
      out_text("fts3snipor "); out_text(label); out_text(" value=[");
      out_text(t ? t : "NULL"); out_text("]\n");
      if(expect && streq(t, expect)) matched = 1;
    }
  }
  out_text("fts3snipor "); out_text(label); out_text(" rows=");
  out_uint((unsigned)n);
  out_text(" expected_value="); out_uint((unsigned)matched); out_text("\n");
  sqlite3_finalize(st);
  if(n==0){
    out_text("fts3snipor WARNING "); out_text(label);
    out_text(" returned 0 rows -- the nested-OR path was NOT taken."
             " Is SQLITE_ENABLE_FTS3_PARENTHESIS defined for this group?\n");
  }
  return n;
}

static int run_case(void){
  if (repro_init()) return 1;
  sqlite3 *db=0; char *e=0; int rc;

  rc = sqlite3_open(":memory:", &db);
  rcline("open", rc);
  if(rc!=SQLITE_OK){ sqlite3_close(db); return 1; }

  /* upstream fts3snippet2.test case 2.1 verbatim */
  rc = sqlite3_exec(db,
      "CREATE VIRTUAL TABLE t0 USING fts3("
      "col0 INTEGER PRIMARY KEY,col1 VARCHAR(8),col2 BINARY,col3 BINARY);", 0,0,&e);
  rcline("create", rc);
  if(e){ out_text("fts3snipor create err ("); out_text(e); out_text(")\n"); sqlite3_free(e); e=0; }
  if(rc!=SQLITE_OK){ sqlite3_close(db); return 1; }

  rc = sqlite3_exec(db, "INSERT INTO t0 VALUES ('one', '1234','aaaa','bbbb');", 0,0,&e);
  rcline("insert", rc);
  if(e){ out_text("fts3snipor insert err ("); out_text(e); out_text(")\n"); sqlite3_free(e); e=0; }

  /* case 2.2 verbatim: the nested OR. snippet() is the upstream reproducer; offsets()
   * reaches the same freed block through a different reader. */
  int n1 = run_q(db, "snippet",
      "SELECT snippet(t0) FROM t0 WHERE t0 MATCH '(def AND (one NEAR abc)) OR one'",
      "<b>one</b>");
  int n2 = run_q(db, "offsets",
      "SELECT offsets(t0) FROM t0 WHERE t0 MATCH '(def AND (one NEAR abc)) OR one'",
      "0 1 0 3 0 3 0 3");

  if(n1>0 && n2>0) out_text("fts3snipor nested-OR path reached on both queries\n");

  sqlite3_close(db);
  out_text("fts3snipor NOTRAP done\n"); return 0;
}
REPRO322_MAIN("fts3snipor")
