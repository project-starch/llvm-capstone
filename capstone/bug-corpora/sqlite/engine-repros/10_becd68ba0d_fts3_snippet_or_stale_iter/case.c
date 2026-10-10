/* row6 / sqlite-becd68ba0d -- fts3EvalNextRow() nested-OR branch keeps evaluating
 * phrase nodes whose doclist was freed because bEof was not set on the exhausted
 * side (fts3.c:5134). Fixed 3.32.0. CONTROL: snippet() over an expression with
 * nested OR phrases; on unprotected capstone completes -> NOTRAP.
 * NOTE: build in fts3 group. */
#include "repro322_common.h"
static int run_case(void){
  if (repro_init()) return 1;
  sqlite3 *db=0; int rc=sqlite3_open(":memory:",&db); if(rc)return FAILRC("open",rc);
  rc=sqlite3_exec(db,"CREATE VIRTUAL TABLE ft USING fts3(a);"
    "INSERT INTO ft VALUES('alpha beta gamma');"
    "INSERT INTO ft VALUES('beta delta epsilon');"
    "INSERT INTO ft VALUES('gamma epsilon zeta');",0,0,0);
  if(rc){out_text("fts3snipor setup rc=");out_uint((unsigned)rc);out_text(" (");out_text(sqlite3_errmsg(db));out_text(")\n");}
  sqlite3_stmt*st=0;
  if(sqlite3_prepare_v2(db,"SELECT snippet(ft) FROM ft WHERE ft MATCH '(alpha OR beta) OR (gamma OR epsilon)'",-1,&st,0)==SQLITE_OK){
    int n=0; while(sqlite3_step(st)==SQLITE_ROW) n++;
    out_text("fts3snipor rows=");out_uint((unsigned)n);out_text("\n"); sqlite3_finalize(st);
  } else { out_text("fts3snipor prepare err (");out_text(sqlite3_errmsg(db));out_text(")\n"); }
  sqlite3_close(db);
  out_text("fts3snipor NOTRAP done\n"); return 0;
}
REPRO322_MAIN("fts3snipor")
