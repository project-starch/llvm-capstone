/* row22 / sqlite-2026-06-08 (b677c5afd4) -- fts5ExprNearIsMatch reads a lookahead
 * reader from each poslist buffer while WriterAppend rewrites the same buffer in
 * place; the append may realloc, so the reader reads freed memory (fts5_expr.c
 * 560/596/627). Fixed 3.53.3. CONTROL: run a NEAR/phrase query that drives
 * fts5ExprNearIsMatch. On unprotected capstone completes -> NOTRAP.
 * NOTE: build in fts5 group. */
#include "repro322_common.h"
static int run_case(void){
  if (repro_init()) return 1;
  sqlite3 *db=0; int rc=sqlite3_open(":memory:",&db); if(rc)return FAILRC("open",rc);
  rc=sqlite3_exec(db,"CREATE VIRTUAL TABLE ft USING fts5(a);"
    "INSERT INTO ft VALUES('the quick brown fox jumps over the lazy dog quick brown');"
    "INSERT INTO ft VALUES('quick brown quick brown quick brown fox fox fox');",0,0,0);
  if(rc){out_text("fts5near setup rc=");out_uint((unsigned)rc);out_text("\n");}
  sqlite3_stmt*st=0;
  if(sqlite3_prepare_v2(db,"SELECT rowid FROM ft WHERE ft MATCH 'NEAR(quick brown, 3)'",-1,&st,0)==SQLITE_OK){
    int n=0; while(sqlite3_step(st)==SQLITE_ROW) n++;
    out_text("fts5near rows=");out_uint((unsigned)n);out_text("\n");
    sqlite3_finalize(st);
  } else { out_text("fts5near prepare err (");out_text(sqlite3_errmsg(db));out_text(")\n"); }
  sqlite3_close(db);
  out_text("fts5near NOTRAP done\n"); return 0;
}
REPRO322_MAIN("fts5near")
