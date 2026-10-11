/* row23 / sqlite-2026-08-17 (a87bd2471a9c) -- fts3BestSnippet() saves poslist
 * pointers (pHead/pTail) per phrase; a later incremental phrase under an OR restarts
 * the NEAR group (fts3EvalRestart, fts3.c:5540), freeing/reloading already-visited
 * doclists, so the saved pointers dangle while scoring (fts3_snippet.c:507/479).
 * Distinct from becd68ba0d. Trunk-only fix. CONTROL: snippet() over an OR of phrases
 * read incrementally; on unprotected capstone completes -> NOTRAP.
 * NOTE: build in fts3 group. */
#include "repro322_common.h"
static int run_case(void){
  if (repro_init()) return 1;
  sqlite3 *db=0; int rc=sqlite3_open(":memory:",&db); if(rc)return FAILRC("open",rc);
  rc=sqlite3_exec(db,"CREATE VIRTUAL TABLE ft USING fts4(a);"
    "INSERT INTO ft VALUES('sun moon star sun moon');"
    "INSERT INTO ft VALUES('moon star cloud moon star');"
    "INSERT INTO ft VALUES('star sun cloud star sun');",0,0,0);
  if(rc){out_text("fts3snip setup rc=");out_uint((unsigned)rc);out_text(" (");out_text(sqlite3_errmsg(db));out_text(")\n");}
  sqlite3_stmt*st=0;
  if(sqlite3_prepare_v2(db,"SELECT snippet(ft) FROM ft WHERE ft MATCH '\"sun moon\" OR \"star sun\"'",-1,&st,0)==SQLITE_OK){
    int n=0; while(sqlite3_step(st)==SQLITE_ROW) n++;
    out_text("fts3snip rows=");out_uint((unsigned)n);out_text("\n"); sqlite3_finalize(st);
  } else { out_text("fts3snip prepare err (");out_text(sqlite3_errmsg(db));out_text(")\n"); }
  sqlite3_close(db);
  out_text("fts3snip NOTRAP done\n"); return 0;
}
REPRO322_MAIN("fts3snip")
