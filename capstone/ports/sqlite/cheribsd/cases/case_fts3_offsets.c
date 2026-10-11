/* row13 / sqlite-51dd67080a -- sqlite3Fts3Offsets() stores a poslist pointer per
 * phrase, then an incremental phrase triggers a NEAR-group restart (fts3EvalRestart,
 * fts3.c:5540) that frees/reloads already-visited doclists, dangling the saved
 * pointers (fts3_snippet.c:1544). Fixed 3.49.0. CONTROL: offsets() over an OR of
 * phrases read incrementally; on unprotected capstone completes -> NOTRAP.
 * NOTE: build in fts3 group. */
#include "repro322_common.h"
static int run_case(void){
  if (repro_init()) return 1;
  sqlite3 *db=0; int rc=sqlite3_open(":memory:",&db); if(rc)return FAILRC("open",rc);
  rc=sqlite3_exec(db,"CREATE VIRTUAL TABLE ft USING fts3(a);"
    "INSERT INTO ft VALUES('red green blue red green');"
    "INSERT INTO ft VALUES('green blue yellow green blue');"
    "INSERT INTO ft VALUES('blue red yellow blue red');",0,0,0);
  if(rc){out_text("fts3offsets setup rc=");out_uint((unsigned)rc);out_text("\n");}
  sqlite3_stmt*st=0;
  if(sqlite3_prepare_v2(db,"SELECT offsets(ft) FROM ft WHERE ft MATCH '\"red green\" OR \"blue red\"'",-1,&st,0)==SQLITE_OK){
    int n=0; while(sqlite3_step(st)==SQLITE_ROW) n++;
    out_text("fts3offsets rows=");out_uint((unsigned)n);out_text("\n"); sqlite3_finalize(st);
  } else { out_text("fts3offsets prepare err (");out_text(sqlite3_errmsg(db));out_text(")\n"); }
  sqlite3_close(db);
  out_text("fts3offsets NOTRAP done\n"); return 0;
}
REPRO322_MAIN("fts3offsets")
