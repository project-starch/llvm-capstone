/* DIAGNOSTIC: fine-grained markers to locate where fts3 stops. Run with --tail. */
#include "repro322_common.h"
static int run_case(void){
  out_text("D:start\n");
  if (repro_init()){ out_text("D:init-FAIL\n"); return 1; }
  out_text("D:init-ok\n");
  sqlite3 *db=0; int rc=sqlite3_open(":memory:",&db);
  out_text("D:open rc="); out_uint((unsigned)rc); out_text("\n");
  if(rc) return 1;
  rc=sqlite3_exec(db,"CREATE VIRTUAL TABLE ft USING fts3(a);",0,0,0);
  out_text("D:create rc="); out_uint((unsigned)rc); out_text(" ("); out_text(sqlite3_errmsg(db)); out_text(")\n");
  rc=sqlite3_exec(db,"INSERT INTO ft VALUES('hello world foo');",0,0,0);
  out_text("D:ins1 rc="); out_uint((unsigned)rc); out_text("\n");
  rc=sqlite3_exec(db,"INSERT INTO ft VALUES('goodbye cruel world');",0,0,0);
  out_text("D:ins2 rc="); out_uint((unsigned)rc); out_text("\n");
  sqlite3_stmt*st=0;
  rc=sqlite3_prepare_v2(db,"SELECT offsets(ft) FROM ft WHERE ft MATCH 'world'",-1,&st,0);
  out_text("D:prep rc="); out_uint((unsigned)rc); out_text("\n");
  if(rc==SQLITE_OK){
    int n=0,sr; while((sr=sqlite3_step(st))==SQLITE_ROW){ n++; out_text("D:row "); out_uint((unsigned)n); out_text("\n"); }
    out_text("D:step-end sr="); out_uint((unsigned)sr); out_text(" rows="); out_uint((unsigned)n); out_text("\n");
    sqlite3_finalize(st);
  }
  sqlite3_close(db);
  out_text("D:NOTRAP done\n"); return 0;
}
REPRO322_MAIN("fts3diag")
