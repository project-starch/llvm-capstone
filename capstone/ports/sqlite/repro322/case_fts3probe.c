/* FTS3/4 feasibility probe: build an fts3 table, insert, run a MATCH + snippet/
 * offsets. If it compiles and completes, FTS3 bugs are reachable. */
#include "repro322_common.h"
static int run_case(void) {
  if (repro_init()) return 1;
  sqlite3 *db = 0;
  int rc = sqlite3_open(":memory:", &db);
  if (rc != SQLITE_OK) return FAILRC("open", rc);
  rc = sqlite3_exec(db, "CREATE VIRTUAL TABLE ft USING fts3(a);"
                        "INSERT INTO ft VALUES('hello world foo');"
                        "INSERT INTO ft VALUES('goodbye cruel world');", 0,0,0);
  if (rc != SQLITE_OK) { out_text("fts3probe exec rc="); out_uint((unsigned)rc);
    out_text(" ("); out_text(sqlite3_errmsg(db)); out_text(")\n"); }
  sqlite3_stmt *st=0;
  if (sqlite3_prepare_v2(db,"SELECT offsets(ft) FROM ft WHERE ft MATCH 'world'",-1,&st,0)==SQLITE_OK){
    int n=0; while(sqlite3_step(st)==SQLITE_ROW) n++; out_text("fts3probe rows="); out_uint((unsigned)n); out_text("\n");
    sqlite3_finalize(st);
  } else { out_text("fts3probe prepare err ("); out_text(sqlite3_errmsg(db)); out_text(")\n"); }
  sqlite3_close(db);
  out_text("fts3probe NOTRAP done\n");
  return 0;
}
REPRO322_MAIN("fts3probe")
