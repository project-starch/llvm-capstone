/* FTS5 feasibility probe: create an fts5 table, insert, query MATCH. If this
 * builds (FTS5 compiled) and runs to completion, FTS5 bugs are reachable. */
#include "repro322_common.h"
static int run_case(void) {
  if (repro_init()) return 1;
  sqlite3 *db = 0;
  int rc = sqlite3_open(":memory:", &db);
  if (rc != SQLITE_OK) return FAILRC("open", rc);
  rc = sqlite3_exec(db, "CREATE VIRTUAL TABLE ft USING fts5(a);"
                        "INSERT INTO ft VALUES('hello world');"
                        "INSERT INTO ft VALUES('goodbye world');", 0,0,0);
  if (rc != SQLITE_OK) { out_text("fts5probe exec rc="); out_uint((unsigned)rc);
    out_text(" ("); out_text(sqlite3_errmsg(db)); out_text(")\n"); }
  sqlite3_stmt *st=0;
  if (sqlite3_prepare_v2(db,"SELECT a FROM ft WHERE ft MATCH 'world' ORDER BY rank",-1,&st,0)==SQLITE_OK){
    int n=0; while(sqlite3_step(st)==SQLITE_ROW) n++; out_text("fts5probe rows="); out_uint((unsigned)n); out_text("\n");
    sqlite3_finalize(st);
  }
  sqlite3_close(db);
  out_text("fts5probe NOTRAP done\n");
  return 0;
}
REPRO322_MAIN("fts5probe")
