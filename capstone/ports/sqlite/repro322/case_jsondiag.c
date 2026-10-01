/* Diagnostic: how far does JSON1 actually work inside the freestanding domain?
 * json_each() returned ZERO rows there while the identical SQL fires an ASan
 * heap-use-after-free on a host 3.22.0 build, so something upstream of the bug
 * path is failing. Probe scalar JSON first, then the eponymous vtab. */
#include "repro322_common.h"
static void q(sqlite3 *db, const char *sql, const char *tag){
  sqlite3_stmt *st=0;
  int rc = sqlite3_prepare_v2(db, sql, -1, &st, 0);
  out_text(tag); out_text(": ");
  if(rc!=SQLITE_OK){ out_text("PREPARE-ERR ("); out_text(sqlite3_errmsg(db)); out_text(")\n"); return; }
  rc = sqlite3_step(st);
  if(rc==SQLITE_ROW){
    const unsigned char *z = sqlite3_column_text(st,0);
    out_text(z?(const char*)z:"(null)");
  } else if(rc==SQLITE_DONE){ out_text("(no rows)"); }
  else { out_text("step rc="); out_uint((unsigned)(rc<0?-rc:rc)); }
  out_text("\n");
  sqlite3_finalize(st);
}
static int run_case(void){
  if (repro_init()) return 1;
  sqlite3 *db=0; int rc=sqlite3_open(":memory:",&db); if(rc)return FAILRC("open",rc);
  q(db,"SELECT json_valid('[1,2]')",            "json_valid");
  q(db,"SELECT json_type('[1,2]')",             "json_type");
  q(db,"SELECT json_array_length('[1,2,3]')",   "json_array_length");
  q(db,"SELECT json_extract('[7,8]','$[0]')",   "json_extract");
  q(db,"SELECT count(*) FROM json_each('[1,2,3]')",  "json_each count");
  q(db,"SELECT value FROM json_each('[1,2,3]')",     "json_each first value");
  q(db,"SELECT json FROM json_each('[1,2,3]')",       "json_each json col");
  q(db,"SELECT count(*) FROM json_tree('[1,2]')",     "json_tree count");
  /* Narrow down the failing shape: TVF argument taken from a COLUMN rather than a literal. */
  sqlite3_exec(db,"CREATE TABLE u(v); INSERT INTO u VALUES('[1,2]'); INSERT INTO u VALUES('[3,4]');",0,0,0);
  q(db,"SELECT count(*) FROM u",                                        "plain table count");
  q(db,"SELECT count(*) FROM u JOIN json_each(u.v) j",                  "TVF col-ref JOIN");
  q(db,"SELECT count(*) FROM u, json_each(u.v) j",                      "TVF col-ref comma");
  q(db,"SELECT (SELECT count(*) FROM json_each(t.v)) FROM u t LIMIT 1", "TVF correlated subq");
  q(db,"SELECT count(*) FROM json_each((SELECT v FROM u LIMIT 1))",      "TVF scalar subq arg");
  q(db,"SELECT j.value FROM u JOIN json_each(u.v) j LIMIT 1",            "TVF col-ref value");
  sqlite3_close(db);
  out_text("jsondiag NOTRAP done\n"); return 0;
}
REPRO322_MAIN("jsondiag")
