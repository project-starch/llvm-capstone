/* row1 / sqlite-2639ddc474 -- fts5VocabInstanceNext() keeps stepping the fts5
 * iterator after EOF, dereferencing an iterator whose backing data was released
 * (fts5_vocab.c:427). Fixed 3.26.0. CONTROL: full-scan an fts5vocab table so the
 * EOF transition runs; on unprotected capstone it completes -> NOTRAP.
 * Public, already-fixed bug; collected for the temporal-safety study.
 * NOTE: place beside repro322_common.h in ports/sqlite/repro322/ to build. */
#include "repro322_common.h"
static int run_case(void){
  if (repro_init()) return 1;
  sqlite3 *db=0; int rc=sqlite3_open(":memory:",&db); if(rc)return FAILRC("open",rc);
  rc=sqlite3_exec(db,
    "CREATE VIRTUAL TABLE ft USING fts5(a);"
    "INSERT INTO ft VALUES('alpha beta');"
    "INSERT INTO ft VALUES('beta gamma');"
    "INSERT INTO ft VALUES('gamma delta');"
    "CREATE VIRTUAL TABLE vv USING fts5vocab('ft','row');",0,0,0);
  if(rc){out_text("fts5vocabeof setup rc=");out_uint((unsigned)rc);out_text(" (");out_text(sqlite3_errmsg(db));out_text(")\n");}
  sqlite3_stmt*st=0;
  if(sqlite3_prepare_v2(db,"SELECT term,doc,cnt FROM vv",-1,&st,0)==SQLITE_OK){
    int n=0; while(sqlite3_step(st)==SQLITE_ROW) n++;   /* steps through EOF */
    out_text("fts5vocabeof rows=");out_uint((unsigned)n);out_text("\n");
    sqlite3_finalize(st);
  }
  sqlite3_close(db);
  out_text("fts5vocabeof NOTRAP done\n"); return 0;
}
REPRO322_MAIN("fts5vocabeof")
