/* row11 / sqlite-dee0359ddb -- fts3SegReaderNext() points pReader->zTerm straight
 * into the pending-terms hash key memory (fts3_write.c:1333); an optimize() during
 * the scan frees that hash, so zTerm dangles. Fixed 3.37.0. CONTROL: keep a
 * non-empty pending hash, open a SELECT scan, run optimize() mid-scan.
 * NOTE: build in fts3 group. */
#include "repro322_common.h"
static int run_case(void){
  if (repro_init()) return 1;
  sqlite3 *db=0; int rc=sqlite3_open(":memory:",&db); if(rc)return FAILRC("open",rc);
  rc=sqlite3_exec(db,"CREATE VIRTUAL TABLE ft USING fts3(a);"
    "INSERT INTO ft VALUES('apple apricot avocado');"
    "INSERT INTO ft VALUES('banana blueberry');"
    "INSERT INTO ft VALUES('cherry cranberry');",0,0,0);   /* leaves a pending-terms hash */
  if(rc){out_text("fts3zterm setup rc=");out_uint((unsigned)rc);out_text("\n");}
  sqlite3_stmt*st=0;
  if(sqlite3_prepare_v2(db,"SELECT a FROM ft WHERE ft MATCH 'a*'",-1,&st,0)==SQLITE_OK){
    int n=0;
    while(sqlite3_step(st)==SQLITE_ROW){
      n++;
      if(n==1){ sqlite3_exec(db,"INSERT INTO ft(ft) VALUES('optimize');",0,0,0); } /* frees pending hash mid-scan */
    }
    out_text("fts3zterm rows=");out_uint((unsigned)n);out_text("\n"); sqlite3_finalize(st);
  } else { out_text("fts3zterm prepare err (");out_text(sqlite3_errmsg(db));out_text(")\n"); }
  sqlite3_close(db);
  out_text("fts3zterm NOTRAP done\n"); return 0;
}
REPRO322_MAIN("fts3zterm")
