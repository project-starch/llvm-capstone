/* row3 / sqlite-25e3073741 -- fts5MultiIterNew caches a raw Fts5Structure*; a
 * table write under an active cursor releases it, dangling the pointer;
 * fts5MultiIterFree releases it again (fts5_index.c). Fixed 3.27.0. CONTROL:
 * open a scan cursor, INSERT into the same table while the cursor is live, keep
 * stepping. On unprotected capstone completes -> NOTRAP.
 * NOTE: build in fts5 group (repro322_fts_stubs.c + math decl). */
#include "repro322_common.h"
static int run_case(void){
  if (repro_init()) return 1;
  sqlite3 *db=0; int rc=sqlite3_open(":memory:",&db); if(rc)return FAILRC("open",rc);
  rc=sqlite3_exec(db,"CREATE VIRTUAL TABLE ft USING fts5(a);"
    "INSERT INTO ft VALUES('one two');INSERT INTO ft VALUES('two three');"
    "INSERT INTO ft VALUES('three four');INSERT INTO ft VALUES('four five');",0,0,0);
  if(rc){out_text("fts5structwrite setup rc=");out_uint((unsigned)rc);out_text("\n");}
  sqlite3_stmt*st=0;
  if(sqlite3_prepare_v2(db,"SELECT rowid,a FROM ft WHERE ft MATCH 'two OR three OR four'",-1,&st,0)==SQLITE_OK){
    int n=0;
    while(sqlite3_step(st)==SQLITE_ROW){
      n++;
      if(n==1){ /* write while the cursor's structure is shared/active */
        sqlite3_exec(db,"INSERT INTO ft VALUES('two three four five six');",0,0,0);
      }
    }
    out_text("fts5structwrite rows=");out_uint((unsigned)n);out_text("\n");
    sqlite3_finalize(st);
  }
  sqlite3_close(db);
  out_text("fts5structwrite NOTRAP done\n"); return 0;
}
REPRO322_MAIN("fts5structwrite")
