/* row24 probe -- confirm ext/misc/spellfix.c builds+links+registers in the domain
 * and editdist3() runs. The UAF (row24) needs OOM at the 2nd editdist3 registration;
 * this probe first proves feasibility of the extension itself. Build spellfix.c with
 * -DSQLITE_CORE so SQLITE_EXTENSION_INIT2 is a no-op and we call the init directly. */
#include "repro322_common.h"
int sqlite3_spellfix_init(sqlite3*, char**, const void*);

static int run_case(void){
  if (repro_init()) return 1;
  sqlite3 *db=0; int rc=sqlite3_open(":memory:",&db); if(rc)return FAILRC("open",rc);
  rc = sqlite3_spellfix_init(db, 0, 0);
  out_text("spellfix init rc="); out_uint((unsigned)(rc<0?-rc:rc)); out_text("\n");
  sqlite3_stmt*st=0;
  if(sqlite3_prepare_v2(db,"SELECT editdist3('kitten','sitting')",-1,&st,0)==SQLITE_OK){
    int n=0; while(sqlite3_step(st)==SQLITE_ROW){ n++; }
    out_text("spellfix editdist3 rows="); out_uint((unsigned)n); out_text("\n");
    sqlite3_finalize(st);
  } else { out_text("spellfix editdist3 prepare err ("); out_text(sqlite3_errmsg(db)); out_text(")\n"); }
  sqlite3_close(db);
  out_text("spellfixprobe NOTRAP done\n"); return 0;
}
REPRO322_MAIN("spellfixprobe")
