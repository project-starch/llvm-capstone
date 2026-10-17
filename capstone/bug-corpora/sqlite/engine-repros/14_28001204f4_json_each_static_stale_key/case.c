/* NEW-1 / sqlite-28001204f4 -- json_each/json_tree hand the SQL layer their .json
 * column as SQLITE_STATIC, pointing at the cursor's OWN malloc'd copy of the input:
 *
 *     case JEACH_JSON:                                     (json1.c, 3.22.0)
 *       sqlite3_result_text(ctx, p->sParse.zJson, -1, SQLITE_STATIC);
 *
 *     jsonEachFilter():      p->zJson = sqlite3_malloc64(n+1)
 *     jsonEachCursorReset(): sqlite3_free(p->zJson)
 *
 * SQLITE_STATIC means no copy is made, and sqlite3VdbeMemCopy() keeps MEM_Static
 * as-is, so a min()/max() accumulator retains the raw pointer. Driving json_each
 * from the right of a join re-filters it per outer row; each jsonEachFilter calls
 * jsonEachCursorReset and frees the PREVIOUS zJson while the accumulator still
 * points into it. minmaxStep() then compares against freed memory.
 * Fixed 3.45.2 (SQLITE_TRANSIENT). Ext: JSON1.
 *
 * RUNTIME-CONFIRMED on host SQLite 3.22.0 under ASan with exactly this SQL:
 *   heap-use-after-free in minmaxStep -> vdbeCompareMemString -> binCollFunc.
 * group_concat does NOT fire (it copies); min/max do.
 *
 * Expected on unprotected Capstone: the freed bytes are compared as a STRING
 * (memcmp), not dereferenced as a capability, so no tag check is tripped and the
 * read should be SILENT -> NOTRAP. That silence is the control-arm result.
 * NOTE -DSQLITE_DQS=0: SQL string literals must be single-quoted.
 * Build in the json group (-DSQLITE_ENABLE_JSON1). */
#include "repro322_common.h"
static int run_case(void){
  if (repro_init()) return 1;
  sqlite3 *db=0; int rc=sqlite3_open(":memory:",&db); if(rc)return FAILRC("open",rc);
  char *e=0;
  rc=sqlite3_exec(db,
    "CREATE TABLE u(v);"
    "INSERT INTO u VALUES('[1,2]');"
    "INSERT INTO u VALUES('[3,4]');"
    "INSERT INTO u VALUES('[5,6]');",0,0,&e);
  if(rc){out_text("jsoneachstatic setup rc=");out_uint((unsigned)rc);
         out_text(" (");out_text(e?e:"");out_text(")\n");sqlite3_free(e);e=0;}

  /* Both json_each arguments are LITERALS on purpose. A table-valued function whose
   * argument is a COLUMN reference yields zero rows in this domain (verified separately),
   * so the host-side shape `json_each(u.v)` cannot be used here. Crossing TWO literal
   * json_each calls gives the same effect: the INNER cursor is re-filtered once per outer
   * row, and each jsonEachFilter calls jsonEachCursorReset -> sqlite3_free(p->zJson) while
   * the max() accumulator still holds a MEM_Static pointer into the previous zJson. */
  sqlite3_stmt *chk=0;
  if(sqlite3_prepare_v2(db,
       "SELECT count(*) FROM json_each('[1,2]') j1, json_each('[3,4,5]') j2",-1,&chk,0)==SQLITE_OK){
    if(sqlite3_step(chk)==SQLITE_ROW){
      out_text("jsoneachstatic cross_rows=");out_uint((unsigned)sqlite3_column_int(chk,0));out_text("\n");
    }
    sqlite3_finalize(chk); chk=0;
  } else { out_text("jsoneachstatic cross count prepare err (");out_text(sqlite3_errmsg(db));out_text(")\n"); }

  out_text("jsoneachstatic before max(inner json) over TVF cross join\n");
  sqlite3_stmt *st=0;
  if(sqlite3_prepare_v2(db,
       "SELECT max(j2.json) FROM json_each('[1,2]') j1, json_each('[3,4,5]') j2",-1,&st,0)==SQLITE_OK){
    int n=0;
    while(sqlite3_step(st)==SQLITE_ROW){
      const unsigned char *z=sqlite3_column_text(st,0);
      n++; out_text("  max=");out_text(z?(const char*)z:"(null)");out_text("\n");
    }
    out_text("jsoneachstatic rows=");out_uint((unsigned)n);out_text("\n");
    sqlite3_finalize(st);
  } else { out_text("jsoneachstatic prepare err (");out_text(sqlite3_errmsg(db));out_text(")\n"); }

  sqlite3_close(db);
  out_text("jsoneachstatic NOTRAP done\n"); return 0;
}
REPRO322_MAIN("jsoneachstatic")
