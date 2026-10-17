/* NEW-2 / json_each root/path column -- the SIBLING the 3.45.2 fix left behind,
 * STILL UNFIXED UPSTREAM.
 *
 * jsonEachColumn()'s default arm returns the cursor's own buffer as SQLITE_STATIC,
 * exactly like the JEACH_JSON arm did before 3.45.2:
 *
 *     default: {                                    (3.22.0 ext/misc/json1.c:2139)
 *       const char *zRoot = p->zRoot;
 *       if( zRoot==0 ) zRoot = "$";
 *       sqlite3_result_text(ctx, zRoot, -1, SQLITE_STATIC);
 *
 *     jsonEachFilter():      p->zRoot = sqlite3_malloc64(n+1)     (:2244)
 *     jsonEachCursorReset(): sqlite3_free(p->zRoot)               (:1963)
 *
 * The 3.45.2 fix (sqlite-28001204f4) converted ONLY JEACH_JSON to SQLITE_TRANSIENT;
 * current trunk still returns p->path.zBuf with SQLITE_STATIC here, so this arm is
 * live upstream today. zRoot is non-NULL only for the TWO-argument form
 * json_each(X, PATH), which is why the SQL below passes a path.
 *
 * Same trigger as case_json_each_static.c: cross TWO json_each calls with LITERAL
 * arguments so the INNER cursor is re-filtered once per outer row. A table-valued
 * function whose argument is a COLUMN reference yields zero rows in this domain
 * (see case_jsondiag.c), so the host's join-on-a-column shape cannot be used.
 *
 * RUNTIME-CONFIRMED on host SQLite 3.22.0 under ASan with this exact SQL:
 *   heap-use-after-free in memcmp <- binCollFunc <- vdbeCompareMemString <- minmaxStep.
 * max(j2.path) and the json_tree form fire identically.
 *
 * Expected on unprotected Capstone: the freed bytes are compared as a STRING (memcmp),
 * not dereferenced as a capability, so no tag check is tripped -> silent NOTRAP.
 * NOTE -DSQLITE_DQS=0: SQL string literals must be single-quoted (JSON keys keep their
 * double quotes INSIDE the single-quoted SQL literal).
 * Build in the json group (-DSQLITE_ENABLE_JSON1). */
#include "repro322_common.h"

#define XJOIN "FROM json_each('{\"a\":[1,2]}','$.a') j1, json_each('{\"bb\":[3,4,5]}','$.bb') j2"

static int run_case(void){
  if (repro_init()) return 1;
  sqlite3 *db=0; int rc=sqlite3_open(":memory:",&db); if(rc)return FAILRC("open",rc);
  sqlite3_stmt *st=0;

  /* Reachability probe 1: the cross join must really produce rows, otherwise a PASS
   * would be empty -- the trap that killed the first draft of the sibling case. */
  if(sqlite3_prepare_v2(db,"SELECT count(*) " XJOIN,-1,&st,0)==SQLITE_OK){
    if(sqlite3_step(st)==SQLITE_ROW){
      out_text("jsoneachroot cross_rows=");out_uint((unsigned)sqlite3_column_int(st,0));out_text("\n");
    }
    sqlite3_finalize(st); st=0;
  } else { out_text("jsoneachroot count prepare err (");out_text(sqlite3_errmsg(db));out_text(")\n"); }

  /* Reachability probe 2: zRoot must be non-NULL, i.e. the 2-arg form really took the
   * malloc'd-path branch rather than the literal "$" fallback. */
  if(sqlite3_prepare_v2(db,"SELECT j2.root, j2.value " XJOIN,-1,&st,0)==SQLITE_OK){
    int k=0;
    while(sqlite3_step(st)==SQLITE_ROW && k<3){
      const unsigned char *a=sqlite3_column_text(st,0), *b=sqlite3_column_text(st,1);
      out_text("  pair: root=");out_text(a?(const char*)a:"(null)");
      out_text(" value=");out_text(b?(const char*)b:"(null)");out_text("\n"); k++;
    }
    sqlite3_finalize(st); st=0;
  }

  /* The bug: max() keeps a MEM_Static pointer into the inner cursor's zRoot; the next
   * jsonEachFilter resets that cursor and frees it; minmaxStep then compares against it. */
  out_text("jsoneachroot before max(inner root) over TVF cross join\n");
  if(sqlite3_prepare_v2(db,"SELECT max(j2.root) " XJOIN,-1,&st,0)==SQLITE_OK){
    int n=0;
    while(sqlite3_step(st)==SQLITE_ROW){
      const unsigned char *z=sqlite3_column_text(st,0);
      n++; out_text("  max=");out_text(z?(const char*)z:"(null)");out_text("\n");
    }
    out_text("jsoneachroot rows=");out_uint((unsigned)n);out_text("\n");
    sqlite3_finalize(st); st=0;
  } else { out_text("jsoneachroot prepare err (");out_text(sqlite3_errmsg(db));out_text(")\n"); }

  sqlite3_close(db);
  out_text("jsoneachroot NOTRAP done\n"); return 0;
}
REPRO322_MAIN("jsoneachroot")
