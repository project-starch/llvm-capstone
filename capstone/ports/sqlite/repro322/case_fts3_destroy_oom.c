/* new-4 / sqlite-129371553c -- fts3DestroyMethod() dereferences a freed Fts3Table
 * after a nested OOM.
 *
 * fts3DestroyMethod() drops the five shadow tables through fts3DbExec():
 *
 *     fts3DbExec(&rc, db, "DROP TABLE IF EXISTS %Q.'%q_content'", zDb, p->zName);
 *     fts3DbExec(&rc, db, "DROP TABLE IF EXISTS %Q.'%q_segments'", zDb, p->zName);
 *     fts3DbExec(&rc, db, "DROP TABLE IF EXISTS %Q.'%q_segdir'",   zDb, p->zName);
 *     ...
 *
 * If one of those statements hits OOM, sqlite3VdbeHalt() -> sqlite3RollbackAll()
 * -> sqlite3ResetAllSchemasOfConnection() -> sqlite3VtabUnlockList() ->
 * fts3DisconnectMethod() sqlite3_free()s the Fts3Table -- i.e. a schema reset runs
 * underneath the object's own destructor. Control then returns to
 * fts3DestroyMethod, which goes on to the NEXT fts3DbExec. Its body returns early
 * because rc!=SQLITE_OK, but `p->zName` is still evaluated at the call site, and p
 * is freed. Fixed 3.27.0 by collapsing the five calls into one.
 *
 * Host ASan oracle on 3.22.0 (lookaside off, default page size, upstream's minimal
 * CREATE-then-DROP shape, counted OOM injection):
 *   READ of size 8, 40 bytes into a 592-byte region   (Fts3Table.zName)
 *     use   fts3DestroyMethod:149364 / :149365
 *     free  fts3DisconnectMethod <- sqlite3VtabUnlock <- sqlite3VtabUnlockList
 *             <- sqlite3ResetAllSchemasOfConnection <- sqlite3RollbackAll
 *             <- sqlite3VdbeHalt
 *     alloc fts3InitVtab
 *   40 of 300 injection points hit, in two contiguous runs: 167-184 and 259-280.
 *   The two runs are the two distinct fts3DbExec call sites that can be the use.
 *
 * WHY THIS DOMAIN CANNOT OBSERVE THE BUG ITSELF, and what it establishes instead.
 * The dangling read is Fts3Table.zName at offset 40. memsys5 overwrites only the
 * first 8 bytes of a freed block (its in-band Mem5Link freelist ints), so offset 40
 * keeps both its bytes and its capability tag: the read is necessarily SILENT on
 * base Capstone. It is also unobservable from inside the domain -- the value is
 * discarded (fts3DbExec returns before using it), and the post-DROP state is
 * identical whether or not the injection landed in the window: drop_rc=7 and the
 * table still present for EVERY injection point from 1 to 292 on the host. An
 * allocator-level witness does not separate them either, because distinguishing
 * in-window from out-of-window means knowing whether fts3DestroyMethod was still on
 * the stack when the free happened, which the allocator cannot see.
 *
 * So this domain does not claim a per-point correspondence. It establishes that its
 * allocation sequence is the SAME sequence the host oracle swept, by reporting two
 * numbers that are fixed by SQLite's code path:
 *     - the DROP's total allocation count   (host: 292)
 *     - the transition point where drop_rc stops being SQLITE_NOMEM and becomes
 *       SQLITE_OK because the injection point is past the end (host: last 7 at 292,
 *       first 0 at 293)
 * If those match, the host's window indices transfer. The domain then sweeps every
 * injection point and runs to NOTRAP, which is the control-arm result: the
 * use-after-free is silent on unprotected Capstone.
 *
 * SQLITE_UNTESTABLE removes sqlite3_test_control, so OOM comes from a counting
 * wrapper installed over memsys5. SQLITE_CONFIG_HEAP sets sqlite3GlobalConfig.m to
 * memsys5's methods, SQLITE_CONFIG_GETMALLOC copies them out, and
 * SQLITE_CONFIG_MALLOC installs the wrapper -- so the order below matters and this
 * case does its own init instead of calling repro_init().
 * NOTE: -DSQLITE_DQS=0, so SQL string literals must be single-quoted. */
#include "repro322_common.h"

static sqlite3_mem_methods g_base;
static int g_armed = 0, g_nAlloc = 0, g_failAt = -1;

static void *wrapMalloc(int n){
  if( g_armed && ++g_nAlloc >= g_failAt ) return 0;
  return g_base.xMalloc(n);
}
static void *wrapRealloc(void *p, int n){
  if( g_armed && ++g_nAlloc >= g_failAt ) return 0;
  return g_base.xRealloc(p, n);
}

static void rcline(const char *what, int rc){
  out_text("fts3destroyoom "); out_text(what); out_text(" rc=");
  out_uint((unsigned)(rc<0?-rc:rc)); out_text("\n");
}

/* One iteration: fresh connection, create the fts3 table, then DROP it with the
 * allocator armed to fail at the g_failAt'th allocation and every one after.
 * Returns the DROP's rc; *pn gets the allocation count the DROP reached. */
static int one_drop(int failAt, int *pn, int *pCreateRc){
  sqlite3 *db = 0; char *e = 0;
  int rc = sqlite3_open(":memory:", &db);
  if( rc!=SQLITE_OK ){ if(db) sqlite3_close(db); *pn=0; *pCreateRc=rc; return -1; }
  int crc = sqlite3_exec(db, "CREATE VIRTUAL TABLE t1 USING fts3(a, b);", 0,0,&e);
  if(e){ sqlite3_free(e); e=0; }
  *pCreateRc = crc;
  if( crc!=SQLITE_OK ){ sqlite3_close(db); *pn=0; return -2; }

  g_nAlloc = 0; g_failAt = failAt; g_armed = 1;
  rc = sqlite3_exec(db, "DROP TABLE t1;", 0,0,&e);
  g_armed = 0;
  if(e){ sqlite3_free(e); e=0; }
  *pn = g_nAlloc;
  sqlite3_close(db);
  return rc;
}

static int run_case(void){
  /* own init: heap -> getmalloc -> malloc -> initialize */
  int rc = sqlite3_config(SQLITE_CONFIG_HEAP, sqlite_heap, (int)sizeof(sqlite_heap), 64);
  rcline("config-heap", rc);
  if(rc!=SQLITE_OK) return rc;
  rc = sqlite3_config(SQLITE_CONFIG_GETMALLOC, &g_base);
  rcline("getmalloc", rc);
  if(rc!=SQLITE_OK) return rc;
  if( g_base.xMalloc==0 ){ out_text("fts3destroyoom ERROR no memsys5 methods\n"); return 1; }
  sqlite3_mem_methods mm = g_base;
  mm.xMalloc = wrapMalloc; mm.xRealloc = wrapRealloc;
  rc = sqlite3_config(SQLITE_CONFIG_MALLOC, &mm);
  rcline("set-malloc", rc);
  if(rc!=SQLITE_OK) return rc;
  rc = sqlite3_initialize();
  rcline("initialize", rc);
  if(rc!=SQLITE_OK) return rc;

  /* self-check: the wrapper really is in the path (arm it at 1 and watch a
   * trivial allocation fail) */
  g_nAlloc = 0; g_failAt = 1; g_armed = 1;
  void *probe = sqlite3_malloc(64);
  g_armed = 0;
  out_text("fts3destroyoom wrapper active="); out_uint((unsigned)(probe==0)); out_text("\n");
  if(probe) sqlite3_free(probe);

  /* phase 1: unarmed -- how many allocations does the DROP make? */
  int n = 0, crc = 0;
  int base_rc = one_drop(1 << 30, &n, &crc);
  out_text("fts3destroyoom unarmed create_rc="); out_uint((unsigned)(crc<0?-crc:crc));
  out_text(" drop_rc="); out_uint((unsigned)(base_rc<0?-base_rc:base_rc));
  out_text(" drop_allocs="); out_uint((unsigned)n); out_text("\n");
  out_text("fts3destroyoom   (host oracle: drop_allocs=292)\n");
  if( base_rc!=SQLITE_OK ){
    out_text("fts3destroyoom WARNING unarmed DROP did not succeed; sweep is meaningless\n");
    out_text("fts3destroyoom NOTRAP done\n"); return 0;
  }

  /* phase 2: sweep every injection point, plus a margin past the end */
  int total = n, k;
  int limit = total + 8; if( limit > 512 ) limit = 512;
  int n_nomem = 0, n_ok = 0, n_other = 0, n_setupfail = 0;
  int last_nomem = -1, first_ok = -1;
  for(k = 1; k <= limit; k++){
    int nn = 0, cc = 0;
    int r = one_drop(k, &nn, &cc);
    if( cc!=SQLITE_OK ){ n_setupfail++; if(n_setupfail<=3){
        out_text("fts3destroyoom setup failed at k="); out_uint((unsigned)k);
        out_text(" create_rc="); out_uint((unsigned)(cc<0?-cc:cc)); out_text("\n"); }
      continue; }
    if( r==SQLITE_NOMEM ){ n_nomem++; last_nomem = k; }
    else if( r==SQLITE_OK ){ n_ok++; if(first_ok<0) first_ok = k; }
    else { n_other++; if(n_other<=3){
        out_text("fts3destroyoom unexpected rc at k="); out_uint((unsigned)k);
        out_text(" rc="); out_uint((unsigned)(r<0?-r:r)); out_text("\n"); } }
    if( (k % 64)==0 ){
      out_text("fts3destroyoom   k="); out_uint((unsigned)k);
      out_text(" mem_used="); out_uint((unsigned)sqlite3_memory_used()); out_text("\n");
    }
  }

  out_text("fts3destroyoom swept="); out_uint((unsigned)limit);
  out_text(" nomem="); out_uint((unsigned)n_nomem);
  out_text(" ok="); out_uint((unsigned)n_ok);
  out_text(" other="); out_uint((unsigned)n_other);
  out_text(" setupfail="); out_uint((unsigned)n_setupfail); out_text("\n");
  out_text("fts3destroyoom last_nomem_at="); out_uint((unsigned)(last_nomem<0?0:last_nomem));
  out_text(" first_ok_at="); out_uint((unsigned)(first_ok<0?0:first_ok)); out_text("\n");
  out_text("fts3destroyoom   (host oracle: last_nomem_at=292 first_ok_at=293)\n");
  out_text("fts3destroyoom   (host oracle UAF window: 167-184 and 259-280)\n");
  if( n_setupfail ) out_text("fts3destroyoom WARNING some iterations could not set up (arena leak?)\n");

  out_text("fts3destroyoom NOTRAP done\n");
  return 0;
}
REPRO322_MAIN("fts3destroyoom")
