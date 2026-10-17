/* case_spellfix_oom.c -- CheriBSD version. Upstream fix 2026-07-26: spellfix1Register's
 * last step, editDist3Install, frees the shared pConfig when its create_function_v2
 * fails -- while editdist3/2, registered a moment earlier on that same pConfig, stays
 * registered. Calling editdist3('kitten','sitting') then reads the freed config in
 * editDist3FindLang. Bug state = init returns SQLITE_NOMEM AND editdist3/2 is still
 * preparable; both halves together, neither alone.
 *
 * WHY THIS DIFFERS FROM THE CAPSTONE VERSION.
 * That one starved the arena: it drained to about k*64 free bytes and swept k = 1..40.
 * On CheriBSD that found nothing, and the reason is not the range -- widening the drain
 * to 96 KiB in 64-byte steps showed a SHARP transition with no window at all:
 *     free <= 1536 B  ->  rc=SQLITE_NOMEM, callable=0   (nothing registered)
 *     free >= 1600 B  ->  rc=SQLITE_OK,    callable=1   (everything registered)
 * memsys5 is a buddy allocator, so a drained arena leaves one coalesced power-of-two
 * region and the ALLOCATABLE capacity jumps in powers of two. Adding 64 free bytes does
 * not buy "one more allocation", so no amount of total-free-bytes tuning lands between
 * "(2) succeeded" and "(3) failed".
 *
 * So this version uses the lever that gives exact control, the same counting-wrapper
 * technique case_fts3_destroy_oom.c uses: fail the k'th allocation (and every one after)
 * and sweep k over the init's whole allocation range. The bug and the trigger are
 * unchanged; only the OOM mechanism is, from "arena pressure" to "fail allocation k".
 *
 * SQLITE_UNTESTABLE removes sqlite3_test_control, so the wrapper goes over memsys5:
 * CONFIG_HEAP installs memsys5, CONFIG_GETMALLOC copies its methods out, CONFIG_MALLOC
 * installs the wrapper -- order matters, so this case does its own init, not repro_init().
 * NOTE: -DSQLITE_DQS=0, so SQL string literals must be single-quoted.
 */
#include "repro322_common.h"

int sqlite3_spellfix_init(sqlite3*, char**, const void*);

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
static void wrapFree(void *p){ g_base.xFree(p); }
static int  wrapSize(void *p){ return g_base.xSize(p); }
static int  wrapRoundup(int n){ return g_base.xRoundup(n); }
static int  wrapInit(void *a){ return g_base.xInit(a); }
static void wrapShutdown(void *a){ g_base.xShutdown(a); }

/* one init attempt with the allocator armed to fail at the failAt'th allocation.
 * *pn gets the allocation count the init reached; *pcallable whether editdist3/2
 * survived. Returns 1 when this is the bug state. */
static int attempt(int failAt, int *prc, int *pn, int *pcallable, int do_call, int *prows){
  sqlite3 *db = 0;
  if( sqlite3_open(":memory:", &db)!=SQLITE_OK ){ sqlite3_close(db); return -1; }

  g_nAlloc = 0; g_failAt = failAt; g_armed = 1;
  int rc = sqlite3_spellfix_init(db, 0, 0);
  g_armed = 0;
  int n = g_nAlloc;

  /* disarm before probing: prepare() needs memory, otherwise "not callable" would be
   * reported for the wrong reason. */
  sqlite3_stmt *st = 0;
  int callable = (sqlite3_prepare_v2(db, "SELECT editdist3('kitten','sitting')", -1, &st, 0)==SQLITE_OK);

  if(prc) *prc = rc; if(pn) *pn = n; if(pcallable) *pcallable = callable;
  if(prows) *prows = 0;
  int is_bug = (rc==SQLITE_NOMEM && callable);
  if( is_bug && do_call ){
    /* prepare() does not invoke the function, so editDist3FindLang can only be
     * reached here -- over the freed pConfig. */
    int rows = 0; while( sqlite3_step(st)==SQLITE_ROW ) rows++;
    if(prows) *prows = rows;
  }
  if(st) sqlite3_finalize(st);
  sqlite3_close(db);
  return is_bug;
}

static int run_case(void){
  int rc = sqlite3_config(SQLITE_CONFIG_HEAP, sqlite_heap, (int)sizeof(sqlite_heap), 64);
  if( rc!=SQLITE_OK ){ out_text("spellfixoom config-heap rc="); out_uint((unsigned)rc); out_text("\n"); return rc; }
  /* Lookaside off: a lookaside slot never reaches the wrapper, so with it on the allocation
   * whose failure leaves editdist3 registered over a freed config cannot be failed (8 of the
   * init's 16 allocations reach the wrapper on Capstone, and the sweep finds no bug state). */
  rc = sqlite3_config(SQLITE_CONFIG_LOOKASIDE, 0, 0);
  if( rc!=SQLITE_OK ){ out_text("spellfixoom lookaside-off rc="); out_uint((unsigned)rc); out_text("\n"); return rc; }
  rc = sqlite3_config(SQLITE_CONFIG_GETMALLOC, &g_base);
  if( rc!=SQLITE_OK ){ out_text("spellfixoom getmalloc rc="); out_uint((unsigned)rc); out_text("\n"); return rc; }
  sqlite3_mem_methods w; w.xMalloc=wrapMalloc; w.xFree=wrapFree; w.xRealloc=wrapRealloc;
  w.xSize=wrapSize; w.xRoundup=wrapRoundup; w.xInit=wrapInit; w.xShutdown=wrapShutdown;
  w.pAppData=g_base.pAppData;
  rc = sqlite3_config(SQLITE_CONFIG_MALLOC, &w);
  if( rc!=SQLITE_OK ){ out_text("spellfixoom set-malloc rc="); out_uint((unsigned)rc); out_text("\n"); return rc; }
  rc = sqlite3_initialize();
  if( rc!=SQLITE_OK ){ out_text("spellfixoom initialize rc="); out_uint((unsigned)rc); out_text("\n"); return rc; }

  /* wrapper live? arm at 1 and expect the very next malloc to fail */
  g_nAlloc=0; g_failAt=1; g_armed=1;
  void *probe = sqlite3_malloc(64);
  g_armed=0;
  out_text("spellfixoom wrapper active="); out_uint((unsigned)(probe==0)); out_text("\n");
  if(probe) sqlite3_free(probe);

  /* phase 1: unarmed -- how many allocations does a full init make? */
  int n_unarmed=0, cal=0;
  (void)attempt(1<<30, &rc, &n_unarmed, &cal, 0, 0);
  out_text("spellfixoom unarmed init rc="); out_uint((unsigned)(rc<0?-rc:rc));
  out_text(" allocs="); out_uint((unsigned)n_unarmed);
  out_text(" callable="); out_uint((unsigned)cal); out_text("\n");
  if( rc!=SQLITE_OK || !cal ){
    out_text("spellfixoom WARNING unarmed init did not fully succeed; sweep is meaningless\n");
    out_text("spellfixoom NOTRAP done\n"); return 0;
  }

  /* phase 2: fail allocation k and every one after, for every k in the init's range */
  int k, hit=0, limit=n_unarmed+4, n_nomem=0, n_ok=0, n_cal=0;
  for(k=1; k<=limit && !hit; k++){
    int n=0, callable=0, rows=0;
    int r = attempt(k, &rc, &n, &callable, 0, 0);
    if(r<0) continue;
    if(rc==SQLITE_NOMEM) n_nomem++; else if(rc==SQLITE_OK) n_ok++;
    if(callable) n_cal++;
    if(r==1){
      /* Print the probe BEFORE the faulting call. The freed-config read may trap, and a
       * marker emitted afterwards would be lost -- which is exactly what happened on the
       * first CheriBSD run of this case: it faulted with no record of which failAt hit. */
      out_text("spellfixoom BUG STATE at failAt="); out_uint((unsigned)k);
      out_text(" rc=NOMEM callable=1 (init allocs="); out_uint((unsigned)n);
      out_text("); now calling editdist3 over the freed config\n");
      (void)attempt(k, &rc, &n, &callable, 1, &rows);
      out_text("spellfixoom editdist3 rows="); out_uint((unsigned)rows);
      out_text(" (returned, so the freed read did not trap)\n");
      hit=1;
    }
  }
  out_text("spellfixoom swept="); out_uint((unsigned)(limit));
  out_text(" nomem="); out_uint((unsigned)n_nomem);
  out_text(" ok="); out_uint((unsigned)n_ok);
  out_text(" callable="); out_uint((unsigned)n_cal); out_text("\n");
  if(!hit) out_text("spellfixoom NO BUG STATE FOUND in allocation-index sweep\n");
  out_text("spellfixoom NOTRAP done\n");
  return 0;
}
REPRO322_MAIN("spellfixoom")
