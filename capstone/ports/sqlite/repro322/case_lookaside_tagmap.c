/* The lookaside counterpart of case_mem5tagmap.c, and the reason the two differ matters.
 *
 * memsys5 keeps its freelist IN BAND as two ints:
 *     struct Mem5Link { int next; int prev; };            /_ 8 bytes of SCALAR _/
 * so freeing a block clears the capability tag of granule 0 (measured: exactly 1 granule
 * of up to 64). That is what incidentally catches 3 of this corpus's 7 faults.
 *
 * lookaside also keeps its freelist IN BAND, but as a POINTER:
 *     if( isLookaside(db, p) ){
 *       LookasideSlot *pBuf = (LookasideSlot*)p;
 *       pBuf->pNext = db->lookaside.pFree;                /_ a VALID capability _/
 *       db->lookaside.pFree = pBuf;
 *       return;                                           /_ never reaches memsys5 _/
 *     }
 * (the memset(0xaa) beside it is SQLITE_DEBUG only, and this port does not set it.)
 *
 * PREDICTION this case tests: a freed lookaside slot keeps a VALID tag at granule 0,
 * because what was written there is a real pointer to another slot. If so, a dangling
 * read of a freed lookaside slot does NOT fault on base Capstone -- it silently reaches
 * a different, perfectly valid slot. That is strictly worse than the memsys5 case and it
 * is the opposite of what catches the rtree/backup faults.
 *
 * HOW THE SLOTS ARE MADE OBSERVABLE. setupLookaside() accepts a caller-supplied buffer
 * (`pStart = pBuf`), so this case hands SQLite its own static array and therefore knows
 * every slot address exactly: pool + i*sz. It also writes pNext into offset 0 of every
 * slot at install time, building the init list -- so the tag map can be read immediately.
 *
 * Note the corpus default is -DSQLITE_DEFAULT_LOOKASIDE=0,0, i.e. lookaside OFF. This case
 * does not depend on that: sqlite3_db_config installs a pool at runtime regardless, and the
 * case reports sqlite3_db_status so the configuration is verified rather than assumed.
 * NOTE: -DSQLITE_DQS=0, so SQL string literals must be single-quoted. */
#include "repro322_common.h"

#define LA_SZ   128            /* ROUNDDOWN8, and must exceed sizeof(LookasideSlot*) */
#define LA_CNT  40
static unsigned char la_pool[LA_SZ * LA_CNT] __attribute__((aligned(16)));

static void kv(const char *k, unsigned long v){
  out_text("latag "); out_text(k); out_text("="); out_uint(v); out_text("\n");
}

/* a live allocation whose capability we can copy around */
static void *g_target;

/* Report the tag of granule 0 of each of the first n slots, plus a total. */
static void slot_tag_map(const char *label, int nslots){
  int i, ntag = 0;
  out_text("latag "); out_text(label); out_text(" granule0 tags: ");
  for(i = 0; i < nslots; i++){
    void **slot = (void **)(la_pool + (unsigned)i * LA_SZ);
    int t = __builtin_capstone_cap_get_tag(slot[0]) ? 1 : 0;
    if(t) ntag++;
    if(i < 10){ out_uint((unsigned)i); out_text(":"); out_uint((unsigned)t); out_text(" "); }
  }
  if(nslots > 10) out_text("...");
  out_text("\n");
  out_text("latag "); out_text(label); out_text(" tagged=");
  out_uint((unsigned)ntag); out_text(" of "); out_uint((unsigned)nslots); out_text("\n");
}

static int run_case(void){
  if (repro_init()) return 1;

  kv("sizeof_pointer", (unsigned long)sizeof(void *));
  kv("slot_size", LA_SZ);
  kv("slot_count", LA_CNT);

  g_target = sqlite3_malloc(64);
  if(!g_target){ out_text("latag ERROR target alloc failed\n"); return 1; }

  /* Baseline: every granule of the pool holds a valid capability of OURS. */
  {
    unsigned i, ngran = sizeof(la_pool) / sizeof(void *);
    void **p = (void **)la_pool;
    for(i = 0; i < ngran; i++) p[i] = g_target;
    slot_tag_map("baseline(ours)", LA_CNT);
  }

  sqlite3 *db = 0; char *e = 0;
  int rc = sqlite3_open(":memory:", &db);
  kv("open_rc", (unsigned long)(rc < 0 ? -rc : rc));
  if(rc != SQLITE_OK){ sqlite3_close(db); return 1; }

  /* Install OUR buffer as the lookaside pool. setupLookaside walks it and writes
   * pNext into offset 0 of every slot, so the next map shows what the allocator put
   * there rather than what we did. */
  rc = sqlite3_db_config(db, SQLITE_DBCONFIG_LOOKASIDE, la_pool, LA_SZ, LA_CNT);
  kv("db_config_lookaside_rc", (unsigned long)(rc < 0 ? -rc : rc));
  if(rc != SQLITE_OK){
    out_text("latag WARNING could not install the pool; the rest establishes nothing\n");
    sqlite3_close(db); out_text("latag NOTRAP done\n"); return 0;
  }

  /* THE MEASUREMENT. Every slot is on the free/init list right now, so granule 0 of each
   * holds the allocator's pNext. memsys5 in the same state has tag 0 there. */
  slot_tag_map("after-install(allocator pNext)", LA_CNT);

  /* Verify the pool is really serving allocations, not just installed. */
  rc = sqlite3_exec(db, "CREATE TABLE t(a,b);", 0,0,&e);
  kv("setup_rc", (unsigned long)(rc < 0 ? -rc : rc));
  if(e){ sqlite3_free(e); e = 0; }
  {
    int cur = 0, hi = 0;
    sqlite3_db_status(db, SQLITE_DBSTATUS_LOOKASIDE_USED, &cur, &hi, 0);
    out_text("latag lookaside slots_in_use="); out_uint((unsigned)(cur<0?0:cur));
    out_text(" high_water="); out_uint((unsigned)(hi<0?0:hi));
    out_text(hi > 0 ? "  (ACTIVE)\n" : "  (NOT SERVING -- the rest establishes nothing)\n");
    if(hi == 0){ sqlite3_close(db); out_text("latag NOTRAP done\n"); return 0; }
  }

  /* Put OUR OWN bytes into a slot through a public API: binding a small text value with
   * SQLITE_TRANSIENT goes sqlite3VdbeMemGrow -> sqlite3DbMallocRaw -> a lookaside slot. */
  sqlite3_stmt *st = 0;
  rc = sqlite3_prepare_v2(db, "INSERT INTO t VALUES(?1,?2)", -1, &st, 0);
  kv("prepare_rc", (unsigned long)(rc < 0 ? -rc : rc));
  if(rc == SQLITE_OK){
    int k;
    for(k = 0; k < 8; k++){
      sqlite3_bind_text(st, 1, "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", 32, SQLITE_TRANSIENT);
      sqlite3_bind_text(st, 2, "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB", 32, SQLITE_TRANSIENT);
      sqlite3_step(st);
      sqlite3_reset(st);          /* releases the Mem buffers back into the pool */
    }
    sqlite3_finalize(st);
  }

  /* After all that traffic every slot is back on the free list, so granule 0 again holds
   * an allocator pNext -- written by sqlite3DbFreeNN this time, not by setupLookaside. */
  slot_tag_map("after-use-and-free", LA_CNT);

  /* THE DANGLING READ.
   *
   * NOT slot 0. setupLookaside builds the list from the LOW address upward:
   *     p = pStart;
   *     for(i=cnt-1; i>=0; i--){ p->pNext = pInit; pInit = p; p = &p[sz]; }
   * so the very first slot it touches gets pNext = 0, i.e. slot 0 is the list TAIL and
   * holds a NULL capability, whose tag is 0. It is the one slot out of 40 that is
   * untagged, and reading it would wrongly suggest the memsys5 shape. Pick a middle slot.
   *
   * Query the tag before dereferencing so this case returns either way rather than dying
   * on a bad load. */
  {
    int iSlot = LA_CNT / 2;
    void **slot = (void **)(la_pool + (unsigned)iSlot * LA_SZ);
    void *stale = slot[0];
    int tagged = __builtin_capstone_cap_get_tag(stale) ? 1 : 0;
    out_text("latag dangling slot"); out_uint((unsigned)iSlot);
    out_text(" granule0 tagged="); out_uint((unsigned)tagged); out_text("\n");
    { void *tail = ((void **)la_pool)[0];
      out_text("latag   (slot0 is the list tail, NULL, tag=");
      out_uint((unsigned)(__builtin_capstone_cap_get_tag(tail) ? 1 : 0));
      out_text(" -- the single exception)\n"); }
    if(tagged){
      /* Safe BECAUSE it is tagged: a freed lookaside slot's first word is a real pointer
       * to another slot, so this silently reads a different, valid object. That is the
       * whole point -- memsys5 would have faulted here. */
      unsigned char *q = (unsigned char *)stale;
      unsigned sum = 0, i;
      for(i = 0; i < 16; i++) sum += q[i];
      out_text("latag dereferenced the stale pointer WITHOUT faulting, byte-sum=");
      out_uint(sum);
      out_text("  <== a freed lookaside slot aliases another LIVE slot\n");
    } else {
      out_text("latag the stale pointer is UNTAGGED, so a dereference would fault"
               " -- same shape as memsys5\n");
    }
  }

  sqlite3_close(db);
  sqlite3_free(g_target);
  out_text("latag NOTRAP done\n"); return 0;
}
REPRO322_MAIN("latag")
