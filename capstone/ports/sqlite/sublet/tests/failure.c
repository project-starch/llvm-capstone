/* Include the patched amalgamation to inspect the allocator's own lists. */
#include "sqlite3-capstone.c"
#include <assert.h>
extern int life_fail, life_calls, life_renew_fail;
static unsigned char heap[1<<20] __attribute__((aligned(2048)));
static unsigned char slots[512] __attribute__((aligned(16)));
int main(int argc, char **argv){
  sqlite3 db = {0};
  void *p, *q;
  LookasideSlot *head;
  unsigned count;
  (void)argv;
  sqlite3_config(SQLITE_CONFIG_HEAP, heap, sizeof heap, 32);
  life_calls = 0; life_fail = 1;
  assert(memsys5Init(0)==SQLITE_NOMEM);
  life_fail = 0;
  assert(sqlite3_initialize()==SQLITE_OK);
  p = memsys5Malloc(64);
  assert(p); memset(p, 42, 64);
  if(argc>1){
    memsys5Free(p);
    return *(volatile unsigned char *)p;
  }
  /* The next derive fails before the buddy lists change. */
  count = mem5.currentCount;
  life_fail = life_calls + 1;
  assert(memsys5Malloc(128)==0);
  assert(mem5.currentCount==count);
  life_fail = life_calls + 1;
  assert(memsys5Realloc(p, 256)==0);
  assert(*(unsigned char *)p==42);
  assert(mem5.currentCount==count);
  life_fail = 0;
  q = memsys5Malloc(128); assert(q); memsys5Free(q);
  life_renew_fail = 1;
  assert(memsys5Realloc(p, 64)==0);
  assert(*(unsigned char *)p==42);
  life_renew_fail = 0;
  memsys5Free(p);
  assert(setupLookaside(&db, slots, 64, 8)==SQLITE_OK);
  head = db.lookaside.pInit;
  life_fail = life_calls + 1;
  assert(sqlite3DbMallocRawNN(&db, 32)==0);
  assert(db.mallocFailed && db.lookaside.pInit==head);
  sqlite3OomClear(&db); life_fail = 0;
  p = sqlite3DbMallocRawNN(&db, 32); assert(p);
  life_renew_fail = 1;
  assert(sqlite3DbRealloc(&db, p, 32)==0);
  assert(db.mallocFailed);
  life_renew_fail = 0; sqlite3OomClear(&db);
  sqlite3DbFree(&db, p);
  head = db.lookaside.pFree;
  life_fail = life_calls + 1;
  assert(sqlite3DbMallocRawNN(&db, 32)==0);
  assert(db.mallocFailed && db.lookaside.pFree==head);
  sqlite3OomClear(&db); life_fail = 0;
  sqlite3LookasideEnd(&db);
  life_fail = life_calls + 1;
  assert(setupLookaside(&db, slots, 64, 8)==SQLITE_OK);
  assert(db.lookaside.bDisable && !db.lookaside.pLife);
  count = mem5.currentCount;
  life_fail = life_calls + 2;
  assert(setupLookaside(&db, 0, 64, 8)==SQLITE_OK);
  assert(db.lookaside.bDisable && !db.lookaside.bMalloced);
  assert(mem5.currentCount==count);
  life_fail = 0;
#ifdef TEST_PAGE
  {
    PCache1 *cache = (PCache1 *)pcache1Create(1024, 64, 1);
    PgHdr1 *page;
    assert(cache); pcache1Cachesize((sqlite3_pcache *)cache, 10);
    /* Allocation may derive backing blocks first. Fail the assignment itself. */
    page = pcache1FetchNoMutex((sqlite3_pcache *)cache, 1, 2);
    assert(page); pcache1Unpin((sqlite3_pcache *)cache, &page->page, 1);
    life_fail = life_calls + 1;
    assert(pcache1FetchNoMutex((sqlite3_pcache *)cache, 2, 2)==0);
    assert(cache->nPage==0);
    life_fail = 0;
    page = pcache1FetchNoMutex((sqlite3_pcache *)cache, 2, 2);
    assert(page && page->page.pBuf);
    pcache1Unpin((sqlite3_pcache *)cache, &page->page, 1);
    pcache1Destroy((sqlite3_pcache *)cache);
  }
#endif
  sqlite3_shutdown();
  puts("LIFETIME sqlite PASS");
  return 0;
}
