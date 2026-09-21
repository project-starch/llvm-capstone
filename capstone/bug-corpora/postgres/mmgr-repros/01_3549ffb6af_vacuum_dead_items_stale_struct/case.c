#include "corpus.h"

PG_CASE(1) {
/* Row 2 -- vacuum dead_items, fix 3549ffb6af (2025-10-03). dead_items_reset
 * (vacuumlazy.c:2905) delegates to the parallel reset at :2911 and RETURNS at
 * :2912 without refreshing vacrel->dead_items, so the caller keeps pointing at
 * the store that was just destroyed and recreated. The upstream fix adds that
 * refresh before the return.
 *
 * The sibling at case 2 is a different defect at the same site: the callee's
 * own stale local, fixed ten months earlier. */
  unsigned char *ts = MemoryContextAlloc(pg_root, 64);
  ts[0] = 17;
  pg_held = ts; /* vacrel->dead_items, never updated */
  pfree(ts);    /* TidStoreDestroy */
  unsigned char *fresh = MemoryContextAlloc(pg_root, 64); /* TidStoreCreate */
  fresh[0] = 19;
  pg_mark();
  (void)pg_probe(pg_held);
}
