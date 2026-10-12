#include "corpus.h"

/* NEGATIVE CONTROL, under -DPG_NEGATIVE_CONTROL.
 *
 * SCHEMA rule 5: every oracle needs one. This arm reports a fault, and a
 * fault is only this case's result if it depends on the defect. The control
 * keeps the allocation and free traffic BYTE FOR BYTE and changes only the
 * one access that is invalid, so what is tested is whether the pattern alone
 * makes the arm fault. It must complete.
 *
 * Replacing the access rather than deleting it is deliberate: an arm that
 * faults because the sequence ran at all would still be let off by a control
 * that simply does less work.
 */

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
#ifdef PG_NEGATIVE_CONTROL
  (void)pg_probe(fresh); /* the LIVE replacement, same read, valid pointer */
#else
  (void)pg_probe(pg_held); /* the stale alias -- the defect */
#endif
}
