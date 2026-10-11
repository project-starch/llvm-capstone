#include "corpus.h"

PG_CASE(2) {
/* Row 3 -- vacuum dead_items, fix 83ce20d671 (2024-12-04), in the CALLEE.
 * parallel_vacuum_reset_dead_items (vacuumparallel.c:473) holds a local
 * `dead_items`, destroys the store through it at :483 -- TidStoreDestroy
 * deletes the store's child context at tidstore.c:340 and pfrees the struct at
 * :342 -- recreates into pvs->dead_items at :484, and then reads the destroyed
 * local twice at :488-489.
 *
 * Reduced to that order: delete the child context, free the struct, reallocate
 * the same size class, read the stale local. */
  MemoryContext rt = pg_aset_child(pg_root, "TID storage");
  unsigned char *ts = MemoryContextAlloc(pg_root, 64);
  ts[0] = 23;
  pg_held = ts;
  MemoryContextDelete(rt); /* MemoryContextDelete(ts->rt_context) */
  pfree(ts);               /* pfree(ts) */
  unsigned char *fresh = MemoryContextAlloc(pg_root, 64);
  fresh[0] = 29;
  pg_mark();
  (void)pg_probe(pg_held);
}
