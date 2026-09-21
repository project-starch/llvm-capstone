#include "corpus.h"

PG_CASE(5) {
/* Row 6 -- WindowAgg, fix 9d5ce4f1a00a. release_partition resets the
 * partition context in bulk; the top-level branch does not NULL the by-ref
 * results, so ecxt_aggvalues[] keeps pointing into it. The array itself lives
 * longer, which is what makes the stale read reachable. */
  MemoryContext partcontext = pg_aset_child(pg_root, "WindowAgg Partition");
  unsigned char *value = MemoryContextAlloc(partcontext, 64);
  value[0] = 47;
  pg_held = value; /* econtext->ecxt_aggvalues[wfuncno] */
  MemoryContextReset(partcontext); /* release_partition */
  unsigned char *fresh = MemoryContextAlloc(partcontext, 64);
  fresh[0] = 53;
  pg_mark();
  (void)pg_probe(pg_held);
}
