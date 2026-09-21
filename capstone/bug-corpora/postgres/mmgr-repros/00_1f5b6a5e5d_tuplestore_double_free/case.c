#include "corpus.h"

PG_CASE(0) {
/* Row 1 -- tuplestore, bug #19438, fix 1f5b6a5e5d.
 * dumptuples() frees tuples as it writes them but clears memtupdeleted only
 * after the loop, so a WRITETUP that throws leaves memtuples[] holding freed
 * chunks; tuplestore_end then frees them a second time.
 *
 * The stale access is the second pfree itself, so there is no read probe: the
 * manager faults reading the revoked chunk's header before any bookkeeping
 * runs. This case's oracle therefore accepts a fault anywhere, unlike the
 * others. */
  unsigned char *memtuples[2];
  memtuples[0] = MemoryContextAlloc(pg_root, 64);
  memtuples[1] = MemoryContextAlloc(pg_root, 64);
  memtuples[0][0] = 11;
  pfree(memtuples[0]); /* WRITETUP freed it, then threw */
  pg_mark();
  pfree(memtuples[0]); /* the end walk, from index 0 */
  pg_held = memtuples[1];
}
