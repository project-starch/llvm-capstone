#include "corpus.h"

PG_CASE(6) {
/* Row 7 -- pgoutput, fix a61592253e. entry_cxt is a grandchild of the decoding
 * context; an error tears that down, while RelationSyncCache lives in
 * CacheMemoryContext and keeps pointing into the dead arena. */
  MemoryContext decoding = pg_aset_child(pg_root, "logical decoding");
  MemoryContext entry_cxt = pg_aset_child(decoding, "entry");
  unsigned char *filter = MemoryContextAlloc(entry_cxt, 64);
  filter[0] = 59;
  pg_held = filter;              /* the cache entry's pointer */
  MemoryContextDelete(decoding); /* the error path, taking the grandchild */
  pg_mark();
  (void)pg_probe(pg_held);
}
