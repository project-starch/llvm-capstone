#include "corpus.h"

/* NEGATIVE CONTROL, under -DPG_NEGATIVE_CONTROL.
 *
 * SCHEMA rule 5: every oracle needs one. This arm reports a fault, and a
 * fault is only this case's result if it depends on the defect. The control
 * keeps the context teardown and changes the one access that is invalid, so
 * what is tested is whether the pattern alone makes the arm fault. It must
 * complete.
 *
 * UNLIKE THE OTHER FOUR, this control is not byte-for-byte the same traffic:
 * the case reads a pointer into the context it just deleted, and there is no
 * surviving object to read instead, so the control allocates one. It is one
 * extra allocation from pg_root, after the teardown. Everything the defect
 * depends on -- the grandchild context, the allocation inside it, the delete
 * that takes it -- happens identically in both.
 *
 * Replacing the access rather than deleting it is deliberate: an arm that
 * faults because the sequence ran at all would still be let off by a control
 * that simply does less work.
 */

PG_CASE(3) {
/* Row 7 -- pgoutput, fix a61592253e. entry_cxt is a grandchild of the decoding
 * context; an error tears that down, while RelationSyncCache lives in
 * CacheMemoryContext and keeps pointing into the dead arena. */
  MemoryContext decoding = pg_aset_child(pg_root, "logical decoding");
  MemoryContext entry_cxt = pg_aset_child(decoding, "entry");
  unsigned char *filter = MemoryContextAlloc(entry_cxt, 64);
  filter[0] = 59;
  pg_held = filter;              /* the cache entry's pointer */
  MemoryContextDelete(decoding); /* the error path, taking the grandchild */
#ifdef PG_NEGATIVE_CONTROL
  /* The same teardown, then a read of something the teardown did not reach.
   * pg_root outlives it, so this allocation is valid where filter is not. */
  unsigned char *live = MemoryContextAlloc(pg_root, 64);
  live[0] = 59;
  pg_mark();
  (void)pg_probe(live);
#else
  pg_mark();
  (void)pg_probe(pg_held); /* into the deleted arena -- the defect */
#endif
}
