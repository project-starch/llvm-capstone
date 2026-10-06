#include "corpus.h"

PG_CASE(4) {
/* Row 5 -- free_child_join_sjinfo, bug #18806, fix 727bc6ac33f6. The child
 * SpecialJoinInfo shares its relid sets with the parent and frees them
 * unconditionally, once per partition pair. */
  unsigned char *relids = MemoryContextAlloc(pg_root, 64);
  relids[0] = 41;
  unsigned char *child = relids; /* the child's copy of the pointer */
  pg_held = relids;              /* the parent still owns this */
  pfree(child);                  /* bms_free, in the child's cleanup */
  unsigned char *fresh = MemoryContextAlloc(pg_root, 64);
  fresh[0] = 43;
  pg_mark();
  (void)pg_probe(pg_held);
}
