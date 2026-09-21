#include "corpus.h"

PG_CASE(7) {
/* Row 8 -- reorderbuffer, fix 9e0b4b1ab5. The change record comes from a Slab
 * context, so the free list is LIFO with one chunk size and the next
 * allocation returns the identical address every time.
 *
 * The CHECK that reuse really happened runs BEFORE the marker, so the marker's
 * presence is itself evidence that the same address came back. */
  MemoryContext change_context = SlabContextCreate(
      pg_root, "Change", SLAB_DEFAULT_BLOCK_SIZE, PG_CHANGE_BYTES);
  unsigned char *specinsert =
      MemoryContextAlloc(change_context, PG_CHANGE_BYTES);
  specinsert[0] = 61;
  pg_held = specinsert; /* change = specinsert, the loop cursor */
  pfree(specinsert);    /* ReorderBufferReturnChange */
  unsigned char *successor =
      MemoryContextAlloc(change_context, PG_CHANGE_BYTES);
  successor[0] = 67;
  CHECK(successor == (unsigned char *)pg_held, 0xbad90004);
  pg_mark();
  (void)pg_probe(pg_held);
}
