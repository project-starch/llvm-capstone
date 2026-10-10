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

PG_CASE(4) {
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
#ifdef PG_NEGATIVE_CONTROL
  /* successor is the SAME ADDRESS the CHECK above just proved came back, read
   * through the live capability instead of the stale one. On a temporal arm
   * that is the sharpest control there is: identical address, valid issue. */
  (void)pg_probe(successor);
#else
  (void)pg_probe(pg_held); /* the stale cursor -- the defect */
#endif
}
