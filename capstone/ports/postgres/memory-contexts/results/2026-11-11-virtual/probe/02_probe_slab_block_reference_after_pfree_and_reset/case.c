#include "slabref.h"
/* A Slab block is filled, so it sits on the full list. One chunk is pfree'd: SlabFree moves the block to another
 * list, through the block pointer it derived from the chunk. Then the context is reset: SlabReset frees the block
 * through the list link. Is the reference malloc returned for the block still live? */
PG_CASE(2) {
  static void *held[64];
  MemoryContext s = SlabContextCreate(pg_root, "slab", 1024, 80);
  MemoryContext other = SlabContextCreate(pg_root, "neighbour", 1024, 80);
  int n = slab_chunks_per_block(s, 1024, 80);
  for (int i = 0; i < n; i++) held[i] = MemoryContextAlloc(s, 80);
  ((unsigned char *)MemoryContextAlloc(other, 80))[0] = 9;   /* keeps a 1 KiB malloc neighbour in use */
  unsigned char *block = slab_block_reference(s, held[0]);
  (void)pg_probe(block);
  printf("PROBE block reference read before pfree and reset\n");
  pfree(held[0]);
  MemoryContextReset(s);
  pg_held = block;
  pg_mark();
  (void)pg_probe(pg_held);
  printf("PROBE block reference STILL LIVE after reset\n");
}
