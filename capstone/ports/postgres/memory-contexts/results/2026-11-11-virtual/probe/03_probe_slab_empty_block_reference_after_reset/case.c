#include "slabref.h"
/* The same with every chunk pfree'd: the empty block goes onto the context's list of empty blocks, again through
 * SlabFree's block pointer, and SlabReset frees it from there. */
PG_CASE(3) {
  static void *held[64];
  MemoryContext s = SlabContextCreate(pg_root, "slab", 1024, 80);
  MemoryContext other = SlabContextCreate(pg_root, "neighbour", 1024, 80);
  int n = slab_chunks_per_block(s, 1024, 80);
  for (int i = 0; i < n; i++) held[i] = MemoryContextAlloc(s, 80);
  ((unsigned char *)MemoryContextAlloc(other, 80))[0] = 9;
  unsigned char *block = slab_block_reference(s, held[0]);
  (void)pg_probe(block);
  printf("PROBE block reference read before pfree and reset\n");
  for (int i = 0; i < n; i++) pfree(held[i]);
  MemoryContextReset(s);
  pg_held = block;
  pg_mark();
  (void)pg_probe(pg_held);
  printf("PROBE block reference STILL LIVE after reset\n");
}
