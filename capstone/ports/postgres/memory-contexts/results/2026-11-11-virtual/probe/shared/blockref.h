#include "corpus.h"
#include <stdio.h>
/* The allocator's own reference to the single-chunk block of a large AllocSet chunk: the one
 * malloc returned, which AllocSetAllocLarge links right behind the head of the context's block
 * list (set->blocks->next). AllocSetContext keeps `blocks` directly after its header, and an
 * AllocBlockData starts with aset, prev, next. */
static inline void *block_reference(MemoryContext context, void *chunk_pointer) {
  void **blocks = (void **)((char *)context + TYPEALIGN(sizeof(void *), sizeof(MemoryContextData)));
  void **head = *blocks;
  void *block = head[2];
  CHECK(block != NULL, 1);
  CHECK((uintptr_t)block < (uintptr_t)chunk_pointer, 2);
  CHECK((uintptr_t)chunk_pointer - (uintptr_t)block <= 256, 3);
  return block;
}
