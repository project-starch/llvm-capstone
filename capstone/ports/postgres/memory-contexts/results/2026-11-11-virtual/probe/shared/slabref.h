#include "corpus.h"
#include "lib/ilist.h"
#include <stdio.h>
/* The link the Slab context holds to the block of `chunk`: &block->node, as the list it sits on stores it. Taken
 * before any pfree it is the pointer malloc returned for the block, moved to the node. SlabContext is private to
 * slab.c; its fields up to the lists are mirrored here (release layout, no MEMORY_CONTEXT_CHECKING) and checked
 * against the sizes the context was created with. */
typedef struct SlabMirror {
  MemoryContextData header;
  uint32 chunkSize, fullChunkSize, blockSize;
  int32 chunksPerBlock, curBlocklistIndex, blocklist_shift;
  dclist_head emptyblocks;
  dlist_head blocklist[3];
} SlabMirror;
static inline int slab_chunks_per_block(MemoryContext s, Size block_size, Size chunk_size) {
  SlabMirror *m = (SlabMirror *)s;
  CHECK(m->blockSize == block_size && m->chunkSize == chunk_size, 1);
  CHECK(m->chunksPerBlock > 1 && m->chunksPerBlock < 64, 2);
  return m->chunksPerBlock;
}
static inline void *slab_block_reference(MemoryContext s, void *chunk) {
  SlabMirror *m = (SlabMirror *)s;
  dlist_head *lists[4] = {&m->emptyblocks.dlist, &m->blocklist[0], &m->blocklist[1], &m->blocklist[2]};
  for (int i = 0; i < 4; i++) {
    dlist_iter it;
    dlist_foreach(it, lists[i]) {
      uintptr_t node = (uintptr_t)it.cur;
      if (node < (uintptr_t)chunk && (uintptr_t)chunk - node < m->blockSize)
        return it.cur;
    }
  }
  pg_give_up(3);
}
