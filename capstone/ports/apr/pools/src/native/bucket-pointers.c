/* Hosted and CheriBSD: the bucket allocator's seam with plain pointers and no
 * authority. The same records and the same LIFO as the domain adapter, so
 * the native arms carve the same pieces in the same order; a piece keeps its
 * address for the run, which is what upstream gives every consumer. */
#include "port.h"
#include <capstone/capability-slot.h>
#include "apr_shim.h"
#include "apr_allocator.h"

#define APRB_BLOCKS 8192
#define APRB_PIECES 65536
#define APRB_LISTS 256

struct block {
  uintptr_t address, cursor, end;
  void *header;
  unsigned live;
};
struct piece {
  uintptr_t address;
  size_t size;
  int block, next_free;
  unsigned live, filed;
};
struct list {
  uintptr_t address;
  int head;
};
static struct block *blocks;
static struct piece *pieces;
static struct list *lists;
static unsigned nblocks, npieces, nlists;
static unsigned long carved, reissues, files;

static void ensure(void) {
  if (blocks)
    return;
  blocks = aprp_meta_calloc(APRB_BLOCKS, sizeof *blocks);
  pieces = aprp_meta_calloc(APRB_PIECES, sizeof *pieces);
  lists = aprp_meta_calloc(APRB_LISTS, sizeof *lists);
  if (!blocks || !pieces || !lists)
    aprp_fail(530);
}
static struct block *block_for(const void *header) {
  uintptr_t address = (uintptr_t)header;
  for (unsigned i = nblocks; i-- > 0;)
    if (blocks[i].live && blocks[i].address == address)
      return &blocks[i];
  aprp_fail(531);
}
static struct piece *piece_for(const void *node) {
  uintptr_t address = (uintptr_t)node;
  for (unsigned i = npieces; i-- > 0;)
    if (pieces[i].live && pieces[i].address == address)
      return &pieces[i];
  aprp_fail(532);
}
static struct list *list_for(const void *list) {
  uintptr_t address = (uintptr_t)list;
  for (unsigned i = 0; i < nlists; ++i)
    if (lists[i].address == address)
      return &lists[i];
  if (nlists == APRB_LISTS)
    aprp_fail(533);
  struct list *l = &lists[nlists++];
  l->address = address;
  l->head = -1;
  return l;
}
void *aprb_block_lend(void *block) {
  ensure();
  if (nblocks == APRB_BLOCKS)
    aprp_fail(534);
  struct block *b = &blocks[nblocks++];
  capstone_cap_slot unused;
  b->header = aprp_node_lend(block, &unused);
  b->address = (uintptr_t)b->header;
  apr_memnode_t *header = b->header;
  b->cursor = (uintptr_t)header->first_avail;
  b->end = (uintptr_t)header->endp;
  if (b->cursor != b->address + APR_MEMNODE_T_SIZE || b->end <= b->cursor)
    aprp_fail(535);
  b->live = 1;
  return b->header;
}
void *aprb_carve(void *block, size_t size) {
  struct block *b = block_for(block);
  if (!size || (size & 15) || size > b->end - b->cursor)
    aprp_fail(536);
  if (npieces == APRB_PIECES)
    aprp_fail(537);
  struct piece *p = &pieces[npieces++];
  p->address = b->cursor;
  p->size = size;
  p->block = (int)(b - blocks);
  p->next_free = -1;
  b->cursor += size;
  apr_memnode_t *header = b->header;
  header->first_avail = (char *)b->header + (b->cursor - b->address);
  p->live = 1;
  ++carved;
  return (void *)p->address;
}
void aprb_file(void *list, void *node) {
  struct piece *p = piece_for(node);
  if (p->filed)
    aprp_fail(538);
  struct list *l = list_for(list);
  p->filed = 1;
  p->next_free = l->head;
  l->head = (int)(p - pieces);
  ++files;
}
void *aprb_reissue(void *list, apr_memnode_t **memnode) {
  struct list *l = list_for(list);
  if (l->head < 0)
    return NULL;
  struct piece *p = &pieces[l->head];
  l->head = p->next_free;
  p->next_free = -1;
  p->filed = 0;
  *memnode = blocks[p->block].header;
  ++reissues;
  return (void *)p->address;
}
void aprb_blocks_returning(void *chain) {
  for (apr_memnode_t *header = chain; header; header = header->next) {
    struct block *b = block_for(header);
    for (unsigned i = 0; i < npieces; ++i)
      if (pieces[i].live && pieces[i].block == (int)(b - blocks))
        pieces[i].live = 0;
    b->live = 0;
  }
  for (unsigned i = 0; i < nlists; ++i) {
    int *link = &lists[i].head;
    while (*link >= 0) {
      if (!pieces[*link].live)
        *link = pieces[*link].next_free;
      else
        link = &pieces[*link].next_free;
    }
  }
}
/* The labelled read, a capability-base load on CheriBSD so a revoked tag
 * faults here; a plain load elsewhere. */
void *aprb_probe(void *mem) {
#if defined(__CHERI_PURE_CAPABILITY__)
  unsigned long value;
  void *out;
  __asm__ volatile(".globl aprb_free_probe\naprb_free_probe:\nclbu %0, 0(%2)\ncmove %1, %2\n"
                   : "=&r"(value), "=C"(out)
                   : "C"(mem)
                   : "memory");
  return out;
#else
  (void)*(volatile unsigned char *)mem;
  return mem;
#endif
}
void aprb_stats(unsigned long *out_pieces, unsigned long *out_reissues, unsigned long *out_files) {
  *out_pieces = carved;
  *out_reissues = reissues;
  *out_files = files;
}
