/* Virtual profile, nested: authority for apr-util's bucket allocator, one
 * CDERIVE child per piece it carves, below the node the pool port issued.
 *
 * The records, the per-list LIFO and every address are those of the pointer
 * adapter (src/native/bucket-pointers.c), so both virtual arms carve the same
 * pieces in the same order. What this one adds:
 *   carve    the piece is a CDERIVE child of the block (the node's issued
 *            child, which aprp_node_lend hands over), bounded to the piece;
 *   file     CREVOKE(block, piece) -- apr_bucket_free's pointer, and every copy
 *            of it, is dead; the freelist link stays here, by index, because a
 *            link written through the revoked piece would fault;
 *   reissue  a fresh child of the block over the same bytes.
 * When the block chain goes back to APR the pool port's release revokes the
 * block, and every piece goes with it; the records are dropped first. */
#include "port.h"
#include <capstone/capability.h>
#include <capstone/capability-slot.h>
#include "apr_shim.h"
#include "apr_allocator.h"

#define APRB_BLOCKS 8192
#define APRB_PIECES 65536
#define APRB_LISTS 256

struct block {
  uintptr_t address, cursor, end;
  void *header; /* the node's issued child: upstream's alias and every piece's parent */
  unsigned live;
};
struct piece {
  uintptr_t address;
  void *handle; /* the live child while out, NULL while filed */
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
static void *derive_piece(struct piece *p) {
  struct block *b = &blocks[p->block];
  return capstone_cap_derive(b->header, p->address - b->address, p->size);
}
void *aprb_block_lend(void *block) {
  ensure();
  if (nblocks == APRB_BLOCKS)
    aprp_fail(534);
  struct block *b = &blocks[nblocks++];
  capstone_cap_slot unused;
  b->header = aprp_node_lend(block, &unused);
  b->address = (uintptr_t)b->header;
  if (__builtin_capstone_cap_get_base(b->header) != b->address)
    aprp_fail(539);
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
  p->handle = derive_piece(p);
  ++carved;
  return p->handle;
}
void aprb_file(void *list, void *node) {
  struct piece *p = piece_for(node);
  if (p->filed || !p->handle)
    aprp_fail(538);
  struct list *l = list_for(list);
  capstone_cap_revoke_child(blocks[p->block].header, p->handle);
  p->handle = NULL;
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
  p->handle = derive_piece(p);
  *memnode = blocks[p->block].header;
  ++reissues;
  return p->handle;
}
void aprb_blocks_returning(void *chain) {
  for (apr_memnode_t *header = chain; header; header = header->next) {
    struct block *b = block_for(header);
    for (unsigned i = 0; i < npieces; ++i)
      if (pieces[i].live && pieces[i].block == (int)(b - blocks)) {
        pieces[i].live = 0;
        pieces[i].handle = NULL;
      }
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
/* The labelled read: a plain load, which faults HERE on a revoked piece. */
void *aprb_probe(void *mem) {
  (void)*(volatile unsigned char *)mem;
  return mem;
}
void aprb_stats(unsigned long *out_pieces, unsigned long *out_reissues, unsigned long *out_files) {
  *out_pieces = carved;
  *out_reissues = reissues;
  *out_files = files;
}
