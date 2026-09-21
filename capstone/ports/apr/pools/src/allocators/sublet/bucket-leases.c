/* Capstone: authority for apr-util's bucket allocator, one piece per node it
 * carves, under the handle the pool port keeps senior to the block.
 *
 * The bucket allocator is a client of APR's allocator: it takes 8 KiB blocks,
 * carves SMALL_NODE_SIZE nodes from them front to back, files a freed node on
 * its own LIFO freelist and returns whole blocks when it is destroyed. Large
 * nodes are whole APR nodes and never come here. Upstream still decides which
 * block, which node and in which order -- the hooks replace a bump and two
 * list operations with calls that do the same thing under authority.
 *
 * A block arrives from aprp_node_lend: its header retaken as the alias the
 * allocator keeps, the rest LINEAR in the block's record. A carve is a split
 * off that rest and a take; the piece's handle stays in its record. A file is
 * a give (mode 1) and a push onto the list's freelist, kept here by index
 * because a link written into a freed node would be written into a revoked
 * region; a reissue pops the same node and takes a fresh alias. When the
 * block chain goes back to APR the pool port's release revokes the senior
 * handle, and every piece dies with it; the records are dropped first.
 *
 * mode 0 (spatial): a piece is a shrink of the block's alias and keeps it
 * across the freelist -- the allocator as upstream ships it. Same layout,
 * same addresses, in both modes. */
#include "port.h"
#include "apr_shim.h"
#include "apr_allocator.h"
#include <sublet/sublet.h>

#define APRB_BLOCKS 8192
#define APRB_PIECES 65536
#define APRB_LISTS 256

struct block {
  capstone_cap_slot rest; /* what is left to carve, linear (mode 1) */
  uintptr_t address, cursor, end;
  void *alias; /* the header alias upstream keeps */
  unsigned live;
};
struct piece {
  capstone_cap_slot handle; /* the piece's handle while out, its region while filed */
  uintptr_t address;
  void *alias;
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
  b->address = (uintptr_t)block;
  b->alias = aprp_node_lend(block, &b->rest);
  apr_memnode_t *header = b->alias;
  b->cursor = (uintptr_t)header->first_avail;
  b->end = (uintptr_t)header->endp;
  if (b->cursor != b->address + APR_MEMNODE_T_SIZE || b->end <= b->cursor)
    aprp_fail(535);
  b->live = 1;
  return b->alias;
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
  if (aprp_mode()) {
    sublet_carve(&b->rest, b->cursor + size, &p->handle);
    p->alias = sublet_take(&p->handle);
  } else {
    p->alias = __builtin_capstone_cap_shrink(b->alias, b->cursor, b->cursor + size);
  }
  b->cursor += size;
  /* first_avail, as the bump would have left it, written through the header
   * alias so upstream's end-of-block test keeps reading what it always read. */
  apr_memnode_t *header = b->alias;
  header->first_avail = (char *)b->alias + (b->cursor - b->address);
  p->live = 1;
  ++carved;
  return p->alias;
}
void aprb_file(void *list, void *node) {
  struct piece *p = piece_for(node);
  if (p->filed)
    aprp_fail(538);
  struct list *l = list_for(list);
  if (aprp_mode())
    sublet_give(&p->handle);
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
  if (aprp_mode())
    p->alias = sublet_take(&p->handle);
  *memnode = blocks[p->block].alias;
  ++reissues;
  return p->alias;
}
/* The chain is walked through the header aliases before the pool port revokes
 * them; the pieces' handles are junior to the block's and need no revoke of
 * their own, only to be forgotten. A filed piece leaves its list's freelist. */
void aprb_blocks_returning(void *chain) {
  for (apr_memnode_t *header = chain; header; header = header->next) {
    struct block *b = block_for(header);
    for (unsigned i = 0; i < npieces; ++i) {
      struct piece *p = &pieces[i];
      if (!p->live || p->block != (int)(b - blocks))
        continue;
      capstone_cap_clear(&p->handle);
      p->live = 0;
    }
    capstone_cap_clear(&b->rest);
    b->live = 0;
  }
  /* Freelists: drop entries whose pieces are gone. */
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
/* The first read apr_bucket_free makes through the pointer it is handed. */
void *aprb_probe(void *mem) {
  unsigned long value;
  void *out;
  /* The pointer comes back through the asm, so what follows depends on it. */
  __asm__ volatile(".globl aprb_free_probe\naprb_free_probe:\nlbu %0, 0(%2)\nmovc %1, %2\n"
                   : "=&r"(value), "=r"(out)
                   : "r"(mem)
                   : "memory");
  return out;
}
void aprb_stats(unsigned long *out_pieces, unsigned long *out_reissues, unsigned long *out_files) {
  *out_pieces = carved;
  *out_reissues = reissues;
  *out_files = files;
}
