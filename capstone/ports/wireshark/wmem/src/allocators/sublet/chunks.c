/* The chunk port's authority in a Capstone domain. See src/shared/chunks.h.
 *
 * Protected mode, per block:
 *   open    the block arrives LINEAR from the backing; a senior handle is taken
 *           on it BEFORE anything is split, the header bytes are carved off, and
 *           the rest is the first chunk's region;
 *   split   a free chunk's region is split at the new boundary;
 *   issue   sublet_take: the slot keeps the handle, the caller gets an alias;
 *   retire  sublet_give: one revoke, and the slot holds the region again;
 *   reset   sublet_give_to on the senior handle: ONE revoke ends every chunk of
 *           the block, whatever it was split into, and the block is whole again;
 *   close   the same revoke, then the region goes back to the backing.
 * Chunks are never joined again: two neighbours can be rejoined only through a
 * handle taken before the split that separated them, and for an allocator that
 * carves from the front that handle also covers the live chunks between them. */
#include "chunks.h"
#include "regions.h"
#include <string.h>
static unsigned temporal;
static struct wm_chunk_counts counts = {.magic = WM_CHUNK_COUNTS_MAGIC};
void wm_chunks_init(unsigned t) { temporal = t; }
static void epoch(struct wm_block_auth *b, size_t hdr) {
  sublet_handle(&b->region, &b->senior);
  sublet_carve(&b->region, b->base + hdr, &b->dead);
}
void wm_block_open(struct wm_block_auth *b, size_t size, size_t hdr) {
  ++counts.opens;
  b->size = size;
  if (!temporal) {
    b->alias = wm_sys_alloc(size);
    b->base = wm_addr(b->alias);
    return;
  }
  b->alias = NULL;
  b->base = wm_block_acquire(size, &b->region);
  epoch(b, hdr);
}
void wm_block_reset(struct wm_block_auth *b, size_t hdr, size_t dropped) {
  ++counts.resets;
  counts.dropped += dropped;
  if (!temporal)
    return;
  unsigned long before = sublet_stats.revoke;
  sublet_give_to(&b->senior, &b->region);
  counts.reset_revokes += sublet_stats.revoke - before;
  epoch(b, hdr);
}
void wm_chunk_adopt(struct wm_block_auth *b, struct wm_chunk_auth *first) {
  if (temporal)
    capstone_cap_move(&b->region, &first->slot);
  first->wide = NULL;
}
void wm_block_close(struct wm_block_auth *b) {
  ++counts.closes;
  if (!temporal) {
    wm_sys_free(b->alias);
    b->alias = NULL;
    return;
  }
  unsigned long before = sublet_stats.revoke;
  sublet_give_to(&b->senior, &b->region);
  counts.close_revokes += sublet_stats.revoke - before;
  capstone_cap_clear(&b->dead);
  wm_block_release(b->base, &b->region);
}
void wm_chunk_split(struct wm_chunk_auth *c, size_t at,
                    struct wm_chunk_auth *upper) {
  ++counts.splits;
  upper->wide = NULL;
  if (temporal)
    sublet_split(&c->slot, at, &upper->slot);
}
void wm_chunk_issue(struct wm_block_auth *b, struct wm_chunk_auth *c,
                    size_t base, size_t len) {
  ++counts.issues;
  if (!temporal) {
    char *p = (char *)b->alias + (base - b->base);
    c->wide = __builtin_capstone_cap_shrink(p, base, base + len);
    return;
  }
  c->wide = sublet_take(&c->slot);
}
void *wm_chunk_bytes(struct wm_chunk_auth *c, size_t at, size_t n) {
  char *p = (char *)c->wide + (at - wm_addr(c->wide));
  return __builtin_capstone_cap_shrink(p, at, at + n);
}
void wm_chunk_retire(struct wm_chunk_auth *c) {
  ++counts.retires;
  if (temporal) {
    sublet_give(&c->slot);
    c->wide = NULL;
  }
}
void wm_chunk_forget(struct wm_chunk_auth *c) {
  capstone_cap_clear(&c->slot);
  c->wide = NULL;
}
void wm_block_forget(struct wm_block_auth *b) {
  capstone_cap_clear(&b->senior);
  capstone_cap_clear(&b->dead);
  capstone_cap_clear(&b->region);
  b->alias = NULL;
}
void wm_chunk_report(void *report_page) {
  counts.revokes = sublet_stats.revoke;
  counts.inits = sublet_stats.init;
  memcpy((unsigned char *)report_page + 128, &counts, sizeof counts);
}
