/* The chunk port: authority for the block allocator's chunks, one region each.
 *
 * Included by the patched wmem_allocator_block.c under WMEM_PORT_CHUNKS only.
 * The region-granular hooks give a whole block one region, so a chunk freed
 * inside a live block keeps its authority until the block is reset. Here the
 * block is held LINEAR and every chunk is carved from it as a region of its
 * own, so a chunk free is a revoke. The allocator's own headers cannot stay in
 * the block once its ranges are carved away, so they live beside it; this
 * interface is everything the allocator still needs from the block itself.
 *
 * Both modes run the same allocator and the same layout. The spatial mode
 * lends the whole block and narrows every object from it, revoking nothing;
 * the protected mode carves, and revokes at a chunk free, a reset and a close. */
#ifndef WM_CHUNKS_H
#define WM_CHUNKS_H
#include "port.h"
#ifdef WM_DOMAIN
#include <sublet/sublet.h>
struct wm_chunk_auth {
  capstone_cap_slot slot; /* the chunk's region while free, its handle while out */
  void *wide;             /* an alias over the whole chunk while it is out */
};
struct wm_block_auth {
  capstone_cap_slot senior; /* taken before the first split: covers every chunk */
  capstone_cap_slot dead;   /* the block-header bytes, which no chunk covers */
  capstone_cap_slot region; /* the whole block, between epochs */
  void *alias;              /* the lent block (spatial) or a jumbo's region */
  size_t base, size;
};
#else
struct wm_chunk_auth {
  void *wide;
};
struct wm_block_auth {
  void *alias;
  size_t base, size;
};
#endif
/* What the protected mode did, for the report page (offset 128). */
struct wm_chunk_counts {
  uint64_t magic, opens, resets, closes, reset_revokes, close_revokes,
      dropped, retires, splits, issues, revokes, inits, region_revokes,
      region_inits;
};
#define WM_CHUNK_COUNTS_MAGIC UINT64_C(0x53544e554f434b43)
static inline size_t wm_addr(const void *p) { return (size_t)(uintptr_t)p; }
void wm_chunks_init(unsigned temporal);
/* Storage for the allocator's headers, which may not live in the block. */
void *wm_meta_alloc(size_t n);
/* A new block of `size` bytes. Its first chunk, [base+hdr, base+size), waits
 * in the block until wm_chunk_adopt gives it to that chunk's header. */
void wm_block_open(struct wm_block_auth *b, size_t size, size_t hdr);
/* A reset: every chunk of the block dies with ONE revoke, and the first chunk
 * waits again. `dropped` is how many chunks that revoke ended. */
void wm_block_reset(struct wm_block_auth *b, size_t hdr, size_t dropped);
/* The first chunk takes the authority open or reset left in the block. */
void wm_chunk_adopt(struct wm_block_auth *b, struct wm_chunk_auth *first);
/* Give the block back to the system: one revoke, then the storage. */
void wm_block_close(struct wm_block_auth *b);
/* Split a FREE chunk's authority at `at`: c keeps the lower part. */
void wm_chunk_split(struct wm_chunk_auth *c, size_t at,
                    struct wm_chunk_auth *upper);
/* Hand out the free chunk [base, base+len); afterwards c->wide covers it. */
void wm_chunk_issue(struct wm_block_auth *b, struct wm_chunk_auth *c,
                    size_t base, size_t len);
/* A pointer to bytes [at, at+n) of an issued chunk, bounded to exactly them. */
void *wm_chunk_bytes(struct wm_chunk_auth *c, size_t at, size_t n);
/* The chunk comes back. In the protected mode every alias of it dies. */
void wm_chunk_retire(struct wm_chunk_auth *c);
/* Clear authority that died with its block, before the record is reused. */
void wm_chunk_forget(struct wm_chunk_auth *c);
void wm_block_forget(struct wm_block_auth *b);
/* A pointer handed back to the allocator faults here, on its own authority,
 * before anything is looked up by its address. */
void wm_handback_probe(const void *p);
/* Write the counts to the report page, beyond the header (domain only). */
void wm_chunk_report(void *report_page);
#endif
