/* Borrow one linear block from the system allocator, aligned.
 *
 * WHY THIS EXISTS. Every nested allocator adapted to the capability discipline
 * takes its backing region as one LINEAR capability and most of them require it
 * ALIGNED: pymalloc wants its arena on a 16 KiB pool boundary, memcached's
 * authority layer on MCP_GRAIN, APR's lease layer on a page. The heap's lend
 * entry point takes no alignment -- runtime/virtual/heap-musl.c calls the
 * service with alignment 0 -- so a caller that needs one has to make it.
 *
 * It USED to come out aligned by accident. The previous virtual heap rounded a
 * request up to a power of two and acquired each arena aligned to its own size,
 * so a 64 MiB block was 64 MiB aligned and every adapter's check passed. The
 * musl mallocng policy that replaced it makes no such promise, and on
 * 2026-10-08 four nested arms refused to start -- `the lent arena is not a
 * pool-aligned linear region` -- because the port had been relying on that
 * accident. Relying on it again would be the same bug waiting for the next
 * allocator change, so the alignment is now MADE rather than hoped for.
 *
 * HOW. Borrow size + alignment, split off the head below the first aligned
 * address and the tail above size, and hand back the middle. Both offcuts stay
 * in their slots for the life of the process: they are linear capabilities and
 * dropping one would leak authority the heap's senior handle still covers, so
 * the heap can reclaim the whole lend when it frees the block, and nothing here
 * has to.
 *
 * The caller passes slots for the offcuts. Keeping them is deliberate: a linear
 * capability has exactly one place, and that place should be visible in the
 * caller's own storage rather than hidden in a static here.
 */
#ifndef CAPSTONE_PORTS_BORROW_ALIGNED_BLOCK_H
#define CAPSTONE_PORTS_BORROW_ALIGNED_BLOCK_H

#include <capstone/capability.h>
#include <stddef.h>

unsigned long __capstone_sublet_malloc_linear(size_t, capstone_cap_slot *);

/* Returns the aligned region in [out], its base as the value, and 0 on failure.
 * [head] and [tail] receive the offcuts and must outlive the region. */
static inline unsigned long
capstone_borrow_aligned_block(size_t size, unsigned long alignment,
                              capstone_cap_slot *out, capstone_cap_slot *head,
                              capstone_cap_slot *tail)
{
  if (!alignment || (alignment & (alignment - 1)))
    return 0;                                   /* a power of two, or nothing */
  if (!__capstone_sublet_malloc_linear(size + alignment, out))
    return 0;
  if (capstone_cap_type(out) != CAPSTONE_CAP_LINEAR)
    return 0;
  unsigned long base = capstone_cap_base(out);
  unsigned long end = capstone_cap_end(out);
  unsigned long aligned = (base + alignment - 1) & ~(alignment - 1);
  if (end - aligned < size)
    return 0;
  if (aligned != base) {
    /* [out] keeps the head, the region moves to [head]; then swap them back so
     * the caller's [out] is always the region. */
    capstone_cap_split(out, aligned, head);
    capstone_cap_move(out, tail);               /* park the head in [tail] */
    capstone_cap_move(head, out);               /* the region is in [out] */
    capstone_cap_move(tail, head);              /* the head ends up in [head] */
  }
  if (capstone_cap_end(out) != aligned + size)
    capstone_cap_split(out, aligned + size, tail);
  return aligned;
}

#endif
