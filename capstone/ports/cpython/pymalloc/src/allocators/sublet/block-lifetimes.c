/* pymalloc's blocks as Sublet child lifetimes (CDERIVE / CREVOKE).
 *
 * The region pymalloc's arenas live in is one non-linear capability. The
 * adapter derives a child of it, which carries MANAGE, and from that one child
 * per arena; every block a client receives is a child of its arena, bounded to
 * the request. Releasing a block revokes that child: every copy of the client's
 * pointer dies, and the block's bytes stay reachable through the arena's own
 * capability, which is what obmalloc uses for its pool headers and free lists.
 * Releasing an arena revokes the arena's child, and with it every block still
 * derived from it. Nothing is carved, so there are no per-pool or per-block
 * records: obmalloc's own bookkeeping already says where each block is, and
 * CREVOKE refuses a reference that is not a live direct child of the arena.
 *
 * Allocator bookkeeping, not the hardware, keeps live blocks from overlapping
 * (docs/design/virtual-capstone/isa.md, "Sublet child lifetimes").
 *
 * Requests over 512 bytes do not reach pymalloc's pools. With
 * PYMALLOC_SYSTEM_MALLOC (hosted builds) they go to the system allocator, which
 * on the virtual profile is musl mallocng: exact bounds, lifetime retired on
 * free. Without it (the freestanding domain) they are derived from the upper
 * half of the region and never reused.
 */
#include "port.h"
#include <string.h>
#include <capstone/capability.h>
#ifdef PYMALLOC_SYSTEM_MALLOC
#include <stdlib.h>
#endif
#ifdef PYMALLOC_GAP_OBSERVER
#include <stdio.h>
#include "../../../../../experiments/study/reuse-gap-observer.h"
#define PYM_REUSE_SLOTS (1u << 17)
static struct reuse_gap_slot pym_reuse_slots[PYM_REUSE_SLOTS];
static struct reuse_gap_observer pym_reuse;
static unsigned observer_inplace_resize;
#endif

#define ARENA_SIZE 1048576UL
#define POOL_SIZE 16384UL
#ifdef PYMALLOC_SYSTEM_MALLOC
#define SMALL_BYTES PYM_ARENA_BYTES
#else
#define SMALL_BYTES (PYM_ARENA_BYTES / 2)
#endif
#define ARENA_COUNT (SMALL_BYTES / ARENA_SIZE)

struct arena {
  void *cap; /* the arena's child: MANAGE over its blocks, data over its bytes */
  unsigned live;
};
static struct arena arenas[ARENA_COUNT];
static void *root;       /* the region's child: MANAGE over the arenas */
static uintptr_t region_base;
static unsigned protected_mode;
static size_t arena_allocations, arena_releases;
#ifndef PYMALLOC_SYSTEM_MALLOC
static uintptr_t large_cursor;
#endif
size_t pym_metadata_used(void);

void pym_lifetime_init(void *region) {
#ifdef PYMALLOC_GAP_OBSERVER
  reuse_gap_init(&pym_reuse, pym_reuse_slots, PYM_REUSE_SLOTS);
#endif
  capstone_cap_slot slot;
  capstone_cap_store(&slot, region);
  region_base = capstone_cap_base(&slot);
  if (capstone_cap_end(&slot) - region_base != PYM_ARENA_BYTES ||
      (region_base & (POOL_SIZE - 1)))
    pym_fail(501);
  /* CDERIVE needs a non-linear parent. A freestanding domain is handed its
   * region linear; a hosted build passes an ordinary malloc'd object. */
  if (capstone_cap_type(&slot) == CAPSTONE_CAP_LINEAR)
    region = capstone_cap_delinearize(&slot);
  else if (capstone_cap_type(&slot) != CAPSTONE_CAP_NONLINEAR)
    pym_fail(501);
  root = capstone_cap_derive(region, 0, PYM_ARENA_BYTES);
#ifndef PYMALLOC_SYSTEM_MALLOC
  large_cursor = region_base + SMALL_BYTES;
#endif
}
void pym_set_mode(unsigned mode) {
  if (mode > 1)
    pym_fail(502);
  protected_mode = mode;
}
static int in_small(uintptr_t address) {
  return address >= region_base && address - region_base < SMALL_BYTES;
}
static struct arena *arena_of(uintptr_t address) {
  if (!in_small(address))
    return NULL;
  struct arena *a = &arenas[(address - region_base) / ARENA_SIZE];
  return a->live ? a : NULL;
}
static uintptr_t arena_base(const struct arena *a) {
  return region_base + (uintptr_t)(a - arenas) * ARENA_SIZE;
}
/* The arena's own capability at ADDRESS: obmalloc's view, never a client's. */
static void *inside(struct arena *a, uintptr_t address) {
  return (char *)a->cap + (address - arena_base(a));
}

void *pym_arena_alloc(void *ctx, size_t n) {
  (void)ctx;
  if (n != ARENA_SIZE)
    pym_fail(504);
  for (unsigned i = 0; i < ARENA_COUNT; ++i) {
    struct arena *a = &arenas[i];
    if (a->live)
      continue;
    a->cap = capstone_cap_derive(root, (unsigned long)i * ARENA_SIZE, ARENA_SIZE);
    a->live = 1;
    ++arena_allocations;
    return a; /* an opaque token; obmalloc takes the address separately */
  }
  return NULL;
}
uintptr_t pym_arena_address(void *token) {
  return arena_base(token);
}
void *pym_arena_pointer(uintptr_t address) {
  struct arena *a = arena_of(address);
  if (!a || arena_base(a) != address)
    pym_fail(505);
  return a;
}
void pym_arena_free(void *ctx, void *token, size_t n) {
  (void)ctx;
  struct arena *a = token;
  if (!a->live || n != ARENA_SIZE)
    pym_fail(506);
  capstone_cap_revoke_child(root, a->cap); /* the arena and every block in it */
  a->cap = NULL;
  a->live = 0;
  ++arena_releases;
}
void *pym_pool_create(uintptr_t address, size_t overhead) {
  struct arena *a = arena_of(address);
  if (!a || (address & (POOL_SIZE - 1)) || overhead >= POOL_SIZE)
    pym_fail(507);
  return inside(a, address);
}
void *pym_pool_pointer(const void *ptr) {
  uintptr_t pool = (uintptr_t)ptr & ~(POOL_SIZE - 1);
  struct arena *a = arena_of(pool);
  return a ? inside(a, pool) : NULL;
}
void pym_pool_reclass(void *header, size_t size, size_t overhead) {
  if (!arena_of((uintptr_t)header) || (size & 15) || overhead >= POOL_SIZE)
    pym_fail(508);
}
void *pym_block_pointer(void *header, size_t offset) {
  if (offset >= POOL_SIZE)
    pym_fail(511);
  return (char *)header + offset;
}
void pym_validate(void *ptr) {
  /* A released block's reference faults here: its node is dead. */
  if (arena_of((uintptr_t)ptr) && protected_mode)
    (void)*(volatile unsigned char *)ptr;
}
void *pym_issue(void *ptr, size_t requested) {
  uintptr_t address = (uintptr_t)ptr;
  struct arena *a = arena_of(address);
  if (!a)
    return ptr; /* a large block: the system allocator's object as it is */
  size_t n = requested ? requested : 1;
  void *client = protected_mode
      ? capstone_cap_derive(a->cap, address - arena_base(a), n)
      : __builtin_capstone_cap_shrink(ptr, address, address + n);
#ifdef PYMALLOC_GAP_OBSERVER
  if (!observer_inplace_resize) {
    reuse_gap_attempt(&pym_reuse);
    reuse_gap_issue(&pym_reuse, (uint64_t)address, (uint64_t)n);
  }
#endif
  return client;
}
void *pym_release(void *ptr) {
  uintptr_t address = (uintptr_t)ptr;
  struct arena *a = arena_of(address);
  if (!a)
    return ptr;
#ifdef PYMALLOC_GAP_OBSERVER
  if (!observer_inplace_resize)
    reuse_gap_release(&pym_reuse, (uint64_t)address);
#endif
  if (protected_mode)
    capstone_cap_revoke_child(a->cap, ptr);
  return inside(a, address);
}
void *pym_resize(void *ptr, size_t requested) {
#ifdef PYMALLOC_GAP_OBSERVER
  observer_inplace_resize = 1;
#endif
  void *client = pym_issue(pym_release(ptr), requested);
#ifdef PYMALLOC_GAP_OBSERVER
  observer_inplace_resize = 0;
  reuse_gap_resize(&pym_reuse, (uint64_t)(uintptr_t)client,
                   (uint64_t)(requested ? requested : 1));
#endif
  return client;
}
size_t pym_requested(void *ptr) {
  pym_validate(ptr);
  capstone_cap_slot slot;
  capstone_cap_store(&slot, ptr);
  return capstone_cap_end(&slot) - capstone_cap_base(&slot);
}
#ifdef PYMALLOC_SYSTEM_MALLOC
void *pym_user_raw_malloc(size_t n) {
  return malloc(n ? n : 1);
}
void pym_user_raw_free(void *ptr) {
  free(ptr);
}
void *pym_user_raw_realloc(void *ptr, size_t n) {
  return realloc(ptr, n ? n : 1);
}
#else
void *pym_user_raw_malloc(size_t n) {
  size_t rounded = n ? (n + 15) & ~(size_t)15 : 16;
  if (rounded > region_base + PYM_ARENA_BYTES - large_cursor)
    return NULL;
  void *p = capstone_cap_derive(root, large_cursor - region_base, rounded);
  large_cursor += rounded;
  return p;
}
void pym_user_raw_free(void *ptr) {
  capstone_cap_revoke_child(root, ptr); /* the bytes are not reused */
}
void *pym_user_raw_realloc(void *ptr, size_t n) {
  void *q = pym_user_raw_malloc(n);
  if (!q)
    return NULL;
  size_t old = pym_requested(ptr);
  memcpy(q, ptr, n < old ? n : old);
  pym_user_raw_free(ptr);
  return q;
}
#endif
void pym_backing_stats(struct pym_header *h) {
  h->arenas = arena_allocations;
  h->arena_frees = arena_releases;
  h->metadata = pym_metadata_used();
}
#ifdef PYMALLOC_GAP_OBSERVER
void pym_gap_report(void) {
  fprintf(stderr, "PYM_REUSE_GAP attempts=%llu issues=%llu releases=%llu "
          "reuses=%llu distinct=%llu capacity=%u error=%u bins=",
          (unsigned long long)pym_reuse.attempts,
          (unsigned long long)pym_reuse.issues,
          (unsigned long long)pym_reuse.releases,
          (unsigned long long)pym_reuse.reuses,
          (unsigned long long)pym_reuse.distinct_starts,
          PYM_REUSE_SLOTS, pym_reuse.error);
  for (unsigned i = 0; i < 32; ++i)
    fprintf(stderr, "%s%llu", i ? "," : "",
            (unsigned long long)pym_reuse.bins[i]);
  fprintf(stderr, "\n");
}
#endif
