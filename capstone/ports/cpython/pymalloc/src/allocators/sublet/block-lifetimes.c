/* Each arena is split into pools, each pool into a persistent header and a
 * revocable body, and each body into size-class blocks. Upstream pymalloc still
 * chooses the pool and free-list order. These records retain authority only;
 * address lookup never authorizes a client's release without a lease check.
 */
#include "port.h"
#include <string.h>
#include <sublet/sublet.h>

#define ARENA_SIZE 1048576UL
#define POOL_SIZE 16384UL
#define ARENA_COUNT 32
#define POOL_COUNT 64
#define LARGE_COUNT 4096
struct block {
  capstone_cap_slot region;
  void *alias, *client;
  size_t size, requested;
  unsigned active;
};
struct pool {
  capstone_cap_slot body, remaining;
  void *header;
  struct block *blocks;
  uintptr_t base;
  size_t size, overhead, carved, capacity;
};
struct arena {
  capstone_cap_slot region, remaining;
  struct pool *pools;
  uintptr_t base;
  size_t carved;
  unsigned live;
};
static capstone_cap_slot small_remaining, large_remaining;
static uintptr_t small_cursor, small_end, large_cursor, large_end;
static struct arena *arenas;
static struct block *large;
static unsigned large_used, protected_mode;
static size_t arena_allocations, arena_releases;
size_t pym_metadata_used(void);

void pym_lifetime_init(void *region) {
  capstone_cap_store(&small_remaining, region);
  small_cursor = capstone_cap_base(&small_remaining);
  small_end = small_cursor + PYM_ARENA_BYTES / 2;
  if (capstone_cap_type(&small_remaining) != CAPSTONE_CAP_LINEAR ||
      capstone_cap_end(&small_remaining) - small_cursor != PYM_ARENA_BYTES ||
      (small_cursor & (POOL_SIZE - 1)))
    pym_fail(501);
  sublet_split(&small_remaining, small_end, &large_remaining);
  large_cursor = small_end;
  large_end = small_cursor + PYM_ARENA_BYTES;
}
void pym_set_mode(unsigned mode) {
  if (mode > 1)
    pym_fail(502);
  protected_mode = mode;
  arenas = pym_raw_calloc(ARENA_COUNT, sizeof *arenas);
  large = pym_raw_calloc(LARGE_COUNT, sizeof *large);
  if (!arenas || !large)
    pym_fail(503);
}
static struct arena *arena_for(uintptr_t address) {
  for (unsigned i = 0; i < ARENA_COUNT; ++i)
    if (arenas[i].live && address >= arenas[i].base &&
        address - arenas[i].base < ARENA_SIZE)
      return &arenas[i];
  return NULL;
}
static struct pool *pool_for(uintptr_t address) {
  struct arena *a = arena_for(address);
  if (!a)
    return NULL;
  size_t index = (address - a->base) / POOL_SIZE;
  return index < a->carved ? &a->pools[index] : NULL;
}
void *pym_arena_alloc(void *ctx, size_t n) {
  (void)ctx;
  if (n != ARENA_SIZE)
    pym_fail(504);
  for (unsigned i = 0; i < ARENA_COUNT; ++i) {
    struct arena *a = &arenas[i];
    if (a->live)
      continue;
    if (!a->base) {
      if (small_cursor > small_end - n)
        return NULL;
      a->pools = pym_raw_calloc(POOL_COUNT, sizeof *a->pools);
      if (!a->pools)
        return NULL;
      a->base = small_cursor;
      small_cursor += n;
      sublet_carve(&small_remaining, small_cursor, &a->region);
    }
    a->carved = 0;
    a->live = 1;
    sublet_take_linear(&a->region, &a->remaining);
    ++arena_allocations;
    /* An opaque token, not an alias spanning the arena's children. */
    return a;
  }
  return NULL;
}
uintptr_t pym_arena_address(void *token) {
  return ((struct arena *)token)->base;
}
void *pym_arena_pointer(uintptr_t address) {
  struct arena *a = arena_for(address);
  if (!a || a->base != address)
    pym_fail(505);
  return a;
}
void pym_arena_free(void *ctx, void *token, size_t n) {
  (void)ctx;
  struct arena *a = token;
  if (!a->live || n != ARENA_SIZE)
    pym_fail(506);
  sublet_give(&a->region);
  capstone_cap_clear(&a->remaining);
  /* Reclaim only metadata through the separate metadata heap. Payload
   * capabilities below the arena have all been revoked already. */
  for (size_t i = 0; i < a->carved; ++i) {
    pym_raw_free(a->pools[i].blocks);
    memset(&a->pools[i], 0, sizeof a->pools[i]);
  }
  a->live = 0;
  ++arena_releases;
}
void *pym_pool_create(uintptr_t address, size_t overhead) {
  struct arena *a = arena_for(address);
  if (!a || address != a->base + a->carved * POOL_SIZE ||
      a->carved == POOL_COUNT)
    pym_fail(507);
  struct pool *p = &a->pools[a->carved++];
  capstone_cap_slot pool, header;
  p->base = address;
  p->overhead = overhead;
  sublet_carve(&a->remaining, address + POOL_SIZE, &pool);
  sublet_carve(&pool, address + overhead, &header);
  p->header = capstone_cap_delinearize(&header);
  capstone_cap_move(&pool, &p->body);
  return p->header;
}
void *pym_pool_pointer(const void *ptr) {
  struct pool *p = pool_for((uintptr_t)ptr);
  return p ? p->header : NULL;
}
void pym_pool_reclass(void *header, size_t size, size_t overhead) {
  struct pool *p = pool_for((uintptr_t)header);
  if (!p || p->header != header || (size & 15) || p->overhead != overhead)
    pym_fail(508);
  if (p->size) {
    for (size_t i = 0; i < p->carved; ++i)
      if (p->blocks[i].active)
        pym_fail(509);
    sublet_give(&p->body);
    capstone_cap_clear(&p->remaining);
    pym_raw_free(p->blocks);
  }
  p->size = size;
  p->carved = 0;
  p->capacity = (POOL_SIZE - overhead) / size;
  p->blocks = pym_raw_calloc(p->capacity, sizeof *p->blocks);
  if (!p->blocks)
    pym_fail(510);
  sublet_take_linear(&p->body, &p->remaining);
}
void *pym_block_pointer(void *header, size_t offset) {
  struct pool *p = pool_for((uintptr_t)header);
  if (!p || offset < p->overhead || (offset - p->overhead) % p->size)
    pym_fail(511);
  size_t index = (offset - p->overhead) / p->size;
  if (index >= p->capacity || index > p->carved)
    pym_fail(512);
  struct block *b = &p->blocks[index];
  if (index == p->carved) {
    sublet_carve(&p->remaining, p->base + offset + p->size, &b->region);
    b->size = p->size;
    b->alias = sublet_take(&b->region);
    ++p->carved;
  }
  return b->alias;
}
static struct block *block_for(void *ptr) {
  uintptr_t address = (uintptr_t)ptr;
  struct pool *p = pool_for(address);
  if (p) {
    if (address < p->base + p->overhead || !p->size)
      pym_fail(513);
    size_t offset = address - p->base - p->overhead;
    if (offset % p->size || offset / p->size >= p->carved)
      pym_fail(514);
    return &p->blocks[offset / p->size];
  }
  for (unsigned i = 0; i < large_used; ++i)
    if ((uintptr_t)large[i].alias == address)
      return &large[i];
  pym_fail(515);
}
static int same_authority(void *a, void *b) {
  capstone_cap_slot x, y;
  capstone_cap_store(&x, a);
  capstone_cap_store(&y, b);
  if (capstone_cap_type(&x) != CAPSTONE_CAP_NONLINEAR ||
      capstone_cap_type(&y) != CAPSTONE_CAP_NONLINEAR)
    return 0;
  const volatile uint64_t *xx = (const volatile uint64_t *)&x;
  const volatile uint64_t *yy = (const volatile uint64_t *)&y;
  return xx[0] == yy[0] && xx[1] == yy[1];
}
void pym_validate(void *ptr) {
  struct block *b = block_for(ptr);
  if (!b->active || !same_authority(ptr, b->client))
    pym_fail(516);
}
void *pym_issue(void *ptr, size_t requested) {
  struct block *b = block_for(ptr);
  if (b->active || requested > b->size)
    pym_fail(517);
  if (protected_mode) {
    sublet_give(&b->region);
    b->alias = sublet_take(&b->region);
  }
  b->requested = requested;
  b->active = 1;
  uintptr_t base = (uintptr_t)b->alias;
  b->client = __builtin_capstone_cap_shrink(b->alias, base,
                                            base + (requested ? requested : 1));
  return b->client;
}
void *pym_release(void *ptr) {
  pym_validate(ptr);
  struct block *b = block_for(ptr);
  if (protected_mode) {
    sublet_give(&b->region);
    b->alias = sublet_take(&b->region);
  }
  b->active = 0;
  b->client = NULL;
  return b->alias;
}
void *pym_resize(void *ptr, size_t requested) {
  return pym_issue(pym_release(ptr), requested);
}
size_t pym_requested(void *ptr) {
  pym_validate(ptr);
  return block_for(ptr)->requested;
}
void *pym_user_raw_malloc(size_t n) {
  if (n > PYM_ARENA_BYTES / 2 - 16)
    return NULL;
  size_t rounded = n ? (n + 15) & ~(size_t)15 : 16;
  for (unsigned i = 0; i < large_used; ++i)
    if (!large[i].active && large[i].size == rounded)
      return large[i].alias;
  if (large_used == LARGE_COUNT || large_cursor > large_end - rounded)
    return NULL;
  struct block *b = &large[large_used++];
  large_cursor += rounded;
  sublet_carve(&large_remaining, large_cursor, &b->region);
  b->size = rounded;
  b->alias = sublet_take(&b->region);
  return b->alias;
}
void pym_user_raw_free(void *ptr) {
  (void)ptr; /* pym_release already returned the slot */
}
void *pym_user_raw_realloc(void *ptr, size_t n) {
  struct block *b = block_for(ptr);
  if (n <= b->size && n > b->size / 2)
    return pym_resize(ptr, n);
  void *raw = pym_user_raw_malloc(n);
  if (!raw)
    return NULL;
  void *q = pym_issue(raw, n);
  memcpy(q, ptr, n < b->requested ? n : b->requested);
  pym_release(ptr);
  return q;
}
void pym_backing_stats(struct pym_header *h) {
  h->arenas = arena_allocations;
  h->arena_frees = arena_releases;
  h->metadata = pym_metadata_used();
}
