/* Trusted pymalloc adapter. Synchronous revocation completes before freed
 * storage becomes an in-band free-list link. Only in-place realloc needs a
 * capability-preserving snapshot. This policy favors simple lifetime rules,
 * not sweep throughput; all sweep/copy work is reported explicitly. */
#include "port.h"
#include <cheri/cheric.h>
#include <cheri/revoke.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#ifndef CHERI_PERM_POISON
#error "Use the published PoisonCap SDK and matching kernel"
#endif
#define ARENA_SIZE 1048576UL
#define POOL_SIZE 16384UL
#define ARENA_COUNT 32
#define POOL_COUNT 64
#define LARGE_COUNT 4096
struct block {
  unsigned char *manager;
  void *client;
  size_t size, requested;
  unsigned active;
};
struct pool {
  unsigned char *header;
  struct block *blocks;
  size_t size, overhead, carved, capacity;
};
struct arena {
  unsigned char *base;
  struct pool *pools;
  size_t carved;
  unsigned live;
};
static unsigned char *backing;
static size_t small_cursor, large_cursor;
static struct arena *arenas;
static struct block *large;
static unsigned large_used, protected_mode, initialized;
static size_t arena_allocations, arena_releases, reclassifications;
static size_t sweeps, poison_bytes, cleared_bytes, zeroed_bytes, copied_bytes, snapshot_peak;
size_t pym_metadata_used(void);

void pym_lifetime_init(void *region) {
  if (initialized || !feature_present("cheri_caprevoke_poison") ||
      !cheri_gettag(region) || ((uintptr_t)region & (POOL_SIZE - 1)) ||
      cheri_getlen(region) < PYM_ARENA_BYTES ||
      !(cheri_getperm(region) & CHERI_PERM_POISON))
    pym_fail(701);
  backing = region;
  large_cursor = PYM_ARENA_BYTES / 2;
  initialized = 1;
}
void pym_set_mode(unsigned mode) {
  if (!initialized || mode > 1 || arenas)
    pym_fail(702);
  protected_mode = mode;
  arenas = pym_raw_calloc(ARENA_COUNT, sizeof *arenas);
  large = pym_raw_calloc(LARGE_COUNT, sizeof *large);
  if (!arenas || !large)
    pym_fail(703);
}
static struct arena *arena_for(uintptr_t address) {
  for (unsigned i = 0; i < ARENA_COUNT; ++i) {
    struct arena *a = &arenas[i];
    uintptr_t base = (uintptr_t)a->base;
    if (a->live && address >= base && address - base < ARENA_SIZE)
      return a;
  }
  return NULL;
}
static struct pool *pool_for(uintptr_t address) {
  struct arena *a = arena_for(address);
  if (!a)
    return NULL;
  size_t i = (address - (uintptr_t)a->base) / POOL_SIZE;
  return i < a->carved ? &a->pools[i] : NULL;
}
static void *manager_pointer(unsigned char *p, size_t n) {
  void *q = cheri_setboundsexact(p, n);
  if (!cheri_gettag(q))
    pym_fail(704);
  return q;
}
/* The retained manager capabilities carry poison authority. Client pointers
 * lose it before publication and are cleared by the published kernel revoker. */
static void invalidate(unsigned char *manager, size_t n) {
  if (!protected_mode)
    return;
  if (!n || (n & 15) || ((uintptr_t)manager & 15))
    pym_fail(705);
  for (size_t i = 0; i < n; i += 16) {
    void *word = manager + i;
    __asm__ volatile("cpoison %0, 0(%0)" : : "C"(word) : "memory");
  }
  poison_bytes += n;
  struct cheri_revoke_syscall_info info = {0};
  if (cheri_revoke(CHERI_REVOKE_LAST_PASS | CHERI_REVOKE_IGNORE_START |
                   CHERI_REVOKE_TAKE_STATS, 0, &info))
    pym_fail(706); /* Never reuse storage after a failed sweep. */
  ++sweeps;
  for (size_t i = 0; i < n; i += 16) {
    void *word = manager + i;
    __asm__ volatile("cclearpoison %0, 0(%0)" : : "C"(word) : "memory");
  }
  cleared_bytes += n;
  /* cclearpoison resets access state but leaves the stored poison capability.
   * The kernel scans that payload on later sweeps. Remove it before publishing
   * fresh client authority, including zero-size allocations that do no stores.
   * In-place realloc restores its capability-preserving snapshot afterwards. */
  memset(manager, 0, n);
  zeroed_bytes += n;
}
void *pym_arena_alloc(void *ctx, size_t n) {
  (void)ctx;
  if (n != ARENA_SIZE)
    pym_fail(707);
  for (unsigned i = 0; i < ARENA_COUNT; ++i) {
    struct arena *a = &arenas[i];
    if (a->live)
      continue;
    if (!a->base) {
      if (small_cursor > PYM_ARENA_BYTES / 2 - n)
        return NULL;
      a->pools = pym_raw_calloc(POOL_COUNT, sizeof *a->pools);
      if (!a->pools)
        return NULL;
      a->base = backing + small_cursor;
      small_cursor += n;
    }
    a->carved = 0;
    a->live = 1;
    ++arena_allocations;
    return a; /* Opaque allocator token, not a client capability. */
  }
  return NULL;
}
uintptr_t pym_arena_address(void *token) {
  return (uintptr_t)((struct arena *)token)->base;
}
void *pym_arena_pointer(uintptr_t address) {
  struct arena *a = arena_for(address);
  if (!a || (uintptr_t)a->base != address)
    pym_fail(708);
  return a;
}
void pym_arena_free(void *ctx, void *token, size_t n) {
  (void)ctx;
  struct arena *a = token;
  if (!a->live || n != ARENA_SIZE)
    pym_fail(709);
  for (size_t i = 0; i < a->carved; ++i)
    for (size_t j = 0; j < a->pools[i].carved; ++j)
      if (a->pools[i].blocks[j].active)
        pym_fail(710);
  invalidate(a->base, n);
  for (size_t i = 0; i < a->carved; ++i) {
    pym_raw_free(a->pools[i].blocks);
    memset(&a->pools[i], 0, sizeof a->pools[i]);
  }
  a->live = 0;
  ++arena_releases;
}
void *pym_pool_create(uintptr_t address, size_t overhead) {
  struct arena *a = arena_for(address);
  if (!a || address != (uintptr_t)a->base + a->carved * POOL_SIZE ||
      a->carved == POOL_COUNT || overhead >= POOL_SIZE || (overhead & 15))
    pym_fail(711);
  struct pool *p = &a->pools[a->carved++];
  p->header = a->base + (address - (uintptr_t)a->base);
  p->overhead = overhead;
  return p->header;
}
void *pym_pool_pointer(const void *ptr) {
  struct pool *p = pool_for((uintptr_t)ptr);
  return p ? p->header : NULL;
}
void pym_pool_reclass(void *header, size_t size, size_t overhead) {
  struct pool *p = pool_for((uintptr_t)header);
  if (!p || p->header != header || !size || size > 512 || (size & 15) ||
      p->overhead != overhead)
    pym_fail(712);
  if (p->size) {
    for (size_t i = 0; i < p->carved; ++i)
      if (p->blocks[i].active)
        pym_fail(713);
    invalidate(p->header + overhead, POOL_SIZE - overhead);
    pym_raw_free(p->blocks);
    ++reclassifications;
  }
  p->size = size;
  p->carved = 0;
  p->capacity = (POOL_SIZE - overhead) / size;
  p->blocks = pym_raw_calloc(p->capacity, sizeof *p->blocks);
  if (!p->blocks)
    pym_fail(714);
}
void *pym_block_pointer(void *header, size_t offset) {
  struct pool *p = pool_for((uintptr_t)header);
  if (!p || !p->size || offset < p->overhead ||
      (offset - p->overhead) % p->size)
    pym_fail(715);
  size_t i = (offset - p->overhead) / p->size;
  if (i >= p->capacity || i > p->carved)
    pym_fail(716);
  struct block *b = &p->blocks[i];
  if (i == p->carved) {
    b->manager = manager_pointer(p->header + offset, p->size);
    b->size = p->size;
    ++p->carved;
  }
  return b->manager;
}
static struct block *block_for(void *ptr) {
  uintptr_t address = (uintptr_t)ptr;
  struct pool *p = pool_for(address);
  if (p) {
    uintptr_t start = (uintptr_t)p->header + p->overhead;
    if (address < start || !p->size || (address - start) % p->size ||
        (address - start) / p->size >= p->carved)
      pym_fail(717);
    return &p->blocks[(address - start) / p->size];
  }
  for (unsigned i = 0; i < large_used; ++i)
    if ((uintptr_t)large[i].manager == address)
      return &large[i];
  pym_fail(718);
}
void pym_validate(void *ptr) {
  /* Lookup selects a record; only the exact current, tagged client authority
   * authorizes release. A stale scalar address must never free a new lease. */
  if (!cheri_gettag(ptr))
    pym_fail(719);
  struct block *b = block_for(ptr);
  if (!b->active || !__builtin_cheri_equal_exact(ptr, b->client))
    pym_fail(719);
}
void *pym_issue(void *ptr, size_t requested) {
  struct block *b = block_for(ptr);
  if (b->active || requested > b->size)
    pym_fail(720);
  void *p = cheri_setbounds(b->manager, requested ? requested : 1);
  if (!cheri_gettag(p) || cheri_getbase(p) != cheri_getaddress(b->manager) ||
      cheri_getlen(p) > b->size)
    pym_fail(721);
  b->requested = requested;
  b->active = 1;
  b->client = cheri_clearperm(p, CHERI_PERM_POISON | CHERI_PERM_SW_VMEM);
  return b->client;
}
void *pym_release(void *ptr) {
  pym_validate(ptr);
  struct block *b = block_for(ptr);
  invalidate(b->manager, b->size);
  b->active = 0;
  b->client = NULL;
  return b->manager;
}
void *pym_resize(void *ptr, size_t requested) {
  pym_validate(ptr);
  struct block *b = block_for(ptr);
  if (requested > b->size)
    pym_fail(722);
  void *snapshot = NULL;
  if (protected_mode) {
    snapshot = pym_raw_malloc(b->size);
    if (!snapshot)
      return NULL; /* Failed realloc leaves the original lease intact. */
    memcpy(snapshot, b->manager, b->size);
    copied_bytes += b->size;
    if (b->size > snapshot_peak)
      snapshot_peak = b->size;
  }
  void *manager = pym_release(ptr);
  if (snapshot) {
    memcpy(manager, snapshot, b->size);
    copied_bytes += b->size;
    pym_raw_free(snapshot);
  }
  return pym_issue(manager, requested);
}
size_t pym_requested(void *ptr) {
  pym_validate(ptr);
  return block_for(ptr)->requested;
}
void *pym_user_raw_malloc(size_t n) {
  if (n > PYM_ARENA_BYTES / 2 - 16)
    return NULL;
  size_t rounded = CHERI_REPRESENTABLE_LENGTH(n ? (n + 15) & ~(size_t)15 : 16);
  for (unsigned i = 0; i < large_used; ++i)
    if (!large[i].active && large[i].size == rounded)
      return large[i].manager;
  size_t mask = CHERI_REPRESENTABLE_ALIGNMENT_MASK(rounded);
  size_t offset = (large_cursor + ~mask) & mask;
  if (large_used == LARGE_COUNT || offset > PYM_ARENA_BYTES - rounded)
    return NULL;
  struct block *b = &large[large_used++];
  b->manager = manager_pointer(backing + offset, rounded);
  b->size = rounded;
  large_cursor = offset + rounded;
  return b->manager;
}
void pym_user_raw_free(void *ptr) { (void)ptr; }
void *pym_user_raw_realloc(void *ptr, size_t n) {
  pym_validate(ptr);
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
uint64_t pym_decision_checksum(void) { return 0; }
void pym_backing_stats(struct pym_header *h) {
  h->arenas = arena_allocations;
  h->arena_frees = arena_releases;
  h->metadata = pym_metadata_used();
  printf("PYM_POISONCAP mode=%u sweeps=%zu poison_bytes=%zu clear_bytes=%zu "
         "zeroed_bytes=%zu copied_bytes=%zu snapshot_peak=%zu reclasses=%zu pointer_bytes=%zu\n",
         protected_mode, sweeps, poison_bytes, cleared_bytes, zeroed_bytes, copied_bytes,
         snapshot_peak, reclassifications, sizeof(void *));
}
