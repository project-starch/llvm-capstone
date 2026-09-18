/* Reusable backing storage for the real allocator and its raw fallback.
 * Arena authority is retained as a pointer, never reconstructed from an
 * integer. The private metadata heap is separate from pymalloc arenas in both
 * ABIs.
 */
#include "port.h"
#include <string.h>
struct raw_block {
  struct raw_block *next;
  size_t size;
};
static unsigned char *metadata;
static size_t used;
static struct raw_block *available;
#ifndef PYMALLOC_DOMAIN
static unsigned char *arena;
static size_t arena_used, arena_count, arena_releases;
static struct {
  unsigned char *base;
  unsigned live;
} arenas[32];
static uint64_t decisions;
#endif
void pym_backing_init(void *m, void *a) {
  metadata = m;
#ifndef PYMALLOC_DOMAIN
  arena = a;
  /* Pools must be 16 KiB aligned; arenas need not be 1 MiB aligned. */
  arena_used = (-(uintptr_t)a) & 16383;
#else
  (void)a;
#endif
}
void *pym_raw_malloc(size_t n) {
  if (n > PYM_META_BYTES - 64)
    return NULL;
  n = (n + 63) & ~(size_t)63;
  if (!n)
    n = 64;
  struct raw_block **link = &available;
  while (*link) {
    struct raw_block *b = *link;
    if (b->size == n) {
      *link = b->next;
      return (unsigned char *)b + 64;
    }
    link = &b->next;
  }
  if (used > PYM_META_BYTES - 64 - n)
    return NULL;
  struct raw_block *b = (void *)(metadata + used);
  b->size = n;
  used += n + 64;
  return (unsigned char *)b + 64;
}
void pym_raw_free(void *p) {
  if (!p)
    return;
  struct raw_block *b = (void *)((unsigned char *)p - 64);
  b->next = available;
  available = b;
}
void *pym_raw_calloc(size_t k, size_t n) {
  if (n && k > SIZE_MAX / n)
    return NULL;
  void *p = pym_raw_malloc(k * n);
  if (p)
    memset(p, 0, k * n);
  return p;
}
void *pym_raw_realloc(void *p, size_t n) {
  if (!p)
    return pym_raw_malloc(n);
  struct raw_block *b = (void *)((unsigned char *)p - 64);
  if (n <= b->size && n > b->size / 2)
    return p;
  void *q = pym_raw_malloc(n);
  if (q) {
    memcpy(q, p, n < b->size ? n : b->size);
    pym_raw_free(p);
  }
  return q;
}
#ifndef PYMALLOC_DOMAIN
void *pym_arena_alloc(void *ctx, size_t n) {
  (void)ctx;
  if (n != 1048576)
    pym_fail(201);
  for (unsigned i = 0; i < 32; ++i) {
    if (arenas[i].live)
      continue;
    if (!arenas[i].base) {
      if (arena_used > PYM_ARENA_BYTES - n)
        return NULL;
      arenas[i].base = arena + arena_used;
      arena_used += n;
    }
    arenas[i].live = 1;
    ++arena_count;
    return arenas[i].base;
  }
  return NULL;
}
void *pym_arena_pointer(uintptr_t address) {
  for (unsigned i = 0; i < 32; ++i)
    if (arenas[i].live && (uintptr_t)arenas[i].base == address)
      return arenas[i].base;
  pym_fail(202);
}
void pym_arena_free(void *ctx, void *p, size_t n) {
  (void)ctx;
  if (n != 1048576)
    pym_fail(203);
  for (unsigned i = 0; i < 32; ++i)
    if (arenas[i].live && arenas[i].base == p) {
      arenas[i].live = 0;
      ++arena_releases;
      return;
    }
  pym_fail(204);
}
void *pym_pool_pointer(const void *p) {
  uintptr_t address = (uintptr_t)p & ~(uintptr_t)16383;
  for (unsigned i = 0; i < 32; ++i) {
    uintptr_t base = (uintptr_t)arenas[i].base;
    if (arenas[i].live && address >= base && address - base < 1048576)
      return arenas[i].base + (address - base);
  }
  return NULL; /* raw fallback: the radix tree rejects this address */
}
void pym_backing_stats(struct pym_header *h) {
  h->arenas = arena_count;
  h->arena_frees = arena_releases;
  h->metadata = used;
}
uintptr_t pym_arena_address(void *p) { return (uintptr_t)p; }
void *pym_pool_create(uintptr_t address, size_t overhead) {
  (void)overhead;
  return pym_pool_pointer((void *)address);
}
void pym_pool_reclass(void *p, size_t size, size_t overhead) {
  (void)p;
  (void)size;
  (void)overhead;
}
void *pym_block_pointer(void *p, size_t offset) {
  return (unsigned char *)p + offset;
}
void *pym_issue(void *p, size_t n) {
  (void)n;
  return p;
}
void *pym_release(void *p) { return p; }
void *pym_resize(void *p, size_t n) {
  (void)n;
  return p;
}
size_t pym_requested(void *p) {
  (void)p;
  return SIZE_MAX;
}
void pym_validate(void *p) { (void)p; }
void *pym_user_raw_malloc(size_t n) { return pym_raw_malloc(n); }
void pym_user_raw_free(void *p) { pym_raw_free(p); }
void *pym_user_raw_realloc(void *p, size_t n) { return pym_raw_realloc(p, n); }
void pym_set_mode(unsigned mode) {
  if (mode)
    pym_fail(205);
}
void pym_observe(unsigned op, void *ptr, size_t size) {
  if (!ptr)
    return;
  uint64_t location = (uintptr_t)ptr - (uintptr_t)metadata + PYM_ARENA_BYTES;
  for (unsigned i = 0; i < 32; ++i) {
    uintptr_t base = (uintptr_t)arenas[i].base;
    if (arenas[i].live && (uintptr_t)ptr >= base &&
        (uintptr_t)ptr - base < 1048576) {
      location = i * 1048576UL + (uintptr_t)ptr - base;
      break;
    }
  }
  decisions = decisions * 33 ^ (location + size * 13 + op);
}
uint64_t pym_decision_checksum(void) { return decisions; }
#else
size_t pym_metadata_used(void) { return used; }
void pym_observe(unsigned op, void *ptr, size_t size) {
  (void)op;
  (void)ptr;
  (void)size;
}
#endif
