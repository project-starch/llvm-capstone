/*
 * cap_heap.c -- a bounded heap allocator for Capstone domains, built on
 * umm_malloc (Ralph Hempel's embedded allocator, MIT, vendored under umm/).
 *
 * Why umm and not dlmalloc: a PureCap capability pointer is 16 bytes (and must
 * stay 16-aligned to keep its tag) while size_t is 8. dlmalloc stores its
 * free-list links (fd/bk/...) *in-band* as capability pointers and navigates
 * chunks with hand-computed SIZE_T_SIZE offset arithmetic, which assumes
 * sizeof(ptr)==sizeof(size_t). On PureCap that breaks the moment a freed chunk
 * enters a bin -> corruption -> abort (see history/04-07-2026_*_dlmalloc-purecap
 * -metadata-incompat.md). umm instead keeps ALL metadata as uint16_t block
 * *indices*; it never stores a capability in heap metadata, so it is PureCap-safe
 * by construction. The only port change is 16-byte alignment (umm/umm_malloc_cfg.h).
 *
 * Capability bounding (C1) is done with a thin boundary shim, NOT by changing the
 * allocator:
 *   - malloc/calloc/realloc: call umm (returns a WIDE arena pointer), then SHRINK
 *     the result to exactly the request at the return.
 *   - free/realloc: RE-WIDEN the incoming user pointer back to a full-arena
 *     capability (cincoffset off the retained wide arena cap) before handing it to
 *     umm -- so umm's block bookkeeping runs unchanged on wide caps.
 * Untrusted in-domain code only ever holds the narrowed capability, so an
 * over-read/-write past an allocation faults; the allocator is the heap TCB.
 *
 * umm's realloc-move copies via memmove/memcpy; the RV8/BEEBS builds link the
 * tag-preserving mem* (beebs_freestanding_string.c), so moving a capability-bearing
 * block preserves tags with no allocator change.
 *
 * Arena is a 16-byte-aligned static array; size via -DCAPSTONE_HEAP_SIZE. Block
 * size is 32 B (umm index ceiling: 32767 blocks -> ~1 MB max per 32 B block; raise
 * -DUMM_BLOCK_BODY_SIZE for larger heaps).
 */

#include <stddef.h> /* size_t */
#include "umm/umm_malloc.h"

#ifndef CAPSTONE_HEAP_SIZE
#define CAPSTONE_HEAP_SIZE 65536
#endif

static char cap_arena[CAPSTONE_HEAP_SIZE] __attribute__((aligned(16)));
static int cap_heap_ready;

/* umm's single-heap init path references these externs (umm_init/umm_multi_init).
 * We init explicitly via umm_init_heap(), but define them so the unused init path
 * still links, and point them at our arena for good measure. */
void *UMM_MALLOC_CFG_HEAP_ADDR;
uint32_t UMM_MALLOC_CFG_HEAP_SIZE;

static void cap_heap_init(void) {
  UMM_MALLOC_CFG_HEAP_ADDR = cap_arena;
  UMM_MALLOC_CFG_HEAP_SIZE = (uint32_t)CAPSTONE_HEAP_SIZE;
  umm_init_heap(cap_arena, (size_t)CAPSTONE_HEAP_SIZE);
  cap_heap_ready = 1;
}

/* ---- capability boundary shim ---- */

/* Re-widen a narrowed user pointer to a full-arena capability at the same
 * address, via the retained wide arena capability (cincoffset). The user
 * pointer's cursor is read only as a scalar address -- no forging. */
static inline void *cap_rewiden(void *user_p) {
  unsigned long abase = __builtin_capstone_cap_get_cursor((char *)cap_arena);
  unsigned long addr = __builtin_capstone_cap_get_cursor((char *)user_p);
  return (void *)(cap_arena + (addr - abase));
}

/* Narrow a wide arena pointer to exactly [p, p+n). */
static inline void *cap_narrow(void *p, size_t n) {
  unsigned long c = __builtin_capstone_cap_get_cursor((char *)p);
  return __builtin_capstone_cap_shrink(p, c, c + n);
}

void *malloc(size_t n) {
  if (!cap_heap_ready)
    cap_heap_init();
  if (n == 0)
    return 0;
  void *p = umm_malloc(n);
  if (!p)
    return 0;
  return cap_narrow(p, n);
}

void free(void *user_p) {
  if (!user_p)
    return;
  umm_free(cap_rewiden(user_p));
}

void *realloc(void *user_p, size_t n) {
  if (!cap_heap_ready)
    cap_heap_init();
  if (!user_p)
    return malloc(n);
  if (n == 0) {
    free(user_p);
    return 0;
  }
  void *p = umm_realloc(cap_rewiden(user_p), n);
  if (!p)
    return 0;
  return cap_narrow(p, n);
}

void *calloc(size_t nmemb, size_t size) {
  if (!cap_heap_ready)
    cap_heap_init();
  size_t total = nmemb * size;
  if (total == 0)
    return 0;
  /* guard the nmemb*size overflow umm_calloc does not check */
  if (size != 0 && total / size != nmemb)
    return 0;
  void *p = umm_calloc(nmemb, size);
  if (!p)
    return 0;
  return cap_narrow(p, total);
}

/* RV8 harness compatibility: reset the arena at benchmark start. Re-inits umm so
 * a re-run starts from a single free block. */
void rv8_arena_init(void) { cap_heap_init(); }
