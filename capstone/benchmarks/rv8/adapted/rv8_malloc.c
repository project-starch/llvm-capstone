/*
 * Minimal 16-byte-aligned bump allocator for RV8 domain benchmarks.
 *
 * Many rv8-bench programs `malloc` records that hold capability fields (e.g.
 * dhrystone's `RecordType.PtrComp`), so every allocation must be 16-byte aligned
 * or the stored capability loses its tag (the constraint that bit dtoa's bigint
 * arena). The backing store is a 16-aligned static array; each allocation gets a
 * 16-byte header recording its requested size (so `realloc` can copy), and the
 * returned payload is 16-aligned. `free` is a no-op (benchmarks allocate a
 * bounded amount); `rv8_arena_init` resets the arena and must be called from
 * initialise_benchmark. Size the arena per benchmark via -DRV8_HEAP_SIZE.
 */
#include "rv8_capstone_preamble.h"

#ifndef RV8_HEAP_SIZE
#define RV8_HEAP_SIZE 65536
#endif

#define RV8_HDR 16 /* header bytes; keeps payload 16-aligned */

static char rv8_heap[RV8_HEAP_SIZE] __attribute__((aligned(16)));
static size_t rv8_off;

void rv8_arena_init(void) { rv8_off = 0; }

void *malloc(size_t n) {
  size_t payload = (n + 15u) & ~(size_t)15u; /* round to 16 */
  size_t need = payload + RV8_HDR;
  if (n == 0 || rv8_off + need > RV8_HEAP_SIZE)
    return NULL;
  char *base = &rv8_heap[rv8_off]; /* 16-aligned: heap aligned + rv8_off mult of 16 */
  *(size_t *)base = n;             /* record requested size in the header */
  rv8_off += need;
  return base + RV8_HDR;
}

void *realloc(void *p, size_t n) {
  if (!p)
    return malloc(n);
  size_t old = *(size_t *)((char *)p - RV8_HDR);
  void *np = malloc(n);
  if (!np)
    return NULL;
  size_t copy = old < n ? old : n;
  /* Inline byte copy so this allocator stays self-contained (no dependency on
     the string lib, which not every benchmark build links). */
  char *d = (char *)np;
  const char *s = (const char *)p;
  for (size_t i = 0; i < copy; i++)
    d[i] = s[i];
  return np;
}

void free(void *p) { (void)p; }
