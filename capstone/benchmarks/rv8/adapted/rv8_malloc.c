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
  char *p = base + RV8_HDR;
  /* Object-granularity heap bounds (the malloc analogue of
     -capstone-shrink-globals): narrow the returned capability to exactly the
     requested n bytes so an over-read/-write past the allocation faults. The
     block lies within rv8_heap, so SHRINK's monotonicity holds. (For n >= 4 KiB
     the in-register bounds are exact; a store/reload may round outward by one
     representable grain -- see design/capability-bounds-model.md.) The size
     header at p-16 is recovered by realloc through the wide arena capability
     below, never through this narrowed user pointer. */
  unsigned long b = __builtin_capstone_cap_get_cursor(p);
  return __builtin_capstone_cap_shrink(p, b, b + n);
}

void *realloc(void *p, size_t n) {
  if (!p)
    return malloc(n);
  /* p is narrowed to its own [payload, payload+old) and cannot reach its size
     header at p-16. Recover it through the wide arena capability: the header
     lives at rv8_heap[poff - RV8_HDR], where poff is p's offset in the arena. */
  unsigned long poff = __builtin_capstone_cap_get_cursor((char *)p) -
                       __builtin_capstone_cap_get_cursor((char *)rv8_heap);
  size_t old = *(size_t *)&rv8_heap[poff - RV8_HDR];
  void *np = malloc(n);
  if (!np)
    return NULL;
  size_t copy = old < n ? old : n; /* copy <= old, so reads of p stay in bounds */
  /* Inline byte copy so this allocator stays self-contained (no dependency on
     the string lib, which not every benchmark build links). */
  char *d = (char *)np;
  const char *s = (const char *)p;
  for (size_t i = 0; i < copy; i++)
    d[i] = s[i];
  return np;
}

void free(void *p) { (void)p; }
