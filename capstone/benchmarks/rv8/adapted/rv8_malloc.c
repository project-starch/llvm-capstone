/*
 * Minimal 16-byte-aligned bump allocator for RV8 domain benchmarks.
 *
 * Many rv8-bench programs `malloc` small records; on Capstone those records can
 * hold capability fields (e.g. dhrystone's `RecordType.PtrComp`), so every
 * allocation must be 16-byte aligned or the stored capability loses its tag
 * (the same constraint that bit dtoa's bigint arena). The backing store is a
 * 16-aligned static array and sizes are rounded up to 16 bytes (integer
 * rounding only -- no pointer forging), so the bump pointer stays 16-aligned.
 * `free` is a no-op (benchmarks allocate a bounded amount and never need reuse);
 * `rv8_arena_init` resets the arena and must be called from initialise_benchmark.
 */
#include "rv8_capstone_preamble.h"

#ifndef RV8_HEAP_SIZE
#define RV8_HEAP_SIZE 65536
#endif

static char rv8_heap[RV8_HEAP_SIZE] __attribute__((aligned(16)));
static size_t rv8_off;

void rv8_arena_init(void) { rv8_off = 0; }

void *malloc(size_t n) {
  n = (n + 15u) & ~(size_t)15u; /* round to 16; keeps rv8_off 16-aligned */
  if (n == 0 || rv8_off + n > RV8_HEAP_SIZE)
    return NULL;
  void *p = &rv8_heap[rv8_off]; /* rv8_heap is 16-aligned; tag-preserving GEP */
  rv8_off += n;
  return p;
}

void free(void *p) { (void)p; }
