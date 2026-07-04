// Authority suite: an allocation REUSED after free() is re-narrowed, so an
// out-of-bounds read of the reused block still traps.
//
// Unlike heap_oob.c (which inlines a bump allocator), this probe drives the REAL
// vendored allocator (umm_malloc behind the cap_heap.c narrow/re-widen shim, the
// same code the RV8 domains link). We malloc a 64-byte block, free it, then
// malloc another 64-byte block. umm's best-fit returns the just-freed slot, and
// the shim SHRINK-narrows the reused pointer to exactly [q, q+64). Reading q[100]
// leaves the reused object (but stays inside umm's arena), so a non-narrowing
// allocator would silently read adjacent heap -- here it traps. This proves free
// -> reuse does not defeat object-granularity heap bounds (C1).
//
// (Temporal safety -- catching a use of the STALE pointer p after free -- is a
// phase-2 revoke-on-free property, not tested here.)
//
// Oracle: bounds-fault.

#include <stddef.h>

// The suite links no libc; umm uses mem*. These probes store only ints (no
// capabilities), so a plain byte-loop is sufficient (no tag preservation needed).
void *memset(void *d, int c, size_t n) {
  unsigned char *p = d;
  while (n--) { *p++ = (unsigned char)c; }
  return d;
}
void *memcpy(void *d, const void *s, size_t n) {
  unsigned char *a = d; const unsigned char *b = s;
  while (n--) { *a++ = *b++; }
  return d;
}
void *memmove(void *d, const void *s, size_t n) {
  unsigned char *a = d; const unsigned char *b = s;
  if (a < b) { while (n--) { *a++ = *b++; } }
  else { a += n; b += n; while (n--) { *--a = *--b; } }
  return d;
}

// The real allocator, compiled into this TU (umm defines umm_*; cap_heap defines
// malloc/free/realloc + the narrow/re-widen shim). Quoted includes resolve
// relative to each file's own directory.
#include "../../../benchmarks/rv8/adapted/umm/umm_malloc.c"
#include "../../../benchmarks/rv8/adapted/cap_heap.c"

void domain_main(unsigned *res, unsigned func) {
  (void)func;
  unsigned char *p = malloc(64);
  p[0] = 0xAB;                             // use the first allocation
  free(p);                                 // return the slot to umm's free list
  unsigned char *q = malloc(64);           // umm best-fit reuses p's slot
  q[0] = 0xCD;                             // in-bounds write to the reused block
  volatile unsigned idx = 100;             // q[100]: past the 64-byte reused block
  *res = 0x5EB00000u | (unsigned)q[idx];   // OOB for the narrowed reuse -> trap
}
