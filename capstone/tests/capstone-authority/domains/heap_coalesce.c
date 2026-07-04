// Authority suite: memory formed by COALESCING two freed adjacent blocks is
// re-narrowed when reallocated, so an out-of-bounds read still traps.
//
// Drives the REAL vendored allocator (umm_malloc behind cap_heap.c). With a tight
// 384-byte arena we allocate three 64-byte blocks a,b,c (c stays live, bounding
// the hole from above), then free a and b -- umm coalesces them into one interior
// free region. A 120-byte malloc then cannot fit in the tiny trailing free space,
// so best-fit carves d from the coalesced a+b hole. The shim SHRINK-narrows d to
// [d, d+120); reading d[200] leaves d (into c's still-live region, but inside the
// arena), so it traps. This proves umm's boundary-tag coalescing does not defeat
// object-granularity heap bounds (C1).
//
// Oracle: bounds-fault.

#include <stddef.h>

#define CAPSTONE_HEAP_SIZE 384  // tight arena: force the 120B alloc to reuse a+b

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

#include "../../../benchmarks/rv8/adapted/umm/umm_malloc.c"
#include "../../../benchmarks/rv8/adapted/cap_heap.c"

void domain_main(unsigned *res, unsigned func) {
  (void)func;
  unsigned char *a = malloc(64);
  unsigned char *b = malloc(64);
  unsigned char *c = malloc(64);   // stays live: upper bound of the coalesced hole
  (void)c;
  free(a);                          // free a, then b -> umm coalesces a+b
  free(b);
  unsigned char *d = malloc(120);   // carved from the coalesced a+b region
  d[0] = 0xEF;                      // in-bounds write
  volatile unsigned idx = 200;      // d[200]: past the 120-byte narrowed block
  *res = 0x60B00000u | (unsigned)d[idx];  // OOB for the coalesced reuse -> trap
}
