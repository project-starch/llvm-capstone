// Authority suite: out-of-bounds read of a SHRINK-narrowed heap allocation.
//
// The allocator narrows the returned pointer to exactly the requested 64 bytes,
// even though the backing arena (4096 bytes) extends well past it. Reading
// p[100] therefore leaves the allocation but stays inside the arena -- so under
// a non-narrowing allocator (the old behaviour) it would silently read an
// adjacent allocation, but with object-granularity heap bounds it traps.
//
// This is the heap analogue of global_oob.c and the malloc-side demonstration
// of the granularity contribution.
//
// Oracle: bounds-fault.

typedef unsigned long size_t;

static char arena[4096] __attribute__((aligned(16)));
static size_t off;

static void *halloc(size_t n) {
  char *base = &arena[off];
  off += (n + 15u) & ~(size_t)15u;
  unsigned long b = __builtin_capstone_cap_get_cursor(base);
  return __builtin_capstone_cap_shrink(base, b, b + n); // narrow to [base, base+n)
}

void domain_main(unsigned *res, unsigned func) {
  (void)func;
  unsigned char *p = halloc(64);
  (void)halloc(64);                     // a second allocation after p in the arena
  volatile unsigned idx = 100;          // p[100]: past the 64-byte allocation
  *res = 0x4EB00000u | (unsigned)p[idx]; // OOB for the narrowed block -> trap
}
