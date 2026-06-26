// Authority suite: positive control for object-granularity HEAP bounds.
//
// A bump allocator narrows each returned pointer to exactly the requested size
// with __builtin_capstone_cap_shrink (the malloc analogue of
// -capstone-shrink-globals; see rv8_malloc.c). In-bounds access to the
// allocation works (no false trap). Pairs with heap_oob.c.
//
// Oracle: ok, retval = 0x4EA00028 (p[40] == 40 == 0x28).

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
  for (int i = 0; i < 64; i++)
    p[i] = (unsigned char)i;
  *res = 0x4EA00000u | (unsigned)p[40]; // in-bounds read of the 64-byte block
}
