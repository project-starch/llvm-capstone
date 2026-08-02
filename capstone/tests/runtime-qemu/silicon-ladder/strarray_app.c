/* MINIMAL REPRODUCER for the untagged-cincoffset codegen bug (2026-08-02).

   A straight-line local array of structs whose first member is a string literal makes the
   compiler emit `cincoffset` with an UNTAGGED base. QEMU catches it as
   `helper_cscincoffset: Assertion rs1_v->tag failed`; the RTL does not check a cincoffset
   base, so on silicon the instruction yields a garbage pointer and execution continues.

   Reduced from sqlite3RegisterBuiltinFunctions, which builds exactly this shape. Needs no
   board, no monitor and no SQLite. Eight elements is enough -- the fault is NOT size- or
   register-pressure-dependent (verified at N=8/32/48/56).  */
struct fd { const char *z; void *p1; void *p2; unsigned char f; };

/* Freestanding memset: the struct has 15 bytes of tail padding after `f`, and the
   aggregate initialiser zero-fills it with a memset call. No libc here. */
void *memset(void *d, int c, unsigned long n)
{
  unsigned char *p = (unsigned char *)d;
  while (n--) *p++ = (unsigned char)c;
  return d;
}

static unsigned build_str_array(void)
{
  struct fd a[] = {
    { "fn0", (void *)0, (void *)0, (unsigned char)0 },
    { "fn1", (void *)0, (void *)0, (unsigned char)1 },
    { "fn2", (void *)0, (void *)0, (unsigned char)2 },
    { "fn3", (void *)0, (void *)0, (unsigned char)3 },
    { "fn4", (void *)0, (void *)0, (unsigned char)4 },
    { "fn5", (void *)0, (void *)0, (unsigned char)5 },
    { "fn6", (void *)0, (void *)0, (unsigned char)6 },
    { "fn7", (void *)0, (void *)0, (unsigned char)7 },
  };
  unsigned n = (unsigned)(sizeof(a) / sizeof(a[0])), s = 0, i;
  for (i = 0; i < n; i++)
    s += (unsigned)(unsigned char)a[i].z[2];   /* the digit character */
  return s + n;
}

void domain_main(unsigned *res, unsigned func) { (void)func; *res = build_str_array(); }
