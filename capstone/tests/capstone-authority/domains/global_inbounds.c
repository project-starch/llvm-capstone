// Authority suite: positive control -- in-bounds access to a global array.
//
// Under test: ordinary, correct global access works (no false trap). Pairs with
// global_oob.c as the in-bounds half of the granularity before/after demo.
//
// Oracle: ok, retval = 0x2110C028 (a[40] == 40 == 0x28).

static unsigned char a[64];

void domain_main(unsigned *res, unsigned func) {
  (void)func;
  for (int i = 0; i < 64; i++)
    a[i] = (unsigned char)i;
  *res = 0x2110C000u | (unsigned)a[40];
}
