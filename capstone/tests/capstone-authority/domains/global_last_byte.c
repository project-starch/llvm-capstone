// Authority suite: last-valid-byte access to a narrowed global object.
//
// Under test: the upper bound is exclusive, so byte 63 of a 64-byte global
// remains accessible after object-granularity SHRINK.
//
// Oracle: ok, retval = 0x2201003F.

static unsigned char object[64];

void domain_main(unsigned *res, unsigned func) {
  (void)func;
  for (unsigned i = 0; i < 64; ++i)
    object[i] = (unsigned char)i;
  volatile unsigned index = 63;
  *res = 0x22010000u | (unsigned)object[index];
}
