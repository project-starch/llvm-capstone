// Authority suite: in-bounds access of a SHRINK-narrowed DYNAMIC alloca.
//
// n is volatile so __builtin_alloca(n) is a runtime-sized (dynamic) alloca,
// lowered via ISD::DYNAMIC_STACKALLOC -- which never reaches a FrameIndex, so
// the fixed-object narrowing (narrowToFrameObjectBounds) does not cover it.
// With -capstone-shrink-stack on, lowerDYNAMIC_STACKALLOC narrows the returned
// pointer to [base, base+alloc_size) while the real sp keeps broad bounds.
// buf[40] is in bounds, so this must return its value cleanly.
//
// Oracle: ok, 0x5D000028 (1560281128).

void domain_main(unsigned *res, unsigned func) {
  (void)func;
  volatile unsigned n = 48;
  volatile unsigned char *buf = (volatile unsigned char *)__builtin_alloca(n);
  for (unsigned i = 0; i < n; i++)
    buf[i] = (unsigned char)i;
  volatile unsigned idx = 40; // in bounds for alloca(48)
  *res = 0x5D000000u | (unsigned)buf[idx];
}
