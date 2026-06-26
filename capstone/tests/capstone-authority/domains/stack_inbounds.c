// Authority suite: positive control for object-granularity STACK bounds.
//
// With -capstone-shrink-stack (default off; this domain is built with it on), an
// address-taken whole stack object (bare FrameIndex) is narrowed to its size.
// In-bounds access to the local array works (no false trap). Pairs with
// stack_oob.c. `volatile` keeps buf as a real stack object indexed at runtime.
//
// Oracle: ok, retval = 0x57A00028 (buf[40] == 40 == 0x28).

void domain_main(unsigned *res, unsigned func) {
  (void)func;
  volatile unsigned char buf[64];
  for (int i = 0; i < 64; i++)
    buf[i] = (unsigned char)i;
  volatile unsigned idx = 40; // in-bounds; runtime index -> via narrowed &buf
  *res = 0x57A00000u | (unsigned)buf[idx];
}
