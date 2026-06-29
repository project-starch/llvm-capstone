// Authority suite: last-valid-byte access to a narrowed stack object.
//
// Under test: with -capstone-shrink-stack, byte 63 of a 64-byte local remains
// accessible and does not produce a false bounds fault.
//
// Oracle: ok, retval = 0x2202003F.

void domain_main(unsigned *res, unsigned func) {
  (void)func;
  volatile unsigned char object[64];
  for (unsigned i = 0; i < 64; ++i)
    object[i] = (unsigned char)i;
  volatile unsigned index = 63;
  *res = 0x22020000u | (unsigned)object[index];
}
