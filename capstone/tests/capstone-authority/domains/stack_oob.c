// Authority suite: out-of-bounds read of a SHRINK-narrowed STACK object.
//
// buf is a 64-byte local; buf[100] leaves the object but stays within the
// (broad) stack region, so under the default (no stack narrowing) it silently
// reads adjacent stack memory. Built with -capstone-shrink-stack on, &buf is
// narrowed to [buf, buf+64) so buf[100] traps. Stack analogue of global_oob.
//
// Oracle: bounds-fault.

void domain_main(unsigned *res, unsigned func) {
  (void)func;
  volatile unsigned char buf[64];
  for (int i = 0; i < 64; i++)
    buf[i] = (unsigned char)i;
  volatile unsigned idx = 100; // OOB for buf[64]; runtime index via narrowed &buf
  *res = 0x57B00000u | (unsigned)buf[idx];
}
