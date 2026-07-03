// Authority suite: out-of-bounds access of a SHRINK-narrowed DYNAMIC alloca.
//
// n is volatile so __builtin_alloca(n) is a runtime-sized (dynamic) alloca.
// Without narrowing the alloca pointer inherits the whole-stack bounds, so
// buf[200] silently reads adjacent stack memory. With -capstone-shrink-stack on,
// lowerDYNAMIC_STACKALLOC narrows the returned pointer to the allocated region,
// so buf[200] (well past the 48-byte, 16-aligned allocation) traps. Dynamic-
// alloca analogue of stack_oob.
//
// Oracle: bounds-fault.

void domain_main(unsigned *res, unsigned func) {
  (void)func;
  volatile unsigned n = 48;
  volatile unsigned char *buf = (volatile unsigned char *)__builtin_alloca(n);
  for (unsigned i = 0; i < n; i++)
    buf[i] = (unsigned char)i;
  volatile unsigned idx = 200; // OOB for alloca(48); runtime index via narrowed ptr
  *res = 0x57B00000u | (unsigned)buf[idx];
}
