// Authority suite: one-past-end access to a narrowed stack object.
//
// Under test: with -capstone-shrink-stack, dereferencing byte 64 of a 64-byte
// local must be rejected by the object's exclusive upper bound.
//
// Oracle: bounds-fault.

void domain_main(unsigned *res, unsigned func) {
  (void)func;
  volatile unsigned char object[64];
  volatile unsigned index = 64;
  *res = (unsigned)object[index];
}
