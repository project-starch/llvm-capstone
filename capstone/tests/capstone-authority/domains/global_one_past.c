// Authority suite: one-past-end access to a narrowed global object.
//
// Under test: forming a one-past pointer is valid, but dereferencing byte 64 of
// a 64-byte global must be rejected by the object's exclusive upper bound.
//
// Oracle: bounds-fault.

static unsigned char object[64];

void domain_main(unsigned *res, unsigned func) {
  (void)func;
  volatile unsigned index = 64;
  *res = (unsigned)object[index];
}
