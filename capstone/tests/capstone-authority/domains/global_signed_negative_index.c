// Authority suite: signed negative index outside a narrowed global object.
//
// Under test: index -1 lands one byte before the global and must fail the
// capability bound. The C access is deliberately undefined and is runtime
// evidence at -O0.
//
// Oracle: bounds-fault.

static unsigned char object[64];

void domain_main(unsigned *res, unsigned func) {
  (void)func;
  volatile long index = -1;
  *res = (unsigned)object[index];
}
