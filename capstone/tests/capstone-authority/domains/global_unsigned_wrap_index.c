// Authority suite: wrapped unsigned index outside a narrowed global object.
//
// Under test: an all-ones unsigned index wraps the byte address to one byte
// before the object; the resulting access must still fail the capability bound.
// The C access is deliberately undefined and is runtime evidence at -O0.
//
// Oracle: bounds-fault.

static unsigned char object[64];

void domain_main(unsigned *res, unsigned func) {
  (void)func;
  volatile unsigned long index = ~0UL;
  *res = (unsigned)object[index];
}
