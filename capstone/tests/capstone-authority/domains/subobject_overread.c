// Authority suite: over-read from one struct field into its adjacent field.
//
// Under test: global SHRINK currently narrows to the complete struct object,
// not to individual fields. Reading first[8] therefore reaches second[0]
// without crossing the allocation bound, documenting the subobject gap.
// The C access is deliberately undefined and is runtime evidence at -O0.
//
// Oracle today: no-trap-today, retval = 0x220700A5.

struct adjacent_fields {
  unsigned char first[8];
  unsigned char second[8];
};

static struct adjacent_fields object = {{0}, {0xA5}};

void domain_main(unsigned *res, unsigned func) {
  (void)func;
  volatile unsigned index = 8;
  *res = 0x22070000u | (unsigned)object.first[index];
}
