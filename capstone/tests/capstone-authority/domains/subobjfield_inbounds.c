// Authority suite: an in-field access still works with subobject narrowing on.
//
// Built with -fcapstone-subobject-bounds. `first` is narrowed to [&first,+8);
// first[7] is the last valid byte, so it reads correctly and does NOT trap.
//
// Oracle: ok, retval = 0x220B003C = 571146300.

struct adjacent_fields {
  unsigned char first[8];
  unsigned char second[8];
};

static struct adjacent_fields object = {{[7] = 0x3C}, {0xA5}};

void domain_main(unsigned *res, unsigned func) {
  (void)func;
  volatile unsigned index = 7;            // last valid byte of first[]
  *res = 0x220B0000u | (unsigned)object.first[index];
}
