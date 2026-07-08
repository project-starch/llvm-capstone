// Authority suite: subobject-bounds narrowing flips the struct-field over-read
// from no-trap-today to a bounds-fault.
//
// Built with -fcapstone-subobject-bounds (the build script maps subobjfield_* to
// that flag). `first` is a NON-LAST array field, so the frontend narrows its
// capability to [&first, &first+8). Reading first[8] therefore leaves the field
// and traps -- unlike subobject_overread.c (same access, built WITHOUT the flag),
// which documents the un-narrowed no-trap-today gap.
//
// Oracle: bounds-fault.

struct adjacent_fields {
  unsigned char first[8];
  unsigned char second[8];
};

static struct adjacent_fields object = {{0}, {0xA5}};

void domain_main(unsigned *res, unsigned func) {
  (void)func;
  volatile unsigned index = 8;            // one past first[] -> into second[]
  *res = 0x220A0000u | (unsigned)object.first[index]; // OOB for narrowed first[]
}
