// Authority suite: flexible array members are REFUSED by subobject narrowing
// (deliberately over-indexed past their declared size), so a FAM access beyond
// the declared array must NOT trap.
//
// Built with -fcapstone-subobject-bounds. `data[]` is an incomplete/flexible
// array member, so the frontend leaves it un-narrowed. f->data[10] stays inside
// the 20-byte `backing` global (narrowed by object-granularity SHRINK to its own
// 20 bytes), so it reads correctly. If the FAM had been (wrongly) narrowed to its
// declared size 0, this would trap -- the refusal is what keeps it in-bounds.
//
// Oracle: ok, retval = 0x220D005E = 571277406.

struct flex {
  unsigned n;
  unsigned char data[];
};

static unsigned char backing[sizeof(unsigned) + 16] __attribute__((aligned(16)));

void domain_main(unsigned *res, unsigned func) {
  (void)func;
  struct flex *f = (struct flex *)backing;
  volatile unsigned k = 10;               // within backing, past declared data[]
  f->data[k] = 0x5E;
  *res = 0x220D0000u | (unsigned)f->data[k];
}
