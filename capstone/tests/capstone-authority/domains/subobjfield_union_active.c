// Authority suite: union members are REFUSED by subobject narrowing (overlapping
// members), so a normal union-member array access must NOT trap.
//
// Built with -fcapstone-subobject-bounds. `a` is a union member, so the frontend
// leaves it un-narrowed; write+read of a[0] works and does not fault.
//
// Oracle: ok, retval = 0x220C005A = 571211866.

union u_arr {
  unsigned char a[8];
  unsigned long i;
};

static union u_arr obj;

void domain_main(unsigned *res, unsigned func) {
  (void)func;
  obj.a[0] = 0x5A;
  volatile unsigned k = 0;
  *res = 0x220C0000u | (unsigned)obj.a[k];
}
