// Authority suite: regression guard for stack-passed capability arguments.
//
// Under test: the 9th and 10th pointer arguments are passed on the stack (only
// the first 8 go in registers). A previously-fixed backend bug computed the
// outgoing stack slot with an integer add instead of CIncOffset, delivering
// those capabilities UNTAGGED -> dereferencing them tag-faulted. This test
// passes 10 pointers and sums through all of them; if the bug regresses, the
// 9th/10th load traps with "Cap mem access requires capability".
//
// Oracle: ok, retval = 0x09A00037 (sum 1..10 == 55 == 0x37).

static int v[12];

__attribute__((noinline)) static int sink(int *a, int *b, int *c, int *d,
                                          int *e, int *f, int *g, int *h,
                                          int *i, int *j) {
  return *a + *b + *c + *d + *e + *f + *g + *h + *i + *j;
}

void domain_main(unsigned *res, unsigned func) {
  (void)func;
  for (int k = 0; k < 12; k++)
    v[k] = k + 1;
  int s = sink(&v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7], &v[8],
               &v[9]);
  *res = 0x09A00000u | (unsigned)s;
}
