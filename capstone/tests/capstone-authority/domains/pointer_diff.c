// Authority suite: difference of two pointers into the same array.
//
// Under test: pointer subtraction is a pure INTEGER computation (it reads the
// two cursors and subtracts), not a capability operation. It neither traps nor
// produces authority. Answers the PI's "how is ptr difference implemented?".
//
// Oracle: ok, retval = 0x21FF0007 (b - a == 7 elements).

static int arr[16];

void domain_main(unsigned *res, unsigned func) {
  (void)func;
  int *a = &arr[3];
  int *b = &arr[10];
  long d = b - a; // integer difference == 7
  *res = 0x21FF0000u | (unsigned)(d & 0xffff);
}
