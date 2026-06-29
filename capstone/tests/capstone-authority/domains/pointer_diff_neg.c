// Negative pointer difference regression. Pointer subtraction projects both
// capabilities to integer cursors, subtracts them, and performs signed scaling
// by sizeof(int). The result must remain negative.

static int arr[16];

void domain_main(unsigned *res, unsigned func) {
  (void)func;
  int *high = &arr[10];
  int *low = &arr[3];
  long difference = low - high;
  *res = difference == -7 ? 0x21FE0007u : 0u;
}
