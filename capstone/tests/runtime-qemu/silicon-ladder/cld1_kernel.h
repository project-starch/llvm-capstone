#ifndef CLD1_KERNEL_H
#define CLD1_KERNEL_H
/* SIMPLEST POSSIBLE R-14 probe -- no struct, no array, no loop, no strlen.
 * 1 straight-line read(s) of ONE global => 1 x `ldc gp[k]` from the SAME cap-table
 * slot at -O0, and nothing else. Isolates repeated ldc-from-gp with no stack stores,
 * no computed addresses and no control flow at all. Expect 65. */
static char g0[2] = { 'A', 0 };
static unsigned cld1_compute(void)
{
  unsigned r = 0;
  r += (unsigned)(unsigned char)g0[0];
  return r;
}
#endif
