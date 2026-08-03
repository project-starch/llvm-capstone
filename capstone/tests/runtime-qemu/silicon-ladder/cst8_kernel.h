#ifndef CST8_KERNEL_H
#define CST8_KERNEL_H
/* SIMPLEST POSSIBLE R-14 probe -- no struct, no array, no loop, no strlen.
 * 8 loads from ONE slot, each STORED to a distinct local capability (stc to stack at -O0)
 * and read back. Adds the capability STORE that cld8 omits, without any array or loop.
 * Expect 520. */
static char g0[2] = { 'A', 0 };
static unsigned cst8_compute(void)
{
  unsigned r = 0;
  const char *p0 = g0;
  const char *p1 = g0;
  const char *p2 = g0;
  const char *p3 = g0;
  const char *p4 = g0;
  const char *p5 = g0;
  const char *p6 = g0;
  const char *p7 = g0;
  r += (unsigned)(unsigned char)p0[0];
  r += (unsigned)(unsigned char)p1[0];
  r += (unsigned)(unsigned char)p2[0];
  r += (unsigned)(unsigned char)p3[0];
  r += (unsigned)(unsigned char)p4[0];
  r += (unsigned)(unsigned char)p5[0];
  r += (unsigned)(unsigned char)p6[0];
  r += (unsigned)(unsigned char)p7[0];
  return r;
}
#endif
