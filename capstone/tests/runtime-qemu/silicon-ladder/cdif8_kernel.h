#ifndef CDIF8_KERNEL_H
#define CDIF8_KERNEL_H
/* SIMPLEST POSSIBLE R-14 probe -- no struct, no array, no loop, no strlen.
 * 8 loads from 8 DIFFERENT globals => 8 x `ldc` from 8 DISTINCT cap-table slots.
 * Paired with cld8 (8 loads from ONE slot). cld8 fails + cdif8 passes => REPETITION of a
 * single slot is the trigger. Both fail => the COUNT of ldc-from-gp is. Expect 548. */
static char g0[2] = { 'A', 0 };
static char g1[2] = { 'B', 0 };
static char g2[2] = { 'C', 0 };
static char g3[2] = { 'D', 0 };
static char g4[2] = { 'E', 0 };
static char g5[2] = { 'F', 0 };
static char g6[2] = { 'G', 0 };
static char g7[2] = { 'H', 0 };
static unsigned cdif8_compute(void)
{
  unsigned r = 0;
  r += (unsigned)(unsigned char)g0[0];
  r += (unsigned)(unsigned char)g1[0];
  r += (unsigned)(unsigned char)g2[0];
  r += (unsigned)(unsigned char)g3[0];
  r += (unsigned)(unsigned char)g4[0];
  r += (unsigned)(unsigned char)g5[0];
  r += (unsigned)(unsigned char)g6[0];
  r += (unsigned)(unsigned char)g7[0];
  return r;
}
#endif
