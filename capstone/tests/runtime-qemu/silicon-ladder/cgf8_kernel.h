#ifndef CGF8_KERNEL_H
#define CGF8_KERNEL_H
/* Measured: clp16 (16 dynamic `ldc gp[0]` in a loop) PASSES, cdif8 (8 ldc, 8 distinct
 * slots) PASSES, cst8 (1 ldc + 8 capability stores, straight-line) PASSES, and r14hl
 * (loop + computed-address capability stores, but loading from a STACK slot) PASSES.
 * Only r14lp -- which has the cap-table load AND the capability store in the same loop
 * iteration -- fails. These two probe exactly that combination, minus the struct.
 * cgf8: identical loop and identical ldc-from-gp, but the capability is stored to ONE FIXED
 * local (immediate offset) instead of a computed p[i]. cgs8 fails + cgf8 passes => the
 * COMPUTED store address matters, but only in combination with the cap-table load.
 * Expect 520. */
static char g0[2] = { 'A', 0 };
static unsigned cgf8_compute(void)
{
  const char *q; unsigned i, r = 0;
  q = g0;
  for (i = 0; i < 8; i++) { q = g0; r += (unsigned)(unsigned char)q[0]; }
  return r;
}
#endif
