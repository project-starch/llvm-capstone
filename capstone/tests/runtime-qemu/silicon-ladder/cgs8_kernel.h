#ifndef CGS8_KERNEL_H
#define CGS8_KERNEL_H
/* Measured: clp16 (16 dynamic `ldc gp[0]` in a loop) PASSES, cdif8 (8 ldc, 8 distinct
 * slots) PASSES, cst8 (1 ldc + 8 capability stores, straight-line) PASSES, and r14hl
 * (loop + computed-address capability stores, but loading from a STACK slot) PASSES.
 * Only r14lp -- which has the cap-table load AND the capability store in the same loop
 * iteration -- fails. These two probe exactly that combination, minus the struct.
 * cgs8: store to a COMPUTED address p[i] each iteration. Expect 520. */
static char g0[2] = { 'A', 0 };
static unsigned cgs8_compute(void)
{
  const char *p[8]; unsigned i, r = 0;
  for (i = 0; i < 8; i++) p[i] = g0;
  for (i = 0; i < 8; i++) r += (unsigned)(unsigned char)p[i][0];
  return r;
}
#endif
