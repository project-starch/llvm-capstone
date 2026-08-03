#ifndef CLP8_KERNEL_H
#define CLP8_KERNEL_H
/* SIMPLEST R-14 probe: read ONE global 8 time(s) in a loop, accumulate, return.
 * One global, one loop, one accumulator -- no struct, no array, no strlen, no computed
 * store address, no capability store at all. At -O0 the `ldc gp[0]` sits INSIDE the loop,
 * so this executes 8 dynamic loads from the SAME cap-table slot.
 * A loop is required: straight-line repeats (with or without memory barriers) are folded
 * into ONE ldc even at -O0 -- verified by disassembly, so those variants tested nothing.
 * Pairs with cdif8 (8 loads, 8 DISTINCT slots, straight-line). Expect 520. */
static char g0[2] = { 'A', 0 };
static unsigned clp8_compute(void)
{
  unsigned r = 0, i;
  for (i = 0; i < 8; i++) r += (unsigned)(unsigned char)g0[0];
  return r;
}
#endif
