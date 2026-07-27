/* Store->load hazard probe. res[3+d] = value read back at store->load distance d
   (d = 0..15 nops), res[3+16] = the swap-then-recompare shape, res[3+17..20] =
   a second sweep on a different array element to rule out one-slot effects.
   Correct hardware: every distance slot reads 0xBB (=187). */
#include "rawhazard_kernel.h"

static unsigned long rh_rd_mcycle(void) {
  unsigned long v; __asm__ volatile("csrr %0, mcycle" : "=r"(v)); return v;
}

void domain_main(unsigned long *res, unsigned func) {
  (void)func;
  unsigned long c0 = rh_rd_mcycle();
  unsigned h = 2166136261u;
  for (int d = 0; d < 16; d++) {
    unsigned v = rh_probe(0, d);
    res[3 + d] = v;
    h ^= v; h *= 16777619u;
  }
  unsigned sw = rh_swap_recompare();
  res[3 + 16] = sw;
  h ^= sw; h *= 16777619u;
  for (int d = 0; d < 4; d++) {
    unsigned v = rh_probe(2, d);
    res[3 + 17 + d] = v;
    h ^= v; h *= 16777619u;
  }
  unsigned long c1 = rh_rd_mcycle();
  res[0] = (unsigned long)h;
  res[1] = c1 - c0;
  res[2] = 0xD09EUL;
}
