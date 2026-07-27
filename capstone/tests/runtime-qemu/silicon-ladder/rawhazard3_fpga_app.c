/* Store->load DISAMBIGUATION probe (v3). Loop-free.
 *
 * v2 proved a same-address store->load is correct at distance 0 when BOTH use
 * the identical address expression. But insertsort stores through `is_a[j-1]`
 * and then loads through `is_a[j]` after `j--`: the SAME address reached by a
 * DIFFERENT computation. v2 was blind to that by construction.
 *
 * Each probe stores NEW through one index expression and loads back through
 * another that evaluates to the same address. The index is laundered through an
 * opaque asm so the compiler cannot fold the two expressions together.
 *   correct -> 187 (0xBB)      stale/bypassed store -> 170 (0xAA)
 *
 * dbg0: store via [j-1], load via [j] after j--          (insertsort's shape)
 * dbg1: store via [j],   load via [k] where k==j          (two live registers)
 * dbg2: store via [j+1], load via [j] after j++
 * dbg3: control -- store and load both via the SAME expression [j]
 * dbg4: control -- literal index both sides (v2's shape, expected 187)
 */
#include "rawhazard_kernel.h"

static inline long opaque(long v) { __asm__ volatile("" : "+r"(v)); return v; }
static unsigned long rh_rd_mcycle(void) {
  unsigned long v; __asm__ volatile("csrr %0, mcycle" : "=r"(v)); return v;
}

void domain_main(unsigned long *res, unsigned func) {
  (void)func;
  unsigned long c0 = rh_rd_mcycle();
  long j, k;

  /* dbg0 -- insertsort's exact shape */
  rh_a[0] = RH_OLD; __asm__ volatile("" ::: "memory");
  j = opaque(1);
  rh_a[j - 1] = RH_NEW; __asm__ volatile("" ::: "memory");
  j--;
  res[3 + 0] = (unsigned long)rh_a[j];

  /* dbg1 -- two independent registers holding the same index */
  rh_a[1] = RH_OLD; __asm__ volatile("" ::: "memory");
  j = opaque(1); k = opaque(1);
  rh_a[j] = RH_NEW; __asm__ volatile("" ::: "memory");
  res[3 + 1] = (unsigned long)rh_a[k];

  /* dbg2 -- store via [j+1], load via [j] after j++ */
  rh_a[2] = RH_OLD; __asm__ volatile("" ::: "memory");
  j = opaque(1);
  rh_a[j + 1] = RH_NEW; __asm__ volatile("" ::: "memory");
  j++;
  res[3 + 2] = (unsigned long)rh_a[j];

  /* dbg3 -- control: identical expression both sides */
  rh_a[3] = RH_OLD; __asm__ volatile("" ::: "memory");
  j = opaque(3);
  rh_a[j] = RH_NEW; __asm__ volatile("" ::: "memory");
  res[3 + 3] = (unsigned long)rh_a[j];

  /* dbg4 -- control: literal both sides (v2 shape, known good) */
  rh_a[0] = RH_OLD; __asm__ volatile("" ::: "memory");
  rh_a[0] = RH_NEW; __asm__ volatile("" ::: "memory");
  res[3 + 4] = (unsigned long)rh_a[0];

  unsigned long c1 = rh_rd_mcycle();
  res[0] = 0xBEEFUL; res[1] = c1 - c0; res[2] = 0xD09EUL;
}
