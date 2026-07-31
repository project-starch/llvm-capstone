#ifndef RAWHAZARD_KERNEL_H
#define RAWHAZARD_KERNEL_H
/* Minimal probe for a store->load (read-after-write) hazard on the
 * capability-addressed data path.
 *
 * MOTIVATION (board, 2026-07-27). The insertsort_diag rung returned its array
 * state raw and localized the fault precisely:
 *   - stores are FINE      -- the post-init array is byte-correct
 *   - loop bounds are FINE -- the outer loop ran all 9 iterations
 *   - the inner loop ran exactly 9 times instead of 45, i.e. exactly ONE
 *     iteration per outer pass, and the resulting permutation is exactly what
 *     "one swap then exit" produces.
 * The inner loop is `while (is_a[j] < is_a[j-1]) { swap; j--; }`. After the swap
 * it immediately RE-READS the element it just wrote. If that load returns the
 * pre-store value, the comparison is false and the loop exits after one pass --
 * which is exactly the observed behaviour.
 *
 * THE TEST. Write a known value, overwrite it with a second known value, then
 * read it back after a controllable number of filler instructions:
 *
 *     a = OLD;  a = NEW;  <filler>  x = a;
 *
 * x == NEW  -> no hazard at that distance
 * x == OLD  -> the load forwarded/returned STALE data (the hazard)
 *
 * The `memory` clobbers stop the compiler from forwarding the value in a
 * register, so the load is really executed; the fillers are `nop`s, so the only
 * variable is the store->load distance in instructions. Sweeping the distance
 * measures the hazard window rather than just detecting it.
 *
 * Every probe is written into a debug slot RAW -- a checksum could not localize
 * this (the same lesson as gp_diag). Correct hardware returns NEW for every
 * distance, so on a good machine every slot reads 0xBB.
 */

static unsigned int rh_a[4];

#define RH_OLD 0xAAu
#define RH_NEW 0xBBu

/* Store OLD, store NEW, wait `filler` nops, load back. */
static unsigned rh_probe(int k, int filler) {
  rh_a[k] = RH_OLD;
  __asm__ volatile("" ::: "memory");
  rh_a[k] = RH_NEW;
  __asm__ volatile("" ::: "memory");
  for (int f = 0; f < filler; f++)
    __asm__ volatile("nop");
  __asm__ volatile("" ::: "memory");
  return rh_a[k];
}

/* The insertsort inner loop's exact shape, reduced to two elements: swap a pair
   then immediately re-compare them. Correct hardware sees the swapped order. */
static unsigned rh_swap_recompare(void) {
  rh_a[0] = 5;
  rh_a[1] = 9;
  __asm__ volatile("" ::: "memory");
  if (rh_a[1] < rh_a[0]) {          /* false: 9 < 5 */
    unsigned t = rh_a[1];
    rh_a[1] = rh_a[0];
    rh_a[0] = t;
  }
  /* now force the real case: a[1] < a[0] */
  rh_a[0] = 9;
  rh_a[1] = 5;
  __asm__ volatile("" ::: "memory");
  unsigned swaps = 0;
  if (rh_a[1] < rh_a[0]) {          /* true: 5 < 9 -> swap */
    unsigned t = rh_a[1];
    rh_a[1] = rh_a[0];
    rh_a[0] = t;
    swaps++;
    __asm__ volatile("" ::: "memory");
    /* Immediately re-read BOTH just-written elements. Correct: a[0]=5, a[1]=9,
       so the re-test is false. A stale read of a[0] returns 9 and would ALSO
       report false -- so return the raw values, not the comparison. */
  }
  return (swaps << 16) | ((rh_a[0] & 0xff) << 8) | (rh_a[1] & 0xff);
}

#endif
