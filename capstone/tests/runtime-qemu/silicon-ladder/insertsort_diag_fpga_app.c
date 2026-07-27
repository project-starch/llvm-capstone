/* Diagnostic rung: beebs_insertsort, but the array state is returned RAW.
 *
 * WHY THIS EXISTS. `beebs_insertsort` returns a WRONG value on silicon
 * (957879052 vs oracle 271779359) with only ~560 retired instructions -- far too
 * few for the sort that produces the oracle, so the compute is not merely
 * miscomputing, it is barely running. A single FNV checksum cannot say WHICH
 * part is wrong: a checksum provably cannot localize (the same lesson as the
 * earlier gp_diag work).
 *
 * This rung is the same benchmark with the array state written out raw, so one
 * boot answers "are the array WRITES wrong, or is the SORT wrong?".
 *
 *   res[3+0 .. 3+10]  = is_a[0..10] AFTER is_init()   -- the plain stores
 *   res[3+11 .. 3+21] = is_a[0..10] AFTER is_sort()   -- the read-modify-write loop
 *   res[3+22]         = is_verify()
 *   res[3+23]         = outer-loop trip count   (correct = 9)
 *   res[3+24]         = inner-loop trip count   (correct = 45)
 *   res[3+25]         = a re-read of is_a[10] much later (late-read stability)
 *
 * Correct values, from the same header on the native host:
 *   post-init  0 11 10 9 8 7 6 5 4 3 2
 *   post-sort  0 2 3 4 5 6 7 8 9 10 11
 *   verify=1, outer=9, inner=45
 *
 * res[0] stays the ordinary FNV checksum so the harness's oracle gate still
 * applies unchanged; the debug slots are purely additive.
 *
 * Reading these: if the POST-INIT row is already wrong, plain indexed stores
 * through the gp-delivered array capability are broken -- which would be a much
 * bigger finding than a sort bug, and would have to be reconciled with
 * rv8_primes (which stores through such a capability every iteration and is
 * correct). If post-init is right and POST-SORT is wrong, the fault is in the
 * read-modify-write/swap loop. If the trip counts are short, a loop bound or
 * index is being corrupted -- the same signature as the hanging rungs, which
 * would tie all four failures together.
 */
#include "beebs_insertsort_kernel.h"

static unsigned long isd_rd_mcycle(void) {
  unsigned long v;
  __asm__ volatile("csrr %0, mcycle" : "=r"(v));
  return v;
}

/* is_sort, verbatim, plus two trip counters. The counters are the only
   difference from the kernel's is_sort; they do not touch the array. */
static void isd_sort_counted(unsigned long *outer, unsigned long *inner) {
  int i, j;
  unsigned int temp;
  unsigned long no = 0, ni = 0;
  i = 2;
  while (i <= 10) {
    no++;
    j = i;
    while (is_a[j] < is_a[j - 1]) {
      ni++;
      temp = is_a[j];
      is_a[j] = is_a[j - 1];
      is_a[j - 1] = temp;
      j--;
    }
    i++;
  }
  *outer = no;
  *inner = ni;
}

void domain_main(unsigned long *res, unsigned func) {
  (void)func;
  unsigned long c0 = isd_rd_mcycle();

  is_init();
  for (int i = 0; i < 11; i++)
    res[3 + i] = (unsigned long)is_a[i];

  unsigned long outer = 0, inner = 0;
  isd_sort_counted(&outer, &inner);
  for (int i = 0; i < 11; i++)
    res[3 + 11 + i] = (unsigned long)is_a[i];

  int ok = is_verify();
  res[3 + 22] = (unsigned long)ok;
  res[3 + 23] = outer;
  res[3 + 24] = inner;

  /* FNV over the sorted array, identical to is_compute's tail, so res[0] is the
     same oracle the normal rung is gated on. */
  unsigned h = 2166136261u;
  for (int i = 0; i < 11; i++) {
    unsigned v = is_a[i];
    for (int b = 0; b < 4; b++) { h ^= (v >> (8 * b)) & 0xffu; h *= 16777619u; }
  }
  h ^= (unsigned)ok; h *= 16777619u;

  res[3 + 25] = (unsigned long)is_a[10];   /* late re-read, after all the folding */

  unsigned long c1 = isd_rd_mcycle();
  res[0] = (unsigned long)h;
  res[1] = c1 - c0;
  res[2] = 0xD09EUL;
}
