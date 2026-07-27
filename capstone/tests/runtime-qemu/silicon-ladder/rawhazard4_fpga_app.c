/* Does a loop whose EXIT CONDITION reads capability-addressed memory work? (v4)
 *
 * v2/v3 proved straight-line store->load is correct (both same-expression and
 * differently-computed addresses). insertsort's OUTER loop -- register bound --
 * is correct too. Its INNER loop, whose condition is two loads out of the array,
 * runs once instead of 45 times. This isolates that.
 *
 * Every loop carries a register-counter safety bound so a failure RETURNS a
 * diagnostic instead of hanging (a hung domain reports nothing at all).
 *
 *  dbg0 n   memory-condition loop, body writes the tested location  correct 5
 *  dbg1     rh_a[1] after that loop                                 correct 0
 *  dbg2 m   register-condition control                              correct 5
 *  dbg3 r   condition reads memory NOT written by the loop          correct 5
 *  dbg4 q   memory condition + memory body, distinct locations      correct 5
 *
 * n == 50 means the memory condition never went false -> the load never saw the
 * body's writes. n == 1 would mirror insertsort exactly (one trip then exit).
 */
#include "rawhazard_kernel.h"

static inline long opaque(long v) { __asm__ volatile("" : "+r"(v)); return v; }
static unsigned long rh_rd_mcycle(void) {
  unsigned long v; __asm__ volatile("csrr %0, mcycle" : "=r"(v)); return v;
}

void domain_main(unsigned long *res, unsigned func) {
  (void)func;
  unsigned long c0 = rh_rd_mcycle();
  long n = 0, m = 0, r = 0, q = 0;

  /* P1 -- insertsort's shape: condition loads the location the body stores. */
  rh_a[1] = 5;
  __asm__ volatile("" ::: "memory");
  while (rh_a[1] > 0 && n < 50) { rh_a[1] = rh_a[1] - 1; n++; }
  res[3 + 0] = (unsigned long)n;
  res[3 + 1] = (unsigned long)rh_a[1];

  /* P2 -- control: identical loop, register only, no memory in the condition. */
  long c = opaque(5);
  while (c > 0 && m < 50) { c = c - 1; m++; }
  res[3 + 2] = (unsigned long)m;

  /* P3 -- condition reads memory, but the loop body never writes it. */
  rh_a[3] = 1;
  __asm__ volatile("" ::: "memory");
  while (rh_a[3] != 0 && r < 5) { r++; }
  res[3 + 3] = (unsigned long)r;

  /* P4 -- condition reads one location, body writes a DIFFERENT one, and the
     condition location is decremented through a separate index expression. */
  rh_a[0] = 5; rh_a[2] = 0;
  __asm__ volatile("" ::: "memory");
  {
    long j = opaque(1);
    while (rh_a[j - 1] > 0 && q < 50) { rh_a[2] = rh_a[2] + 1; rh_a[j - 1] = rh_a[j - 1] - 1; q++; }
  }
  res[3 + 4] = (unsigned long)q;

  unsigned long c1 = rh_rd_mcycle();
  res[0] = 0xBEEFUL; res[1] = c1 - c0; res[2] = 0xD09EUL;
}
