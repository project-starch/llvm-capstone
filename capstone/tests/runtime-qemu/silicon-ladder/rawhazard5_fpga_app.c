/* Split P4's two variables (v5).
 *
 * v4 result: P1 (literal index, body writes only the tested location) = 5 CORRECT.
 *            P4 (computed index [j-1], body ALSO writes another location) = 1 WRONG
 *                -- one trip then exit, insertsort's exact signature.
 * P4 differs from P1 in exactly two ways. This splits them.
 *
 *  dbg0  A: computed index [j-1], body writes ONLY the tested location   -> isolates the INDEX
 *  dbg1  B: literal index [0], body writes another location first        -> isolates the EXTRA STORE
 *  dbg2  C: P4 exactly, re-run as a positive control                     -> expect the failure (1)
 *  dbg3  D: computed index, extra store AFTER the decrement (ordering)
 *  dbg4  E: computed index via a DIFFERENT expression (j) with j=0
 * All correct = 5. Register safety bound at 50 so nothing hangs.
 */
#include "rawhazard_kernel.h"
static inline long opaque(long v) { __asm__ volatile("" : "+r"(v)); return v; }
static unsigned long rh_rd_mcycle(void){unsigned long v;__asm__ volatile("csrr %0, mcycle":"=r"(v));return v;}

void domain_main(unsigned long *res, unsigned func) {
  (void)func;
  unsigned long c0 = rh_rd_mcycle();
  long a=0,b=0,c=0,d=0,e=0,j;

  /* A -- computed index, no other store */
  rh_a[0]=5; __asm__ volatile("":::"memory"); j=opaque(1);
  while (rh_a[j-1] > 0 && a < 50) { rh_a[j-1] = rh_a[j-1] - 1; a++; }
  res[3+0]=(unsigned long)a;

  /* B -- literal index, extra store to a different location */
  rh_a[0]=5; rh_a[2]=0; __asm__ volatile("":::"memory");
  while (rh_a[0] > 0 && b < 50) { rh_a[2] = rh_a[2] + 1; rh_a[0] = rh_a[0] - 1; b++; }
  res[3+1]=(unsigned long)b;

  /* C -- P4 exactly (positive control: expect 1 on this board) */
  rh_a[0]=5; rh_a[2]=0; __asm__ volatile("":::"memory"); j=opaque(1);
  while (rh_a[j-1] > 0 && c < 50) { rh_a[2] = rh_a[2] + 1; rh_a[j-1] = rh_a[j-1] - 1; c++; }
  res[3+2]=(unsigned long)c;

  /* D -- computed index, extra store AFTER the decrement */
  rh_a[0]=5; rh_a[2]=0; __asm__ volatile("":::"memory"); j=opaque(1);
  while (rh_a[j-1] > 0 && d < 50) { rh_a[j-1] = rh_a[j-1] - 1; rh_a[2] = rh_a[2] + 1; d++; }
  res[3+3]=(unsigned long)d;

  /* E -- computed index with a plain j (no -1), extra store */
  rh_a[0]=5; rh_a[2]=0; __asm__ volatile("":::"memory"); j=opaque(0);
  while (rh_a[j] > 0 && e < 50) { rh_a[2] = rh_a[2] + 1; rh_a[j] = rh_a[j] - 1; e++; }
  res[3+4]=(unsigned long)e;

  unsigned long c1 = rh_rd_mcycle();
  res[0]=0xBEEFUL; res[1]=c1-c0; res[2]=0xD09EUL;
}
