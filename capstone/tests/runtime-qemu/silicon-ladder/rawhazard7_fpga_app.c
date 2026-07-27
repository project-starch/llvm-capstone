/* Last mitigation candidate: CONSTANT-OFFSET addressing (v7).
 *
 * v6 killed fences, cache-line separation, and register hoisting. But every
 * PASSING case so far used a constant-offset load (`lw rd, imm(cap)`), and every
 * FAILING case used a register-indexed one. If walking a pointer with
 * constant offsets is safe, the backend can strength-reduce indexed array
 * accesses into pointer increments and side-step the fault entirely -- which
 * would unblock all four rungs.
 *
 *  dbg0  P0 control: register-indexed condition + other store   expect 1
 *  dbg1  P1 pointer walk, *p in the condition (constant offset)
 *  dbg2  P2 pointer walk, other store also via a pointer
 *  dbg3  P3 pointer walk, pointer advanced by a constant each trip
 *  All correct = 5.
 */
#include "rawhazard_kernel.h"
static inline long opaque(long v) { __asm__ volatile("" : "+r"(v)); return v; }
static unsigned long rh_rd_mcycle(void){unsigned long v;__asm__ volatile("csrr %0, mcycle":"=r"(v));return v;}

void domain_main(unsigned long *res, unsigned func) {
  (void)func;
  unsigned long c0 = rh_rd_mcycle();
  long n, j;

  /* P0 -- control */
  rh_a[0]=5; rh_a[2]=0; __asm__ volatile("":::"memory"); j=opaque(1); n=0;
  while (rh_a[j-1] > 0 && n < 50) { rh_a[2]=rh_a[2]+1; rh_a[j-1]=rh_a[j-1]-1; n++; }
  res[3+0]=(unsigned long)n;

  /* P1 -- condition through a pointer (constant offset 0) */
  rh_a[0]=5; rh_a[2]=0; __asm__ volatile("":::"memory"); n=0;
  { unsigned int *p = &rh_a[0]; __asm__ volatile("" : "+r"(p));
    while (*p > 0 && n < 50) { rh_a[2]=rh_a[2]+1; *p = *p - 1; n++; } }
  res[3+1]=(unsigned long)n;

  /* P2 -- both accesses through pointers */
  rh_a[0]=5; rh_a[2]=0; __asm__ volatile("":::"memory"); n=0;
  { unsigned int *p = &rh_a[0]; unsigned int *o = &rh_a[2];
    __asm__ volatile("" : "+r"(p)); __asm__ volatile("" : "+r"(o));
    while (*p > 0 && n < 50) { *o = *o + 1; *p = *p - 1; n++; } }
  res[3+2]=(unsigned long)n;

  /* P3 -- walking pointer, advanced by a constant, over a 5-element run */
  rh_a[0]=1; rh_a[1]=1; rh_a[2]=1; rh_a[3]=1; __asm__ volatile("":::"memory"); n=0;
  { unsigned int *p = &rh_a[0]; __asm__ volatile("" : "+r"(p));
    long guard = 0;
    while (guard < 4) { if (*p != 0) n++; p = p + 1; guard++; }
    n++; }
  res[3+3]=(unsigned long)n;

  unsigned long c1 = rh_rd_mcycle();
  res[0]=0xBEEFUL; res[1]=c1-c0; res[2]=0xD09EUL;
}
