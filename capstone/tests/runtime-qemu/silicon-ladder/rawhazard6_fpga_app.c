/* Can the fault be WORKED AROUND in the compiler? (v6)
 *
 * Confirmed trigger (v5): a loop whose exit condition loads through a
 * REGISTER-COMPUTED index into the capability-addressed array, where the body
 * ALSO stores to another location. Neither ingredient alone fails.
 *
 * If any variant below returns 5, we have a mitigation we can emit, which would
 * unblock matmult_int / coremark_matrix / crc32 / insertsort -- i.e. take the
 * paper's perf table from 3 rungs to 7.
 *
 *  dbg0  W0 baseline failing shape (control)          expect 1 on this board
 *  dbg1  W1 fence rw,rw before the condition load
 *  dbg2  W2 condition value hoisted into a register   (the legal compiler fix)
 *  dbg3  W3 extra store ALSO register-indexed
 *  dbg4  W4 the two locations 64 B apart (cache line)
 *  dbg5  W5 fence rw,rw after every store in the body
 * All correct = 5.
 */
#include "rawhazard_kernel.h"
static unsigned int rh_far[24];
static inline long opaque(long v) { __asm__ volatile("" : "+r"(v)); return v; }
static unsigned long rh_rd_mcycle(void){unsigned long v;__asm__ volatile("csrr %0, mcycle":"=r"(v));return v;}

void domain_main(unsigned long *res, unsigned func) {
  (void)func;
  unsigned long c0 = rh_rd_mcycle();
  long n, j;

  /* W0 -- baseline failing shape */
  rh_a[0]=5; rh_a[2]=0; __asm__ volatile("":::"memory"); j=opaque(1); n=0;
  while (rh_a[j-1] > 0 && n < 50) { rh_a[2]=rh_a[2]+1; rh_a[j-1]=rh_a[j-1]-1; n++; }
  res[3+0]=(unsigned long)n;

  /* W1 -- fence rw,rw immediately before re-testing the condition */
  rh_a[0]=5; rh_a[2]=0; __asm__ volatile("":::"memory"); j=opaque(1); n=0;
  for (;;) { __asm__ volatile("fence rw,rw" ::: "memory");
             if (!(rh_a[j-1] > 0 && n < 50)) break;
             rh_a[2]=rh_a[2]+1; rh_a[j-1]=rh_a[j-1]-1; n++; }
  res[3+1]=(unsigned long)n;

  /* W2 -- hoist the condition value into a register (no reload per trip) */
  rh_a[0]=5; rh_a[2]=0; __asm__ volatile("":::"memory"); j=opaque(1); n=0;
  { long v = rh_a[j-1];
    while (v > 0 && n < 50) { rh_a[2]=rh_a[2]+1; v = v - 1; rh_a[j-1] = (unsigned)v; n++; } }
  res[3+2]=(unsigned long)n;

  /* W3 -- make the OTHER store register-indexed too */
  rh_a[0]=5; rh_a[2]=0; __asm__ volatile("":::"memory"); j=opaque(1); n=0;
  { long k = opaque(2);
    while (rh_a[j-1] > 0 && n < 50) { rh_a[k]=rh_a[k]+1; rh_a[j-1]=rh_a[j-1]-1; n++; } }
  res[3+3]=(unsigned long)n;

  /* W4 -- put the two locations far apart (separate cache lines) */
  rh_far[0]=5; rh_far[16]=0; __asm__ volatile("":::"memory"); j=opaque(1); n=0;
  while (rh_far[j-1] > 0 && n < 50) { rh_far[16]=rh_far[16]+1; rh_far[j-1]=rh_far[j-1]-1; n++; }
  res[3+4]=(unsigned long)n;

  /* W5 -- fence after every store in the body */
  rh_a[0]=5; rh_a[2]=0; __asm__ volatile("":::"memory"); j=opaque(1); n=0;
  while (rh_a[j-1] > 0 && n < 50) {
    rh_a[2]=rh_a[2]+1; __asm__ volatile("fence rw,rw" ::: "memory");
    rh_a[j-1]=rh_a[j-1]-1; __asm__ volatile("fence rw,rw" ::: "memory"); n++; }
  res[3+5]=(unsigned long)n;

  unsigned long c1 = rh_rd_mcycle();
  res[0]=0xBEEFUL; res[1]=c1-c0; res[2]=0xD09EUL;
}
