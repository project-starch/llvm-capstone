#ifndef CBL4_KERNEL_H
#define CBL4_KERNEL_H
/* SIMPLEST R-14 probe: 4 reads of ONE global separated by memory barriers, so the
 * capability load is NOT CSE'd and the domain really executes 4 x `ldc gp[0]` from the
 * SAME cap-table slot. No struct, no array, no loop, no strlen, no computed address.
 * (Without the barriers the compiler folds all N loads into ONE ldc even at -O0 --
 * verified by disassembly, which is why the first cut of this ladder tested nothing.)
 * Expect 260. */
static char g0[2] = { 'A', 0 };
static unsigned cbl4_compute(void)
{
  unsigned r = 0;
  __asm__ volatile("" ::: "memory");
  r += (unsigned)(unsigned char)g0[0];
  __asm__ volatile("" ::: "memory");
  r += (unsigned)(unsigned char)g0[0];
  __asm__ volatile("" ::: "memory");
  r += (unsigned)(unsigned char)g0[0];
  __asm__ volatile("" ::: "memory");
  r += (unsigned)(unsigned char)g0[0];
  return r;
}
#endif
