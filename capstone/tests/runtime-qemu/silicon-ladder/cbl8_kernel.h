#ifndef CBL8_KERNEL_H
#define CBL8_KERNEL_H
/* SIMPLEST R-14 probe: 8 reads of ONE global separated by memory barriers, so the
 * capability load is NOT CSE'd and the domain really executes 8 x `ldc gp[0]` from the
 * SAME cap-table slot. No struct, no array, no loop, no strlen, no computed address.
 * (Without the barriers the compiler folds all N loads into ONE ldc even at -O0 --
 * verified by disassembly, which is why the first cut of this ladder tested nothing.)
 * Expect 520. */
static char g0[2] = { 'A', 0 };
static unsigned cbl8_compute(void)
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
