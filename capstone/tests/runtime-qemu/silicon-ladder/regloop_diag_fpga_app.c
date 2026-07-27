/* R-6 next probe: does PURE REGISTER looping work on this board?
 *
 * janne_diag's failing nest contains no memory operations at all (verified in
 * the .dom), so R-1 cannot explain it. This walks up from the simplest possible
 * register loop to janne's exact shape, so whichever step first diverges is the
 * trigger. No memory access anywhere except the final res[] writes.
 *
 *   dbg0  simple counted loop, 100 trips                       -> 100
 *   dbg1  nested counted loops 10x10                           -> 100
 *   dbg2  loop with a data-dependent branch in the body        -> 100
 *   dbg3  loop with a multiply in the body (janne has b*3)     -> 100
 *   dbg4  janne's exact nest, bounded, trips counted           -> 21
 *   dbg5  0xREG marker
 *
 * If dbg0 diverges, the finding is far larger than R-1 and everything measured
 * on this board needs re-examining. If dbg0..3 are correct and only dbg4
 * diverges, janne's specific control flow is implicated.
 */
/* One global, touched only OUTSIDE the loops: the build gate requires an
   ldc gp[i] access, and this keeps every loop body register-pure. */
static volatile int rl_touch;

static unsigned long rl_rd_mcycle(void){unsigned long v;__asm__ volatile("csrr %0, mcycle":"=r"(v));return v;}
static inline long opaque(long v){ __asm__ volatile("" : "+r"(v)); return v; }

void domain_main(unsigned long *res, unsigned func) {
  (void)func;
  unsigned long c0 = rl_rd_mcycle();
  rl_touch = 7;

  /* dbg0 -- the simplest register loop there is */
  { long i = opaque(0), n = 0;
    while (i < 100) { i++; n++; }
    res[3+0] = (unsigned long)n; }

  /* dbg1 -- nested */
  { long n = 0, i = opaque(0);
    while (i < 10) { long j = 0; while (j < 10) { j++; n++; } i++; }
    res[3+1] = (unsigned long)n; }

  /* dbg2 -- data-dependent branch in the body */
  { long i = opaque(0), n = 0, acc = 0;
    while (i < 100) { if (acc > 5) acc = acc - 3; else acc = acc + 2; i++; n++; }
    res[3+2] = (unsigned long)n; }

  /* dbg3 -- multiply in the body */
  { long i = opaque(0), n = 0, m = 1;
    while (i < 100) { m = m * 3; if (m > 1000000) m = 1; i++; n++; }
    res[3+3] = (unsigned long)n; }

  /* dbg4 -- janne's exact nest, bounded, counting every body execution */
  { int a = (int)opaque(1), b = (int)opaque(1);
    long n = 0, guard = 0;
    while (a < 30 && guard < 400) {
      while (b < a && guard < 400) {
        if (b > 5) b = b * 3; else b = b + 2;
        if (b >= 10 && b <= 12) a = a + 10; else a = a + 1;
        n++; guard++;
      }
      a = a + 2; b = b - 10; n++; guard++;
    }
    res[3+4] = (unsigned long)n; }

  res[3+5] = 0x2E60UL;
  res[3+6] = (unsigned long)(unsigned)rl_touch;   /* expect 7 */
  unsigned long c1 = rl_rd_mcycle();
  res[0]=0xBEEFUL; res[1]=c1-c0; res[2]=0xD09EUL;
}
