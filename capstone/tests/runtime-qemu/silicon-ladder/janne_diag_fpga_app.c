/* R-6 diagnostic: why does beebs_janne hang, when R-1 predicts it should pass?
 *
 * A hung domain reports NOTHING (the controller prints res[] only after the
 * cscall returns), so the loops carry register safety bounds and the trip counts
 * come back raw. Correct values are computed by the native host from this same
 * file, so any divergence is silicon.
 *
 *   dbg0 outer trips   dbg1 inner trips   dbg2 final a   dbg3 final b
 *   dbg4 jc_iters (the .bss counter)      dbg5 0xD1A6 reached-end marker
 *
 * If a trip count comes back at its safety bound, that loop never terminated.
 * If jc_iters is wrong but the trip counts are right, the .bss counter is the
 * problem rather than the control flow.
 */
static int jd_iters;

static unsigned long jd_rd_mcycle(void){unsigned long v;__asm__ volatile("csrr %0, mcycle":"=r"(v));return v;}

void domain_main(unsigned long *res, unsigned func) {
  (void)func;
  unsigned long c0 = jd_rd_mcycle();
  int a = 1, b = 1;
  __asm__ volatile("" : "+r"(a)); __asm__ volatile("" : "+r"(b));
  jd_iters = 0;
  long outer = 0, inner = 0;

  while (a < 30 && outer < 200) {
    while (b < a && inner < 500) {
      if (b > 5) b = b * 3;
      else       b = b + 2;
      if (b >= 10 && b <= 12) a = a + 10;
      else                    a = a + 1;
      jd_iters++;
      inner++;
    }
    a = a + 2;
    b = b - 10;
    jd_iters++;
    outer++;
  }

  res[3+0] = (unsigned long)outer;
  res[3+1] = (unsigned long)inner;
  res[3+2] = (unsigned long)(unsigned)a;
  res[3+3] = (unsigned long)(unsigned)b;
  res[3+4] = (unsigned long)(unsigned)jd_iters;
  res[3+5] = 0xD1A6UL;

  unsigned long c1 = jd_rd_mcycle();
  res[0] = 0xBEEFUL; res[1] = c1 - c0; res[2] = 0xD09EUL;
}
