/* FPGA domain for rung 's06agg': does the COMPILER'S aggregate struct copy carry S-06?
 *
 * Reports the RAW WORDS as well as the verdict. The first board run returned a conflated 0 and
 * therefore said nothing about which half broke or whether the copy happened at all; a rung that
 * costs a boot should never come back with a number that cannot be read. res[0] is the bitmask
 * (15 = correct), res[3..6] are the destination words as actually read back, res[7..10] the
 * source words -- so "the copy never ran" and "the copy corrupted a half" are distinguishable
 * without another boot.
 *
 * domain_main is written out here rather than reused from ladder_perf_domain.h because the extra
 * slots are the point; the shared header writes only res[0..2]. */
#include "s06agg_kernel.h"

void domain_main(unsigned long *res, unsigned func)
{
  (void)func;
  volatile unsigned long *r = (volatile unsigned long *)res;
  r[1] = 0UL;
  r[2] = 0xD09EUL;                 /* ran-marker, as the ladder host expects */
  r[0] = (unsigned long)s06agg_compute();
  r[3] = s06agg_dst.lo;  r[4] = s06agg_dst.hi;
  r[5] = s06agg_dst.lo2; r[6] = s06agg_dst.hi2;
  r[7] = s06agg_src.lo;  r[8] = s06agg_src.hi;
  r[9] = s06agg_src.lo2; r[10] = s06agg_src.hi2;
}
