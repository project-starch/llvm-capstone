#ifndef OB2_H
#define OB2_H
/* MINIMAL reproducer: capability upper bound is not enforced on stores.
 * Carves a 16-byte capability off the top of the domain's own stack with `split`, verifies its
 * bounds with `lcc`, then stores 8 bytes ONE CAPABILITY PAST THE END and returns.
 * Needs NO cap-table, NO gp, NO interp glue, NO DOMAIN_WINDOW -- it builds its own bounded
 * capability, so it runs on the DEFAULT glue in any domain.
 * The overrun lands in the remaining stack, which this domain owns, so nothing outside the
 * domain is touched: this demonstrates a SUB-BOUND violation only.
 * Return value:
 *    0        the carve did not come out 16 bytes -- probe invalid, ignore
 *    1        bounds are 16 bytes AND the out-of-bounds store COMPLETED  -> NOT ENFORCED
 *    (wedge)  the store trapped -> enforcement works and the finding is wrong
 * `split(rd,rs1,rs2)`: rd = [rs2.cursor, rs1.end), rs1 keeps the lower part. */
#define LCC(rd, rs, f) __asm__ volatile(".insn r 0x5b,0x1,0x4, %0, %1, x" #f : "=r"(rd) : "r"(rs))
static unsigned ob2_compute(void)
{
  void *win;
  unsigned long end = 0, cut = 0, s = 0, e = 0;
  /* cut = sp.end - 16 ; win = split(sp, cut) -> a 16-byte capability at the top of the stack */
  __asm__ volatile(".insn r 0x5b,0x1,0x4, %0, sp, x4" : "=r"(end));      /* end = sp.END   */
  cut = end - 16u;
  __asm__ volatile(".insn r 0x5b,0x1,0x6, %0, sp, %1" : "=r"(win) : "r"(cut)); /* split      */
  LCC(s, win, 3); LCC(e, win, 4);
  if (e - s != 16u) return 0u;                       /* probe did not carve what it intended */
  __asm__ volatile("sd x0, 16(%0)" :: "r"(win) : "memory");   /* 8 bytes PAST the end        */
  return 1u;                                         /* reached => the store did not trap    */
}
#endif
