#ifndef OBN_H
#define OBN_H
/* Is the LSU capability check ACTIVE AT ALL in this domain?
 *
 * Everything measured so far exercises ONE clause of core/load_store_unit.sv's
 * cap_violation_detection block -- the bounds compare (cause 28). The whole block is
 * gated on `capmode_i && ld_st_priv_lvl_i == PRIV_LVL_M`. If our domains do not satisfy
 * that gate, skipping the check is CORRECT hardware behaviour and there is no bug.
 *
 * This probe hits a DIFFERENT clause of the same block: the very first one, `lsu_cap_type
 * == NOT_CAP -> cause 24`. `mv` is an integer op, so its destination carries no capability
 * metadata; storing through it must raise UNEXPECTED_OPERAND if the block is live.
 *
 * The address used is c0's own cursor -- inside this domain's own 16-byte carve -- so if
 * the store is NOT trapped it merely zeroes this domain's own global. Safe either way.
 *
 *   traps   -> block is live, only the bounds clause is broken -> a real silicon defect
 *   returns -> block is inert in our domains -> a gate/config question, NOT a new bug
 */
#define LCC(rd, rs, f) __asm__ volatile(".insn r 0x5b,0x1,0x4, %0, %1, x" #f : "=r"(rd) : "r"(rs))
static char obn_g[2] = { 1, 0 };
static unsigned obn_compute(void)
{
  void *c0;
  unsigned long e = 0, cur = 0, head, addr;
  __asm__ volatile(".insn i 0x5b, 0x3, %0, 0(gp)" : "=r"(c0));
  LCC(e, c0, 4); LCC(cur, c0, 2);
  head = e - cur;
  if (head > 9000u) head = 9000u;
  __asm__ volatile("mv %0, %1" : "=r"(addr) : "r"(c0));   /* integer copy of the cursor */
  __asm__ volatile(
  "nop\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "sd x0, 0(%0)\n"                        /* store through a NOT_CAP base */
  :: "r"(addr) : "memory");
  (void)obn_g;
  return (unsigned)(head * 100u + 57u);
}
#endif
