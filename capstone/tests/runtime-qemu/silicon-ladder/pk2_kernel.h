#ifndef PK2_H
#define PK2_H
/* Settles WHY a 6-byte overrun crossed a 16-byte carve without trapping.
 * During the pad investigation an incomplete fix made each record write 22 bytes into a 16-byte
 * carve. That should have raised a bounds fault (load_store_unit.sv:970-972 is an exact upper-
 * bound check) and did not. Two survivors, and they need different follow-ups:
 *   (a) carves OVERLAP  -- `split` is not narrowing `sp`, so every carve runs to the region end
 *                          and the overrun was legitimately in bounds. Our bug, contained.
 *   (b) carves are ADJACENT and correct -- then the LSU is not enforcing its upper bound on the
 *                          board, which is a hardware finding that outranks R-16.
 * pk showed slot starts are distinct; it never read the ENDS, so it cannot separate these.
 * lcc fields (capstone_dyn_unit.anvil:182-188): 3=start, 4=end.
 * Encoding:  +1 end0==end1 (overlap => a)   +2 end1==start0 (adjacent => b)
 *            +4 start1<start0 (carves descend, as expected)   +10*(end0-start0 == 16)
 * Performs no capability store, so it cannot wedge. */
#define LCC(rd, rs, f) __asm__ volatile(".insn r 0x5b,0x1,0x4, %0, %1, x" #f : "=r"(rd) : "r"(rs))
static char pk2_a[2] = { 1, 0 };
static char pk2_b[2] = { 2, 0 };
static unsigned pk2_compute(void)
{
  void *c0, *c1;
  unsigned long s0=0, s1=0, e0=0, e1=0;
  __asm__ volatile(".insn i 0x5b, 0x3, %0, 0(gp)"  : "=r"(c0));
  __asm__ volatile(".insn i 0x5b, 0x3, %0, 16(gp)" : "=r"(c1));
  LCC(s0, c0, 3); LCC(s1, c1, 3);
  LCC(e0, c0, 4); LCC(e1, c1, 4);
  (void)pk2_a; (void)pk2_b;
  return (unsigned)((e0 == e1) + 2u*(e1 == s0) + 4u*(s1 < s0)
                    + 10u*((e0 - s0) == 16u));
}
#endif
