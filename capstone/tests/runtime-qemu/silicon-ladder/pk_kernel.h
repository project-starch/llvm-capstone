#ifndef PK_H
#define PK_H
/* Direct observation of what the interp record loop actually produced, instead of inferring it
 * from whether a global ended up correct -- two theories have already died that way.
 * Reads cap-table slots 0 and 1 with `ldc gp[i]` and inspects them with `lcc`
 * (capstone_dyn_unit.anvil:182-188: 1=type, 3=start), so the diagnostic itself does not depend
 * on global initialisation.
 * Encoding of the return value:
 *     +1   slot0.start != 0      (record 0 produced a storage capability)
 *     +2   slot1.start != 0      (record 1 produced one)
 *     +4   slot0.start != slot1.start   (they are DISTINCT carves)
 *     +10*type0  +100*type1
 * A healthy build is 7 + 10*t + 100*t. If slot1 is zero or equal to slot0, the record loop is
 * not carving per-record and that is defect 2, observed rather than deduced. */
#define LCC(rd, rs, f) __asm__ volatile(".insn r 0x5b,0x1,0x4, %0, %1, x" #f : "=r"(rd) : "r"(rs))
static char pk_m0[2] = { 1, 0 };
static char pk_m1[2] = { 2, 0 };
static unsigned pk_compute(void)
{
  void *c0, *c1;
  unsigned long b0 = 0, b1 = 0, t0 = 0, t1 = 0;
  __asm__ volatile(".insn i 0x5b, 0x3, %0, 0(gp)"  : "=r"(c0));   /* ldc gp[0] */
  __asm__ volatile(".insn i 0x5b, 0x3, %0, 16(gp)" : "=r"(c1));   /* ldc gp[1] */
  LCC(b0, c0, 3); LCC(b1, c1, 3);
  LCC(t0, c0, 1); LCC(t1, c1, 1);
  (void)pk_m0; (void)pk_m1;
  return (unsigned)((b0 != 0) + 2u*(b1 != 0) + 4u*(b0 != b1)
                    + 10u*(unsigned)t0 + 100u*(unsigned)t1);
}
#endif
