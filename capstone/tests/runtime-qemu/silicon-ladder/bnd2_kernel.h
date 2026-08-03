#ifndef BND2_H
#define BND2_H
/* Full capability-field dump at the failing address, to test the CAPABILITY COMPRESSION
 * hypothesis. bnds only measured end-cursor (1312 B, in bounds). Compression damage would show
 * up as the START having moved ABOVE the cursor, or the bounds having been rounded outward --
 * neither of which end-cursor alone can see.
 * lcc fields (capstone_dyn_unit.anvil:182-188): 1=type, 2=cursor, 3=start, 4=end.
 * Encodes a verdict instead of raw addresses, which do not survive a single unsigned return:
 *    +1   cursor >= start        (cursor not below the base)
 *    +2   cursor + 16 <= end     (the 16-byte store fits)
 *    +4   start is 16-aligned
 *    +100 * captype
 * Expect 3 + 4 + 100*type if the capability is well formed. Performs NO capability store, so
 * it cannot wedge. */
#define lcc_f(rd, rs, f) __asm__ volatile(".insn r 0x5b,0x1,0x4, %0, %1, x" #f : "=r"(rd) : "r"(rs))
static unsigned bnd2_compute(void)
{
  struct kv { const char *z; const char *y; };
  struct kv a[32];
  volatile char pad[1200];
  unsigned long cur = 0, st = 0, en = 0, ty = 0;
  const char *volatile *p;
  unsigned v = 0;
  pad[0] = 1;
  a[0].z = "x0";
  p = &a[3].y;
  lcc_f(cur, p, 2);
  lcc_f(st,  p, 3);
  lcc_f(en,  p, 4);
  lcc_f(ty,  p, 1);
  if (cur >= st)        v += 1u;
  if (cur + 16UL <= en) v += 2u;
  if ((st & 15UL) == 0) v += 4u;
  return v + 100u * (unsigned)ty;
}
#endif
