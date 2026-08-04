#ifndef BNDS_H
#define BNDS_H
/* COMPILER-vs-RTL discriminator.
 * k1200 fails on `stc rs, 0x10(base)` where `base` came from a register-form cincoffset.
 * If the capability's bounds LEGITIMATELY cover [cursor, cursor+16) then the store is a legal
 * access and the fault is an RTL defect. If the bounds do NOT cover it, the compiler handed the
 * hardware a capability too small for the access and the bug is ours.
 * lcc field map (capstone_dyn_unit.anvil:182-188): 1=type, 2=cursor, 3=start, 4=end.
 * Returns the HEADROOM end-cursor (clamped), so >=16 means the +0x10 store is in bounds.
 * Cannot wedge: it performs no capability store at all. */
#define lcc_f(rd, rs, f) __asm__ volatile(".insn r 0x5b,0x1,0x4, %0, %1, x" #f : "=r"(rd) : "r"(rs))
static unsigned bnds_compute(void)
{
  struct kv { const char *z; const char *y; };
  struct kv a[32];
  volatile char pad[1200];
  unsigned long cur = 0, end = 0;
  const char *volatile *p;
  pad[0] = 1;
  a[0].z = "x0";
  p = &a[3].y;                 /* the +0x10 field of a computed element: the failing address */
  lcc_f(cur, p, 2);
  lcc_f(end, p, 4);
  if (end < cur) return 1u;                       /* 1 = END BELOW CURSOR (bounds broken) */
  { unsigned long hr = end - cur;
    if (hr > 900000UL) hr = 900000UL;
    return (unsigned)hr + 10u; }                  /* headroom+10; >=26 means +0x10 is in bounds */
}
#endif
