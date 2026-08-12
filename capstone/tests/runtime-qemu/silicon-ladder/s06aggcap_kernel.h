#ifndef S06AGGCAP_KERNEL_H
#define S06AGGCAP_KERNEL_H
/* Does the guarded aggregate copy still copy a REAL CAPABILITY correctly?
 *
 * WHY THIS EXISTS, AND WHY IT SHOULD HAVE EXISTED FIRST. s06agg proves the guard stops plain
 * data losing its high half. It contains NO POINTER FIELDS, so it would return a clean 15 even
 * if the guard destroyed every capability it touched -- a gate that cannot fire for the failure
 * mode that matters most. That gap was not hypothetical: with the guard enabled, SQLite stopped
 * faulting at vdbeMemClearExternAndSetNull and started faulting at sqlite3_strnicmp with
 * mcause 25 (UNEXPECTED_OPERAND) on `cincoffsetimm a0, a0, 1` -- a pointer loaded from a stack
 * slot that had come back UNTAGGED. That is the signature of a capability that was copied as
 * data, which is precisely what this rung tests and s06agg cannot.
 *
 * THE SUBJECT. One struct assignment over a 32-byte aggregate whose FIRST granule is a genuine
 * capability (a pointer to a real object) and whose SECOND is plain data with a 0x4000-aligned
 * low half -- so the same copy exercises both paths of the guard: the branch NOT taken (store
 * the capability) and the branch taken (skip the store, plain data already written).
 *
 * The capability is checked by USING it, not by inspecting it. A tag that survives an inspection
 * but faults on dereference is not a surviving capability, and LCC's own type query is exactly
 * the thing under test, so it cannot also be the judge.
 *
 * retval bitmask, 15 is a correct copy:
 *
 *   bit 0  the copied pointer is still a CAPABILITY   (lcc type != 7)
 *   bit 1  dereferencing the copied pointer yields the expected value
 *   bit 2  the plain granule's LOW  half survived
 *   bit 3  the plain granule's HIGH half survived
 *
 *   15  correct: capabilities survive AND plain data survives
 *   12  the guard preserved plain data but DESTROYED the capability -- the regression this rung
 *       exists to catch
 *    3  capabilities survive but plain data was lost -- the guard is not firing
 *
 * A domain that faults instead of returning is also a result here: dereferencing an untagged
 * pointer raises UNEXPECTED_OPERAND, so bit 1 can only be observed by surviving to set it. The
 * type query in bit 0 is total and cannot fault, so a return value of 12 with no fault and a
 * return of 0 via a trap are both "the capability did not survive".
 */

typedef struct {
  void *p;            /* granule 0: a genuine capability */
  unsigned long lo;   /* granule 1, low  -- 0x4000-aligned, the S-06 trigger */
  unsigned long hi;   /* granule 1, high -- sentinel */
} s06aggcap_t;

/* Statics, not locals: a 16-byte-aligned local forces dynamic stack realignment, which this
 * backend cannot legalize. Same reasoning as s06agg_kernel.h. */
__attribute__((aligned(16))) static s06aggcap_t s06aggcap_src;
__attribute__((aligned(16))) static s06aggcap_t s06aggcap_dst;
__attribute__((aligned(16))) static unsigned long s06aggcap_target[2];

#define S06AGGCAP_HI    0xCCCC3333CCCC3333UL
#define S06AGGCAP_VALUE 0x5A5A5A5A5A5A5A5AUL

#define S06AGGCAP_LCC_TYPE(out, cap) \
  __asm__ volatile(".insn r 0x5b, 0x1, 0x4, %0, %1, x1" : "=r"(out) : "r"(cap))

static unsigned s06aggcap_compute(void)
{
  unsigned r = 0;
  unsigned long ty = 0;

  s06aggcap_target[0] = S06AGGCAP_VALUE;

  s06aggcap_src.p  = (void *)s06aggcap_target;   /* a real capability */
  s06aggcap_src.lo = 0x4000UL;
  s06aggcap_src.hi = S06AGGCAP_HI;

  s06aggcap_dst.p  = (void *)0;
  s06aggcap_dst.lo = 0xDEADDEADDEADDEADUL;
  s06aggcap_dst.hi = 0xDEADDEADDEADDEADUL;

  /* THE SUBJECT: one aggregate copy carrying a capability AND plain data. */
  s06aggcap_dst = s06aggcap_src;

  /* bit 0 -- is it still a capability at all? The type query is total, so this cannot fault
     even if the answer is "no". */
  S06AGGCAP_LCC_TYPE(ty, s06aggcap_dst.p);
  if (ty != 7UL) r |= 1u;

  /* bit 1 -- USE it. An untagged pointer raises UNEXPECTED_OPERAND here rather than returning,
     which is itself the answer; inspection alone would not prove the capability is usable. */
  if (ty != 7UL) {
    unsigned long *q = (unsigned long *)s06aggcap_dst.p;
    if (q[0] == S06AGGCAP_VALUE) r |= 2u;
  }

  if (s06aggcap_dst.lo == 0x4000UL)     r |= 4u;
  if (s06aggcap_dst.hi == S06AGGCAP_HI) r |= 8u;
  return r;
}
#endif /* S06AGGCAP_KERNEL_H */
