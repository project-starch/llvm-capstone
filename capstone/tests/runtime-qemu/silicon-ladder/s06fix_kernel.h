#ifndef S06FIX_H
#define S06FIX_H
/* S-06, second exposure: the COMPILER emits capability-grained copies of PLAIN DATA for an
 * ordinary struct assignment, entirely outside memcpy.
 *
 * s06copy shows the defect in memcpy's aligned loop, which software can work around because we
 * own memcpy. This rung shows the part software CANNOT work around that way. For
 *
 *     struct { void *p; unsigned long x; unsigned long y; }
 *
 * sizeof is 32 (a pointer is 16 bytes here), so p occupies bytes 0..15 and x,y occupy 16..31.
 * `*d = *s` lowers to TWO capability-grained copies: `ldc/stc 0x0` for the pointer, which is a
 * real capability and therefore safe, and `ldc/stc 0x10` for x AND y together -- 16 bytes of
 * ordinary data. Under S-06 that keeps the low 8 bytes (x) and loses the high 8 (y).
 *
 * RETURN VALUE IS THE VERDICT:
 *   64  both x and y survived      -> no defect
 *   66  y is gone, x is intact     -> S-06 through the compiler's aggregate copy
 *   65  x is gone                  -> not the predicted shape; read before concluding
 *
 * Predicted 66 rather than merely "wrong": the defect keeps the LOW half of each 16-byte chunk,
 * and x is the low half. A rung that returned 65 would mean something other than S-06.
 *
 * The pointer is deliberately NOT dereferenced. Tag survival is already established elsewhere
 * (SQLite stage 171), and a lost tag would wedge instead of returning, taking the x/y answer
 * with it -- this rung is built to always produce a number.
 *
 * Statics, not locals: a 16-byte-aligned local forces dynamic stack realignment, which this
 * backend cannot legalize (clang dies in LegalizeDAG, "Unable to legalize non-vector shift").
 * The volatile reads stop the compiler proving dst == src and folding the check away.
 */

#ifdef S06FIX_WITH_FIXED_MEMCPY
/* A self-contained memcpy carrying the S-06 chunk fix, so this rung can be built with
 * -mllvm -capstone-lower-memops-via-libcall and still link. The ladder links no libc.
 *
 * The chunk sequence is the one validated in capstone-ariane
 * verif/tests/custom/capstone/untagged-ldc-stc-fixup.S arm E: plain-store BOTH halves first,
 * then lay the ldc/stc on top. Correct for both kinds of chunk, and branchless:
 *   - source IS a capability -> its metadata is non-zero, so the stc writes both banks and
 *     restores the tag, overwriting the plain stores;
 *   - source is PLAIN data   -> the ldc yields zero metadata, so the stc degrades to a
 *     single-bank store that never touches the high half, and the plain store survives.
 * Deliberately NOT a byte loop: that would strip capability tags and wedge the core. */
typedef __SIZE_TYPE__ s06size_t;
typedef unsigned long long s06u64_t;

__attribute__((used)) void *memcpy(void *dst, const void *src, s06size_t n) {
  unsigned char *d = (unsigned char *)dst;
  const unsigned char *s = (const unsigned char *)src;
  const s06size_t ps = sizeof(void *);
  s06size_t i = 0, k;
  s06size_t da = ((s06size_t)d) & (ps - 1u);
  s06size_t sa = ((s06size_t)s) & (ps - 1u);
  if (da == sa) {
    s06size_t head = da ? (ps - da) : 0u;
    if (head > n) head = n;
    for (; i < head; i++) d[i] = s[i];
    for (; i + ps <= n; i += ps) {
      for (k = 0; k < ps / sizeof(s06u64_t); k++)
        ((s06u64_t *)(void *)(d + i))[k] = ((const s06u64_t *)(const void *)(s + i))[k];
      *(void **)(d + i) = *(void *const *)(s + i);
    }
  }
  for (; i < n; i++) d[i] = s[i];
  return dst;
}
#endif

struct s06fix_s { void *p; unsigned long x; unsigned long y; };

__attribute__((aligned(16))) static struct s06fix_s s06fix_src;
__attribute__((aligned(16))) static struct s06fix_s s06fix_dst;
__attribute__((aligned(16))) static unsigned long   s06fix_target[4];

#define S06FIX_X 0x1111222233334444UL
#define S06FIX_Y 0x5555666677778888UL

static unsigned s06fix_compute(void)
{
  volatile unsigned long *v;
  unsigned r = 0;

  s06fix_target[0] = 0xABCDEFUL;
  s06fix_src.p = (void *)s06fix_target;      /* chunk 0: a real capability */
  s06fix_src.x = S06FIX_X;                   /* chunk 1 low  half */
  s06fix_src.y = S06FIX_Y;                   /* chunk 1 high half */

  s06fix_dst = s06fix_src;                   /* THE CONSTRUCT: aggregate assignment */

  v = (volatile unsigned long *)&s06fix_dst.x;
  if (*v != S06FIX_X) r |= 1u;
  v = (volatile unsigned long *)&s06fix_dst.y;
  if (*v != S06FIX_Y) r |= 2u;
  return 64u + r;
}
#endif
