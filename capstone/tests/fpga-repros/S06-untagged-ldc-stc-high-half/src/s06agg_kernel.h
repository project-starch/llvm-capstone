#ifndef S06AGG_H
#define S06AGG_H
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
struct s06agg_s { void *p; unsigned long x; unsigned long y; };

__attribute__((aligned(16))) static struct s06agg_s s06agg_src;
__attribute__((aligned(16))) static struct s06agg_s s06agg_dst;
__attribute__((aligned(16))) static unsigned long   s06agg_target[4];

#define S06AGG_X 0x1111222233334444UL
#define S06AGG_Y 0x5555666677778888UL

static unsigned s06agg_compute(void)
{
  volatile unsigned long *v;
  unsigned r = 0;

  s06agg_target[0] = 0xABCDEFUL;
  s06agg_src.p = (void *)s06agg_target;      /* chunk 0: a real capability */
  s06agg_src.x = S06AGG_X;                   /* chunk 1 low  half */
  s06agg_src.y = S06AGG_Y;                   /* chunk 1 high half */

  s06agg_dst = s06agg_src;                   /* THE CONSTRUCT: aggregate assignment */

  v = (volatile unsigned long *)&s06agg_dst.x;
  if (*v != S06AGG_X) r |= 1u;
  v = (volatile unsigned long *)&s06agg_dst.y;
  if (*v != S06AGG_Y) r |= 2u;
  return 64u + r;
}
#endif
