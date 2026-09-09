#ifndef CISTK_KERNEL_H
#define CISTK_KERNEL_H
/* MINIMAL REPRODUCER for the mruby freestanding wedge, reconstructed off the mruby path.
 *
 * WHAT IS BEING TESTED. mrb_open_core dies in stack_clear because c->ci->stack holds a
 * capability whose BOUNDS are 80 bytes -- exactly sizeof(struct RProc) -- while its cursor
 * points into the VM stack. mruby's own cipush was read in the disassembly and is correct:
 * it derives ci->stack from ci[-1].stack with cincoffset and stores it at 0x30. So the wrong
 * capability enters at the root of the chain, and the two capability-shaped things standing
 * next to it in the same struct are ci->proc and ci->blk, both RProc pointers at 0x10/0x20.
 *
 * This kernel reproduces that struct, that copy and that store sequence, and nothing else:
 *
 *   CiProc   5 capabilities = 80 bytes, standing in for struct RProc
 *   CiInfo   96 bytes with proc at 0x10, blk at 0x20, stack at 0x30, exactly mrb_callinfo
 *
 * It does NOT measure a fault. It returns the LENGTH of the capability that ends up in
 * ci->stack, so a wrong answer is a number rather than a dead domain -- the same reason
 * every rung in this directory returns a checksum.
 *
 *   CISTK_STAGE=1  ci[0] = ci_zero (static const struct copy), then ci[0].stack = stk.
 *                  Oracle: sizeof(stk).
 *   CISTK_STAGE=2  stage 1 plus ci[0].proc = &the_proc BEFORE the stack store, which is
 *                  the order stack_init and cipush use. Oracle: sizeof(stk).
 *   CISTK_STAGE=3  cipush shape: ci[1].proc/blk set, then ci[1].stack = ci[0].stack + 4
 *                  derived with pointer arithmetic. Oracle: sizeof(stk).
 *
 * A stage that answers 80 has put an RProc capability into the stack slot and IS the bug.
 */

#ifndef CISTK_STAGE
#define CISTK_STAGE 3
#endif

typedef struct { void *a, *b, *c, *d, *e; } CiProc;          /* 80 bytes */

typedef struct {
  unsigned char n, cci, vis;
  unsigned int  mid;
  const CiProc *proc;                                        /* 0x10 */
  CiProc       *blk;                                         /* 0x20 */
  long         *stack;                                       /* 0x30 */
  const unsigned char *pc;                                   /* 0x40 */
  void         *u;                                           /* 0x50 */
} CiInfo;                                                    /* 96 bytes */

static CiProc cistk_the_proc;
static long   cistk_stk[128];
static CiInfo cistk_ci[4];

/* On the host there is no capability to inspect, so the oracle states the expectation.
   In the domain the real length is read back, which is the whole measurement. */
#ifdef __capstone
#define CISTK_LEN(p) ((unsigned)(__builtin_capstone_cap_get_end((char *)(p)) \
                               - __builtin_capstone_cap_get_base((char *)(p))))
#else
#define CISTK_LEN(p) ((unsigned)CISTK_HOST_LEN)
#endif

/* THE POSITIVE CONTROL. Stage 4 puts the RProc capability into the stack slot on purpose.
   Without it a clean 1024 from stages 1-3 is indistinguishable from a CISTK_LEN that cannot
   report anything else -- a broken builtin, a folded constant, a host-only path taken by
   mistake. If stage 4 does not answer 80, stages 1-3 have measured nothing. */
#ifndef CISTK_HOST_LEN
#if (CISTK_STAGE) == 4
/* 80, not sizeof(CiProc): on the host a pointer is 8 bytes and the struct is 40. The
   number the domain must answer is the TARGET's size, so it is written out. */
#define CISTK_HOST_LEN 80u
#else
#define CISTK_HOST_LEN sizeof(cistk_stk)
#endif
#endif

static unsigned
cistk_compute(void)
{
  static const CiInfo ci_zero = { 0 };

  cistk_ci[0] = ci_zero;                    /* the static-const struct copy */
#if (CISTK_STAGE) >= 2
  cistk_ci[0].proc = &cistk_the_proc;       /* an 80-byte capability, two slots earlier */
  cistk_ci[0].blk  = &cistk_the_proc;
#endif
  cistk_ci[0].stack = cistk_stk;

#if (CISTK_STAGE) == 4
  cistk_ci[0].stack = (long *)&cistk_the_proc;   /* deliberate: the control must answer 80 */
  return CISTK_LEN(cistk_ci[0].stack);
#endif

#if (CISTK_STAGE) >= 3
  cistk_ci[1] = ci_zero;
  cistk_ci[1].proc  = &cistk_the_proc;
  cistk_ci[1].blk   = &cistk_the_proc;
  cistk_ci[1].stack = cistk_ci[0].stack + 4;   /* the cipush derivation */
  return CISTK_LEN(cistk_ci[1].stack);
#else
  return CISTK_LEN(cistk_ci[0].stack);
#endif
}
#endif /* CISTK_KERNEL_H */
