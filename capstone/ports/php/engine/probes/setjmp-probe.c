/* Does capability-aware setjmp/longjmp actually work on this ABI?
 *
 * zend_bailout depends on it, and a wrong jmp_buf layout fails LATE and far from its
 * cause, so this is verified standalone before any engine rung relies on it.
 *
 * Checks, in order:
 *   1 direct setjmp returns 0
 *   2 longjmp(buf, 7) resurfaces at setjmp returning 7
 *   3 longjmp(buf, 0) resurfaces as 1, as C requires
 *   4 callee-saved state survives the unwind (s-registers restored)
 *   5 a CAPABILITY held across the unwind is still dereferenceable -- the tag survived
 *
 * (5) is the one that matters here: if stc/ldc were replaced by sd/ld, or jmp_buf were
 * not 16-aligned, the tag would be stripped and only this check would notice.
 */
#include <setjmp.h>

static jmp_buf jb;
static unsigned char buf[64] __attribute__((aligned(16)));

static void thrower(void) { longjmp(jb, 7); }
static void thrower0(void) { longjmp(jb, 0); }

void domain_main(unsigned *res, unsigned func)
{
    (void)func;
    unsigned code = 0;
    volatile int guard = 0x5A5A;          /* must survive the unwind */
    char *cap = (char *)buf;              /* a capability held across longjmp */
    cap[0] = 'k';

    int r = setjmp(jb);
    if (r == 0) { code |= 1u; thrower(); }
    if (r == 7) { code |= 2u; }
    if (guard == 0x5A5A) { code |= 4u; }
    if (cap[0] == 'k') { code |= 8u; }    /* tag survived: deref did not fault */

    int r0 = setjmp(jb);
    if (r0 == 0) { thrower0(); }
    if (r0 == 1) { code |= 16u; }         /* longjmp(buf,0) must appear as 1 */

    *res = 0xB0u + code;                  /* 0xB0 + 0x1F = 0xCF = 207 == all five passed */
}
