/* --- BUILD-TIME GATE on the outer heap's size -------------------------------
 * umm indexes blocks with a 15-bit field, so a heap of more than 32767 blocks
 * cannot be represented. umm_multi_init_heap does not fail loudly when that is
 * exceeded: it RETURNS with heap->pheap still NULL, and the first malloc then
 * dereferences it. On this target that surfaces as `cause = 24` deep inside
 * umm_malloc_core with no hint of the cause, which is where an afternoon went.
 *
 * That is umm's documented 15-bit block-index limitation -- the same defect this
 * corpus catalogues as entry 31 -- biting the corpus itself. So the combination is
 * checked here, at build time, where it can only be wrong loudly.
 */
#include "umm/umm_malloc_cfg.h"

#ifndef CAPSTONE_HEAP_SIZE
#define CAPSTONE_HEAP_SIZE 65536
#endif

_Static_assert(CAPSTONE_HEAP_SIZE / UMM_BLOCK_BODY_SIZE <= 32767,
               "outer heap exceeds umm's 15-bit block index: raise "
               "UMM_BLOCK_BODY_SIZE or lower CAPSTONE_HEAP_SIZE");
_Static_assert(CAPSTONE_HEAP_SIZE / UMM_BLOCK_BODY_SIZE >= 8,
               "outer heap is below umm's minimum block count");

/* The three libc functions mruby links that neither beebs' string file nor its
 * libm carries. Nothing more: an unused definition here would be a silent claim
 * that something works, and this file exists precisely to keep that claim honest.
 *
 * beebs_freestanding_string.c supplies memcpy, memmove, memset, memcmp, strcmp,
 * strlen and strcpy, and its mem* are the TAG-PRESERVING versions -- which is why
 * it is reused rather than replaced. It does not have memchr or strchr.
 */
#include <stddef.h>

/* Byte-wise on purpose. beebs' memcmp is word-at-a-time and casts pointers to a
 * narrower integer to test alignment; there is no reason to repeat that here for
 * two functions mruby calls on short strings, and a byte loop cannot lose a tag
 * because it never forms a wide load. */
void *
memchr(const void *s, int c, size_t n)
{
    const unsigned char *p = (const unsigned char *)s;
    unsigned char want = (unsigned char)c;

    for (; n; n--, p++)
        if (*p == want)
            return (void *)p;
    return NULL;
}

char *
strchr(const char *s, int c)
{
    char want = (char)c;

    for (;; s++) {
        if (*s == want)
            return (char *)s;
        if (*s == '\0')
            return NULL;   /* c == 0 is handled by the test above, as C requires */
    }
}

/* mruby calls abort() from mrb_assert and from its OOM path. A domain has nowhere
 * to abort TO: there is no host to return an exit status to from here, and the
 * loader reads a return value rather than a signal. Spinning would wedge the
 * domain and produce no result at all, which is the one outcome this project's
 * ladder exists to avoid -- so this traps deliberately instead. A capability fault
 * is reported by the monitor with a cause and a pc, so an abort becomes a
 * diagnosable line rather than silence.
 *
 * ponytail: a trap, not a graceful unwind. Ceiling: the marker protocol cannot say
 * WHICH assert fired. Upgrade path is a global set before the trap and read back
 * by the next stage, the way the tag-check note in the WAMR port works. */
void
abort(void)
{
    for (;;)
        *(volatile unsigned long *)0 = 0;
}

/* --- three libm functions mruby's CORE float arithmetic needs ---------------
 * flodivmod and flo_remainder are in numeric.c, not in the math gem, so dropping
 * mruby-math does not remove them. Built on beebs' floor and ceil rather than
 * written from scratch: those two already handle the infinities, the NaNs and the
 * values too large to have a fractional part, and repeating that here would be
 * three more chances to get it wrong.
 */

double floor(double x);
double ceil(double x);

double
trunc(double x)
{
    return x < 0.0 ? ceil(x) : floor(x);
}

double
round(double x)
{
    /* C's round is half-away-from-zero, not half-to-even. */
    return x < 0.0 ? ceil(x - 0.5) : floor(x + 0.5);
}

double
fmod(double x, double y)
{
    /* NaN for every case the identity below would get wrong: y == 0, either
       operand NaN, or x infinite. `x - x` is non-zero exactly when x is not
       finite, which is the cheapest finite test available without <math.h>, and
       0.0 / 0.0 is the portable way to produce a NaN here. */
    if (y == 0.0 || x != x || y != y || x - x != 0.0)
        return 0.0 / 0.0;

    /* ponytail: the textbook identity, not an exact-remainder algorithm.
       Ceiling: x / y is rounded to 53 bits before it is truncated, so once |x/y|
       exceeds 2^53 this is not the exact IEEE remainder fmod promises -- it can be
       off by a multiple of y, or zero where it must not be. Fine for this corpus,
       which computes no floats at all; NOT fine for a numeric benchmark. Upgrade
       path: repeated subtraction on the exponent, the way musl's fmod does it. */
    return x - trunc(x / y) * y;
}
