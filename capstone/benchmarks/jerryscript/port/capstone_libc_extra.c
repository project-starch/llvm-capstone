/* The two libc functions jerry-core needs that no other port here supplied.
 *
 * Kept in the port rather than in adapted/include because they have bodies: a
 * header full of static inline definitions would emit a private copy into every
 * translation unit that includes it, and under -capstone-gp-captable each copy is
 * a symbol the cap table has to carry.
 */
#include <stdint.h>

/* nextafter: the next representable double from x towards y.
 *
 * Bit-twiddling rather than a libm call, because the domain has no libm. The
 * representation is walked directly: adjacent doubles differ by one in the
 * unsigned magnitude ordering, which is what makes this exact rather than an
 * approximation. Handles the sign boundary, where "next towards y" crosses zero.
 *
 * NaN is returned unchanged for either operand, and x == y returns y, both per C99.
 * Subnormals and the infinity boundary fall out of the magnitude walk without a
 * special case -- that is the point of doing it this way. */
double nextafter(double x, double y) {
    union { double d; uint64_t u; } ax = { x }, ay = { y };
    if (x != x || y != y) {
        return x + y;                       /* NaN propagates */
    }
    if (x == y) {
        return y;                           /* C99: return y, not x, so the sign of a zero follows y */
    }
    if (x == 0.0) {
        ax.u = 1;                           /* smallest subnormal ... */
        ax.u |= (ay.u & 0x8000000000000000ull); /* ... in y's direction */
        return ax.d;
    }
    /* Magnitude ordering: for a positive x, larger bit pattern means larger value;
       for a negative x the order reverses, which the two branches below encode. */
    if ((x < y) == (x > 0.0)) {
        ax.u++;
    } else {
        ax.u--;
    }
    return ax.d;
}

/* rand: a 32-bit xorshift, NOT a cryptographic or statistically strong generator.
 *
 * It exists so Math.random() returns something rather than failing to link. Any
 * measurement that depends on the QUALITY of these numbers is measuring this
 * function, not JerryScript -- and a domain has no entropy source to do better
 * from, which is why the seed is fixed and every run is identical. That
 * reproducibility is a feature here: two runs of the same script must agree, or
 * the corpus cannot compare them. */
static uint32_t rand_state = 0x2545F491u;

void srand(unsigned int seed) {
    rand_state = seed ? (uint32_t) seed : 0x2545F491u;   /* zero would lock the shift */
}

int rand(void) {
    uint32_t x = rand_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rand_state = x;
    return (int) (x & 0x7fffffffu);
}

/* fabs and cbrt: the two of the twenty libm functions jerry-core needs that
   MicroPython's lib/libm_dbl does not provide. */

double fabs(double x) {
    union { double d; uint64_t u; } v = { x };
    v.u &= 0x7fffffffffffffffull;           /* clear the sign bit; exact, no FP op */
    return v.d;
}

/* cbrt: Kahan's magic-constant seed followed by Newton iterations.
 *
 * The seed exploits the exponent field being a scaled logarithm: dividing the
 * biased exponent by three lands within a few percent of the cube root, which two
 * Newton steps in double take to full precision. Written out rather than borrowed
 * because MicroPython's libm has no cbrt, and Math.cbrt is the only caller. */
double cbrt(double x) {
    if (x != x || x == 0.0) {
        return x;                            /* NaN and both zeros pass through */
    }
    int neg = x < 0.0;
    if (neg) {
        x = -x;
    }
    union { double d; uint64_t u; } v = { x };
    if ((v.u >> 52) == 0x7ff) {
        return neg ? -x : x;                 /* infinity */
    }
    /* seed: exponent/3 with the bias re-applied */
    v.u = v.u / 3 + 0x2a9f84fe36d22425ull;
    double r = v.d;
    for (int i = 0; i < 4; i++) {
        r = r - (r - x / (r * r)) / 3.0;     /* Newton on r^3 = x */
    }
    return neg ? -r : r;
}

/* log2. MicroPython's libm has log but not log2.
 *
 * NOT log(x) / M_LN2, which is the obvious form and is visibly wrong in
 * JavaScript: it returns 2.9999999999999996 for Math.log2(8), and a language test
 * that prints it would fail for a reason that has nothing to do with what we are
 * measuring. Splitting off the exponent first makes every exact power of two exact,
 * because the mantissa is then 1.0 and log(1.0) is 0. */
double log(double);

double log2(double x) {
    if (x != x || x < 0.0) {
        return (x - x) / 0.0;                /* NaN */
    }
    if (x == 0.0) {
        return -1.0 / 0.0;
    }
    union { double d; uint64_t u; } v = { x };
    int e = (int) ((v.u >> 52) & 0x7ff);
    if (e == 0x7ff) {
        return x;                             /* +inf */
    }
    if (e == 0) {                             /* subnormal: scale into normal range */
        v.d = x * 9007199254740992.0;         /* 2^53 */
        e = (int) ((v.u >> 52) & 0x7ff) - 53;
    }
    e -= 1023;
    v.u = (v.u & 0x800fffffffffffffull) | 0x3ff0000000000000ull;  /* mantissa in [1,2) */
    /* 1.4426950408889634 = 1/ln(2). For an exact power of two v.d is 1.0 here and
       log(1.0) is 0, so the result is the exponent alone and therefore exact. */
    return (double) e + log (v.d) * 1.4426950408889634;
}
