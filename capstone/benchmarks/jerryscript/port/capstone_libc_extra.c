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
