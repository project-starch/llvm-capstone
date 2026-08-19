/* The two libc functions jerry-core needs that no other port here supplied.
 *
 * Kept in the port rather than in adapted/include because they have bodies: a
 * header full of static inline definitions would emit a private copy into every
 * translation unit that includes it, and under -capstone-gp-captable each copy is
 * a symbol the cap table has to carry.
 */
#include <stdint.h>

/* WHAT USED TO BE HERE, and why it is gone.
 *
 * This file once implemented nextafter, fabs, cbrt and log2, each checked against the
 * host libm before use. All four were then found to be shipped by JerryScript itself:
 * tools/amalgam.py --jerry-math emits jerry-math, upstream's own libm, and the linker
 * reported them as duplicate symbols.
 *
 * The lesson is cheap to state and was not free to learn: the compile probe asks what
 * a candidate NEEDS, and that is not the same question as what it already PROVIDES.
 * Look for a bundled libm before writing one. Nothing was lost but the writing, since
 * upstream's own math is also the more faithful choice for a runtime we are measuring.
 *
 * rand and srand stay: jerry-math has no PRNG, and Math.random reaches rand() through
 * ecma-builtin-math.c. */

/* rand: a 32-bit xorshift, NOT a cryptographic or statistically strong generator.
 *
 * It exists so Math.random() links. Any measurement that depends on the QUALITY of
 * these numbers is measuring this function, not JerryScript -- and a domain has no
 * entropy source to do better from, which is why the seed is fixed and every run is
 * identical. That reproducibility is a feature here: two runs of the same script must
 * agree, or the corpus cannot compare them. */
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
