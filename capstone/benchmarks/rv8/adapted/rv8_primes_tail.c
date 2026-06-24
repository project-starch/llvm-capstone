/*
 * Capstone adapted oracle for RV8 `primes`.
 *
 * rv8-bench's primes sieves up to 33,333,333 (a ~4 MB malloc) and prints the
 * largest prime, 33,333,329. The 4 MB sieve far exceeds the domain's memory, so
 * the build reduces the limit to 100000 (~12.5 KB sieve, fits the bump arena)
 * and turns main() into a value-returning `rv8_primes_run()`.
 *
 * The build also fixes a latent upstream bug: the bit macros use `1 << (p&0x3f)`
 * where `1` is a 32-bit int and the shift can reach 63 -> undefined behaviour
 * (it only happens to work at the original limit). The build rewrites it to
 * `1ull << ...`, matching the obvious intent. With the fix the sieve is correct
 * at any limit: the largest prime <= 100000 is 99991 (independently well-known).
 */
#include "rv8_capstone_preamble.h"

extern int rv8_primes_run(void);
extern void rv8_arena_init(void);

void initialise_benchmark(void) { rv8_arena_init(); }

int benchmark(void) { return rv8_primes_run(); }

int verify_benchmark(int result) { return result == 99991; }
