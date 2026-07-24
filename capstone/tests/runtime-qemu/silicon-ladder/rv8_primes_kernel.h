#ifndef RV8_PRIMES_KERNEL_H
#define RV8_PRIMES_KERNEL_H
/* Silicon-ladder rung 6: RV8 `primes` (rv8-bench) -- the first RV8-family rung.
 *
 * Faithful to rv8-bench's primes sieve, with the same two adaptations the
 * committed Capstone RV8 oracle (benchmarks/rv8/adapted/rv8_primes_tail.c)
 * already documents and uses:
 *   (1) LIMIT reduced from 33,333,333 (a ~4 MB sieve) to 100000, so the sieve
 *       fits the domain arena. The largest prime <= 100000 is 99991 (this is
 *       exactly the value the committed RV8 oracle verifies against).
 *   (2) The upstream `1 << (p & 0x3f)` bit macros are `1ull << ...`: `1` is a
 *       32-bit int there and the shift can reach 63 -> UB (it only happens to
 *       work at the original limit). With `1ull` the sieve is correct at any
 *       limit. This is the same latent-bug fix the RV8 oracle applies.
 *
 * UNLIKE the upstream (which mallocs the sieve), this single-TU ladder kernel
 * uses a file-scope .bss bitmap -- no allocator, no big *initialized* table
 * (the sieve is written at runtime), so it stays inside the silicon-gp model
 * (validated for a 1 KiB .bss table by crc32; this is ~1.6 KB). It exercises a
 * .bss array reached via `ldc gp[i]`, 64-bit shift arithmetic, and a real
 * nested sieve loop. The largest prime found is itself the oracle. */
typedef unsigned long long u64;

#define RV8_PRIMES_LIMIT 100000u

/* Bitmap: one bit per candidate; a set bit marks a composite. */
static u64 rv8_sieve[(RV8_PRIMES_LIMIT >> 6) + 1];

#define RV8_ISCOMP(p)  ((rv8_sieve[(p) >> 6] >> ((p) & 0x3f)) & 1ull)
#define RV8_SETCOMP(p) (rv8_sieve[(p) >> 6] |= (1ull << ((p) & 0x3f)))

static unsigned rv8_primes_run(void) {
  unsigned largest = 2;
  unsigned i, j;
  for (i = 2; i <= RV8_PRIMES_LIMIT; i++) {
    if (!RV8_ISCOMP(i)) {
      largest = i;
      for (j = i + i; j <= RV8_PRIMES_LIMIT; j += i)
        RV8_SETCOMP(j);
    }
  }
  return largest;
}

static unsigned primes_compute(void) { return rv8_primes_run(); }
#endif
