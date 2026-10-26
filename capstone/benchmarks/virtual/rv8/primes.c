/* RV8 primes: upstream main() with limit = RV8_N. Upstream prints the largest prime <= limit and
 * never frees. Two deviations, both correctness fixes and neither capability-specific:
 *  - test/set shift 1ULL, not the int 1 (upstream shifts an int by up to 63: undefined behaviour,
 *    as the physical-profile port also found);
 *  - the sieve is zeroed after malloc (upstream relies on fresh pages; see aes.c).
 * The result is checked by trial division (it is prime, and nothing between it and limit is). */
#include RV8_UPSTREAM
#include "rv8-common.h"
#undef test
#undef set
#define test(p) (primes[p >> 6] & 1ULL << (p & 0x3f))
#define set(p) (primes[p >> 6] |= 1ULL << (p & 0x3f))

static int found;

static void unit(void) {
  int limit = RV8_N;
  size_t primes_size = ((limit >> 6) + 1) * sizeof(uint64_t);
  uint64_t *primes = (uint64_t *)malloc(primes_size);
  memset(primes, 0, primes_size);
  int64_t p = 2, sqrt_limit = (int64_t)sqrt(limit);
  while (p <= limit >> 1) {
    for (int64_t n = 2 * p; n <= limit; n += p)
      if (!test(n))
        set(n);
    while (++p <= sqrt_limit && test(p))
      ;
  }
  found = 0;
  for (int i = limit; i > 0; i--) {
    if (is_prime(i)) {
      found = i;
      break;
    }
  }
  free(primes);
}

static int is_prime_td(int n) {
  if (n < 2)
    return 0;
  for (int d = 2; (long)d * d <= n; d++)
    if (n % d == 0)
      return 0;
  return 1;
}

static int post(unsigned long long *check) {
  *check = (unsigned long long)found;
  if (!is_prime_td(found))
    return 0;
  for (int n = found + 1; n <= RV8_N; n++)
    if (is_prime_td(n))
      return 0;
  return 1;
}

int main(int argc, char **argv) { return bench_run("rv8/primes", argc, argv, unit, post); }
