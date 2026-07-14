/* Real-workload temporal-safety overhead -- CHERI side. Runs the shared
 * binary-search-tree build/lookup/destroy workload (tests/shared/tree_workload.h)
 * under the three CHERI revocation policies, so the CHERI cost is measured on a
 * real pointer-chasing allocation pattern with a live-set, not a synthetic
 * malloc/free loop. Counterpart to tree_capstone on the Capstone side.
 *
 * Same instrumentation as revoke_cost_cheri.c: rdinstret bracket (user+kernel,
 * so the kernel revocation sweep is counted). Config is chosen by the caller via
 * sysctl before exec (see run-in-guest.sh). Output line:
 *   RCTREE mode=<name> keys=<k> rounds=<r> ops=<n> instr=<c> perop=<f>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef TW_KEYS
#define TW_KEYS 2000u
#endif
#ifndef TW_ROUNDS
#define TW_ROUNDS 25u
#endif
#define TW_MALLOC(n) malloc(n)
#define TW_FREE(p) free(p)
#include "../shared/tree_workload.h"

static inline unsigned long rd_count(void) {
  unsigned long v;
#ifdef USE_CYCLE
  __asm__ volatile("rdcycle %0" : "=r"(v));
#else
  __asm__ volatile("rdinstret %0" : "=r"(v));
#endif
  return v;
}

static volatile unsigned long sink;

int main(int argc, char **argv) {
  const char *mode = argc > 1 ? argv[1] : "unknown";

  /* warm the allocator + counter path */
  (void)tw_run(&sink);

  unsigned long t0 = rd_count();
  unsigned long ops = tw_run(&sink);
  unsigned long t1 = rd_count();

  unsigned long instr = t1 - t0;
  double perop = ops ? (double)instr / (double)ops : 0.0;
  printf("RCTREE mode=%s keys=%u rounds=%u ops=%lu instr=%lu perop=%.3f\n",
         mode, TW_KEYS, TW_ROUNDS, ops, instr, perop);
  fflush(stdout);
  return 0;
}
