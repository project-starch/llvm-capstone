/* BEEBS driver for the virtual profile: one harness iteration is exactly the body of BEEBS's own
 * support/main.c --
 *
 *     initialise_benchmark ();
 *     for (i = 0; i < REPEAT_FACTOR; i++) { initialise_benchmark (); result = benchmark (); }
 *     correct = verify_benchmark (result);
 *
 * with REPEAT_FACTOR from support.h (set through -DBOARD_REPEAT_FACTOR). The exit status follows
 * BEEBS's `return !correct`: an iteration fails only when verify_benchmark returns 0. A program
 * whose verify_benchmark returns -1 ("no verification", upstream) is built with
 * -DBEEBS_STATE_CHECK and a state/<name>.c that hashes its output; that hash is the check value,
 * compared against the native build by the runner, and is computed after the bracket. */
#include "support.h"
#include "bench-harness.h"

#ifndef BENCH_ID
#error "BENCH_ID must name the program (beebs/<name>)"
#endif

extern int initialise_benchmark(void);

#ifdef BEEBS_STATE_CHECK
unsigned long long beebs_state_check(int result);
#endif

static volatile int result;
static int correct;

static void unit(void) {
  int i;
  initialise_benchmark();
  for (i = 0; i < REPEAT_FACTOR; i++) {
    initialise_benchmark();
    result = benchmark();
  }
  correct = verify_benchmark(result);
}

static int post(unsigned long long *check) {
#ifdef BEEBS_STATE_CHECK
  *check = beebs_state_check(result);
#else
  *check = (unsigned long long)(unsigned int)result;
#endif
  return correct != 0;
}

int main(int argc, char **argv) {
  return bench_run(BENCH_ID, argc, argv, unit, post);
}
