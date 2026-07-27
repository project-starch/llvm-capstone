#ifndef BEEBS_FIBCALL_KERNEL_H
#define BEEBS_FIBCALL_KERNEL_H
/* Silicon-ladder rung: BEEBS fibcall -- iterative Fibonacci behind a call.
 *
 * Source: Bristol/Embecosm BEEBS `fibcall` (SNU-RT). Verbatim compute.
 *
 * SHAPE PREDICTION under issue R-1 (ref/ISSUES.md): PASS. The kernel touches no
 * array at all -- every value lives in a local, and the only memory traffic is
 * the stack. R-1 needs a load through one capability register while a store
 * through a DIFFERENT capability register into the same object is pending; that
 * shape cannot occur here. A failure would mean R-1 is not the whole story (as
 * R-6 already hints), so this rung is cheap insurance as well as a data point.
 *
 * Why it earns a row: `beebs_prime` is currently the ONLY pure-scalar point in
 * the overhead table, and a single point cannot separate "scalar code is cheap"
 * from "that particular benchmark is cheap".
 *
 * The two globals are load-bearing, not decoration. Upstream fibcall keeps its
 * argument in a `volatile int` for the same reason we must: a rung with NO
 * globals emits no `ldc gp[i]` at all, so it never touches the gp cap-table --
 * which is the ABI this whole table is pricing. The build gate rejects such a
 * rung (`FAIL: no ldc gp[i] global access found`) rather than let it be measured
 * as if it exercised the ABI. `beebs_prime` and `beebs_recursion` carry a
 * volatile global for exactly this reason. */

static volatile int fibcall_n;    /* the argument, as upstream keeps it */
static int fibcall_ans;

static int fibcall_fib(int n) {
  int i, Fnew, Fold, temp, ans;
  Fnew = 1;
  Fold = 0;
  for (i = 2; i <= 30 && i <= n; i++) {
    temp = Fnew;
    Fnew = Fnew + Fold;
    Fold = temp;
  }
  ans = Fnew;
  return ans;
}

static unsigned fibcall_compute(void) {
  unsigned h = 2166136261u;
  /* Upstream computes fib(30) once, which is far too short to measure against
     an interrupt-sized error bar (see the measurements doc: a run that catches
     an interrupt shows ~16k cycles of noise). Sweep every n instead -- same
     kernel, enough work to bracket. */
  for (int rep = 0; rep < 64; rep++) {
    for (int n = 0; n <= 30; n++) {
      fibcall_n = n;                    /* volatile global store */
      fibcall_ans = fibcall_fib(fibcall_n);
      h ^= (unsigned)fibcall_ans;
      h *= 16777619u;
    }
  }
  return h;
}
#endif
