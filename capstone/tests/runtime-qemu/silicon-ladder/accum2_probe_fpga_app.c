/* R-6/R-8 bisect v3. Targets the ONE fact the expint bisect established:
 *   del was correct (3881), the loop ran all 100 trips, and `ans += del` did not
 *   take -- ans stayed at its initial 2.
 * accum_probe already proved plain accumulation, a=a+1, memory accumulators,
 * conditional bodies, nested loops and 1000-trip loops ALL work. So the missing
 * ingredient is something expint/janne have that those nine shapes do not.
 *
 * The candidate this probe tests: an accumulate that follows a branch whose OTHER
 * side contains a NESTED LOOP -- i.e. the update is reached from a join point after
 * divergent control flow of unequal length. accum_probe's dbg4 had an if/else with
 * IDENTICAL trivial arms, which is not the same shape at all.
 *
 *   dbg0  expint's exact shape, ans at end                expect 3883
 *   dbg1  del at i==nm1                                   expect 3881
 *   dbg2  ans IMMEDIATELY after the i==nm1 update         expect 3883
 *   dbg3  count of iterations that executed `ans += del`  expect 100
 *   dbg4  second accumulator updated at the same point    expect 3881
 *   dbg5  same loop, accumulator forced to MEMORY         expect 3883
 *   dbg6  nested-loop trip count                          expect 49
 *   dbg7  branch-with-nested-loop, accumulate +1 only     expect 100
 *   dbg8  same as dbg0 but nested loop hoisted OUT        expect 3883
 *
 * dbg2 vs dbg0 splits "the update never happened" from "it happened and was lost
 * later". dbg5 splits register from memory. dbg8 removes the nested loop from the
 * branch while keeping everything else -- if dbg8 is right and dbg0 wrong, the
 * nested-loop-in-a-branch is the ingredient.
 */
static volatile long a2_sink;

void domain_main(unsigned long *res, unsigned func) {
  (void)func;
  volatile unsigned long *out = res;
  int i, ii;
  int nm1 = 49;
  long n = 100;
  __asm__ volatile("" : "+r"(n));

  /* dbg0..dbg4: expint's exact shape */
  long ans = 2, del = 0, psi = 0, acc2 = 0, ans_at_49 = 0;
  long updates = 0, nest_trips = 0;
  for (i = 1; i <= n; i++) {
    if (i != nm1) {
      del = 0;
    } else {
      psi = 0x00FF;
      for (ii = 1; ii <= nm1; ii++) { psi += ii + nm1; nest_trips++; }
      del = psi;
    }
    ans += del;
    acc2 += del;
    updates++;
    if (i == nm1) ans_at_49 = ans;
  }
  out[3 + 0] = (unsigned long)ans;
  out[3 + 1] = (unsigned long)del;
  out[3 + 2] = (unsigned long)ans_at_49;
  out[3 + 3] = (unsigned long)updates;
  out[3 + 4] = (unsigned long)acc2;

  /* dbg5: identical, accumulator in memory */
  a2_sink = 2;
  long d5 = 0;
  for (i = 1; i <= n; i++) {
    if (i != nm1) { d5 = 0; }
    else { long p = 0x00FF; for (ii = 1; ii <= nm1; ii++) p += ii + nm1; d5 = p; }
    a2_sink = a2_sink + d5;
  }
  out[3 + 5] = (unsigned long)a2_sink;
  out[3 + 6] = (unsigned long)nest_trips;

  /* dbg7: branch containing a nested loop, but accumulate a constant */
  long a7 = 0;
  for (i = 1; i <= n; i++) {
    if (i == nm1) { long p = 0; for (ii = 0; ii < nm1; ii++) p++; a7 += 1; }
    else a7 += 1;
  }
  out[3 + 7] = (unsigned long)a7;

  /* dbg8: expint's shape with the nested loop hoisted out of the branch */
  long p8 = 0x00FF;
  for (ii = 1; ii <= nm1; ii++) p8 += ii + nm1;
  long a8 = 2, d8 = 0;
  for (i = 1; i <= n; i++) { d8 = (i == nm1) ? p8 : 0; a8 += d8; }
  out[3 + 8] = (unsigned long)a8;

  out[0] = (unsigned long)ans;
  out[2] = 0xD09E;
}
