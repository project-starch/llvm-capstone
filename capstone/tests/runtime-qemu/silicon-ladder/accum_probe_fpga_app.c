/* R-8 probe v2. v1 returned res[0] correctly while ALL nine res[3..11] slots read
 * zero, so eight of nine sub-probes produced no data.
 *
 * Cause of that failure, and the reason this version is simpler: v1 pinned two
 * accumulators to specific registers with `register long a __asm__("s1")`. Pinning
 * a callee-saved register across a whole function body is fragile -- the compiler
 * still assumes it owns s1 for its own spills, and `res` itself lives in a0 -- so
 * the most likely explanation for the lost stores is that v1 corrupted its own
 * pointer, not that the board dropped them. Testing register CLASS is a good idea
 * but needs hand-written asm, not a C register variable; it is deferred.
 *
 * v2 keeps only what can be measured safely, writes each slot IMMEDIATELY after the
 * loop that produces it (so a later fault cannot erase an earlier result), and
 * marks the store pointer volatile so nothing is sunk or reordered.
 *
 *   dbg0  plain accumulate, 100 trips              expect 100
 *   dbg1  a = a + 1 form (janne's shape)           expect 100
 *   dbg2  accumulator in MEMORY (volatile global)  expect 100
 *   dbg3  addend from a volatile global            expect 100
 *   dbg4  accumulate INSIDE AN IF -- the shape expint and janne share  expect 100
 *   dbg5  accumulate in a NESTED loop, 10x10       expect 100
 *   dbg6  independent trip counter                 expect 100
 *   dbg7  short loop, 3 trips                      expect 3
 *   dbg8  long loop, 1000 trips                    expect 1000
 *
 * dbg4 and dbg5 are the load-bearing ones: the minimal accumulate ALREADY PASSES on
 * this board, so the fault needs an extra ingredient, and a conditional body and a
 * nested loop are the two both failing kernels have and the passing minimal probe
 * does not. dbg7 vs dbg8 tests trip-count dependence, which would suggest something
 * periodic (a trap) rather than static codegen.
 */
static volatile long ap_one = 1;
static volatile long ap_mem;

void domain_main(unsigned long *res, unsigned func) {
  (void)func;
  volatile unsigned long *out = res;      /* never sink or reorder these stores */
  int i, j;
  long n = 100;
  __asm__ volatile("" : "+r"(n));

  long a = 0;
  for (i = 0; i < n; i++) a += 1;
  out[3 + 0] = (unsigned long)a;

  long b = 0;
  for (i = 0; i < n; i++) b = b + 1;
  out[3 + 1] = (unsigned long)b;

  ap_mem = 0;
  for (i = 0; i < n; i++) ap_mem = ap_mem + 1;
  out[3 + 2] = (unsigned long)ap_mem;

  long c = 0;
  for (i = 0; i < n; i++) c += ap_one;
  out[3 + 3] = (unsigned long)c;

  long d = 0;
  for (i = 0; i < n; i++) { if (i != 49) d += 1; else d += 1; }
  out[3 + 4] = (unsigned long)d;

  long e = 0;
  for (i = 0; i < 10; i++) for (j = 0; j < 10; j++) e += 1;
  out[3 + 5] = (unsigned long)e;

  long f = 0, trips = 0;
  for (i = 0; i < n; i++) { f += 1; trips++; }
  out[3 + 6] = (unsigned long)trips;

  long g = 0;
  for (i = 0; i < 3; i++) g += 1;
  out[3 + 7] = (unsigned long)g;

  long h = 0;
  for (i = 0; i < 1000; i++) h += 1;
  out[3 + 8] = (unsigned long)h;

  out[0] = (unsigned long)a;
  out[2] = 0xD09E;
}
