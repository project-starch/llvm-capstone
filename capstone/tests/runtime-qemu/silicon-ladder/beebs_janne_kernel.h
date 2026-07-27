#ifndef BEEBS_JANNE_KERNEL_H
#define BEEBS_JANNE_KERNEL_H
/* Silicon-ladder rung: BEEBS janne_complex (a *found* benchmark, kept faithful).
 *
 * Source: Bristol/Embecosm BEEBS `janne_complex` -- nested data-dependent WCET
 * loops, integer only. The compute is verbatim; no delin, no adaptation.
 *
 * SHAPE PREDICTION under issue R-1 (ref/ISSUES.md): PASS. The loop conditions are
 * computed entirely from locals (registers); the only global, `jc_iters`, is a
 * single location touched through a single capability register. R-1 needs a load
 * through one capability register with an intervening store through ANOTHER, and
 * that never occurs here. */

static int jc_iters;   /* .bss global, reached via the gp cap-table */

static int jc_complex(int a, int b) {
  while (a < 30) {
    while (b < a) {
      if (b > 5) b = b * 3;
      else       b = b + 2;
      if (b >= 10 && b <= 12) a = a + 10;
      else                    a = a + 1;
      jc_iters++;
    }
    a = a + 2;
    b = b - 10;
    jc_iters++;
  }
  return a * 1000 + b;
}

static unsigned jc_compute(void) {
  jc_iters = 0;
  /* The seeds are laundered through an opaque register: with literal 1,1 the
     whole nest constant-folds at -O1 and the benchmark disappears (the build
     gate catches it as ldc-gp=0). Same trap as beebs_crc32's folded table. */
  int a0 = 1, b0 = 1;
  __asm__ volatile("" : "+r"(a0)); __asm__ volatile("" : "+r"(b0));
  int r = jc_complex(a0, b0);
  unsigned h = 2166136261u;
  h ^= (unsigned)r;        h *= 16777619u;
  h ^= (unsigned)jc_iters; h *= 16777619u;
  return h;
}
#endif
