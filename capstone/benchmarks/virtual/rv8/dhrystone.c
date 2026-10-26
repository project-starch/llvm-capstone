/* RV8 dhrystone: upstream Proc0() unchanged with LOOPS = RV8_N (build.py rewrites the #define).
 * Proc0 brackets its own loop with two gettimeofday calls; they are routed here (-Dgettimeofday=
 * rv8_gettimeofday) and become the harness bracket, so the timed region is exactly upstream's.
 * Proc0's report printf, after the bracket, is silenced (-Dprintf=rv8_silent_printf). The check is
 * Dhrystone's canonical end state. */
#include <sys/time.h>
#include "bench-harness.h"
static int rv8_calls;
int rv8_gettimeofday(struct timeval *tv, void *tz);
/* -Dprintf=rv8_silent_printf (build.py): Proc0's report line is not written. */
int rv8_silent_printf(const char *restrict fmt, ...) {
  (void)fmt;
  return 0;
}
#include RV8_UPSTREAM
#include "rv8-common.h"



int rv8_gettimeofday(struct timeval *tv, void *tz) {
  (void)tz;
  if (rv8_calls++ % 2 == 0) {
    bench_begin();
    tv->tv_sec = 0;
    tv->tv_usec = 0;
  } else {
    bench_end();
    long long ns = bench_window_ns();
    tv->tv_sec = ns / 1000000000LL;
    tv->tv_usec = (ns % 1000000000LL) / 1000;
    if (tv->tv_sec == 0 && tv->tv_usec < 8)
      tv->tv_usec = 8; /* Proc0 divides by benchtime >> 3 */
  }
  return 0;
}

static void unit(void) { Proc0(); }

static int post(unsigned long long *check) {
  /* BoolGlob is not checked: rv8-bench's Proc0 never initialises String1Loc (the published
   * Dhrystone copies "DHRYSTONE PROGRAM, 1'ST STRING" into it), so Func2's strcmp reads stack
   * garbage and BoolGlob is undefined. */
  int ok = IntGlob == 5 && Char1Glob == 'A' && Char2Glob == 'B' && Array1Glob[8] == 7 &&
           Array2Glob[8][7] == RV8_N + 10;
  *check = ((unsigned long long)(unsigned)Array2Glob[8][7] << 32) ^ ((unsigned)IntGlob << 16) ^
           ((unsigned)Char1Glob << 8) ^ (unsigned)Char2Glob;
  return ok;
}

int main(int argc, char **argv) { return bench_run_self("rv8/dhrystone", argc, argv, unit, post); }
