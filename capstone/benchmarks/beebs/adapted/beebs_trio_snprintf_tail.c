/*
 * Capstone adapted oracle for BEEBS `trio-snprintf`.
 *
 * Upstream `src/trio/trio_test.c` under `-DTRIO_SNPRINTF` runs five
 * `trio_snprintf` calls into a scratch buffer and `verify_benchmark` returns -1
 * (no verification — each call overwrites the previous output). This tail
 * replaces that test core with a self-contained oracle: it runs the same five
 * format conversions and checks each formatted string against its expected
 * result, so the domain genuinely exercises and validates trio's integer
 * formatter (`TRIO_FEATURE_FLOAT=0`, so no long-double/fp128 path).
 *
 * `trio_snprintf`/`init_heap` come from the trio library objects (trio.c); the
 * string helpers come from the shared trio stubs. `output` is a plain local
 * buffer written by snprintf (no const initializer, so no Bug #9).
 *
 * Note: the upstream `"%ld"` call passed an `int` (123); reading it back as a
 * `long` through varargs is technically undefined, and upstream never checked
 * the result. This oracle passes `123L` so the conversion is well-defined.
 */
#include "beebs_trio_capstone_preamble.h"

extern int trio_snprintf(char *buffer, size_t max, const char *format, ...);
extern void init_heap(void);

void initialise_benchmark(void) { init_heap(); }

int benchmark(void) {
  char output[50];
  int ok = 1;

  trio_snprintf(output, 50, "%d", 123);
  ok &= (strcmp(output, "123") == 0);
  trio_snprintf(output, 50, "%ld", 123L);
  ok &= (strcmp(output, "123") == 0);
  trio_snprintf(output, 50, "%5d", 123);
  ok &= (strcmp(output, "  123") == 0);
  trio_snprintf(output, 50, "%05x", 123);
  ok &= (strcmp(output, "0007b") == 0);
  trio_snprintf(output, 50, "%*d", 5, 10);
  ok &= (strcmp(output, "   10") == 0);

  return ok;
}

int verify_benchmark(int result) { return result == 1; }
