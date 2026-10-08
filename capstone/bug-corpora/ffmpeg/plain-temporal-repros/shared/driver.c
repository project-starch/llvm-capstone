/* main() for the FFmpeg plain-temporal corpus. One program per case, run twice against the same
 * binary: the control arm (fixed) first, then the buggy one. The allocator under test is the
 * platform's -- that is the whole point of the corpus.
 */
#include "corpus.h"

_Noreturn void fft_fail(unsigned code) {
  fprintf(stderr, "CONTROL-FAILED %u\n", code);
  exit(75); /* an infrastructure failure is never a verdict */
}

/* The labelled accesses, defined ONCE here so the symbol is resolvable from the
 * image and a fault can be required to land at it. */
__attribute__((noinline, used)) unsigned
fft_read_probe(const volatile unsigned char *p) {
  return *p;
}
__attribute__((noinline, used)) void
fft_write_probe(volatile unsigned char *p, unsigned char v) {
  *p = v;
}

int main(int argc, char **argv) {
  int fixed = argc > 1 && !strcmp(argv[1], "fixed");
  if (argc > 2 && atoi(argv[2]) != fft_case_number) {
    fprintf(stderr, "CONTROL-FAILED fixture is case %d, run asked for %s\n",
            fft_case_number, argv[2]);
    return 75;
  }
  struct fft_outcome o = {0};
  printf("case=%d arm=%s\n", fft_case_number, fixed ? "fixed" : "buggy");
  fft_case_run(fixed, &o);
  printf("bytes=%lu freed=%d marker=0x%02x observed=0x%02x aliased=%d damage=%d\n",
         o.bytes, o.freed, o.marker, o.observed, o.aliased, o.damage);
  /* A case that never reached its lifetime ender has measured nothing, whichever
   * arm it is -- that is INCONCLUSIVE, never a pass. */
  if (!o.freed)
    printf("VERDICT INCONCLUSIVE the lifetime ender did not run\n");
  else if (!fixed && o.aliased)
    printf("VERDICT DEFECT-REPRODUCED %s\n", o.defect_text);
  else if (fixed && !o.aliased)
    printf("VERDICT FIXED %s\n", o.fixed_text);
  else
    printf("VERDICT INCONCLUSIVE\n");
  return fixed ? !!o.aliased : !o.aliased;
}
