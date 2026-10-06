/* main() for the wiretap plain-heap corpus. One program per case, run twice against the
 * same binary: the control arm (fixed) first, then the buggy one. There are no
 * arenas and no pools here -- the allocator under test is the system's, which is
 * the whole point of the corpus.
 */
#include "corpus.h"

_Noreturn void wsh_fail(unsigned code) {
  fprintf(stderr, "CONTROL-FAILED %u\n", code);
  exit(75); /* an infrastructure failure is never a verdict */
}

int main(int argc, char **argv) {
  int fixed = argc > 1 && !strcmp(argv[1], "fixed");
  if (argc > 2 && atoi(argv[2]) != wsh_case_number) {
    fprintf(stderr, "CONTROL-FAILED fixture is case %d, run asked for %s\n",
            wsh_case_number, argv[2]);
    return 75;
  }
  struct wsh_outcome o = {0};
  printf("case=%d arm=%s\n", wsh_case_number, fixed ? "fixed" : "buggy");
  wsh_case_run(fixed, &o);
  printf("cap=%lu touched=%lu extent=%ld crossed=%d damage=%d\n", o.cap, o.touched,
         o.extent, o.crossed, o.damage);
  /* The defect reproduces when the buggy arm crossed and the fixed one did not.
   * A case that cannot say which is INCONCLUSIVE, never a pass. */
  if (!fixed && o.crossed)
    printf("VERDICT DEFECT-REPRODUCED %s\n", o.defect_text);
  else if (fixed && !o.crossed)
    printf("VERDICT FIXED %s\n", o.fixed_text);
  else
    printf("VERDICT INCONCLUSIVE\n");
  return fixed ? !!o.crossed : !o.crossed;
}

/* The labelled crossing, defined ONCE here so the symbol is resolvable from the
 * image. corpus.h explains why this is not `static` in the header. */
__attribute__((noinline, used)) unsigned
wsh_read_probe(const volatile unsigned char *p) {
  return *p;
}
