/* Native oracle for the v3 shared-region diagnostic rung.
 *
 * Runs the identical kernel over an ordinary array (so "the shared region" is
 * just memory) and prints the fold, plus each probe's expected raw value so a
 * board capture can be read against it without recomputing anything by hand. */
#include <stdio.h>
#include "gp_diag3_kernel.h"

int main(int argc, char **argv) {
  static unsigned long m[512];
  unsigned h = gpd3_run(m);
  printf("%u\n", h);
  if (argc > 1 && argv[1][0] == '-' && argv[1][1] == 'v') {
    static const char *const names[GPD3_NPROBE] = {
        "A global-array loop", "B res[] loop (SUSPECT)", "C res[] straight-line",
        "D res[] loop, const index", "E local-array loop", "F res[] store loop",
        "G res[] walking pointer", "H res[] nested loop (v2 shape)", "I canary"};
    for (int i = 0; i < GPD3_NPROBE; i++)
      printf("dbg%d=%lu  %s\n", i, m[3 + i], names[i]);
  }
  return 0;
}
