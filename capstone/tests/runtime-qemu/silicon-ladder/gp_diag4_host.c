/* Native oracle for the v4 per-element readback dump: prints the gate value and,
 * with -v, every expected slot so a board capture can be diffed directly. */
#include <stdio.h>
#include "gp_diag4_kernel.h"

int main(int argc, char **argv) {
  static unsigned long m[512];
  unsigned long sum = gpd4_run(m);
  printf("%lu\n", sum);
  if (argc > 1 && argv[1][0] == '-' && argv[1][1] == 'v') {
    static const char *const group[4] = {
        "res[] pass 1", "larr[] local stack", "garr[] global via gp", "res[] pass 2"};
    for (int g = 0; g < 4; g++)
      for (int k = 0; k < GPD4_N; k++)
        printf("dbg%-2d=%-8lu %s[%d]\n", g * 8 + k, m[3 + g * 8 + k], group[g], k);
    printf("dbg32=%lu canary\n", m[3 + 32]);
  }
  return 0;
}
