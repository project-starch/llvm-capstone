/* Native oracle: the same folded value correct hardware must produce. */
#include <stdio.h>
#include "rawhazard_kernel.h"
int main(void) {
  unsigned h = 2166136261u;
  for (int d = 0; d < 16; d++) { unsigned v = rh_probe(0, d); h ^= v; h *= 16777619u; }
  unsigned sw = rh_swap_recompare(); h ^= sw; h *= 16777619u;
  for (int d = 0; d < 4; d++) { unsigned v = rh_probe(2, d); h ^= v; h *= 16777619u; }
  printf("%u\n", h); return 0;
}
