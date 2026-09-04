/* Native oracle for the S-12 minimal repro.
 *
 * The domain's return value is 0xC12A0000 | (arm << 12) | bad. On a machine with no defect --
 * which is what QEMU is for every mechanism under investigation here -- `bad` is 0, because the
 * reload of a slot that was just written with a tagged capability must yield that capability.
 *
 * Computed from the SAME -DS12_ARM the domain was built with, not from the header default. A
 * host built at the default while the domain is built at arm 3 compares two different things,
 * and that comparison either fails for the wrong reason or passes by coincidence -- the exact
 * drift recorded for fdreg_host.c.
 *
 * Arm 2 has no oracle here and must not be built for QEMU: it clears the tag on purpose and the
 * emulator aborts on the resulting type query. It is board-only by construction. */
#include <stdio.h>

#ifndef S12_ARM
#define S12_ARM 1
#endif

int main(void) {
#if S12_ARM == 2
  fprintf(stderr, "s12 arm 2 is BOARD-ONLY: QEMU aborts on a type query of an untagged value "
                  "(op_helper.c:719 asserts the tag before the selector check).\n");
  return 2;
#else
  unsigned expect = 0xC12A0000u | ((unsigned)S12_ARM << 12) | 0u;
  printf("%u\n", expect);
  return 0;
#endif
}
