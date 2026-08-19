/* Native oracle for the nestalloc rung: the SAME source on hardware with no
   capability checking whatsoever. It is the unprotected baseline on purpose.
   The rung asserts the domain agrees with it, which is the measurement:
   agreement means the use-after-free was not trapped. */
#include <stdio.h>
#define NEST_STALE_OFFSET 0u
#include "nestalloc_kernel.h"
int main(void) { printf("%u\n", nest_run()); return 0; }
