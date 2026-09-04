/* Native oracle for the spatial rung: the same source on hardware with no
   capability checking at all, built at the same NEST_SPATIAL_OFFSET as the domain.
   Agreement between the two IS the measurement: it means the overflow out of one
   sub-allocated block into another was not trapped. */
#include <stdio.h>
#include "nestalloc_kernel.h"
int main(void) { printf("%u\n", nest_spatial_run()); return 0; }
