/* Native oracle for the insertsort_diag rung: same checksum as beebs_insertsort,
   since the diagnostic slots are additive and res[0] is unchanged. */
#include <stdio.h>
#include "beebs_insertsort_kernel.h"
int main(void) { printf("%u\n", is_compute()); return 0; }
