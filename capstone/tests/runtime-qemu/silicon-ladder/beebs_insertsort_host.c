/* Native oracle: prints the same checksum the domain must return. */
#include <stdio.h>
#include "beebs_insertsort_kernel.h"
int main(void) { printf("%u\n", is_compute()); return 0; }
