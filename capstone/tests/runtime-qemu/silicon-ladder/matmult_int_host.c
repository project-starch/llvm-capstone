/* Native oracle: prints the same checksum the domain must return. */
#include <stdio.h>
#include "matmult_int_kernel.h"
int main(void) { printf("%u\n", mm_compute()); return 0; }
