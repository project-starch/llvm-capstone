/* Native oracle: prints the same CoreMark matrix crc16 the domain must return. */
#include <stdio.h>
#include "coremark_matrix_kernel.h"
int main(void) { printf("%u\n", coremark_matrix_compute()); return 0; }
