/* Native oracle: prints the same crc the domain must return. */
#include <stdio.h>
#include "beebs_crc32_kernel.h"
int main(void) { printf("%u\n", crc_compute()); return 0; }
