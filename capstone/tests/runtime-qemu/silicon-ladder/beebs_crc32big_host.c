/* Native oracle: same crc value as beebs_crc32 (identical table). */
#include <stdio.h>
#include "beebs_crc32big_kernel.h"
int main(void) { printf("%u\n", crc_compute()); return 0; }
