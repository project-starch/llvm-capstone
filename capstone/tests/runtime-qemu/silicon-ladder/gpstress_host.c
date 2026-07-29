/* Native oracle: prints the same checksum the domain must return. */
#include <stdio.h>
#include "gpstress_kernel.h"
int main(void) { printf("%u\n", gpstress_compute()); return 0; }
