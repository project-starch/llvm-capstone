/* Native oracle: prints the same checksum the domain must return. */
#include <stdio.h>
#include "beebs_recursion_kernel.h"
int main(void) { printf("%u\n", rec_compute()); return 0; }
