/* Native oracle: prints the same checksum the domain must return. */
#include <stdio.h>
#include "beebs_prime_kernel.h"
int main(void) { printf("%u\n", prime_compute()); return 0; }
