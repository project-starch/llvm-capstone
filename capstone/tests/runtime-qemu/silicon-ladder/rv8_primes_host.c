/* Native oracle: prints the same largest prime the domain must return (99991). */
#include <stdio.h>
#include "rv8_primes_kernel.h"
int main(void) { printf("%u\n", primes_compute()); return 0; }
