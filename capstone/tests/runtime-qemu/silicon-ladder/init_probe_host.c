#include <stdio.h>
#include "init_probe_kernel.h"
int main(void) { printf("%u\n", ip_compute()); return 0; }
