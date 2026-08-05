#include <stdio.h>
#include "mtvfault_kernel.h"
int main(void) { printf("%u\n", mtvfault_expect()); return 0; }
