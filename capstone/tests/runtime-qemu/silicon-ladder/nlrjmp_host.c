/* Native oracle for the nlrjmp rung: the same kernel over the C library's setjmp. */
#include <stdio.h>
#include "nlrjmp_kernel.h"
int main(void) { printf("%u\n", nlrjmp_compute()); return 0; }
