/* Native oracle for the fill-cost control rung. */
#define FILLCOST_NATIVE_ORACLE 1   /* declares this build as the x86 oracle; the domain guard is __CAPSTONE__ */
#include <stdio.h>
#include "fillnop_kernel.h"
int main(void){printf("%u\n", fillnop_compute()); return 0;}
