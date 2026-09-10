/* Native oracle for the fillsd rung. */
#define FILLCOST_NATIVE_ORACLE 1
#include <stdio.h>
#include "fillsd_kernel.h"
int main(void){printf("%u\n", fillsd_compute()); return 0;}
