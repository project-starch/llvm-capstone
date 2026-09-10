/* Native oracle for the fillwarm rung. */
#define FILLCOST_NATIVE_ORACLE 1
#include <stdio.h>
#include "fillwarm_kernel.h"
int main(void){printf("%u\n", fillwarm_compute()); return 0;}
