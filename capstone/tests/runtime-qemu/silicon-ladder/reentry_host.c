#include <stdio.h>
#include "reentry_kernel.h"
int main(void){unsigned a=reentry_compute();unsigned b=reentry_compute();printf("%u %u\n",a,b);return 0;}
