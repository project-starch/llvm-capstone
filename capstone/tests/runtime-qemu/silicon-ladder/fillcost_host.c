/* Native oracle for the reclaim-fill cost rung. The fill's VALUE is layout-independent -- it
   counts every 64th slot that reads back zero -- so a native build produces the same number even
   though its pointers are 8 bytes rather than a capability's 16. */
#include <stdio.h>
#include "fillcost_kernel.h"
int main(void){printf("%u\n", fillcost_compute()); return 0;}
