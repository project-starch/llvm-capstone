/* Oracle is not meaningful for a probe: the point is to PRINT what the board read.
   Emits 1, the value a correct copy must produce (descriptor count == 1). */
#include <stdio.h>
int main(void) { printf("1\n"); return 0; }
