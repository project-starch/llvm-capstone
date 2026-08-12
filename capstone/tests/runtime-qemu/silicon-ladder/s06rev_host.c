/* Native oracle: the value returned when BOTH arms survive, which is what QEMU should give.
   On silicon a different answer -- or no answer at all -- is the actual experiment. */
#include <stdio.h>
int main(void) { printf("11\n"); return 0; }
