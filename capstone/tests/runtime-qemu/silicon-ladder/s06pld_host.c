/* Native oracle: LCC does not exist on the host, so print the EXPECTED verdict -- every spill
   survives on a correct machine. */
#include <stdio.h>
int main(void) { printf("65535\n"); return 0; }
