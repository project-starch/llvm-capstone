/* Native oracle for the s06agg rung. A correct machine copies both granules intact, so both
   bits are set and the answer is 3. This is what the compiler fix must make the board return. */
#include <stdio.h>
#include "s06agg_kernel.h"
int main(void) { printf("%u\n", s06agg_compute()); return 0; }
