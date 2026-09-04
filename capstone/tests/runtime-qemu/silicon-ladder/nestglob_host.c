/* Native oracle: the same source with no capability hardware at all, built at the
   same NEST_GLOBAL_OFFSET. For the in-bounds arm agreement is the sanity check;
   for the out-of-bounds arm agreement would mean the overflow went UNTRAPPED. */
#include <stdio.h>
#include "nestalloc_kernel.h"
int main(void) { printf("%u\n", nest_global_run()); return 0; }
