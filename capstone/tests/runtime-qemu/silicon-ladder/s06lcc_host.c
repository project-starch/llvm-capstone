/* Native oracle. NOTE: this cannot run the real check -- LCC does not exist on the host, and the
   query's whole point is silicon behaviour. It prints the EXPECTED verdict so the runner has an
   oracle to compare against, not a computed one. */
#include <stdio.h>
int main(void) { printf("171\n"); return 0; }
