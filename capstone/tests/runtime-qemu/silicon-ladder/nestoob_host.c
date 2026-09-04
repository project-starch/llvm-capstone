/* Oracle for the control arm. Deliberately not run natively: the native build of
   an out-of-bounds write is undefined behaviour and its value would mean nothing.
   34 is what the domain would return if the store were ALLOWED, so a rung that
   "passes" here is a control failure and must be read as such. */
#include <stdio.h>
int main(void) { printf("34\n"); return 0; }
