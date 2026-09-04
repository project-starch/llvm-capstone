/* Native oracle for the rc_const0 arm of the R-20 pair: the loop stores the index into a
   global array and accumulates it, so the answer is 0+1+...+63 = 2016. */
#include <stdio.h>
static long acc[64];
int main(void) {
  long n = 64, s = 0;
  int i;
  for (i = 0; i < n; i++) { acc[i] = i; s += acc[i]; }
  printf("%lu\n", (unsigned long)s);
  return 0;
}
