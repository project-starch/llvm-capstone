/* Native oracle for the rc_p1 arm of the R-20 pair: the loop stores index+1 into a global
   array and accumulates it, so the answer is 1+2+...+64 = 2080. The only source difference
   from rc_const0 is the stored value, which is the variable the pair isolates. */
#include <stdio.h>
static long acc[64];
int main(void) {
  long n = 64, s = 0;
  int i;
  for (i = 0; i < n; i++) { acc[i] = i + 1; s += acc[i]; }
  printf("%lu\n", (unsigned long)s);
  return 0;
}
