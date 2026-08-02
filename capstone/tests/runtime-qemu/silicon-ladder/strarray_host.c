/* Native oracle for strarray_app.c -- same computation, host compiler. */
#include <stdio.h>
struct fd { const char *z; void *p1; void *p2; unsigned char f; };
int main(void)
{
  struct fd a[] = {
    { "fn0", (void *)0, (void *)0, (unsigned char)0 },
    { "fn1", (void *)0, (void *)0, (unsigned char)1 },
    { "fn2", (void *)0, (void *)0, (unsigned char)2 },
    { "fn3", (void *)0, (void *)0, (unsigned char)3 },
    { "fn4", (void *)0, (void *)0, (unsigned char)4 },
    { "fn5", (void *)0, (void *)0, (unsigned char)5 },
    { "fn6", (void *)0, (void *)0, (unsigned char)6 },
    { "fn7", (void *)0, (void *)0, (unsigned char)7 },
  };
  unsigned n = (unsigned)(sizeof(a) / sizeof(a[0])), s = 0, i;
  for (i = 0; i < n; i++)
    s += (unsigned)(unsigned char)a[i].z[2];
  printf("%u\n", s + n);
  return 0;
}
