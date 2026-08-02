/* Native oracle for r14b_app.c -- same computation, host compiler. Expect 16. */
#include <stdio.h>
#include <string.h>
struct kv { const char *z; const char *y; };
int main(void)
{
  struct kv a[64]; unsigned i; int ok = 0;
  a[0].z="ltrim"; a[0].y="aaa0"; a[1].z="rtrim"; a[1].y="aaa1";
  a[2].z="trim";  a[2].y="aaa2"; a[3].z="max";   a[3].y="aaa3";
  for (i=4;i<64;i++){ a[i].z="filler"; a[i].y="fill"; }
  for (i=0;i<16;i++)
    if (a[i].z && a[i].y && strlen(a[i].z)>0 && strlen(a[i].y)>0) ok++;
  printf("%d\n", ok);
  return 0;
}
