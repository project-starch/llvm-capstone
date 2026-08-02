/* Native oracle for r14a_app.c -- same computation, host compiler. Expect 16. */
#include <stdio.h>
#include <string.h>
struct kv { const char *z; const char *y; };
int main(void)
{
  struct kv a[64]; unsigned i; int ok = 0;
  a[0].z="ltrim";  a[0].y="aaa0";   a[1].z="rtrim";  a[1].y="aaa1";
  a[2].z="trim";   a[2].y="aaa2";   a[3].z="max";    a[3].y="aaa3";
  a[4].z="min";    a[4].y="aaa4";   a[5].z="typeof"; a[5].y="aaa5";
  a[6].z="length"; a[6].y="aaa6";   a[7].z="instr";  a[7].y="aaa7";
  a[8].z="substr"; a[8].y="aaa8";   a[9].z="upper";  a[9].y="aaa9";
  a[10].z="lower"; a[10].y="aab0";  a[11].z="coalesce"; a[11].y="aab1";
  a[12].z="hex";   a[12].y="aab2";  a[13].z="unhex"; a[13].y="aab3";
  a[14].z="quote"; a[14].y="aab4";  a[15].z="replace"; a[15].y="aab5";
  for (i=16;i<64;i++){ a[i].z="filler"; a[i].y="fill"; }
  for (i=0;i<16;i++)
    if (a[i].z && a[i].y && strlen(a[i].z)>0 && strlen(a[i].y)>0) ok++;
  printf("%d\n", ok);
  return 0;
}
