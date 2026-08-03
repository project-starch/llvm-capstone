/* Native oracle for r14sl. Expect 4. */
#include <stdio.h>
#include <string.h>
struct kv_sl { const char *z; const char *y; };
int main(void)
{
  struct kv_sl a[64]; unsigned i; int ok = 0;
  a[0].z = "x0"; a[0].y = "y0";
  a[1].z = "x0"; a[1].y = "y0";
  a[2].z = "x0"; a[2].y = "y0";
  a[3].z = "x0"; a[3].y = "y0";
  for (i = 0; i < 4; i++)
    if (a[i].z && a[i].y && strlen(a[i].z) > 0 && strlen(a[i].y) > 0) ok++;
  printf("%d\n", ok);
  return 0;
}
