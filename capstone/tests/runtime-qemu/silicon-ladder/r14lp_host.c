/* Native oracle for r14lp. Expect 4. */
#include <stdio.h>
#include <string.h>
struct kv_lp { const char *z; const char *y; };
int main(void)
{
  struct kv_lp a[64]; unsigned i; int ok = 0;
  for (i = 0; i < 4; i++) { a[i].z = "x0"; a[i].y = "y0"; }
  for (i = 0; i < 4; i++)
    if (a[i].z && a[i].y && strlen(a[i].z) > 0 && strlen(a[i].y) > 0) ok++;
  printf("%d\n", ok);
  return 0;
}
