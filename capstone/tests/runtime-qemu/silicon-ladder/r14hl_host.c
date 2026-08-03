/* Native oracle for r14hl. Expect 4. */
#include <stdio.h>
#include <string.h>
struct kv_hl { const char *z; const char *y; };
int main(void)
{
  struct kv_hl a[64]; unsigned i; int ok = 0;
  const char *z0 = "x0"; const char *y0 = "y0";
  for (i = 0; i < 4; i++) { a[i].z = z0; a[i].y = y0; }
  for (i = 0; i < 4; i++)
    if (a[i].z && a[i].y && strlen(a[i].z) > 0 && strlen(a[i].y) > 0) ok++;
  printf("%d\n", ok);
  return 0;
}
