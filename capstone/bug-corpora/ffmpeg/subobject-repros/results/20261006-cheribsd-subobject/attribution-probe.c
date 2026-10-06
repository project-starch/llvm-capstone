/* Platform or fixture? The subobject fixtures' own FIXED arm faults under
   _RUNTIME_REVOCATION_ENABLE=1. This is the same compiler, SDK, link mode and
   platform with none of the pool library, so if it survives, the fault belongs to
   the fixture or to libffmpeg-pool, not to running under revocation. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(void) {
  char *p = malloc(4096); memset(p, 'x', 4096);
  char *q = realloc(p, 8192);
  free(q);
  void *r = malloc(64); free(r);
  printf("ATTRIB-OK alloc/realloc/free all survived under this revocation setting\n");
  return 0;
}
