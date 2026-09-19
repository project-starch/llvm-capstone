#include <cheri/cheric.h>
#include <malloc_np.h>
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv) {
  (void)argv;
  volatile unsigned char *p = malloc(64);
  if (!p)
    return 2;
  p[0] = 19;
  if (p[0] != 19)
    return 3;
  printf("HEAP_PROBE runtime=%u type=%ld tag=%u bytes=%zu\n",
         (unsigned)malloc_revoke_enabled(), cheri_gettype(p),
         (unsigned)cheri_gettag(p), (size_t)cheri_getlen(p));
  fflush(stdout);
  free((void *)p);
  if (argc > 1) {
    puts("HEAP_PROBE stale-access-ready");
    fflush(stdout);
    return p[0] == 19 ? 0 : 4;
  }
  return 0;
}
