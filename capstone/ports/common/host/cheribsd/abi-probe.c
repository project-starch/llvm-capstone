/* Positive ABI/runtime control and a paired exact-boundary fault. */
#include <cheri/cheric.h>
#include <malloc_np.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef __CHERI_PURE_CAPABILITY__
#error "This probe must be compiled for CHERI purecap"
#endif
int main(int argc, char **argv) {
  unsigned char *allocation = malloc(32);
  if (!allocation)
    return 1;
  volatile unsigned char *bounded = cheri_setboundsexact(allocation, 16);
  if (!cheri_gettag(bounded) || cheri_getlen(bounded) != 16)
    return 1;
  bounded[15] = 41;
  if (bounded[15] != 41)
    return 1;
  printf("CHERI_ABI pointer_bytes=%zu runtime_revocation=%u\n",
         sizeof(void *), (unsigned)malloc_revoke_enabled());
  fflush(stdout);
  if (argc == 2 && strcmp(argv[1], "oob") == 0) {
    puts("CHERI_BOUNDARY_READY");
    fflush(stdout);
    unsigned value = bounded[16];
    printf("UNEXPECTED_ACCESS %u\n", value);
    return 1;
  }
  free(allocation);
  return argc == 1 ? 0 : 2;
}
