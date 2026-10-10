/* What bounds does CheriBSD's malloc hand out for THIS corpus's request sizes?
 *
 * The calloc length table measured for the memcached/wireshark rows is NOT
 * transferable: it was taken at 1, 9, 16, 17 and 8192, and a size class is a
 * step function. Each ffmpeg plain-heap case predicts a CATCH conditional on its
 * own crossing leaving the USABLE allocation, so its own request size has to be
 * read rather than interpolated.
 *
 * Sizes come from argv, so the probe is reusable and the run records which sizes
 * were actually asked for instead of burying them in the source.
 */
#include <stdio.h>
#include <stdlib.h>
#include <cheriintrin.h>

int main(int argc, char **argv) {
  if (argc < 2) {
    printf("CAPBOUNDS NO-SIZES-GIVEN\n"); /* never a silent zero */
    return 75;
  }
  for (int i = 1; i < argc; i++) {
    size_t request = (size_t)strtoull(argv[i], NULL, 10);
    void *p = calloc(1, request);
    if (!p) {
      printf("CAPBOUNDS request=%zu ALLOC-FAILED\n", request);
      continue;
    }
    size_t len = cheri_length_get(p);
    unsigned long base = (unsigned long)cheri_base_get(p);
    unsigned long addr = (unsigned long)cheri_address_get(p);
    /* slack is what decides every CATCH prediction in this corpus: a crossing
     * that stays inside it is NOT caught, as memcached/plain-heap-repros/00
     * measured the hard way. */
    printf("CAPBOUNDS request=%zu length=%zu slack=%zu base=%#lx addr_eq_base=%d\n",
           request, len, len - request, base, addr == base);
    free(p);
  }
  printf("CAPBOUNDS DONE\n");
  return 0;
}
