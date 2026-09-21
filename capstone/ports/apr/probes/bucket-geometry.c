/* The shim claims APR_HAS_MMAP does not move APR_BUCKET_ALLOC_SIZE, because
 * apr_bucket is the union's largest member either way. This checks it instead
 * of asserting it, and prints the geometry a port has to preserve. */
#include "apr_bucket_shim.h"
#include <stdio.h>
int main(void) {
  printf("sizeof(apr_bucket)          %zu\n", sizeof(apr_bucket));
  printf("sizeof(apr_bucket_heap)     %zu\n", sizeof(apr_bucket_heap));
  printf("sizeof(apr_bucket_pool)     %zu\n", sizeof(apr_bucket_pool));
  printf("sizeof(apr_bucket_file)     %zu\n", sizeof(apr_bucket_file));
#if APR_HAS_MMAP
  printf("sizeof(apr_bucket_mmap)     %zu\n", sizeof(apr_bucket_mmap));
#endif
  printf("sizeof(union)               %zu   (APR_HAS_MMAP=%d)\n",
         sizeof(apr_bucket_structs), APR_HAS_MMAP);
  printf("APR_BUCKET_ALLOC_SIZE       %zu\n", (size_t)APR_BUCKET_ALLOC_SIZE);
  return 0;
}
