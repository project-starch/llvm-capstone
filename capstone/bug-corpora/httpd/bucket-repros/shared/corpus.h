/* What a case in this corpus needs. The contract is the one in
 * ../../cpython/pymalloc-repros/SCHEMA.md; this header is its APR-bucket seam.
 *
 * Real here: apr-util's apr_buckets_alloc.c and APR's apr_pools.c, both
 * upstream byte for byte through the port's shims. Reduced: brigades, filters,
 * connections and requests -- none of which changes which storage the allocator
 * is asked for or when it is asked to take it back.
 */
#ifndef APRB_CORPUS_H
#define APRB_CORPUS_H

#include "apr_bucket_shim.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define APRB_CASE(n)                                                           \
  const int aprb_case_number = (n);                                            \
  int aprb_case_run(int fixed, apr_pool_t *root)

extern const int aprb_case_number;
int aprb_case_run(int fixed, apr_pool_t *root);

/* free() is interposed and counted, so "nothing reaches malloc" is measured in
 * each case rather than inherited from the census. */
extern unsigned long freed_to_malloc;

_Noreturn void aprb_fail(unsigned code);
#define CHECK(c, n)                                                            \
  do {                                                                         \
    if (!(c))                                                                  \
      aprb_fail(n);                                                            \
  } while (0)

#define APRB_VERDICT(defect, held, defect_text, fixed_text)                    \
  printf("VERDICT %s\n", (defect)  ? "DEFECT-REPRODUCED " defect_text          \
                         : (held)  ? "FIXED " fixed_text                       \
                                   : "INCONCLUSIVE")

/* The part of a brigade these defects touch: a holder, on a pool, of pointers
 * into bucket-allocator storage. apr_brigade.c is not ported and is not needed
 * -- what every one of these cases turns on is which pool the holder is on and
 * which allocator the storage came from. */
#define APRB_SLOTS 8
struct brigade {
  apr_pool_t *pool;              /* the brigade's own lifetime */
  void *bucket[APRB_SLOTS];      /* storage from a bucket allocator */
  apr_bucket_alloc_t *from[APRB_SLOTS]; /* which allocator issued it */
  int n;
};
#endif
