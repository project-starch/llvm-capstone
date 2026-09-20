/* What a case in this corpus needs. The contract is the one in
 * ../../cpython/pymalloc-repros/SCHEMA.md; this header is its APR seam.
 * shared/driver.c supplies main() and the root pool; a case supplies its
 * sequence inside APR_CASE(NN) and reports its own verdict line. */
#ifndef APR_CORPUS_H
#define APR_CORPUS_H

#include "apr_pools.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define APR_CASE(n)                                                            \
  const int apr_case_number = (n);                                             \
  int apr_case_run(int fixed, apr_pool_t *root)

extern const int apr_case_number;
int apr_case_run(int fixed, apr_pool_t *root);

_Noreturn void apr_corpus_fail(unsigned code);
#define CHECK(c, n)                                                            \
  do {                                                                         \
    if (!(c))                                                                  \
      apr_corpus_fail(n);                                                      \
  } while (0)

#define APR_VERDICT(defect_reproduced, fixed_held, defect_text, fixed_text)    \
  printf("VERDICT %s\n", (defect_reproduced) ? "DEFECT-REPRODUCED " defect_text \
                         : (fixed_held)      ? "FIXED " fixed_text              \
                                             : "INCONCLUSIVE")
#endif
