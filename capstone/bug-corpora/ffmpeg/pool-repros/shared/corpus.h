/* What a case in this corpus needs, so a case.c is a complete translation unit.
 *
 * The contract is the one in
 * ../../cpython/pymalloc-repros/SCHEMA.md; this header is its FFmpeg seam.
 * shared/driver.c supplies main(), the arenas and the pool; a case supplies
 * its sequence inside FF2_CASE(NN) and reports its own verdict line.
 */
#ifndef FF2_CORPUS_H
#define FF2_CORPUS_H

#include "libavutil/buffer.h"
#include "trace.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define POOL_BYTES 64
#define PLANE_BYTES 32

/* A case declares the number its directory carries. The driver refuses a
 * fixture that names another case rather than silently running it. */
#define FF2_CASE(n)                                                            \
  const int ff2_case_number = (n);                                             \
  int ff2_case_run(int fixed)

extern const int ff2_case_number;
int ff2_case_run(int fixed);

/* The pool the driver created. Real libavutil/buffer.c throughout. */
extern AVBufferPool *g_pool;

_Noreturn void ff2_fail(unsigned code);
#define CHECK(c, n)                                                            \
  do {                                                                         \
    if (!(c))                                                                  \
      ff2_fail(n);                                                             \
  } while (0)

/* Every case prints one of these and nothing else, so a run is result lines. */
#define FF2_VERDICT(defect_reproduced, fixed_held, defect_text, fixed_text)    \
  printf("VERDICT %s\n", (defect_reproduced)  ? "DEFECT-REPRODUCED " defect_text \
                         : (fixed_held)       ? "FIXED " fixed_text             \
                                              : "INCONCLUSIVE")
#endif
