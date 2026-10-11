/* What a case in this corpus needs, so a case.c is a complete translation unit.
 *
 * SUB-OBJECT corpus. Copied verbatim from ../pool-repros/shared/corpus.h, which
 * is the contract's FFmpeg seam, because each corpus owns a private copy and the
 * FF2_* infrastructure macros come from the port's replay-config and must not be
 * renamed. The pools below are created by the driver and UNUSED by every case
 * here: these defects cross a bound INSIDE ONE ALLOCATION, between two members
 * of a struct, so the allocator they need is av_refstruct_allocz and nothing
 * else. The pools stay so this corpus links the identical, proven driver.
 *
 * The contract is ../../SCHEMA.md. shared/driver.c supplies main() and the
 * arenas; a case supplies its sequence inside FF2_CASE(NN) and reports its own
 * verdict line.
 */
#ifndef FF2_CORPUS_H
#define FF2_CORPUS_H

#include "libavutil/buffer.h"
#include "libavutil/refstruct.h"
#include "trace.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define POOL_BYTES 64
#define PLANE_BYTES 32
/* One entry of a per-frame side table, the shape VVC's tab_dmvr_mvf and rpl_tab
 * have: pooled through AVRefStructPool rather than AVBufferPool. */
#define TAB_BYTES 32

/* A case declares the number its directory carries. The driver refuses a
 * fixture that names another case rather than silently running it. */
#define FF2_CASE(n)                                                            \
  const int ff2_case_number = (n);                                             \
  int ff2_case_run(int fixed)

extern const int ff2_case_number;
int ff2_case_run(int fixed);

/* The pools the driver created. Real libavutil/buffer.c and libavutil/refstruct.c
 * throughout -- a case reduces its consumer, never the allocator.
 *   g_pool    AVBufferPool      the payload pool, e.g. a frame's planes
 *   g_refpool AVRefStructPool   the side-table pool; a release returns the entry
 *                               to pool->available_entries and the next
 *                               av_refstruct_pool_get() hands the same one back
 * A case that does not use g_refpool is unaffected by its existence. */
extern AVBufferPool *g_pool;
extern AVRefStructPool *g_refpool;

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
