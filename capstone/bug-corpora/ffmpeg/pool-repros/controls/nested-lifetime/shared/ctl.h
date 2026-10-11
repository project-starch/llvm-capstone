/* What a nested-lifetime control needs, so a case.c is a complete translation unit.
 *
 * Pre-registration: capstone/docs/plans/2026-10-11-nested-lifetime-controls.md. A control drives
 * FFmpeg's own AVBufferPool (libavutil/buffer.c, unchanged on the release arm, with patch 0003 on
 * the protected arm) through its API and makes every access of interest through a labelled probe
 * in its own case.c, announced by a CTL-ACCESS line that names the call site (file:line) and the
 * reference it uses. shared/driver.c supplies main() and the pool.
 *
 * Output, one fact per line, so a run is result lines:
 *   case=NN control=<name> arm=<protected|release> mode=<run|stale>
 *   CTL-PTR <name> cursor= base= end=          a pointer's capability metadata (LCC)
 *   CTL-ACCESS id= op= ref= site=file:line     printed immediately before the probe call
 *   CTL-ACCESS-DONE id= ...                    printed only if the probe returned
 *   CTL-CHECK <what> ok|FAIL ...
 *   CTL-RESULT completed                       the control's last line, exit 0
 * A failed CHECK prints CONTROL-FAILED and exits 75: an infrastructure failure, never a reading.
 */
#ifndef FF2_CTL_H
#define FF2_CTL_H

#include "libavutil/buffer.h"
/* BufferPoolEntry: the pool's own record of an entry. Patch 0003 adds sublet_life and
 * sublet_issued to it; the release arm's header has neither. */
#include "libavutil/buffer_internal.h"
#include "trace.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef CTL_PROTECTED
#error "build with -DCTL_PROTECTED=1 (FFPOOL_SUBLET=ON, patch 0003) or -DCTL_PROTECTED=0 (release)"
#endif

#define POOL_BYTES 64

#define CTL_CASE(n, name)                                                      \
  const int ctl_case_number = (n);                                             \
  const char ctl_case_name[] = name;                                           \
  int ctl_case_run(int stale)

extern const int ctl_case_number;
extern const char ctl_case_name[];
int ctl_case_run(int stale);

/* The pool the driver created: av_buffer_pool_init(POOL_BYTES, NULL), so every entry's
 * storage is av_buffer_alloc(POOL_BYTES), i.e. av_malloc -> the process's malloc. */
extern AVBufferPool *g_pool;

_Noreturn void ff2_fail(unsigned code);
#define CHECK(c, n, ...)                                                       \
  do {                                                                         \
    if (!(c)) {                                                                \
      printf("CONTROL-FAILED %u: ", (unsigned)(n));                            \
      printf(__VA_ARGS__);                                                     \
      putchar('\n');                                                           \
      ff2_fail(n);                                                             \
    }                                                                          \
  } while (0)

/* LCC: read a capability's metadata without dereferencing it. The selector is the register
 * number of rs2 (x2 cursor, x3 base, x4 end), as capstone/capability.h uses it. */
#define CTL_LCC(p, n)                                                          \
  ({                                                                           \
    unsigned long ctl_r_;                                                      \
    __asm__ volatile(".insn r 0x5b, 0x1, 0x04, %0, %1, x" #n "\n"              \
                     : "=r"(ctl_r_)                                            \
                     : "r"((const void *)(p)));                                \
    ctl_r_;                                                                    \
  })
#define CTL_CURSOR(p) CTL_LCC(p, 2)

static inline void ctl_ptr(const char *name, const void *p) {
  printf("CTL-PTR %s cursor=0x%lx base=0x%lx end=0x%lx\n", name, CTL_LCC(p, 2), CTL_LCC(p, 3),
         CTL_LCC(p, 4));
  fflush(stdout);
}

/* The call site of a probe, as file:line. CTL_FILE is the case directory's case.c. */
#define CTL_ANNOUNCE(id, op, ref)                                              \
  do {                                                                         \
    printf("CTL-ACCESS id=%s op=%s ref=%s site=%s:%d\n", id, op, ref,         \
           CTL_FILE, __LINE__);                                                \
    fflush(stdout);                                                            \
  } while (0)

#endif
