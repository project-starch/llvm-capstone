/* What a case in this corpus needs, so a case.c is a complete translation unit.
 *
 * PLANE corpus. A sibling of ../subobject-repros and ../pool-repros, and separate
 * from both on purpose:
 *
 *   pool-repros       storage an AVBufferPool handed out and took back  (temporal)
 *   subobject-repros  a bound between two MEMBERS of one av_malloc      (spatial, not nested)
 *   plane-repros      a bound between two PLANES of one AVBuffer        (spatial, NESTED)
 *
 * The third is the shape the inventory's FFmpeg nested-spatial cell needs and had
 * none of. av_frame_get_buffer carves every plane of a frame out of a SINGLE
 * AVBuffer -- measured, not assumed: a YUVA420P frame reports buf[1] == NULL and
 * 960 to 1024 bytes of slack after the alpha plane inside that one buffer. So a
 * read that leaves a plane stays inside the allocation, exactly as a wmem chunk
 * overread stays inside its block, and per-malloc bounds are in bounds for it.
 *
 * The contract is ../../SCHEMA.md. shared/driver.c supplies main(); a case
 * supplies its sequence inside FFP_CASE(NN) and fills the outcome the driver
 * prints.
 */
#ifndef FFP_CORPUS_H
#define FFP_CORPUS_H

#include "libavutil/frame.h"
#include "libavutil/imgutils.h"
#include "libavutil/pixdesc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FFP_CASE(n)                                                            \
  const int ffp_case_number = (n);                                             \
  void ffp_case_run(int fixed, struct ffp_outcome *o)

/* What a case observed. `crossed` is the claim the corpus exists to measure: the
 * read left its PLANE. `contained` is what makes it nested rather than a plain
 * overflow -- it stayed inside the frame's single AVBuffer. A case that cannot
 * assert both is inconclusive, never a pass. */
struct ffp_outcome {
  int crossed;                 /* the access left the plane */
  int contained;               /* ... and stayed inside the one AVBuffer */
  int damage;                  /* the consequence the upstream report describes */
  long plane_slack;            /* bytes between the plane's end and the buffer's */
  const char *defect_text, *fixed_text;
};

extern const int ffp_case_number;
void ffp_case_run(int fixed, struct ffp_outcome *o);

_Noreturn void ffp_fail(unsigned code);
#define CHECK(c, n)                                                            \
  do {                                                                         \
    if (!(c))                                                                  \
      ffp_fail(n);                                                             \
  } while (0)

/* The crossing, labelled so a sanitiser's report -- if one ever fires -- can be
 * required to land HERE. It is not expected to: the read stays inside the
 * allocation, which is the whole point of the row.
 *
 * DEFINED ONCE IN driver.c, NOT static here, and that is load-bearing rather
 * than style. `supervise` (the CheriBSD observer) resolves the probe by NAME
 * from the ELF symbol table, so that a fault can be REQUIRED to land inside it
 * instead of merely somewhere in the program. A `static` definition has
 * internal linkage: `used` keeps the compiler from discarding it, but the name
 * never reaches the symbol table, so every arm would come back UNRESOLVED --
 * a whole platform run spent on an instrument that cannot report. The sibling
 * corpora already do it this way (plain-heap-repros/shared/driver.c defines
 * ffh_read_probe); this corpus was header-static only because it had no
 * platform runner yet. */
unsigned ffp_read_probe(const volatile unsigned char *p);
#define read_probe ffp_read_probe

/* The carve-bounds remedy, FFP_CARVE_BOUNDS (added 2026-10-09). av_frame_get_buffer carves every
 * plane out of ONE AVBuffer by pointer arithmetic (libavutil/frame.c) and narrows none of them, so a
 * read one row past a plane lands in the same buffer. With the switch, the case hands the consumer
 * the plane narrowed to its own rows, linesize * height -- what frame.c would do if it narrowed at
 * the carve. Without it this is the identity, so every other arm builds byte-identically. */
#if defined(FFP_CARVE_BOUNDS)
__attribute__((unused)) static unsigned char *ffp_carve(unsigned char *p, unsigned long len) {
#if defined(__CHERI_PURE_CAPABILITY__)
  return __builtin_cheri_bounds_set(p, len);
#elif defined(__CAPSTONE__)
  unsigned long long at = __builtin_capstone_cap_get_cursor(p);
  return __builtin_capstone_cap_shrink(p, at, at + len);
#else
#error "FFP_CARVE_BOUNDS needs a capability target: CHERI purecap or Capstone"
#endif
}
#else
/* The identity as a macro, not a function, so the build without the switch is byte-identical. */
#define ffp_carve(p, len) (p)
#endif

#endif
