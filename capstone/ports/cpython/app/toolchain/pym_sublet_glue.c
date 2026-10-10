/* What the Sublet arm of the interpreter needs beyond the shared adapter.
 *
 * The adapter (ports/cpython/pymalloc/src/allocators/sublet/block-lifetimes.c)
 * and its metadata heap (src/shared/backing.c) are program-independent and
 * compiled straight into this port's runtime directory; patch 0014 makes
 * obmalloc call them. Two things they expect from the program:
 *
 *   pym_fail()          the adapter's abort: the code is printed and the
 *                       process exits with it. A silent abort would be
 *                       indistinguishable from the interpreter crashing on its
 *                       own.
 *   the two regions     pymalloc's arena region, PYM_ARENA_BYTES on a pool
 *                       boundary, and PYM_META_BYTES for the metadata heap. Both
 *                       are ordinary objects from the system allocator (musl
 *                       mallocng): the adapter derives a child of the arena
 *                       region and every arena and block below it (CDERIVE), so
 *                       nothing has to be lent linear.
 *
 * MODE. pym_set_mode(0) is spatial: request-bounded pointers, no per-object
 * revocation. pym_set_mode(1) is sublet: every block is a child lifetime,
 * revoked when pymalloc takes it back. One image carries both, chosen at run
 * time, because the two arms of a matched pair must differ in exactly one
 * thing -- an image built twice differs in its layout too (a single changed
 * string moved 44743 bytes of this port's image, measured). The value comes
 * from the environment.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "port.h"

#ifdef PYMALLOC_GAP_OBSERVER
void pym_gap_report(void);
#endif

#ifndef CPY_SUBLET_MODE_ENV
#define CPY_SUBLET_MODE_ENV "CPY_SUBLET_MODE"
#endif

static int sublet_started;

_Noreturn void pym_fail(unsigned code) {
  fprintf(stderr, "CPY-SUBLET-FAIL code=%u\n", code);
  fflush(stderr);
  exit(code ? (int)code : 1);
}

/* Called once, before the first allocation that can reach pymalloc. Returns the
 * mode it set, or -1 when the regions could not be allocated -- which must be loud
 * rather than a silent fallback to an unprotected heap: a run that reported
 * "sublet" while allocating unprotected memory would be the worst possible
 * result, indistinguishable from the discipline failing to catch anything. */
int cpy_sublet_init(void) {
  if (sublet_started)
    return -2;
  void *payload = aligned_alloc(16384, PYM_ARENA_BYTES);
  void *metadata = malloc(PYM_META_BYTES);
  if (!payload || !metadata) {
    fprintf(stderr, "CPY-SUBLET-FAIL no regions (payload=%d metadata=%d)\n",
            payload != 0, metadata != 0);
    fflush(stderr);
    return -1;
  }
  pym_lifetime_init(payload);
  pym_backing_init(metadata, 0);
  const char *want = getenv(CPY_SUBLET_MODE_ENV);
  unsigned mode = (want && want[0] == '1') ? 1u : 0u;
  pym_set_mode(mode);
#ifdef PYMALLOC_GAP_OBSERVER
  if (atexit(pym_gap_report))
    pym_fail(735);
#endif
  sublet_started = 1;
  fprintf(stderr, "CPY-SUBLET mode=%u (%s)\n", mode,
          mode ? "sublet: every free revokes" : "spatial: bounds only");
  fflush(stderr);
  return (int)mode;
}
