/* What the Sublet arm of the interpreter needs beyond the shared adapter.
 *
 * The adapter (ports/cpython/pymalloc/src/allocators/sublet/block-lifetimes.c)
 * and its metadata heap (src/shared/backing.c) are program-independent and
 * compiled straight into this port's runtime directory; patch 0014 makes
 * obmalloc call them. Two things they expect from the program, which the
 * component port's own domain entry supplies and this port must supply itself:
 *
 *   pym_fail()          the adapter's abort. The component port returns to its
 *                       host through a saved frame; here the domain has stdio
 *                       over the hostcall and an exit() the runtime delivers, so
 *                       the code is printed and the domain exits with it. A
 *                       silent abort would be indistinguishable from the
 *                       interpreter crashing on its own.
 *   the two regions     the adapter's init contract, which is strict:
 *                       pym_lifetime_init wants a LINEAR capability of exactly
 *                       PYM_ARENA_BYTES, 16 KiB aligned, and splits it in half
 *                       into small-object and raw-fallback space;
 *                       pym_backing_init wants PYM_META_BYTES of writable
 *                       memory for the records, which must NOT live in freed
 *                       payload, since that payload is revoked.
 *
 * Both arrive through the runtime's existing path: hostcall.c compiled with
 * CAPSTONE_PROGRAM_REGIONS parks every region past the two host-call ones and
 * hands them over one at a time through __capstone_region(). HC_PROGRAM_REGIONS
 * is 2, which is exactly what is needed, so no runtime change -- index 0 is the
 * payload, index 1 the metadata, in the order the host shares them.
 *
 * MODE. pym_set_mode(0) is spatial: request-bounded pointers, no per-object
 * revocation. pym_set_mode(1) is sublet: every issue and release does
 * sublet_give then sublet_take, so a freed block's alias is dead. One image
 * carries both, chosen at run time, because the two arms of a matched pair must
 * differ in exactly one thing -- an image built twice differs in its layout too
 * (a single changed string moved 44743 bytes of this port's image, measured).
 * The value comes from the environment the run script writes before each run.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The adapter's own header is not on this TU's include path (it belongs to the
 * component port's build); these are the four entries this file uses. */
void pym_lifetime_init(void *region);
void pym_backing_init(void *metadata, void *arena);
void pym_set_mode(unsigned mode);
void *__capstone_region(unsigned index);

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
 * mode it set, or -1 when the regions were not shared -- which must be loud
 * rather than a silent fallback to an unprotected heap: a run that reported
 * "sublet" while allocating unprotected memory would be the worst possible
 * result, indistinguishable from the discipline failing to catch anything. */
int cpy_sublet_init(void) {
  if (sublet_started)
    return -2;
  void *payload = __capstone_region(0);
  void *metadata = __capstone_region(1);
  if (!payload || !metadata) {
    fprintf(stderr, "CPY-SUBLET-FAIL no regions (payload=%d metadata=%d); "
                    "the host must share the payload and metadata regions\n",
            payload != 0, metadata != 0);
    fflush(stderr);
    return -1;
  }
  /* Order: the payload's slot first (it only uses the capability primitives),
   * then the metadata heap, because pym_set_mode allocates its arena table out
   * of it. */
  pym_lifetime_init(payload);
  pym_backing_init(metadata, 0);
  const char *want = getenv(CPY_SUBLET_MODE_ENV);
  unsigned mode = (want && want[0] == '1') ? 1u : 0u;
  pym_set_mode(mode);
  sublet_started = 1;
  fprintf(stderr, "CPY-SUBLET mode=%u (%s)\n", mode,
          mode ? "sublet: every free revokes" : "spatial: bounds only");
  fflush(stderr);
  return (int)mode;
}
