// Control: the REVOKE must not be over-broad. After the domain revokes its own
// arena, unrelated authority it still holds -- here its own .bss, reached through
// the domain's gp-derived capability -- must keep working, and the domain must
// survive its own revoke rather than being wedged by it.
//
// The arena is written AND read back through the held capability before the
// revoke fires, so this probe also witnesses that the grant was live. It has to
// do that read-back itself: this is the one OK probe that revokes the whole
// arena, and once that lineage is dead the controller must not touch its Linux
// mapping of the region either (see README.md, "Gap found"). The retval carries
// the verdict, so nothing outside the domain needs to look at the bytes.
//
// Expected: no fault, retval 0x22310033. (0x223100bb would mean the read-back
// through the grant did not see the sentinel -- grant broken, not revoke.)
#include "probe_domain.h"

static volatile char unrelated[64];

void domain_main(void *arg, unsigned func) {
  if (probe_receive(arg, func))
    return;

  unsigned *res = (unsigned *)arg;

  void *A = probe_arena;
  void *R = __builtin_capstone_cap_mrev(A);
  void *D = __builtin_capstone_cap_delin(A);
  volatile char *buf = (volatile char *)D;

  buf[PROBE_OFFSET] = (char)PROBE_SENTINEL_LIVE; /* live use through the grant */
  volatile char live = buf[PROBE_OFFSET];        /* ...and read it back */

  __builtin_capstone_cap_revoke(R); /* sweeps the arena lineage */

  /* Different provenance: the domain's own data segment, not a descendant of
   * the arena's revocation node. Untouched by the sweep. */
  unrelated[PROBE_OFFSET] = 0x33;
  volatile char v = unrelated[PROBE_OFFSET];

  if ((unsigned char)live != PROBE_SENTINEL_LIVE) {
    *res = (PROBE_RET_UNRELATED_OK & ~0xffu) | 0xbbu; /* grant was never live */
    return;
  }
  *res = (PROBE_RET_UNRELATED_OK & ~0xffu) | (unsigned char)v; /* 0x22310033 */
}
