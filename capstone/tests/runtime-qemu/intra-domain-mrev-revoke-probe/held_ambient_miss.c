// Provenance control: is the arena reachable by any route OTHER than the
// delivered capability? If it were, a revoke could be silently sidestepped -- the
// exact hazard the task-005 spike found for a .bss arena (mrev_ambient_miss:
// the domain's gp already covered those bytes, so the sweep missed them).
//
// Here the arena is a monitor-granted region outside the domain's image, so the
// only authority over it is the capability the monitor handed us. This probe
// proves the negative: take the arena's base ADDRESS as a plain integer, forge a
// pointer from it, and dereference. inttoptr yields an UNTAGGED value, so the
// deref must trap -- there is no ambient second path to those bytes.
//
// No mrev and no revoke, so the fault cannot be blamed on revocation. The probe
// first writes through the real capability, which proves the bytes are mapped
// and writable; only then does it try the forged address. That ordering is what
// makes this a proof of "no second path" rather than "the region isn't there".
//
// Expected: fault, cause 24 "Cap mem access requires capability".
#include "probe_domain.h"

void domain_main(void *arg, unsigned func) {
  if (probe_receive(arg, func))
    return;

  unsigned *res = (unsigned *)arg;

  void *A = probe_arena;
  volatile char *buf = (volatile char *)A;
  buf[PROBE_OFFSET] = (char)PROBE_SENTINEL_LIVE; /* the bytes ARE reachable... */

  /* ...through the capability. Now try the same bytes by raw address. */
  volatile unsigned long base = __builtin_capstone_cap_get_base(A);
  volatile char *ambient = (volatile char *)base; /* inttoptr -> untagged */
  volatile char v = ambient[PROBE_OFFSET];        /* no authority -> FAULT */

  *res = PROBE_RET_AMBIENT_NOTRAP | (unsigned char)v; /* unreached if it faults */
}
