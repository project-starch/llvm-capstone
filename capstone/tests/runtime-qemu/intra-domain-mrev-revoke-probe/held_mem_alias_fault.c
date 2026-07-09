// The cached alias lives in a memory-resident struct field across the REVOKE --
// the shape a stale SQLite handle (or a register-allocator spill) produces. The
// reload (ldc) must observe the revoked node and clear the tag.
//
// Expected: fault, cause 24 "Cap mem access requires capability" -- NOT 25.
// helper_reg_set_cap_compressed drops the tag when the reloaded capability's
// rev-tree node has been revoked, so the deref takes the untagged path rather
// than the revoked-node path.
//
// Cause 24 is ambiguous on its own (a cleared tag could also mean the capability
// was consumed before the deref). held_no_revoke_ok.c is its control: same
// receive + mrev + delin, no revoke, deref succeeds.
#include "probe_domain.h"

static struct {
  void *p;
  unsigned long generation;
} handle;

void domain_main(void *arg, unsigned func) {
  if (probe_receive(arg, func))
    return;

  unsigned *res = (unsigned *)arg;

  void *A = probe_arena;
  void *R = __builtin_capstone_cap_mrev(A);
  void *D = __builtin_capstone_cap_delin(A);

  handle.p = D; /* alias parked in memory (stc) */
  handle.generation = 1;
  *((volatile char *)D + PROBE_OFFSET) = (char)PROBE_SENTINEL_LIVE; /* live: ok */

  __builtin_capstone_cap_revoke(R);

  volatile char *buf = (volatile char *)handle.p; /* reload the alias (ldc) */
  volatile char v = buf[PROBE_OFFSET];            /* USE-AFTER-REVOKE -> FAULT */

  *res = PROBE_RET_MEM_NOTRAP | (unsigned char)v; /* unreached if it faults */
}
