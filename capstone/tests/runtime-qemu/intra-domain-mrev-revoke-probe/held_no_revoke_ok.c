// Control for held_revoke_fault.c: identical, except there is NO revoke.
//
// Without it, "the post-revoke deref faulted" would not be evidence that the
// REVOKE caused the fault. The delivery path itself could have been at fault:
// the arena is a LINEAR capability that crosses a domain-entry boundary through
// a .bss slot, and a linear capability passed by value is silently consumed
// (task-005, finding C3). This probe runs the whole receive / mrev / delin /
// deref sequence and dereferences the alias twice, proving the alias is live and
// the bytes are reachable when nothing revokes them.
//
// Expected: no fault, retval 0x2230005e.
#include "probe_domain.h"

void domain_main(void *arg, unsigned func) {
  if (probe_receive(arg, func))
    return;

  unsigned *res = (unsigned *)arg;

  void *A = probe_arena;
  void *R = __builtin_capstone_cap_mrev(A);
  void *D = __builtin_capstone_cap_delin(A);
  volatile char *buf = (volatile char *)D;

  buf[PROBE_OFFSET] = (char)PROBE_SENTINEL_LIVE;
  (void)R;                             /* the rev handle is minted, never fired */
  volatile char v = buf[PROBE_OFFSET]; /* second deref: still live */

  *res = PROBE_RET_NO_REVOKE_OK | (unsigned char)v; /* reached: 0x2230005e */
}
