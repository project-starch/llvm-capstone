// Phase 0, THE MECHANISM: a freed allocation's cached alias faults.
//
//   arena = <monitor grant>
//   a = rof_malloc(64); write/read a          -- allocation A, live
//   b = rof_malloc(64); write/read b          -- allocation B, live
//   rof_free(a)                               -- revoke A's node
//   read a                                    -- USE AFTER FREE -> must FAULT
//
// Each allocation is its own SPLIT sub-capability with its own revocation node,
// so freeing A revokes ONLY A. This is the allocator analogue of the held-cap
// mechanism, but the pointer that faults is one the allocator RETURNED, revoked
// by the allocator's own free -- no wrapper.
//
// Expected: FAULT. Cause is opt-level dependent, exactly as in the held-cap
// probe: -O0 spills the alias so the post-free reload clears the tag (cause 24);
// -O1/-O2 keep it in a register (cause 25, self-proving). alloc_no_free_ok is
// the control for the cause-24 case.
#include "rof_probe_domain.h"

void domain_main(void *arg, unsigned func) {
  if (rof_probe_receive(arg, func))
    return;

  unsigned *res = (unsigned *)arg;

  rof_init(rof_probe_grant);

  volatile char *a = (volatile char *)rof_malloc(64);
  volatile char *b = (volatile char *)rof_malloc(64);
  a[0] = (char)ROF_PROBE_SENTINEL_A;
  b[0] = (char)ROF_PROBE_SENTINEL_B;
  volatile char liveA = a[0];
  (void)liveA;

  rof_free((void *)a); /* revoke A only */

  volatile char v = a[0]; /* USE AFTER FREE -> FAULT */
  *res = ROF_RET_USE_AFTER_FREE_NOTRAP | (unsigned char)v; /* unreachable */
}
