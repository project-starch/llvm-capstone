// Phase 0 control for hier_child_revoked_fault: WITHOUT the parent close, the
// child alias reads back. Proves the fault was the parent's revoke, not the
// allocator handing out a dead capability, and disambiguates the -O0 cause-24
// fault (tag gone, which a plain spill reload also yields).
//
// Expected: no fault, retval 0x0872005e.
#include "hier_probe_domain.h"

void domain_main(void *arg, unsigned func) {
  if (hier_probe_receive(arg, func))
    return;

  unsigned *res = (unsigned *)arg;

  hier_init(hier_probe_grant);

  hconn a;
  volatile char *child = (volatile char *)hier_probe_open_with_child(&a);
  child[0] = (char)HIER_PROBE_SENTINEL_A;

  volatile char v = child[0]; /* no close: still live */

  *res = (HIER_RET_NO_CLOSE_OK & ~0xffu) | (unsigned char)v; /* 0x0872005e */
}
