// Phase 0 control for hier_two_handles_one_node: the same two handles on one node,
// and NEITHER is revoked. The alias must read back its sentinel.
//
// Without this, a fault in the probe could mean either that revoking the second parent
// reached the alias -- the finding -- or that taking two handles on one node broke the
// node outright, which would say nothing about revocation scope. That is the
// distinction the probe cannot make on its own.
//
// Expected: no fault, retval 0x0875005e.
#include "hier_probe_domain.h"
#include "../../../sublet/sublet.h"

#define HIER_RET_TWO_HANDLES_NO_GIVE 0x08750000u

void domain_main(void *arg, unsigned func) {
  if (hier_probe_receive(arg, func))
    return;

  unsigned *res = (unsigned *)arg;

  sublet_cap region, h1, h2;
  sublet_store(&region, hier_probe_grant);
  if (sublet_type(&region) != SUBLET_TYPE_LIN) {
    *res = HIER_RET_TWO_HANDLES_NO_GIVE | 0xFFu;
    return;
  }

  sublet_handle(&region, &h1);
  sublet_handle(&region, &h2);

  volatile char *p = (volatile char *)sublet_take(&region);
  p[0] = (char)HIER_PROBE_SENTINEL_A;

  volatile char after = p[0]; /* no revoke at all: must read back */
  *res = HIER_RET_TWO_HANDLES_NO_GIVE | (unsigned char)after;
}
