// Phase 0, THE SCOPING PROPERTY: closing one connection revokes ONLY its
// subtree; a sibling connection and its child survive. This is what makes the
// cascade HIERARCHICAL rather than a global heap wipe.
//
//   hier_open(a); childA = malloc from A      -- connection A + its child
//   hier_open(b); childB = malloc from B      -- connection B + its child
//   hier_close(a)                             -- revoke A's subtree only
//   read childB                               -- B's child survives A's close
//   malloc a fresh child from B; write/read   -- B still fully usable
//
// A's sub-arena and B's sub-arena are independent SPLITs off the main arena, so
// A's senior node is not an ancestor of B's children. If hier_close(a) had
// revoked a node senior to B (or to the main arena), reading childB or
// allocating on B would fault.
//
// Expected: no fault, retval 0x0873003c (B's sentinel; ...00bb if B read wrong).
#include "hier_probe_domain.h"

void domain_main(void *arg, unsigned func) {
  if (hier_probe_receive(arg, func))
    return;

  unsigned *res = (unsigned *)arg;

  hier_init(hier_probe_grant);

  hconn a, b;
  volatile char *childA = (volatile char *)hier_probe_open_with_child(&a);
  volatile char *childB = (volatile char *)hier_probe_open_with_child(&b);
  childA[0] = (char)HIER_PROBE_SENTINEL_A;
  childB[0] = (char)HIER_PROBE_SENTINEL_B;

  hier_close(&a); /* revoke A's subtree only */

  volatile char vb = childB[0]; /* B's child survives A's close */

  hier_activate(&b);
  volatile char *childB2 = (volatile char *)rof_malloc(64); /* B still usable */
  hier_deactivate(&b);
  childB2[0] = (char)0x77;
  volatile char vb2 = childB2[0];

  if ((unsigned char)vb != HIER_PROBE_SENTINEL_B || (unsigned char)vb2 != 0x77) {
    *res = (HIER_RET_SIBLING_SURVIVES_OK & ~0xffu) | 0xbbu;
    return;
  }
  *res = (HIER_RET_SIBLING_SURVIVES_OK & ~0xffu) | (unsigned char)vb; /* 0x0873003c */
}
