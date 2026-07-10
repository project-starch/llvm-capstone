// Phase 0, THE MECHANISM: a child allocation faults when its PARENT connection
// is closed (its senior MREV handle revoked).
//
//   hier_open(a)            -- connection A: sub-arena SPLIT off main, MREV'd
//   child = rof_malloc(64)  -- carved from A's sub-arena: a SPLIT DESCENDANT of
//                              A's senior revocation node
//   write/read child        -- live
//   hier_close(a)           -- REVOKE A's senior node (parent teardown)
//   read child              -- USE AFTER PARENT CLOSE -> must FAULT
//
// This is the hierarchical cascade: destroying the parent revokes the child's
// authority even though nothing freed the child's memory (models SQLite's zombie
// close, which does not free a live statement). If REVOKE of the parent's senior
// node does NOT reach a child SPLIT off the sub-arena after the MREV, this probe
// RETURNS instead of faulting -- and that is the checkpoint's negative finding.
//
// Expected: FAULT. Cause is opt-level dependent (as in revoke-on-free-probe):
// -O0 spills the child alias so the post-revoke reload clears the tag (cause 24,
// with hier_no_close_ok as control); -O1/-O2 keep it register-held (cause 25,
// self-proving).
#include "hier_probe_domain.h"

void domain_main(void *arg, unsigned func) {
  if (hier_probe_receive(arg, func))
    return;

  unsigned *res = (unsigned *)arg;

  hier_init(hier_probe_grant);

  hconn a;
  volatile char *child = (volatile char *)hier_probe_open_with_child(&a);
  child[0] = (char)HIER_PROBE_SENTINEL_A;
  volatile char live = child[0];
  (void)live;

  hier_close(&a); /* revoke the parent connection's senior node */

  volatile char v = child[0]; /* USE AFTER PARENT CLOSE -> FAULT */
  *res = HIER_RET_CHILD_REVOKED_NOTRAP | (unsigned char)v; /* unreachable */
}
